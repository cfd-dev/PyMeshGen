import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录添加到sys.path - DO THIS FIRST before any imports
sys.path.insert(0, project_root)  # Add project root first

# Now import modules using proper package structure
from fileIO.read_cas import parse_fluent_msh
from data_structure.front2d import construct_initial_front
from meshsize import QuadtreeSizing
from adfront2.adfront2 import Adfront2
from adfront2.adlayers2 import Adlayers2
from visualization.mesh_visualization import Visualization
from data_structure.parameters import Parameters
from optimize.optimize import edge_swap, laplacian_smooth
from utils.timer import TimeSpan
from utils.message import info, debug, verbose, gui_log, gui_progress

# 混合网格生成相关导入
from adfront2.adfront2_hybrid import Adfront2Hybrid
from optimize.optimize import optimize_hybrid_grid


def _cell_node_count(cell):
    if cell is None:
        return 0
    if hasattr(cell, "node_ids") and cell.node_ids is not None:
        return len(cell.node_ids)
    if hasattr(cell, "nodes") and cell.nodes is not None:
        return len(cell.nodes)
    if isinstance(cell, (list, tuple)):
        return len(cell)
    return 0


def _extract_node_ids(cell):
    if cell is None:
        return []
    if hasattr(cell, "node_ids") and cell.node_ids is not None:
        return list(cell.node_ids)
    if hasattr(cell, "nodes") and cell.nodes is not None:
        ids = []
        for node in cell.nodes:
            node_idx = getattr(node, "idx", None)
            if node_idx is not None:
                ids.append(node_idx)
        return ids
    if isinstance(cell, (list, tuple)):
        return list(cell)
    return []


def _iter_cells_for_analysis(mesh_data):
    if hasattr(mesh_data, "cell_container") and mesh_data.cell_container:
        return mesh_data.cell_container
    if hasattr(mesh_data, "cells") and mesh_data.cells:
        return mesh_data.cells
    return []


def _log_parameters_debug_summary(parameters):
    debug(
        f"[参数调试] debug_level={parameters.debug_level}, mesh_type={parameters.mesh_type}, "
        f"input={parameters.input_file}, output={parameters.output_file}"
    )
    debug(f"[参数调试] 部件数量: {len(parameters.part_params)}")
    for i, part in enumerate(parameters.part_params):
        part_cfg = getattr(part, "part_params", None)
        if part_cfg is None:
            debug(f"[参数调试] 部件{i+1}: {part.part_name} (无part_params)")
            continue
        debug(
            f"[参数调试] 部件{i+1}: {part.part_name}, "
            f"PRISM_SWITCH={part_cfg.PRISM_SWITCH}, "
            f"multi_direction={part_cfg.multi_direction}, "
            f"max_layers={part_cfg.max_layers}, "
            f"full_layers={part_cfg.full_layers}, "
            f"first_height={part_cfg.first_height}, "
            f"max_size={part_cfg.max_size}"
        )


def _log_mesh_debug_summary(mesh_data, summary_name):
    cells_for_analysis = _iter_cells_for_analysis(mesh_data)
    tri_count = 0
    quad_count = 0
    other_count = 0
    layer_stats = {}
    quad_samples = []

    for cell_idx, cell in enumerate(cells_for_analysis):
        n_nodes = _cell_node_count(cell)
        if n_nodes == 4:
            quad_count += 1
            if len(quad_samples) < 5:
                quad_samples.append((cell_idx, _extract_node_ids(cell)))
        elif n_nodes == 3:
            tri_count += 1
        else:
            other_count += 1

        layer = getattr(cell, "layer", None)
        if layer is None:
            continue
        if layer not in layer_stats:
            layer_stats[layer] = {"quads": 0, "tris": 0, "others": 0}
        if n_nodes == 4:
            layer_stats[layer]["quads"] += 1
        elif n_nodes == 3:
            layer_stats[layer]["tris"] += 1
        else:
            layer_stats[layer]["others"] += 1

    debug(
        f"[{summary_name}] 节点数量: {len(mesh_data.node_coords)}, "
        f"单元数量: {len(cells_for_analysis)}"
    )
    debug(
        f"[{summary_name}] 单元类型统计: "
        f"四边形={quad_count}, 三角形={tri_count}, 其他={other_count}"
    )

    for layer in sorted(layer_stats.keys()):
        stats = layer_stats[layer]
        debug(
            f"[{summary_name}] 第{layer}层: "
            f"{stats['quads']}个四边形 + {stats['tris']}个三角形 + {stats['others']}个其他"
        )

    if not quad_samples:
        return

    verbose(f"[{summary_name}] 四边形节点样本(最多5个):")
    for cell_idx, node_ids in quad_samples:
        verbose(f"[{summary_name}]   单元{cell_idx}: 节点{node_ids}")
        coord_text = []
        for node_id in node_ids:
            if not isinstance(node_id, int):
                continue
            if not (0 <= node_id < len(mesh_data.node_coords)):
                continue
            coord = mesh_data.node_coords[node_id]
            if len(coord) >= 2:
                coord_text.append(f"{node_id}:({coord[0]:.6f}, {coord[1]:.6f})")
            else:
                coord_text.append(f"{node_id}:{coord}")
        if coord_text:
            verbose(f"[{summary_name}]     坐标: {'; '.join(coord_text)}")


def generate_mesh(parameters, mesh_data=None, parts=None, gui_instance=None):
    """核心网格生成函数

    调用底层算法生成网格，支持多种调用方式：
    1. 通过parameters参数传递配置文件路径，从文件中读取网格数据
    2. 直接通过mesh_data参数传递网格数据
    3. 直接通过parts参数传递Part列表，直接使用part中的front_list

    Args:
        parameters: Parameters对象，包含网格生成的配置参数
        mesh_data: 可选，直接传入的网格数据对象，可以是Unstructured_Grid对象、字典或其他类型
        parts: 可选，直接传入的Part列表，优先级高于mesh_data
        gui_instance: 可选，GUI实例，用于在网格生成过程中输出信息和显示中间结果

    Returns:
        生成的网格数据
    """
    # 设置GUI实例到消息系统，确保所有info, error, warning等消息都能输出到GUI
    if gui_instance:
        from utils.message import set_gui_instance
        set_gui_instance(gui_instance)

    # 开始计时
    global_timer = TimeSpan("PyMeshGen开始运行...")

    # 建立可视化对象
    if gui_instance and hasattr(gui_instance, 'ax') and gui_instance.ax:
        # 在GUI模式下，传入GUI的绘图区域
        visual_obj = Visualization(parameters.viz_enabled, gui_instance.ax)
    else:
        # 在命令行模式下，不传入ax参数
        visual_obj = Visualization(parameters.viz_enabled)
    
    # 输出信息到GUI
    if gui_instance:
        mesh_type_str = "三角形/四边形混合网格" if parameters.mesh_type == 3 else "三角形网格"
        gui_log(gui_instance, f"开始生成{mesh_type_str}...")
        gui_progress(gui_instance, 0)  # 初始化参数

    # 读入边界网格
    gui_log(gui_instance, "开始读取输入网格数据...")
    gui_progress(gui_instance, 1)  # 开始读取输入网格数据

    input_grid = None
    front_heap = None
    
    # 优先级1: 直接使用parts参数中的front_list
    if parts is not None:
        info("使用直接传入的parts参数，直接提取front_list")
        from data_structure.basic_elements import Part
        if isinstance(parts, list) and len(parts) > 0:
            # 将传入的parts添加到parameters.part_params中，供adlayers2使用
            parameters.part_params = parts
            info(f"已将 {len(parts)} 个部件添加到parameters.part_params")
            
            front_heap = []
            front_idx = 0
            for part in parts:
                if hasattr(part, 'front_list'):
                    for front in part.front_list:
                        front.idx = front_idx
                        front_idx += 1
                        front_heap.append(front)
            import heapq
            heapq.heapify(front_heap)
            info(f"从parts中提取了 {len(front_heap)} 个阵面")
    
    # 优先级2: 使用直接传入的网格数据
    elif mesh_data is not None:
        info("使用直接传入的网格数据")
        from utils.data_converter import convert_to_internal_mesh_format
        input_grid = convert_to_internal_mesh_format(mesh_data)
    # 优先级3: 尝试从参数中获取文件路径
    elif parameters.input_file and isinstance(parameters.input_file, str) and os.path.exists(parameters.input_file):
        # 真实文件路径，正常解析
        input_grid = parse_fluent_msh(parameters.input_file)
    else:
        # 尝试从GUI获取当前网格数据
        if gui_instance and hasattr(gui_instance, 'current_mesh'):
            current_mesh = gui_instance.current_mesh
            info("使用GUI中的当前网格数据")
            from utils.data_converter import convert_to_internal_mesh_format
            input_grid = convert_to_internal_mesh_format(current_mesh)
        else:
            raise ValueError("无法获取有效的网格数据")

    # 如果front_heap已经从parts或connectors中提取，跳过construct_initial_front
    if front_heap is None and input_grid is not None:
        # If the input grid contains parts information, log it and update parameters
        if hasattr(input_grid, 'parts_info') and input_grid.parts_info:
            info(f"检测到 {len(input_grid.parts_info)} 个部件信息")
            for part_name in input_grid.parts_info.keys():
                info(f"  - 部件: {part_name}")

            # Update parameters based on mesh parts information
            parameters.update_part_params_from_mesh(input_grid)
        
        # 在GUI模式下清除之前的绘图内容
        if gui_instance and hasattr(gui_instance, 'ax') and gui_instance.ax:
            gui_instance.ax.clear()
            
        visual_obj.plot_mesh(input_grid, boundary_only=True)
        
        gui_log(gui_instance, f"已读取输入网格文件: {parameters.input_file}")

        # 构造初始阵面
        gui_log(gui_instance, "开始构造初始阵面...")
        gui_progress(gui_instance, 2)  # 开始构造初始阵面

        front_heap = construct_initial_front(input_grid)
        
        gui_log(gui_instance, "初始阵面构造完成")
    elif front_heap is not None:
        # front_heap已经从parts或connectors中提取
        gui_log(gui_instance, "直接使用parts/connectors中的front_list，跳过构造初始阵面")
        gui_progress(gui_instance, 2)  # 跳过构造初始阵面

    _log_parameters_debug_summary(parameters)

    # 计算网格尺寸场
    gui_log(gui_instance, "开始计算网格尺寸场...")
    gui_progress(gui_instance, 3)  # 开始计算网格尺寸场

    sizing_system = QuadtreeSizing(
        initial_front=front_heap,
        max_size=4,
        resolution=0.1,
        decay=1.2,
        visual_obj=visual_obj
    )
    # sizing_system.draw_bgmesh()
    
    gui_log(gui_instance, "网格尺寸场计算完成")

    unstr_grid_list = []
    # 推进生成边界层网格
    gui_log(gui_instance, "开始生成边界层网格...")
    gui_progress(gui_instance, 4)  # 开始生成边界层网格

    adlayers = Adlayers2(
        boundary_front=front_heap,
        sizing_system=sizing_system,
        param_obj=parameters,
        visual_obj=visual_obj,
    )
    boundary_grid, front_heap = adlayers.generate_elements()
    _log_mesh_debug_summary(boundary_grid, "边界层网格")
    unstr_grid_list.append(boundary_grid)
    
    gui_log(gui_instance, "边界层网格生成完成")

    # 推进生成网格
    gui_log(gui_instance, "开始推进生成网格...")
    gui_progress(gui_instance, 5)  # 开始推进生成网格

    # 根据网格类型选择不同的生成算法
    if parameters.mesh_type == 3:  # 三角形/四边形混合网格
        adfront2 = Adfront2Hybrid(
            boundary_front=front_heap,
            sizing_system=sizing_system,
            node_coords=boundary_grid.node_coords,
            param_obj=parameters,
            visual_obj=visual_obj,
        )
        triangular_grid = adfront2.generate_elements()
    else:  # 三角形网格
        adfront2 = Adfront2(
            boundary_front=front_heap,
            sizing_system=sizing_system,
            node_coords=boundary_grid.node_coords,
            param_obj=parameters,
            visual_obj=visual_obj,
        )
        triangular_grid = adfront2.generate_elements()
    
    gui_log(gui_instance, "网格生成完成")

    # 网格质量优化
    gui_log(gui_instance, "开始优化网格质量...")
    gui_progress(gui_instance, 6)  # 开始优化网格质量

    triangular_grid = edge_swap(triangular_grid)
    
    if parameters.mesh_type == 3:  # 三角形/四边形混合网格
        # 合并三角形生成混合网格
        hybrid_grid = triangular_grid.merge_elements(
            method=parameters.triangle_to_quad_method
        )
        # 优化混合网格
        hybrid_grid = optimize_hybrid_grid(hybrid_grid)
        unstr_grid_list.append(hybrid_grid)
    else:  # 三角形网格
        triangular_grid = laplacian_smooth(triangular_grid, 3)
        unstr_grid_list.append(triangular_grid)
    
    gui_log(gui_instance, "网格质量优化完成")

    # 合并各向同性网格和边界层网格
    gui_log(gui_instance, "开始合并网格...")
    gui_progress(gui_instance, 7)  # 开始合并网格

    global_unstr_grid = unstr_grid_list[0]

    # 确保原始网格的parts_info被保留（包含原始边界部件信息）
    if hasattr(input_grid, 'parts_info') and input_grid.parts_info:
        global_unstr_grid.parts_info = input_grid.parts_info

    for unstr_grid in unstr_grid_list[1:]:
        global_unstr_grid.merge(unstr_grid)

    _log_mesh_debug_summary(global_unstr_grid, "最终网格")
    
    gui_log(gui_instance, "网格合并完成")

    # 可视化
    global_unstr_grid.visualize_unstr_grid_2d(visual_obj)
    
    # 在GUI模式下更新画布
    if gui_instance and hasattr(gui_instance, 'canvas') and gui_instance.canvas:
        gui_instance.canvas.draw()

    # 输出网格信息
    global_unstr_grid.summary(gui_instance)
    # global_unstr_grid.quality_histogram(gui_instance.ax if gui_instance else None)
    
    gui_log(gui_instance, "网格信息输出完成")

    # 输出网格文件
    # 对于非GUI运行，默认要输出网格
    should_save = parameters.auto_output or gui_instance is None
    
    if should_save:
        # 如果输出路径为空，设置默认路径
        output_path = parameters.output_file
        # 处理output_path可能是列表的情况
        if isinstance(output_path, list):
            if not output_path or not output_path[0]:
                output_path = ["./out/mesh.vtk"]
            else:
                output_path = output_path[0]
        else:
            if not output_path:
                output_path = ["./out/mesh.vtk"]
        
        gui_log(gui_instance, "开始保存网格文件...")
        gui_progress(gui_instance, 8)  # 开始保存网格文件

        global_unstr_grid.save_to_vtkfile(output_path)
        
        if gui_instance:
            gui_log(gui_instance, f"网格文件已保存至: {output_path}")
        else:
            # 非GUI运行时，输出到控制台
            print(f"网格文件已保存至: {output_path}")
    else:
        gui_log(gui_instance, "自动输出网格已禁用，未保存网格文件")

    # 保留原始部件信息以便后续修改部件参数
    if gui_instance:
        # 将优化后的网格对象设置到GUI实例中
        gui_instance.mesh_data = global_unstr_grid
        # 保存原始参数配置，以便后续可以重新配置部件参数
        gui_instance.original_parameters = parameters

    # 结束计时
    global_timer.show_to_console("程序运行正常退出.")

    # 输出信息到GUI
    gui_log(gui_instance, "程序运行正常退出")

    # 重置消息系统中的GUI实例，避免影响后续操作
    if gui_instance:
        from utils.message import set_gui_instance
        set_gui_instance(None)
    
    return global_unstr_grid

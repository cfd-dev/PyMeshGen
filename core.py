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
from utils.message import info

# 混合网格生成相关导入
from adfront2.adfront2_hybrid import Adfront2Hybrid
from optimize.optimize import optimize_hybrid_grid


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

    def _gui_log(message):
        if gui_instance:
            gui_instance.append_info_output(message)

    def _gui_progress(step):
        if gui_instance and hasattr(gui_instance, '_update_progress'):
            gui_instance._update_progress(step)

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
        _gui_log(f"开始生成{mesh_type_str}...")
        _gui_progress(0)  # 初始化参数

    # 读入边界网格
    _gui_log("开始读取输入网格数据...")
    _gui_progress(1)  # 开始读取输入网格数据

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
        
        _gui_log(f"已读取输入网格文件: {parameters.input_file}")

        # 构造初始阵面
        _gui_log("开始构造初始阵面...")
        _gui_progress(2)  # 开始构造初始阵面

        front_heap = construct_initial_front(input_grid)
        
        _gui_log("初始阵面构造完成")
    elif front_heap is not None:
        # front_heap已经从parts或connectors中提取
        _gui_log("直接使用parts/connectors中的front_list，跳过构造初始阵面")
        _gui_progress(2)  # 跳过构造初始阵面

    # 计算网格尺寸场
    _gui_log("开始计算网格尺寸场...")
    _gui_progress(3)  # 开始计算网格尺寸场

    sizing_system = QuadtreeSizing(
        initial_front=front_heap,
        max_size=4,
        resolution=0.1,
        decay=1.2,
        visual_obj=visual_obj
    )
    # sizing_system.draw_bgmesh()
    
    _gui_log("网格尺寸场计算完成")

    unstr_grid_list = []
    # 推进生成边界层网格
    _gui_log("开始生成边界层网格...")
    _gui_progress(4)  # 开始生成边界层网格

    adlayers = Adlayers2(
        boundary_front=front_heap,
        sizing_system=sizing_system,
        param_obj=parameters,
        visual_obj=visual_obj,
    )
    boundary_grid, front_heap = adlayers.generate_elements()
    unstr_grid_list.append(boundary_grid)
    
    _gui_log("边界层网格生成完成")

    # 推进生成网格
    _gui_log("开始推进生成网格...")
    _gui_progress(5)  # 开始推进生成网格

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
    
    _gui_log("网格生成完成")

    # 网格质量优化
    _gui_log("开始优化网格质量...")
    _gui_progress(6)  # 开始优化网格质量

    triangular_grid = edge_swap(triangular_grid)
    
    if parameters.mesh_type == 3:  # 三角形/四边形混合网格
        # 合并三角形生成混合网格
        hybrid_grid = triangular_grid.merge_elements()
        # 优化混合网格
        hybrid_grid = optimize_hybrid_grid(hybrid_grid)
        unstr_grid_list.append(hybrid_grid)
    else:  # 三角形网格
        triangular_grid = laplacian_smooth(triangular_grid, 3)
        unstr_grid_list.append(triangular_grid)
    
    _gui_log("网格质量优化完成")

    # 合并各向同性网格和边界层网格
    _gui_log("开始合并网格...")
    _gui_progress(7)  # 开始合并网格

    global_unstr_grid = unstr_grid_list[0]

    # 确保原始网格的parts_info被保留（包含原始边界部件信息）
    if hasattr(input_grid, 'parts_info') and input_grid.parts_info:
        global_unstr_grid.parts_info = input_grid.parts_info

    for unstr_grid in unstr_grid_list[1:]:
        global_unstr_grid.merge(unstr_grid)
    
    _gui_log("网格合并完成")

    # 可视化
    global_unstr_grid.visualize_unstr_grid_2d(visual_obj)
    
    # 在GUI模式下更新画布
    if gui_instance and hasattr(gui_instance, 'canvas') and gui_instance.canvas:
        gui_instance.canvas.draw()

    # 输出网格信息
    global_unstr_grid.summary(gui_instance)
    # global_unstr_grid.quality_histogram(gui_instance.ax if gui_instance else None)
    
    _gui_log("网格信息输出完成")

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
        
        _gui_log("开始保存网格文件...")
        _gui_progress(8)  # 开始保存网格文件

        global_unstr_grid.save_to_vtkfile(output_path)
        
        _gui_log(f"网格文件已保存至: {output_path}")
        else:
            # 非GUI运行时，输出到控制台
            print(f"网格文件已保存至: {output_path}")
    else:
        _gui_log("自动输出网格已禁用，未保存网格文件")

    # 保留原始部件信息以便后续修改部件参数
    if gui_instance:
        # 将优化后的网格对象设置到GUI实例中
        gui_instance.mesh_data = global_unstr_grid
        # 保存原始参数配置，以便后续可以重新配置部件参数
        gui_instance.original_parameters = parameters

    # 结束计时
    global_timer.show_to_console("程序运行正常退出.")

    # 输出信息到GUI
    _gui_log("程序运行正常退出")

    # 重置消息系统中的GUI实例，避免影响后续操作
    if gui_instance:
        from utils.message import set_gui_instance
        set_gui_instance(None)
    
    return global_unstr_grid

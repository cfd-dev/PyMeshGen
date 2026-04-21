import sys
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录添加到sys.path - DO THIS FIRST before any imports
sys.path.insert(0, project_root)  # Add project root first

# Now import modules using proper package structure
from meshsize import QuadtreeSizing
from adfront2.adlayers2 import Adlayers2
from delaunay import create_bowyer_watson_mesh
from data_structure.parameters import Parameters
from optimize.optimize import edge_swap, edge_collapse, laplacian_smooth
from utils.timer import TimeSpan
from utils.message import info, gui_log, gui_progress

from optimize.optimize import optimize_hybrid_grid
from utils.core_helpers import (
    is_mixed_mesh,
    use_triangle_pipeline_for_qmorph,
    select_delaunay_backend,
    create_interior_generator,
    log_parameters_debug_summary,
    log_mesh_debug_summary,
)
from utils.core_io import (
    create_visualization,
    resolve_input_grid_and_front,
    merge_generated_grids,
    output_and_finalize,
)
from delaunay.postprocess import (
    collect_boundary_edges_from_fronts,
    is_topology_valid,
    recover_boundary_edges_by_swaps,
)
from utils.mesh_utils import deduplicate_grid_cells


def generate_mesh(parameters, mesh_data=None, parts=None, gui_instance=None):
    """核心网格生成流程（统一入口）。

    支持三类输入来源（优先级从高到低）：
    1) `parts`：直接使用外部构造好的front_list；
    2) `mesh_data`：直接使用外部网格对象；
    3) `parameters.input_file`：从算例文件读取。

    网格类型与流程分支：
    - 三角网格（mesh_type == 1 或 2）：
      边界层 -> 三角推进 -> edge swap -> laplacian -> 合并输出。
    - Bowyer-Watson 三角网格（mesh_type == 4）：
      边界层 -> Bowyer-Watson Delaunay 三角剖分 -> edge swap -> laplacian -> 合并输出。
    - 混合网格（mesh_type == 3, 非q_morph）：
      边界层 -> 混合推进(Adfront2Hybrid) -> edge swap -> 三角转四边形 -> 混合优化 -> 合并输出。
    - 混合网格（mesh_type == 3, q_morph）：
      边界层 -> 纯三角推进(Adfront2) -> edge swap -> laplacian(预平滑) -> q_morph合并 -> 混合优化 -> 合并输出。
    """
    # ------------------------------------------------------------------
    # 0) 运行上下文初始化（GUI消息路由、计时器、可视化对象）
    # ------------------------------------------------------------------
    # 设置GUI实例到消息系统，确保所有info/error/warning消息都能输出到GUI
    if gui_instance:
        from utils.message import set_gui_instance
        set_gui_instance(gui_instance)

    # 开始计时
    global_timer = TimeSpan("PyMeshGen开始运行...")

    visual_obj = create_visualization(parameters, gui_instance)
    
    # 输出信息到GUI
    if gui_instance:
        mesh_type_str = (
            "三角形/四边形混合网格"
            if is_mixed_mesh(parameters.mesh_type)
            else "三角形网格"
        )
        gui_log(gui_instance, f"开始生成{mesh_type_str}...")
        gui_progress(gui_instance, 0)  # 初始化参数

    # ------------------------------------------------------------------
    # 1) 输入解析与初始阵面构造（由utils.core_io统一处理）
    # ------------------------------------------------------------------
    input_grid, front_heap = resolve_input_grid_and_front(
        parameters=parameters,
        mesh_data=mesh_data,
        parts=parts,
        gui_instance=gui_instance,
        visual_obj=visual_obj,
    )

    log_parameters_debug_summary(parameters)

    # ------------------------------------------------------------------
    # 2) 构建尺寸场（QuadtreeSizing）
    # ------------------------------------------------------------------
    gui_log(gui_instance, "开始计算网格尺寸场...")
    gui_progress(gui_instance, 3)  # 开始计算网格尺寸场

    sizing_system = QuadtreeSizing(
        initial_front=front_heap,
        max_size=4,
        resolution=0.1,
        decay=getattr(parameters, 'sizing_decay', 1.2),
        visual_obj=visual_obj
    )
    # sizing_system.draw_bgmesh()
    
    gui_log(gui_instance, "网格尺寸场计算完成")

    # ------------------------------------------------------------------
    # 3) 生成边界层网格（Adlayers2）
    #    - Bowyer-Watson 模式（mesh_type == 4）也支持边界层
    # ------------------------------------------------------------------
    unstr_grid_list = []
    
    # 检查是否开启了边界层（任何 part 的 PRISM_SWITCH 不是 'off'）
    # part_params 中存储的是 Part 对象，Part 对象有 part_params 属性（MeshParameters）
    has_boundary_layer = False
    for part_obj in parameters.part_params:
        # Part 对象有 part_params 属性
        if hasattr(part_obj, 'part_params'):
            mesh_param = part_obj.part_params
            if hasattr(mesh_param, 'PRISM_SWITCH') and mesh_param.PRISM_SWITCH != 'off':
                has_boundary_layer = True
                break
    
    if has_boundary_layer:
        gui_log(gui_instance, "开始生成边界层网格...")
        gui_progress(gui_instance, 4)  # 开始生成边界层网格

        adlayers = Adlayers2(
            boundary_front=front_heap,
            sizing_system=sizing_system,
            param_obj=parameters,
            visual_obj=visual_obj,
        )
        boundary_grid, front_heap = adlayers.generate_elements()
        log_mesh_debug_summary(boundary_grid, "边界层网格")
        unstr_grid_list.append(boundary_grid)

        gui_log(gui_instance, "边界层网格生成完成")
    else:
        if parameters.mesh_type == 4:
            info("Bowyer-Watson 模式：无边界层配置，跳过边界层生成")
        boundary_grid = None
        gui_log(gui_instance, "无边界层配置")

    # ------------------------------------------------------------------
    # 4) 生成内层网格
    #    - Bowyer-Watson：mesh_type == 4
    #    - 普通混合：Adfront2Hybrid
    #    - q_morph混合：Adfront2（先纯三角）
    #    - 三角网格：Adfront2
    # ------------------------------------------------------------------
    gui_log(gui_instance, "开始推进生成网格...")
    gui_progress(gui_instance, 5)  # 开始推进生成网格

    # Bowyer-Watson Delaunay 网格生成分支
    if parameters.mesh_type == 4:
        configured_backend = str(
            getattr(parameters, "delaunay_backend", "bowyer_watson")
        ).strip().lower()
        delaunay_backend = select_delaunay_backend(
            parameters,
            has_boundary_layer=has_boundary_layer,
        )
        if has_boundary_layer and delaunay_backend != configured_backend:
            info(
                "Bowyer-Watson 模式：检测到边界层，"
                f"内层三角剖分从 {configured_backend} 切换为 triangle 后端"
            )
        else:
            info(
                f"Bowyer-Watson 模式：使用 {delaunay_backend} Delaunay 三角剖分生成网格"
            )
        
        from data_structure.unstructured_grid import Unstructured_Grid
        from data_structure.basic_elements import Triangle as BWTriangle, NodeElement

        # 调用 Delaunay 后端
        # 启用自动孔洞检测以正确处理内边界（如翼型内部固体区域）
        points, simplices, boundary_mask = create_bowyer_watson_mesh(
            boundary_front=front_heap,
            sizing_system=sizing_system,
            target_triangle_count=None,  # 可选：指定目标三角形数量
            smoothing_iterations=3,
            auto_detect_holes=True,  # 启用自动孔洞检测
            backend=delaunay_backend,
            triangle_point_strategy=getattr(
                parameters,
                "triangle_point_strategy",
                "equilateral",
            ),
        )

        if delaunay_backend != "triangle":
            boundary_edges = collect_boundary_edges_from_fronts(front_heap)
            swapped_simplices, recovered_edges, remaining_edges = recover_boundary_edges_by_swaps(
                points, simplices, boundary_edges
            )
            if recovered_edges > 0:
                if is_topology_valid(points, swapped_simplices):
                    simplices = swapped_simplices
                    info(f"Bowyer-Watson 模式：通过边翻转恢复了 {recovered_edges} 条边界约束边")
                else:
                    info("Bowyer-Watson 模式：边翻转恢复引入拓扑异常，已回退到原始三角剖分结果")
            if remaining_edges:
                info(f"Bowyer-Watson 模式：仍有 {len(remaining_edges)} 条边界约束边未直接恢复")
        
        # 将结果转换为 Unstructured_Grid 对象
        node_coords = points.tolist()
        cell_container = []
        boundary_nodes = set()
        
        for i, simplex in enumerate(simplices):
            # 创建 NodeElement 对象
            nodes = []
            for node_idx in simplex:
                node = NodeElement(
                    coords=node_coords[node_idx],
                    idx=int(node_idx),
                    part_name="interior-node",
                    bc_type="boundary" if boundary_mask[node_idx] else "interior",
                )
                nodes.append(node)
                if boundary_mask[node_idx]:
                    boundary_nodes.add(node)
            
            # 创建 Triangle 对象
            triangle = BWTriangle(
                p1=nodes[0],
                p2=nodes[1],
                p3=nodes[2],
                part_name="interior-triangle",
                idx=i,
            )
            cell_container.append(triangle)
        
        triangular_grid = Unstructured_Grid(
            cell_container=cell_container,
            node_coords=node_coords,
            boundary_nodes=boundary_nodes,
        )
        
        gui_log(gui_instance, "Bowyer-Watson 网格生成完成")
    else:
        if use_triangle_pipeline_for_qmorph(parameters):
            info("q-morph模式：先生成纯三角形网格，再进行q-morph四边形合并")

        adfront2 = create_interior_generator(
            parameters=parameters,
            front_heap=front_heap,
            sizing_system=sizing_system,
            boundary_grid=boundary_grid,
            visual_obj=visual_obj,
        )
        triangular_grid = adfront2.generate_elements()

        gui_log(gui_instance, "网格生成完成")

    # ------------------------------------------------------------------
    # 5) 网格质量优化与单元类型后处理
    #    - Bowyer-Watson：默认跳过后续优化（已经是 Delaunay 最优网格）
    #    - Triangle 后端(mesh_type == 4)：允许轻量 laplacian 平滑
    #    - 公共：edge_swap
    #    - q_morph混合：先laplacian预平滑，再q_morph合并
    #    - 非q_morph混合：直接按配置方法合并（默认greedy）
    #    - 三角网格：laplacian
    # ------------------------------------------------------------------
    gui_log(gui_instance, "开始优化网格质量...")
    gui_progress(gui_instance, 6)  # 开始优化网格质量

    if parameters.mesh_type != 4:
        triangular_grid = edge_swap(triangular_grid)
        removed_duplicates = deduplicate_grid_cells(triangular_grid)
        if removed_duplicates > 0:
            info(f"优化后去重: 删除了 {removed_duplicates} 个重复单元")
        triangular_grid = edge_collapse(triangular_grid)

    # triangular_grid.save_to_vtkfile("./out/debug.vtk")
    
    if parameters.mesh_type == 4 and delaunay_backend == "triangle":
        triangular_grid = laplacian_smooth(triangular_grid, 3)
        info("Triangle 后端：已执行 3 轮 laplacian 平滑")

    # Q-morph算法执行前先确保三角形网格质量
    if use_triangle_pipeline_for_qmorph(parameters):
        triangular_grid = laplacian_smooth(triangular_grid, 3)

    if is_mixed_mesh(parameters.mesh_type):  # 三角形/四边形混合网格
        # 合并三角形生成混合网格
        hybrid_grid = triangular_grid.merge_elements(
            method=parameters.triangle_to_quad_method
        )
        # 优化混合网格
        hybrid_grid = optimize_hybrid_grid(
            hybrid_grid, 
            use_angle_based=True,  # 设置为 True 启用角度优化
            angle_iterations=3
        )
        unstr_grid_list.append(hybrid_grid)
    else:  # 三角形网格
        if parameters.mesh_type != 4:
            triangular_grid = laplacian_smooth(triangular_grid, 3)
            removed_duplicates = deduplicate_grid_cells(triangular_grid)
            if removed_duplicates > 0:
                info(f"平滑后去重: 删除了 {removed_duplicates} 个重复单元")
        unstr_grid_list.append(triangular_grid)
    
    gui_log(gui_instance, "网格质量优化完成")

    # ------------------------------------------------------------------
    # 6) 合并边界层与内层网格
    # ------------------------------------------------------------------
    gui_log(gui_instance, "开始合并网格...")
    gui_progress(gui_instance, 7)  # 开始合并网格

    global_unstr_grid = merge_generated_grids(unstr_grid_list, input_grid)

    log_mesh_debug_summary(global_unstr_grid, "最终网格")
    
    gui_log(gui_instance, "网格合并完成")

    # ------------------------------------------------------------------
    # 7) 输出与收尾（可视化、保存、GUI状态回填、计时结束）
    # ------------------------------------------------------------------
    return output_and_finalize(
        global_unstr_grid=global_unstr_grid,
        parameters=parameters,
        gui_instance=gui_instance,
        visual_obj=visual_obj,
        global_timer=global_timer,
    )

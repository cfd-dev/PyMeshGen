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


def _has_boundary_layer(parameters):
    """Return True when any part enables prism-layer growth."""
    for part_obj in parameters.part_params:
        mesh_param = getattr(part_obj, "part_params", None)
        if getattr(mesh_param, "PRISM_SWITCH", "off") != "off":
            return True
    return False


def _generate_boundary_layer_grid(parameters, front_heap, sizing_system, visual_obj, gui_instance):
    """Generate the boundary-layer grid when enabled."""
    has_boundary_layer = _has_boundary_layer(parameters)
    if not has_boundary_layer:
        if parameters.mesh_type == 4:
            info("Bowyer-Watson 模式：无边界层配置，跳过边界层生成")
        gui_log(gui_instance, "无边界层配置")
        return None, front_heap, False

    gui_log(gui_instance, "开始生成边界层网格...")
    gui_progress(gui_instance, 4)

    adlayers = Adlayers2(
        boundary_front=front_heap,
        sizing_system=sizing_system,
        param_obj=parameters,
        visual_obj=visual_obj,
    )
    boundary_grid, updated_front_heap = adlayers.generate_elements()
    log_mesh_debug_summary(boundary_grid, "边界层网格")
    gui_log(gui_instance, "边界层网格生成完成")
    return boundary_grid, updated_front_heap, True


def _resolve_delaunay_backend(parameters, has_boundary_layer):
    """Select and log the effective mesh_type=4 Delaunay backend."""
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
    return delaunay_backend


def _recover_delaunay_boundary_edges(points, simplices, front_heap):
    """Run the lightweight swap-based boundary recovery used after BW meshing."""
    boundary_edges = collect_boundary_edges_from_fronts(front_heap)
    swapped_simplices, recovered_edges, remaining_edges = recover_boundary_edges_by_swaps(
        points,
        simplices,
        boundary_edges,
    )
    if recovered_edges > 0:
        if is_topology_valid(points, swapped_simplices):
            simplices = swapped_simplices
            info(f"Bowyer-Watson 模式：通过边翻转恢复了 {recovered_edges} 条边界约束边")
        else:
            info("Bowyer-Watson 模式：边翻转恢复引入拓扑异常，已回退到原始三角剖分结果")
    if remaining_edges:
        info(f"Bowyer-Watson 模式：仍有 {len(remaining_edges)} 条边界约束边未直接恢复")
    return simplices


def _build_delaunay_unstructured_grid(points, simplices, boundary_mask):
    """Convert raw triangulation arrays into an Unstructured_Grid."""
    from data_structure.unstructured_grid import Unstructured_Grid

    boundary_nodes_idx = [
        idx for idx, is_boundary in enumerate(boundary_mask) if is_boundary
    ]
    return Unstructured_Grid.from_cells(
        node_coords=points.tolist(),
        cells=simplices.tolist(),
        boundary_nodes_idx=boundary_nodes_idx,
        grid_dimension=2,
    )


def _generate_delaunay_triangular_grid(
    parameters,
    front_heap,
    sizing_system,
    has_boundary_layer,
    gui_instance,
):
    """Generate the mesh_type=4 interior triangular grid."""
    delaunay_backend = _resolve_delaunay_backend(
        parameters,
        has_boundary_layer=has_boundary_layer,
    )
    points, simplices, boundary_mask = create_bowyer_watson_mesh(
        boundary_front=front_heap,
        sizing_system=sizing_system,
        target_triangle_count=None,
        smoothing_iterations=3,
        auto_detect_holes=True,
        backend=delaunay_backend,
        triangle_point_strategy=getattr(
            parameters,
            "triangle_point_strategy",
            "equilateral",
        ),
    )

    if delaunay_backend != "triangle":
        simplices = _recover_delaunay_boundary_edges(points, simplices, front_heap)

    triangular_grid = _build_delaunay_unstructured_grid(
        points,
        simplices,
        boundary_mask,
    )
    gui_log(gui_instance, "Bowyer-Watson 网格生成完成")
    return triangular_grid, delaunay_backend


def _generate_advancing_front_grid(
    parameters,
    front_heap,
    sizing_system,
    boundary_grid,
    visual_obj,
    gui_instance,
):
    """Generate the non-Delaunay interior grid path."""
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
    return triangular_grid


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
    boundary_grid, front_heap, has_boundary_layer = _generate_boundary_layer_grid(
        parameters=parameters,
        front_heap=front_heap,
        sizing_system=sizing_system,
        visual_obj=visual_obj,
        gui_instance=gui_instance,
    )
    if boundary_grid is not None:
        unstr_grid_list.append(boundary_grid)

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
        triangular_grid, delaunay_backend = _generate_delaunay_triangular_grid(
            parameters,
            sizing_system=sizing_system,
            front_heap=front_heap,
            has_boundary_layer=has_boundary_layer,
            gui_instance=gui_instance,
        )
    else:
        triangular_grid = _generate_advancing_front_grid(
            parameters=parameters,
            front_heap=front_heap,
            sizing_system=sizing_system,
            boundary_grid=boundary_grid,
            visual_obj=visual_obj,
            gui_instance=gui_instance,
        )

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

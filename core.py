import sys
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录添加到sys.path - DO THIS FIRST before any imports
sys.path.insert(0, project_root)  # Add project root first

# Now import modules using proper package structure
from meshsize import QuadtreeSizing
from adfront2.adlayers2 import Adlayers2
from data_structure.parameters import Parameters
from optimize.optimize import edge_swap, laplacian_smooth
from utils.timer import TimeSpan
from utils.message import info, gui_log, gui_progress

from optimize.optimize import optimize_hybrid_grid
from utils.core_helpers import (
    is_mixed_mesh,
    use_triangle_pipeline_for_qmorph,
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


def generate_mesh(parameters, mesh_data=None, parts=None, gui_instance=None):
    """核心网格生成流程（统一入口）。

    支持三类输入来源（优先级从高到低）：
    1) `parts`：直接使用外部构造好的front_list；
    2) `mesh_data`：直接使用外部网格对象；
    3) `parameters.input_file`：从算例文件读取。

    网格类型与流程分支：
    - 三角网格（mesh_type != 3）：
      边界层 -> 三角推进 -> edge swap -> laplacian -> 合并输出。
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
        decay=1.2,
        visual_obj=visual_obj
    )
    # sizing_system.draw_bgmesh()
    
    gui_log(gui_instance, "网格尺寸场计算完成")

    # ------------------------------------------------------------------
    # 3) 生成边界层网格（Adlayers2）
    # ------------------------------------------------------------------
    unstr_grid_list = []
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

    # ------------------------------------------------------------------
    # 4) 生成内层网格
    #    - 普通混合：Adfront2Hybrid
    #    - q_morph混合：Adfront2（先纯三角）
    #    - 三角网格：Adfront2
    # ------------------------------------------------------------------
    gui_log(gui_instance, "开始推进生成网格...")
    gui_progress(gui_instance, 5)  # 开始推进生成网格

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
    #    公共：edge_swap
    #    q_morph混合：先laplacian预平滑，再q_morph合并
    #    非q_morph混合：直接按配置方法合并（默认greedy）
    #    三角网格：laplacian
    # ------------------------------------------------------------------
    gui_log(gui_instance, "开始优化网格质量...")
    gui_progress(gui_instance, 6)  # 开始优化网格质量

    triangular_grid = edge_swap(triangular_grid)

    # triangular_grid.save_to_vtkfile("./out/debug.vtk")
    
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
        triangular_grid = laplacian_smooth(triangular_grid, 3)
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

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

    input_grid, front_heap = resolve_input_grid_and_front(
        parameters=parameters,
        mesh_data=mesh_data,
        parts=parts,
        gui_instance=gui_instance,
        visual_obj=visual_obj,
    )

    log_parameters_debug_summary(parameters)

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
    log_mesh_debug_summary(boundary_grid, "边界层网格")
    unstr_grid_list.append(boundary_grid)
    
    gui_log(gui_instance, "边界层网格生成完成")

    # 推进生成网格
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

    # 网格质量优化
    gui_log(gui_instance, "开始优化网格质量...")
    gui_progress(gui_instance, 6)  # 开始优化网格质量

    triangular_grid = edge_swap(triangular_grid)

    # Q-morph算法执行前先确保三角形网格质量
    if use_triangle_pipeline_for_qmorph(parameters):
        triangular_grid = laplacian_smooth(triangular_grid, 3)

    if is_mixed_mesh(parameters.mesh_type):  # 三角形/四边形混合网格
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

    global_unstr_grid = merge_generated_grids(unstr_grid_list, input_grid)

    log_mesh_debug_summary(global_unstr_grid, "最终网格")
    
    gui_log(gui_instance, "网格合并完成")

    return output_and_finalize(
        global_unstr_grid=global_unstr_grid,
        parameters=parameters,
        gui_instance=gui_instance,
        visual_obj=visual_obj,
        global_timer=global_timer,
    )

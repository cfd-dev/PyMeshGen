import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "fileIO"))
sys.path.append(str(Path(__file__).parent / "data_structure"))
sys.path.append(str(Path(__file__).parent / "meshsize"))
sys.path.append(str(Path(__file__).parent / "visualization"))
sys.path.append(str(Path(__file__).parent / "adfront2"))
sys.path.append(str(Path(__file__).parent / "optimize"))
sys.path.append(str(Path(__file__).parent / "utils"))

from read_cas import parse_fluent_msh
from front2d import construct_initial_front
from meshsize import QuadtreeSizing
from adfront2 import Adfront2
from adfront2_hybrid import Adfront2Hybrid
from optimize import (
    edge_swap,
    laplacian_smooth,
    merge_elements,
    hybrid_smooth,
    optimize_hybrid_grid,
)
from adlayers2 import Adlayers2
from mesh_visualization import Visualization
from parameters import Parameters
from utils.timer import TimeSpan
from utils.message import info

# 全局GUI引用
_global_gui_instance = None


def set_gui_instance(gui_instance):
    """设置全局GUI实例"""
    global _global_gui_instance
    _global_gui_instance = gui_instance


def PyMeshGen_mixed(parameters=None):
    # 开始计时
    global_timer = TimeSpan("PyMeshGen开始运行...")

    # 建立参数管理对象
    # if parameters is None:
    #     parameters = Parameters("FROM_MAIN_JSON")

    # 建立可视化对象
    visual_obj = Visualization(parameters.viz_enabled)
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output("开始生成混合网格...")

    # 读入边界网格
    input_grid = parse_fluent_msh(parameters.input_file)
    
    # 在GUI模式下清除之前的绘图内容
    if _global_gui_instance and hasattr(_global_gui_instance, 'ax') and _global_gui_instance.ax:
        _global_gui_instance.ax.clear()
        
    visual_obj.plot_mesh(input_grid, boundary_only=True)
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output(f"已读取输入网格文件: {parameters.input_file}")

    # 构造初始阵面
    front_heap = construct_initial_front(input_grid)
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output("初始阵面构造完成")

    # 计算网格尺寸场
    sizing_system = QuadtreeSizing(
        initial_front=front_heap,
        max_size=4,
        resolution=0.1,
        decay=1.2,
        visual_obj=visual_obj,
    )
    # sizing_system.draw_bgmesh()
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output("网格尺寸场计算完成")

    unstr_grid_list = []
    # 推进生成边界层网格
    adlayers = Adlayers2(
        boundary_front=front_heap,
        sizing_system=sizing_system,
        param_obj=parameters,
        visual_obj=visual_obj,
    )
    # Adlayers2的generate_elements总是返回两个值
    boundary_grid, front_heap = adlayers.generate_elements()
    unstr_grid_list.append(boundary_grid)
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output("边界层网格生成完成")

    # 推进生成网格
    if parameters.mesh_type <= 2:
        adfront2 = Adfront2(
            boundary_front=front_heap,
            sizing_system=sizing_system,
            node_coords=boundary_grid.node_coords,
            param_obj=parameters,
            visual_obj=visual_obj,
        )
        triangular_grid = adfront2.generate_elements()
    elif parameters.mesh_type == 3:
        adfront2 = Adfront2Hybrid(
            boundary_front=front_heap,
            sizing_system=sizing_system,
            node_coords=boundary_grid.node_coords,
            param_obj=parameters,
            visual_obj=visual_obj,
        )
        triangular_grid = adfront2.generate_elements()
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output("网格生成完成")

    # 网格质量优化
    if parameters.mesh_type <= 2:
        # 仅对纯三角形网格进行边交换优化
        triangular_grid = edge_swap(triangular_grid)
        triangular_grid = laplacian_smooth(triangular_grid, 3)
        unstr_grid_list.append(triangular_grid)
    elif parameters.mesh_type == 3:
        # 混合网格不需要边交换优化，直接进行混合网格优化
        hybrid_grid = merge_elements(triangular_grid)
        # hybrid_grid = hybrid_smooth(hybrid_grid, 3)
        hybrid_grid = optimize_hybrid_grid(hybrid_grid)
        unstr_grid_list.append(hybrid_grid)
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output("网格质量优化完成")

    # 合并各向同性网格和边界层网格
    global_unstr_grid = unstr_grid_list[0]
    for unstr_grid in unstr_grid_list[1:]:
        global_unstr_grid.merge(unstr_grid)
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output("网格合并完成")

    # 可视化
    global_unstr_grid.visualize_unstr_grid_2d(visual_obj)
    
    # 在GUI模式下更新画布
    if _global_gui_instance:
        _global_gui_instance.canvas.draw()

    # 输出网格信息
    global_unstr_grid.summary()
    # global_unstr_grid.quality_histogram(_global_gui_instance.ax if _global_gui_instance else None)
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output("网格信息输出完成")

    # 输出网格文件
    global_unstr_grid.save_to_vtkfile(parameters.output_file)
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output(f"网格文件已保存至: {parameters.output_file}")
        
        # 将优化后的网格对象设置到GUI实例中
        _global_gui_instance.mesh_data = global_unstr_grid

    # 结束计时
    global_timer.show_to_console("程序运行正常退出.")
    
    # 输出信息到GUI
    if _global_gui_instance:
        _global_gui_instance.append_info_output("程序运行正常退出")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyMeshGen 非结构网格生成器")
    parser.add_argument(
        "--case",
        type=str,
        default="",
        help="算例文件路径 (默认: 空)",
    )
    args = parser.parse_args()

    # 创建参数对象并应用命令行参数
    params = (
        Parameters("FROM_CASE_JSON", args.case)
        if args.case
        else Parameters("FROM_MAIN_JSON")
    )

    PyMeshGen_mixed(params)

    input("Press Enter to continue...")

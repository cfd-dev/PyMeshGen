import os
import heapq

from fileIO.read_cas import parse_fluent_msh
from data_structure.front2d import construct_initial_front
from visualization.mesh_visualization import Visualization
from utils.message import info, gui_log, gui_progress


def create_visualization(parameters, gui_instance):
    if gui_instance and hasattr(gui_instance, "ax") and gui_instance.ax:
        return Visualization(parameters.viz_enabled, gui_instance.ax)
    return Visualization(parameters.viz_enabled)


def resolve_input_grid_and_front(parameters, mesh_data, parts, gui_instance, visual_obj):
    gui_log(gui_instance, "开始读取输入网格数据...")
    gui_progress(gui_instance, 1)

    input_grid = None
    front_heap = None

    if parts is not None:
        info("使用直接传入的parts参数，直接提取front_list")
        if isinstance(parts, list) and len(parts) > 0:
            parameters.part_params = parts
            info(f"已将 {len(parts)} 个部件添加到parameters.part_params")

            front_heap = []
            front_idx = 0
            for part in parts:
                if hasattr(part, "front_list"):
                    for front in part.front_list:
                        front.idx = front_idx
                        front_idx += 1
                        front_heap.append(front)
            heapq.heapify(front_heap)
            info(f"从parts中提取了 {len(front_heap)} 个阵面")
    elif mesh_data is not None:
        info("使用直接传入的网格数据")
        from utils.data_converter import convert_to_internal_mesh_format

        input_grid = convert_to_internal_mesh_format(mesh_data)
    elif parameters.input_file and isinstance(parameters.input_file, str) and os.path.exists(
        parameters.input_file
    ):
        input_grid = parse_fluent_msh(parameters.input_file)
    else:
        if gui_instance and hasattr(gui_instance, "current_mesh"):
            current_mesh = gui_instance.current_mesh
            info("使用GUI中的当前网格数据")
            from utils.data_converter import convert_to_internal_mesh_format

            input_grid = convert_to_internal_mesh_format(current_mesh)
        else:
            raise ValueError("无法获取有效的网格数据")

    if front_heap is None and input_grid is not None:
        if hasattr(input_grid, "parts_info") and input_grid.parts_info:
            info(f"检测到 {len(input_grid.parts_info)} 个部件信息")
            for part_name in input_grid.parts_info.keys():
                info(f"  - 部件: {part_name}")
            parameters.update_part_params_from_mesh(input_grid)

        if gui_instance and hasattr(gui_instance, "ax") and gui_instance.ax:
            gui_instance.ax.clear()

        visual_obj.plot_mesh(input_grid, boundary_only=True)
        gui_log(gui_instance, f"已读取输入网格文件: {parameters.input_file}")
        gui_log(gui_instance, "开始构造初始阵面...")
        gui_progress(gui_instance, 2)
        front_heap = construct_initial_front(input_grid)
        gui_log(gui_instance, "初始阵面构造完成")
    elif front_heap is not None:
        gui_log(gui_instance, "直接使用parts/connectors中的front_list，跳过构造初始阵面")
        gui_progress(gui_instance, 2)

    return input_grid, front_heap


def merge_generated_grids(unstr_grid_list, input_grid):
    global_unstr_grid = unstr_grid_list[0]
    if hasattr(input_grid, "parts_info") and input_grid.parts_info:
        global_unstr_grid.parts_info = input_grid.parts_info
    for unstr_grid in unstr_grid_list[1:]:
        global_unstr_grid.merge(unstr_grid)
    return global_unstr_grid


def output_and_finalize(global_unstr_grid, parameters, gui_instance, visual_obj, global_timer):
    global_unstr_grid.visualize_unstr_grid_2d(visual_obj)
    if gui_instance and hasattr(gui_instance, "canvas") and gui_instance.canvas:
        gui_instance.canvas.draw()

    global_unstr_grid.summary(gui_instance)
    gui_log(gui_instance, "网格信息输出完成")

    should_save = parameters.auto_output or gui_instance is None
    if should_save:
        output_path = parameters.output_file
        if isinstance(output_path, list):
            if not output_path or not output_path[0]:
                output_path = ["./out/mesh.vtk"]
            else:
                output_path = output_path[0]
        else:
            if not output_path:
                output_path = ["./out/mesh.vtk"]

        gui_log(gui_instance, "开始保存网格文件...")
        gui_progress(gui_instance, 8)
        global_unstr_grid.save_to_vtkfile(output_path)
        if gui_instance:
            gui_log(gui_instance, f"网格文件已保存至: {output_path}")
        else:
            print(f"网格文件已保存至: {output_path}")
    else:
        gui_log(gui_instance, "自动输出网格已禁用，未保存网格文件")

    if gui_instance:
        gui_instance.mesh_data = global_unstr_grid
        gui_instance.original_parameters = parameters

    global_timer.show_to_console("程序运行正常退出.")
    gui_log(gui_instance, "程序运行正常退出")

    if gui_instance:
        from utils.message import set_gui_instance

        set_gui_instance(None)

    return global_unstr_grid

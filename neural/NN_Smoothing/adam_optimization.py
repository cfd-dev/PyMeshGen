import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import trimesh
from pathlib import Path
from math import sqrt
import sys

"""
This code using adam optimization to smooth the mesh. 
The opensource code is originally from the author's GitHub:
https://github.com/yfguo91/meshsmoothing

The code is modified by the author of this repository to make it more readable and usable.
"""

DEFAULT_MOVEMENT_FACTOR = 0.3  # 位移限制因子
DEFAULT_ITERATION_LIMIT = 100  # 最大迭代次数
DEFAULT_LEARNING_RATE = 0.2  # 学习率
DEFAULT_CONVERGENCE_TOLERANCE = 0.1  # 收敛容忍度
DEFAULT_OBJ_FUNC = "L1"  # 目标函数类型
DEFAULT_LR_STEP_SIZE = 1  # 学习率调整步数
DEFAULT_LR_GAMMA = 0.9  # 学习率衰减系数
DEFAULT_INNER_STEPS = 5  # 内层优化步数


def laplacian(ring):
    newpoints = np.mean(ring, axis=0)
    return newpoints


def calc_area(p1, p2, p3):
    """计算三角形面积
    return (0.5 * |v1 x v2|)
    """
    v1 = p2 - p1
    v2 = p3 - p1

    area = torch.cross(v1, v2, dim=0)  # 叉乘计算面积
    area = torch.norm(area) / 2.0  # 计算面积
    return area


def calc_len(p1, p2, p3):
    """计算三角形三边长之和
    return (a + b + c)
    """
    p4 = p2 - p1
    p5 = p3 - p2
    p6 = p1 - p3

    v1 = p4.dot(p4)
    v2 = p5.dot(p5)
    v3 = p6.dot(p6)

    v = torch.sqrt(v1) + torch.sqrt(v2) + torch.sqrt(v3)

    return v


def calc_len2(p1, p2, p3):
    """计算三角形三边长之平方和
    return(a**2 + b**2 + c**2)
    """
    p4 = p2 - p1
    p5 = p3 - p2
    p6 = p1 - p3

    v1 = p4.dot(p4)
    v2 = p5.dot(p5)
    v3 = p6.dot(p6)

    v = v1 + v2 + v3

    return v


def compute_obj_function(cell_indices, points, cells, obj_func="L1"):
    """计算单元质量的L1、L2、Loo
    cell_indices: 单元索引列表
    points: 顶点坐标
    cells: 单元顶点索引
    obj_func: 统计类型，可选值为"L1", "L2", "Loo"
    """
    if obj_func not in ["L1", "L2", "Loo"]:
        raise ValueError("Invalid energy type. Choose from 'L1', 'L2', or 'Loo'.")

    total_energy = 0
    max_energy = 0
    for idx in cell_indices:
        if idx == -1:
            continue

        # 提取三角形顶点坐标
        p1, p2, p3 = points[cells[idx, 0]], points[cells[idx, 1]], points[cells[idx, 2]]
        sum_len2 = calc_len2(p1, p2, p3)
        area = calc_area(p1, p2, p3)

        # 按照三角形质量公式计算quality = 4.0 * sqrt(3.0) * area / a**2 + b**2 + c**2
        # 由于要energy最小化，所以取1-quality
        energy = 1.0 - 4.0 * sqrt(3.0) * area / (sum_len2 + 1e-8)

        if energy > max_energy:
            max_energy = energy

        if obj_func == "L1":
            total_energy = total_energy + energy
        elif obj_func == "L2":
            total_energy = total_energy + energy**2

    if obj_func == "Loo":
        return max_energy
    else:
        return total_energy / len(cell_indices)


def compute_local_cell_size(cell_indices, points, cells):
    """计算节点周围单元的平均尺寸
    cell_indices: 周围单元索引列表
    points: 顶点坐标
    cells: 单元顶点索引
    """
    total_size = 0.0
    valid_cells = 0
    for idx in cell_indices:
        if idx == -1:
            continue
        p1, p2, p3 = points[cells[idx, 0]], points[cells[idx, 1]], points[cells[idx, 2]]
        edge_lengths = [torch.norm(p2 - p1), torch.norm(p3 - p2), torch.norm(p1 - p3)]
        total_size += sum(edge_lengths) / 3  # 单元平均边长
        valid_cells += 1
    return total_size / valid_cells if valid_cells > 0 else 0.0


def init_mesh_and_variables(input_file):
    """
    加载网格并初始化优化所需变量
    返回: points, faces, vertex_faces, original_points, variapoints, cell_size_cache, global_avg_size
    """
    mesh = trimesh.load(input_file)  # 这个网格结构可以方便的寻找点面之间的邻接关系
    points = mesh.vertices  # 网格的点的坐标
    faces = mesh.faces  # 网格的三角片

    boundary_mask = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    boundary_edges = mesh.edges[boundary_mask]  # 边界edges
    boundary_ids = np.unique(boundary_edges.ravel())  # 边界点的索引
    bpindex = np.zeros(len(points), dtype=bool)  # 边界点标记
    bpindex[boundary_ids] = True  # 标记边界点

    non_boundary_indices = np.where(~bpindex)[0]  # 非边界点索引
    vertex_faces = mesh.vertex_faces  # 记录每个点连接的三角片索引
    original_points = points.copy()  # 保存原始点位置

    # 打印网格信息
    print("Input file path: ", input_file)
    print("Mesh loaded:", not (mesh.is_empty))
    print("Mesh information:")
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Edges:", len(mesh.edges))
    print("Reading input mesh file..., DONE!")

    # 创建变量，将所有节点都设置成变量
    variapoints = []
    for i in range(0, points.shape[0]):
        torch_point = torch.from_numpy(points[i])
        tfpoint = Variable(torch_point, requires_grad=True)
        variapoints.append(tfpoint)

    # 预计算所有非边界点的单元尺寸
    cell_size_cache = {}
    global_avg_size = 0.0
    for idx in non_boundary_indices:
        neighbor_cells = [f for f in vertex_faces[idx] if f != -1]
        cell_size_cache[idx] = compute_local_cell_size(
            neighbor_cells, variapoints, faces
        )
        global_avg_size += cell_size_cache[idx]
    global_avg_size /= len(np.where(~bpindex)[0])  # 计算初始全局平均尺寸

    return (
        points,  # 原始点坐标
        faces,  # 原始三角片
        non_boundary_indices,  # 非边界点索引
        vertex_faces,  # 记录每个点连接的三角片索引
        original_points,  # 原始点坐标备份
        variapoints,  # variable变量
        cell_size_cache,  # 非边界点的单元尺寸缓存
        global_avg_size,  # 全局平均尺寸
    )


def limit_displacement(
    non_boundary_indices,
    variapoints,
    prev_variapoints,
    original_points,
    movement_factor,
    cell_size_cache,
):
    """限制节点位移的辅助函数"""
    with torch.no_grad():
        for idx in non_boundary_indices:
            # prev_point = torch.from_numpy(original_points[idx])  # 限制与原始位置的距离
            prev_point = prev_variapoints[idx].data  # 限制与前一步位置的距离
            current_point = variapoints[idx].data
            displacement = current_point - prev_point  # 计算当前位移量

            max_allow_disp = movement_factor * cell_size_cache[idx]

            # 限制位移量
            if torch.norm(displacement) > max_allow_disp:
                clamped_disp = displacement * (
                    max_allow_disp / (torch.norm(displacement) + 1e-12)
                )
                # new_position = original_points[idx] + clamped_disp  # 在原始位置上位移
                new_position = prev_point + clamped_disp  # 在上一步的位移基础上继续位移
                variapoints[idx].data.copy_(new_position)
            else:
                variapoints[idx].data.copy_(current_point)


def update_learning_rate_based_on_size(
    non_boundary_indices,
    param_groups,
    global_size_ref,
    base_lr,
    last_lr,
    cell_size_cache,
):
    """根据单元尺寸动态更新学习率"""
    # 计算动态边界阈值
    dynamic_min = global_size_ref * 0.1  # 基准的10%
    dynamic_max = global_size_ref * 10.0  # 基准的10倍

    for idx, param_group in zip(non_boundary_indices, param_groups):
        # 使用动态阈值限制
        clamped_size = torch.clamp(
            cell_size_cache[idx], min=dynamic_min, max=dynamic_max
        )
        param_group["lr"] = base_lr * clamped_size * last_lr


def build_adam_optimizer(
    variapoints, non_boundary_indices, cell_size_cache, lr, lr_step_size, lr_gamma
):
    """
    构建自适应学习率的Adam优化器和调度器
    返回 optimizer, scheduler, param_groups
    """
    base_lr = lr
    param_groups = []
    for i in non_boundary_indices:
        # 设置参数组，学习率与单元尺寸成比例
        param_groups.append(
            {
                "params": variapoints[i],
                "lr": base_lr * cell_size_cache[i],
            }
        )
    optimizer = optim.Adam(param_groups)  # 使用参数组替代统一学习率

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )
    return optimizer, scheduler, param_groups


def print_optimizer_status(epoch, optimizer, total_energy, start_time, end_time):
    """
    打印当前epoch优化器的学习率和能量等信息
    """
    lrs = [group["lr"] for group in optimizer.param_groups]
    avg_lr = sum(lrs) / len(lrs)
    max_lr = max(lrs)
    min_lr = min(lrs)
    print(
        f"epoch {epoch+1}, lr_min = {min_lr:.3e}, lr_max = {max_lr:.3e}, lr_avg = {avg_lr:.3e}, "
        f"total_energy = {total_energy:.3f}, time elapsed= {(end_time - start_time):.3f}s"
    )


def optimize_single_node(
    idx,
    neighbor_faces,
    variapoints,
    faces,
    obj_func,
    optimizer,
    original_points,
    movement_factor,
    cell_size_cache,
):
    # 只备份当前idx节点的位置
    prev_variapoints = {idx: variapoints[idx].clone().detach()}

    optimizer.zero_grad()

    local_energy = compute_obj_function(neighbor_faces, variapoints, faces, obj_func)
    local_energy.backward()
    optimizer.step()

    # with torch.no_grad():
    #     limit_displacement(
    #         [idx],
    #         variapoints,
    #         prev_variapoints,
    #         original_points,
    #         movement_factor,
    #         cell_size_cache,
    #     )
    return local_energy.item()


def run_element_on_vertex(
    input_file,
    output_file,
    opt_epoch=DEFAULT_ITERATION_LIMIT,
    lr=DEFAULT_LEARNING_RATE,
    conver_tol=DEFAULT_CONVERGENCE_TOLERANCE,
    obj_func=DEFAULT_OBJ_FUNC,
    movement_factor=DEFAULT_MOVEMENT_FACTOR,  # 位移限制因子
    lr_step_size=DEFAULT_LR_STEP_SIZE,  # 每步调整学习率的步数
    lr_gamma=DEFAULT_LR_GAMMA,  # 学习率衰减系数
    inner_steps=DEFAULT_INNER_STEPS,  # 内层优化步数
):
    """使用Adam优化器优化网格，局部优化"""
    ############################################################################
    (
        points,  # 原始点坐标
        faces,  # 原始三角片
        non_boundary_indices,  # 非边界点索引
        vertex_faces,  # 记录每个点连接的三角片索引
        original_points,  # 原始点坐标备份
        variapoints,  # variable变量
        cell_size_cache,  # 非边界点的单元尺寸缓存
        global_avg_size,  # 全局平均尺寸
    ) = init_mesh_and_variables(input_file)

    # 创建自适应学习率参数组
    optimizer, scheduler, param_groups = build_adam_optimizer(
        variapoints, non_boundary_indices, cell_size_cache, lr, lr_step_size, lr_gamma
    )

    # 优化网格
    start_time = time.time()
    prev_energy = 0
    for epoch in range(opt_epoch):
        total_energy = 0
        prev_variapoints = [p.clone().detach() for p in variapoints]

        # 外层：遍历所有非边界节点
        for idx in non_boundary_indices:
            neighbor_faces = [f for f in vertex_faces[idx] if f != -1]

            # 内层：对该节点的局部 patch 进行多步优化
            for istep in range(inner_steps):
                total_energy += optimize_single_node(
                    idx,
                    neighbor_faces,
                    variapoints,
                    faces,
                    obj_func,
                    optimizer,
                    original_points,
                    movement_factor,
                    cell_size_cache,
                )

        # 统一做位移限制
        with torch.no_grad():
            limit_displacement(
                non_boundary_indices,
                variapoints,
                prev_variapoints,
                original_points,
                movement_factor,
                cell_size_cache,
            )

        # 判断是否收敛
        if abs(prev_energy - total_energy) < conver_tol:
            break
        prev_energy = total_energy

        scheduler.step()

        end_time = time.time()
        print_optimizer_status(epoch, optimizer, total_energy, start_time, end_time)

    # 将variapoints转换到points
    for i in non_boundary_indices:
        points[i] = variapoints[i].data.numpy()

    new_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    new_mesh.export(output_file)

    print("Output file path: ", output_file)
    print("Export output mesh file..., DONE!")

    # new_mesh.show(
    #     wireframe=True,
    #     wireframe_color=[0, 0, 0, 1],
    #     background=[1, 1, 1, 1],
    #     smooth=False,
    # )

    return


def run_global_patch(
    input_file,
    output_file,
    opt_epoch=DEFAULT_ITERATION_LIMIT,
    lr=DEFAULT_LEARNING_RATE,
    conver_tol=DEFAULT_CONVERGENCE_TOLERANCE,
    obj_func=DEFAULT_OBJ_FUNC,
    movement_factor=DEFAULT_MOVEMENT_FACTOR,  # 位移限制因子
    lr_step_size=DEFAULT_LR_STEP_SIZE,  # 每步调整学习率的步数
    lr_gamma=DEFAULT_LR_GAMMA,  # 学习率衰减系数
    inner_steps=DEFAULT_INNER_STEPS,  # 内层优化步数
):
    """使用Adam优化器优化网格，全局优化"""
    ############################################################################
    (
        points,
        faces,
        non_boundary_indices,
        vertex_faces,
        original_points,
        variapoints,
        cell_size_cache,
        global_avg_size,
    ) = init_mesh_and_variables(input_file)

    # 创建自适应学习率参数组
    optimizer, scheduler, param_groups = build_adam_optimizer(
        variapoints, non_boundary_indices, cell_size_cache, lr, lr_step_size, lr_gamma
    )

    # 优化网格
    start_time = time.time()
    prev_energy = 0
    for epoch in range(opt_epoch):
        prev_variapoints = [p.clone().detach() for p in variapoints]  # 备份当前点位置

        optimizer.zero_grad()
        total_energy = 0
        for idx in non_boundary_indices:
            total_energy += compute_obj_function(
                vertex_faces[idx], variapoints, faces, obj_func
            )

        # 判断是否收敛
        if abs(prev_energy - total_energy) < conver_tol:
            break
        prev_energy = total_energy

        total_energy.backward()
        optimizer.step()
        scheduler.step()

        # 添加位移限制
        with torch.no_grad():
            limit_displacement(
                non_boundary_indices,
                variapoints,
                prev_variapoints,
                original_points,
                movement_factor,
                cell_size_cache,
            )

        # 训练过程中实时更新学习率
        # with torch.no_grad():
        #     update_learning_rate_based_on_size(
        #         np.where(~bpindex)[0],
        #         optimizer.param_groups,
        #         global_avg_size,
        #         base_lr,
        #         scheduler.get_last_lr()[0],
        #         cell_size_cache,
        #     )

        end_time = time.time()
        print_optimizer_status(epoch, optimizer, total_energy, start_time, end_time)

    # 将variapoints转换到points
    for i in non_boundary_indices:
        points[i] = variapoints[i].data.numpy()

    new_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    new_mesh.export(output_file)

    print("Output file path: ", output_file)
    print("Export output mesh file..., DONE!")

    # new_mesh.show(
    #     wireframe=True,
    #     wireframe_color=[0, 0, 0, 1],
    #     background=[1, 1, 1, 1],
    #     smooth=False,
    # )

    return


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="optimize mesh by optimization")
    parser.add_argument(
        "--input_file", help="The input file.", default=None, required=True
    )
    parser.add_argument(
        "--output_file", help="The output file.", default=None, required=True
    )
    parser.add_argument(
        "--opt_epoch",
        help="the epoch of smoothing.",
        type=int,
        default=DEFAULT_ITERATION_LIMIT,
    )
    parser.add_argument(
        "--lr", help="the learning rate.", type=float, default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument(
        "--conver_tol",
        help="the convergence tolerance.",
        type=float,
        default=DEFAULT_CONVERGENCE_TOLERANCE,
    )
    parser.add_argument(
        "--obj_func", help="the energy type.", type=str, default=DEFAULT_OBJ_FUNC
    )
    parser.add_argument(
        "--movement_factor",
        help="the movement factor.",
        type=float,
        default=DEFAULT_MOVEMENT_FACTOR,
    )
    parser.add_argument(
        "--lr_step_size",
        help="the step size of learning rate.",
        type=int,
        default=DEFAULT_LR_STEP_SIZE,
    )
    parser.add_argument(
        "--lr_gamma",
        help="the gamma of learning rate.",
        type=float,
        default=DEFAULT_LR_GAMMA,
    )
    parser.add_argument(
        "--inner_steps",
        help="the inner steps.",
        type=int,
        default=DEFAULT_INNER_STEPS,
    )
    parser.add_argument(
        "--example_index",
        help="the example index.",
        type=int,
    )
    parser.add_argument(
        "--global_patch",
        help="use global patch.",
        action="store_true",
    )

    return parser.parse_args()


def predefined_examples(example_index):
    """Predefined examples."""
    # TODO example 2, 3, 4优化后节点超出边界，需要修复
    example_dir = Path(__file__).parent / "example"
    examples = {
        1: {
            "input_file": example_dir / "first.stl",
            "output_file": example_dir / "first_opt.stl",
        },
        2: {
            "input_file": example_dir / "second.stl",
            "output_file": example_dir / "second_opt.stl",
        },
        3: {
            "input_file": example_dir / "third.stl",
            "output_file": example_dir / "third_opt.stl",
        },
        4: {
            "input_file": example_dir / "fourth.stl",
            "output_file": example_dir / "fourth_opt.stl",
        },
        5: {
            "input_file": example_dir / "fifth.stl",
            "output_file": example_dir / "fifth_opt.stl",
        },
        6: {
            "input_file": example_dir / "sixth.stl",
            "output_file": example_dir / "sixth_opt.stl",
        },
        7: {
            "input_file": example_dir / "rae2822_bad.stl",
            "output_file": example_dir / "rae2822_opt.stl",
        },
        8: {
            "input_file": example_dir / "30p30n_bad.stl",
            "output_file": example_dir / "30p30n_opt.stl",
        },
        9: {
            "input_file": example_dir / "naca0012_bad.stl",
            "output_file": example_dir / "naca0012_opt.stl",
        },
    }
    if example_index in examples:
        return examples[example_index]
    else:
        raise ValueError(f"Example index {example_index} not found.")


if __name__ == "__main__":
    # 支持直接运行和示例调用
    if len(sys.argv) > 1:
        args = parse_args()
        if getattr(args, "global_patch", False):
            run_global_patch(
                args.input_file,
                args.output_file,
                args.opt_epoch,
                args.lr,
                args.conver_tol,
                args.obj_func,
                args.movement_factor,
                args.lr_step_size,
                args.lr_gamma,
            )
        else:
            run_element_on_vertex(
                args.input_file,
                args.output_file,
                args.opt_epoch,
                args.lr,
                args.conver_tol,
                args.obj_func,
                args.movement_factor,
                args.lr_step_size,
                args.lr_gamma,
                args.inner_steps,
            )
    else:
        # 默认示例参数
        example_index = 1  # 可修改为其他示例
        example_args = predefined_examples(example_index)
        # run_element_on_vertex(
        run_global_patch(
            example_args.get("input_file", "example/first.stl"),
            example_args.get("output_file", "example/first_opt.stl"),
            example_args.get("opt_epoch", DEFAULT_ITERATION_LIMIT),
            example_args.get("lr", DEFAULT_LEARNING_RATE),
            example_args.get("conver_tol", DEFAULT_CONVERGENCE_TOLERANCE),
            example_args.get("obj_func", DEFAULT_OBJ_FUNC),
            example_args.get("movement_factor", DEFAULT_MOVEMENT_FACTOR),
            example_args.get("lr_step_size", DEFAULT_LR_STEP_SIZE),
            example_args.get("lr_gamma", DEFAULT_LR_GAMMA),
            example_args.get("inner_steps", DEFAULT_INNER_STEPS),
        )

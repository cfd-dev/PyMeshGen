import os
import numpy as np
import heapq
from itertools import combinations

import geom_toolkit as geom_tool
from timer import TimeSpan
from message import info, debug, verbose, warning, error
from basic_elements import Triangle, Quadrilateral
from mesh_quality import quadrilateral_quality2

def optimize_hybrid_grid(hybrid_grid):
    """调用外部混合网格优化软件进行优化"""  
    import subprocess
    tmp_file = "./out/tmp_mesh.vtk"
    hybrid_grid.save_to_vtkfile(tmp_file)

    max_iter = 10
    movement_factor = 0.3
    opt_exe = './optimize/laplacian_opt.exe'
    cmd = [opt_exe, tmp_file, str(max_iter), str(movement_factor)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    debug(f'Optimized VTK mesh written to: {tmp_file.replace(".vtk", "_opt.vtk")}')
    
    # 从优化后的VTK文件加载网格
    from fileIO.vtk_io import parse_vtk_msh
    optimized_grid = parse_vtk_msh(tmp_file.replace(".vtk", "_opt.vtk"))
    
    # 如果需要，删除临时文件
    if not __debug__:
        os.remove(tmp_file)
        os.remove(tmp_file.replace(".vtk", "_opt.vtk"))

    return optimized_grid

def merge_elements(unstr_grid):
    """
    合并相邻的三角形单元，形成四边形单元。
    合并条件：
    1. 两个三角形必须共享一条边
    2. 合并后的四边形是凸多边形
    3. 四边形质量高于合并前三角形质量中位数
    """
    timer = TimeSpan("开始合并三角形为四边形...")
    node_coords = unstr_grid.node_coords
    edge_map = {}
    merge_candidates = []

    # 构建边到单元的映射
    for cell_idx, cell in enumerate(unstr_grid.cell_container):
        if not isinstance(cell, Triangle):
            continue
        for i, j in combinations(sorted(cell.node_ids), 2):
            edge = (i, j)
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append(cell_idx)

    # 寻找可合并的三角形对
    for edge, cells in edge_map.items():
        if len(cells) != 2:
            continue

        cell1, cell2 = cells
        tri1 = unstr_grid.cell_container[cell1]
        tri2 = unstr_grid.cell_container[cell2]

        # 获取四个顶点
        common = set(tri1.node_ids) & set(tri2.node_ids)
        if len(common) != 2:
            continue

        a, b = sorted(common)
        c = list(set(tri1.node_ids) - common)[0]
        d = list(set(tri2.node_ids) - common)[0]

        # 凸性检查
        if not geom_tool.is_convex(a, c, b, d, node_coords):
            continue

        # 计算质量增益
        tri1.init_metrics()
        tri2.init_metrics()
        tri_quality = (tri1.quality + tri2.quality) / 2
        quad_quality = quadrilateral_quality2(
            node_coords[a], node_coords[c], node_coords[b], node_coords[d]
        )

        # 质量提升判断（使用最小堆保存优质候选）
        # if quad_quality > tri_quality and quad_quality > 0.3:
        heapq.heappush(merge_candidates, (-quad_quality, (cell1, cell2, a, b, c, d)))

    # 按质量从高到低处理合并
    merged = set()
    num_merged = 0
    while merge_candidates:
        _, (cell1_idx, cell2_idx, a, b, c, d) = heapq.heappop(merge_candidates)

        # 跳过已处理单元
        if cell1_idx in merged or cell2_idx in merged:
            continue

        # 确保新创建的四边形法向指向z轴正方向
        if not geom_tool.is_left2d(node_coords[a], node_coords[b], node_coords[d]):
            a, c, b, d = a, d, b, c

        # 创建新四边形
        new_quad = Quadrilateral(
            node_coords[a],
            node_coords[c],
            node_coords[b],
            node_coords[d],
            "interior",
            len(unstr_grid.cell_container),
            [a, c, b, d],
        )

        # 替换原单元
        unstr_grid.cell_container[cell1_idx] = new_quad
        unstr_grid.cell_container[cell2_idx] = None  # 标记删除

        merged.update([cell1_idx, cell2_idx])
        num_merged += 1

    # 清理被删除的单元
    unstr_grid.cell_container = [c for c in unstr_grid.cell_container if c is not None]

    info(f"成功合并{num_merged}对三角形为四边形")
    timer.show_to_console("四边形合并完成.")
    return unstr_grid


def hybrid_smooth(unstr_grid, max_iter=3):
    """混合平滑算法（结合角度优化和形态优化）"""
    timer = TimeSpan("开始混合平滑优化...")
    node_coords = np.array(unstr_grid.node_coords)
    original_coords = node_coords.copy()
    boundary_nodes = set(unstr_grid.boundary_nodes_list)

    # 新增凸性检查辅助函数
    def is_quad_convex(nodes):
        return geom_tool.is_convex(
            nodes[0], nodes[1], nodes[2], nodes[3], node_coords.tolist()
        )

    for _ in range(max_iter):
        # 存储每个节点的移动向量和权重
        displacements = np.zeros_like(node_coords)
        weights = np.zeros(len(node_coords))

        # 存储四边形原始节点用于回滚检查
        quad_originals = {}
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Quadrilateral):
                quad_originals[id(cell)] = np.array(
                    [node_coords[i] for i in cell.node_ids]
                )

        # 遍历所有单元进行贡献计算
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Quadrilateral):
                # 四边形优化目标：接近矩形（角度优化+边长比优化）
                quad_nodes = cell.node_ids
                original_nodes = quad_originals[id(cell)]

                for i in range(4):
                    prev = quad_nodes[i - 1]
                    curr = quad_nodes[i]
                    next1 = quad_nodes[(i + 1) % 4]
                    next2 = quad_nodes[(i + 2) % 4]

                    # 计算带约束的理想位置
                    ideal_point = compute_rectangular_position(
                        original_nodes[(i - 1) % 4],  # 使用原始坐标计算
                        original_nodes[i],
                        original_nodes[(i + 1) % 4],
                    )

                    displacement = 0.3 * (
                        ideal_point - node_coords[next1]
                    )  # 减小位移系数
                    if next1 not in boundary_nodes:
                        displacements[next1] += displacement
                        weights[next1] += 1.0

            elif isinstance(cell, Triangle):
                # 三角形优化目标：接近等边三角形（角度优化）
                tri_nodes = cell.node_ids
                centroid = np.mean(node_coords[tri_nodes], axis=0)

                for i in range(3):
                    curr = tri_nodes[i]
                    if curr in boundary_nodes:
                        continue

                    # 向质心方向移动（促进等边化）
                    displacement = 0.2 * (centroid - node_coords[curr])
                    displacements[curr] += displacement
                    weights[curr] += 1.0

        # 应用平滑并更新坐标
        new_coords = node_coords.copy()
        for i in range(len(node_coords)):
            if weights[i] > 0 and i not in boundary_nodes:
                new_coords[i] += displacements[i] / weights[i]

        # 凸性检查和修正
        need_rollback = False
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Quadrilateral) and len(cell.node_ids) == 4:
                new_nodes = [new_coords[i] for i in cell.node_ids]
                if not is_quad_convex(new_nodes):
                    need_rollback = True
                    break

        # 只有当所有四边形保持凸性时才接受更新
        if not need_rollback:
            node_coords = new_coords
        else:
            warning("检测到凹四边形，跳过本轮平滑")

        # 限制最大位移防止震荡
        max_disp = np.linalg.norm(node_coords - original_coords, axis=1).max()
        if max_disp < 1e-6:
            break

    # 更新回网格结构
    unstr_grid.node_coords = node_coords.tolist()
    timer.show_to_console("混合平滑完成.")
    return unstr_grid


def compute_rectangular_position(prev, curr, next):
    """计算理想矩形位置（基于相邻三点）"""
    vec1 = curr - prev
    vec2 = next - curr

    # 计算正交修正
    ideal_vec = vec1 - 2 * np.dot(vec1, vec2) / np.dot(vec2, vec2) * vec2
    return curr + ideal_vec


def edge_swap(unstr_grid):
    timer = TimeSpan("开始进行边交换优化...")
    node_coords = unstr_grid.node_coords
    edge_map = {}

    # 构建边到单元的映射
    for cell_idx, cell in enumerate(unstr_grid.cell_container):
        for i, j in combinations(sorted(cell.node_ids), 2):
            edge = (i, j)
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append(cell_idx)

    swapped = True
    num_swapped = 0
    while swapped:
        swapped = False
        # 使用当前edge_map的快照避免迭代修改问题
        for edge, cells in list(edge_map.items()):
            # 仅处理内部边（被两个单元共享）
            if len(cells) != 2:
                continue
            if edge == (194, 24) or edge == (24, 194):
                kkk = 0

            cell1_idx, cell2_idx = cells
            cell1 = unstr_grid.cell_container[cell1_idx]
            cell2 = unstr_grid.cell_container[cell2_idx]
            if not (isinstance(cell1, Triangle) and isinstance(cell2, Triangle)):
                continue  # 非三角形单元跳过

            # 确认公共边
            common_edge = set(cell1.node_ids) & set(cell2.node_ids)
            if len(common_edge) != 2:
                continue  # 数据异常

            a, b = sorted(common_edge)
            other_points = list(
                (set(cell1.node_ids) | set(cell2.node_ids)) - common_edge
            )
            if len(other_points) != 2:
                continue  # 无法构成四边形

            c, d = other_points

            # 凸性检查
            if not geom_tool.is_convex(a, c, b, d, node_coords):
                continue

            # 计算交换前的最小角
            current_min = min(
                geom_tool.calculate_min_angle(cell1, node_coords),
                geom_tool.calculate_min_angle(cell2, node_coords),
            )

            # 交换后的单元
            swapped_cell1 = [a, c, d]
            swapped_cell2 = [b, c, d]

            # 确保构成的单元节点逆时针
            if not geom_tool.is_left2d(node_coords[a], node_coords[c], node_coords[d]):
                swapped_cell1 = [d, c, a]
            if not geom_tool.is_left2d(node_coords[b], node_coords[c], node_coords[d]):
                swapped_cell2 = [d, c, b]

            # 有效性检查
            if not (
                geom_tool.is_valid_triangle(swapped_cell1, node_coords)
                and geom_tool.is_valid_triangle(swapped_cell2, node_coords)
            ):
                continue

            # 计算交换后的最小角
            swapped_min = min(
                geom_tool.calculate_min_angle(swapped_cell1, node_coords),
                geom_tool.calculate_min_angle(swapped_cell2, node_coords),
            )

            # 凸性检查
            # if not geom_tool.is_convex(a, c, b, d, node_coords):
            #     continue

            # 交换条件：最小角优化且不创建新边界边
            if swapped_min > current_min and not (
                c in unstr_grid.boundary_nodes_list
                and d in unstr_grid.boundary_nodes_list
            ):
                # 执行交换
                # 创建新的Triangle对象
                new_cell1 = Triangle(
                    node_coords[swapped_cell1[0]],
                    node_coords[swapped_cell1[1]],
                    node_coords[swapped_cell1[2]],
                    cell1.idx,
                    node_ids=swapped_cell1,
                )
                new_cell2 = Triangle(
                    node_coords[swapped_cell2[0]],
                    node_coords[swapped_cell2[1]],
                    node_coords[swapped_cell2[2]],
                    cell2.idx,
                    node_ids=swapped_cell2,
                )

                unstr_grid.cell_container[cell1_idx] = new_cell1  # 修改点
                unstr_grid.cell_container[cell2_idx] = new_cell2  # 修改点
                swapped = True
                num_swapped += 1

        # 重新构建边映射
        if swapped:
            edge_map = {}
            for cell_idx, cell in enumerate(unstr_grid.cell_container):
                for i, j in combinations(sorted(cell.node_ids), 2):
                    edge = (i, j)
                    if edge not in edge_map:
                        edge_map[edge] = []
                    edge_map[edge].append(cell_idx)

    info(f"共进行了{num_swapped}次边交换.")
    timer.show_to_console("边交换优化完成.")

    return unstr_grid


def laplacian_smooth(unstr_grid, num_iter=10):
    timer = TimeSpan("开始进行laplacian优化...")

    node_coords = unstr_grid.node_coords
    # 将键改为节点索引
    neighbors = {node_ids: set() for node_ids in range(len(node_coords))}

    for cell in unstr_grid.cell_container:
        for i, j in combinations(sorted(cell.node_ids), 2):
            neighbors[i].add(j)
            neighbors[j].add(i)

    # 迭代进行拉普拉斯平滑
    for i in range(num_iter):
        new_coords = []
        for node, coord in enumerate(node_coords):
            # 跳过边界节点（保持固定）
            if node in unstr_grid.boundary_nodes_list:
                new_coords.append(coord)
                continue

            if len(neighbors[node]) == 0:
                new_coords.append(coord)
                continue

            # 计算邻居节点的平均坐标（转换为numpy数组）
            neighbor_coords = np.array([node_coords[n] for n in neighbors[node]])
            avg_coord = np.mean(neighbor_coords, axis=0)

            # 使用numpy进行向量运算
            new_coord = 0.5 * np.array(coord) + 0.5 * avg_coord
            new_coords.append(new_coord.tolist())  # 转换回列表格式

        unstr_grid.node_coords = new_coords
        info(f"第{i+1}轮laplacian优化完成.")
    timer.show_to_console("laplacian优化完成.")

    return unstr_grid

def node_perturbation(unstr_grid, ratio=0.8):
    """对网格节点进行随机扰动
    Args:
        ratio: 扰动比例（相对于单元特征尺寸）
    """
    np.random.seed(42)  # 固定随机种子保证可重复性
    
    # 获取节点坐标和边界信息
    node_coords = np.array(unstr_grid.node_coords)
    boundary_nodes = set(unstr_grid.boundary_nodes_list)
    
    # 计算每个节点的扰动幅度（基于关联单元尺寸）
    node_scale = np.zeros(len(node_coords))
    for cell in unstr_grid.cell_container:
        cell_size = cell.get_element_size()
        for node_id in cell.node_ids:
            node_scale[node_id] = max(node_scale[node_id], cell_size)
    
    # 仅扰动内部节点
    for i in range(len(node_coords)):
        if i in boundary_nodes:
            continue
        
        # 生成随机方向向量
        direction = np.random.normal(size=node_coords.shape[1])
        direction /= np.linalg.norm(direction)  # 单位向量
        
        # 计算扰动幅度
        max_shift = node_scale[i] * ratio
        shift = direction * np.random.uniform(0, max_shift)
        
        # 应用扰动
        node_coords[i] += shift
        
    # 更新回网格结构
    unstr_grid.node_coords = node_coords.tolist()
    info(f"节点扰动完成，最大位移: {np.max(node_scale * ratio):.4f}")
    return unstr_grid
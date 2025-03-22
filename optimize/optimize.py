import numpy as np
from itertools import combinations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
import geometry_info as geo_info


def edge_swap(unstr_grid):
    print(f"开始进行边交换优化...")
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
            if not geo_info.is_convex(a, b, c, d, unstr_grid.node_coords):
                continue

            # 计算交换前的最小角
            current_min = min(
                geo_info.calculate_min_angle(cell1, unstr_grid.node_coords),
                geo_info.calculate_min_angle(cell2, unstr_grid.node_coords),
            )

            # 交换后的单元
            swapped_cell1 = [a, c, d]
            swapped_cell2 = [b, c, d]

            # 有效性检查
            if not (
                geo_info.is_valid_triangle(swapped_cell1, unstr_grid.node_coords)
                and geo_info.is_valid_triangle(swapped_cell2, unstr_grid.node_coords)
            ):
                continue

            # 计算交换后的最小角
            swapped_min = min(
                geo_info.calculate_min_angle(swapped_cell1, unstr_grid.node_coords),
                geo_info.calculate_min_angle(swapped_cell2, unstr_grid.node_coords),
            )

            # 凸性检查
            if not geo_info.is_convex(c, d, a, b, unstr_grid.node_coords):
                continue

            # 交换条件：最小角优化且不创建新边界边
            if swapped_min > current_min and not (
                c in unstr_grid.boundary_nodes_list
                and d in unstr_grid.boundary_nodes_list
            ):
                # 执行交换
                # 创建新的Triangle对象
                new_cell1 = geo_info.Triangle(
                    unstr_grid.node_coords[a],
                    unstr_grid.node_coords[c],
                    unstr_grid.node_coords[d],
                    cell1.idx,
                    node_ids=(a, c, d),
                )
                new_cell2 = geo_info.Triangle(
                    unstr_grid.node_coords[b],
                    unstr_grid.node_coords[c],
                    unstr_grid.node_coords[d],
                    cell2.idx,
                    node_ids=(b, c, d),
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

    print(f"共进行了{num_swapped}次边交换.\n")

    return unstr_grid


def laplacian_smooth(unstr_grid, num_iter=10):
    print(f"开始进行laplacian优化...")
    # 将键改为节点索引
    neighbors = {node_ids: set() for node_ids in range(len(unstr_grid.node_coords))}

    for cell in unstr_grid.cell_container:
        for i, j in combinations(sorted(cell.node_ids), 2):
            neighbors[i].add(j)
            neighbors[j].add(i)

    # 迭代进行拉普拉斯平滑
    for i in range(num_iter):
        new_coords = []
        for node, coord in enumerate(unstr_grid.node_coords):
            # 跳过边界节点（保持固定）
            if node in unstr_grid.boundary_nodes_list:
                new_coords.append(coord)
                continue

            if len(neighbors[node]) == 0:
                new_coords.append(coord)
                continue

            # 计算邻居节点的平均坐标（转换为numpy数组）
            neighbor_coords = np.array(
                [unstr_grid.node_coords[n] for n in neighbors[node]]
            )
            avg_coord = np.mean(neighbor_coords, axis=0)

            # 使用numpy进行向量运算
            new_coord = 0.5 * np.array(coord) + 0.5 * avg_coord
            new_coords.append(new_coord.tolist())  # 转换回列表格式

        unstr_grid.node_coords = new_coords
        print(f"第{i}轮laplacian优化完成.")
    print(f"laplacian优化完成.\n")

    return unstr_grid

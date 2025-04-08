import numpy as np
from itertools import combinations
import geom_toolkit as geom_tool
from utils.timer import TimeSpan
from message import info, debug, verbose, warning, error
from basic_elements import Triangle


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

import numpy as np
from itertools import combinations


def edge_swap(unstr_grid):
    edge_map = {}

    # 构建边到单元的映射
    for cell_idx, cell in enumerate(unstr_grid.cell_nodes):
        for i, j in combinations(sorted(cell), 2):
            edge = (i, j)
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append(cell_idx)

    # 遍历所有共享边
    swapped = True
    while swapped:
        swapped = False
        for edge, cells in edge_map.items():
            if len(cells) != 2:
                continue

            # 修复点：正确识别四边形顶点
            tri1 = unstr_grid.cell_nodes[cells[0]]
            tri2 = unstr_grid.cell_nodes[cells[1]]

            # 找到公共边（已排序的edge）
            common_edge = set(tri1) & set(tri2)
            other_points = list((set(tri1) | set(tri2)) - common_edge)

            if len(other_points) != 2:  # 确保构成四边形
                continue

            a, b = sorted(common_edge)
            c, d = other_points

            # 修复点：正确计算四边形对角线的角度
            def triangle_angles(p1, p2, p3):
                # 计算三角形三个内角
                v1 = np.subtract(unstr_grid.node_coords[p2], unstr_grid.node_coords[p1])
                v2 = np.subtract(unstr_grid.node_coords[p3], unstr_grid.node_coords[p1])
                v3 = np.subtract(unstr_grid.node_coords[p3], unstr_grid.node_coords[p2])

                angle1 = np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                angle2 = np.arccos(
                    np.dot(-v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
                )
                angle3 = np.pi - angle1 - angle2
                return np.rad2deg([angle1, angle2, angle3])

            # 计算当前对角线的最小角
            current_min = min(
                min(triangle_angles(a, b, c)), min(triangle_angles(a, d, b))
            )

            # 计算交换后的最小角
            swapped_min = min(
                min(triangle_angles(a, c, d)), min(triangle_angles(b, c, d))
            )

            if swapped_min > current_min:
                # 执行边交换
                unstr_grid.cell_nodes[cells[0]] = [a, c, d]
                unstr_grid.cell_nodes[cells[1]] = [b, c, d]
                swapped = True

        # 需要重新构建边映射
        if swapped:
            edge_map = {}
            for cell_idx, cell in enumerate(unstr_grid.cell_nodes):
                for i, j in combinations(sorted(cell), 2):
                    edge = (i, j)
                    if edge not in edge_map:
                        edge_map[edge] = []
                    edge_map[edge].append(cell_idx)

    return unstr_grid


def laplacian_smooth(unstr_grid):
    # 将键改为节点索引
    neighbors = {node_idx: set() for node_idx in range(len(unstr_grid.node_coords))}

    for cell in unstr_grid.cell_nodes:
        for i, j in combinations(sorted(cell), 2):
            neighbors[i].add(j)
            neighbors[j].add(i)

    # 迭代进行拉普拉斯平滑
    num_iterations = 10
    for _ in range(num_iterations):
        new_coords = []
        for node, coord in enumerate(unstr_grid.node_coords):
            # 跳过边界节点（保持固定）
            if node in unstr_grid.boundary_nodes:
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

    return unstr_grid

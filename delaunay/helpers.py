"""
Bowyer-Watson 网格生成器 - 辅助工具函数

包含：
- 边界环提取（从 Front 对象中检测外边界和孔洞）
- 公共接口函数 create_bowyer_watson_mesh
"""

import numpy as np
from typing import List, Tuple, Optional

from utils.message import verbose
from utils.timer import TimeSpan
from utils.geom_toolkit import point_in_polygon


def _extract_boundary_loops_from_fronts(boundary_front) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """从 boundary_front 中提取边界环（loop）。

    算法：
    1. 将所有 Front 边组织为邻接表
    2. 追踪连续的边形成环
    3. 使用包含关系区分外边界和孔洞

    返回:
        (outer_loops, hole_loops): 外边界环列表和孔洞环列表
    """
    adjacency = {}
    node_coords = {}

    for front in boundary_front:
        if len(front.node_elems) < 2:
            continue
        node1 = front.node_elems[0]
        node2 = front.node_elems[1]
        hash1, hash2 = node1.hash, node2.hash
        coord1, coord2 = node1.coords, node2.coords
        node_coords[hash1] = coord1
        node_coords[hash2] = coord2
        adjacency.setdefault(hash1, []).append(hash2)
        adjacency.setdefault(hash2, []).append(hash1)

    visited_edges = set()
    visited_nodes = set()
    loops = []

    for start_node in adjacency:
        if start_node in visited_nodes:
            continue
        loop = []
        current = start_node
        prev = None

        while True:
            if current in node_coords:
                loop.append(node_coords[current])
                visited_nodes.add(current)
            if prev is not None:
                visited_edges.add(tuple(sorted([prev, current])))

            neighbors = adjacency.get(current, [])
            next_node = None
            for neighbor in neighbors:
                if tuple(sorted([current, neighbor])) not in visited_edges:
                    next_node = neighbor
                    break
            if next_node is None:
                break
            prev = current
            current = next_node
            if current == start_node:
                break

        if len(loop) >= 3:
            loops.append(np.array(loop))

    if not loops:
        return [], []

    centroids = [np.mean(loop, axis=0) for loop in loops]

    def polygon_area(loop):
        area = 0.0
        n = len(loop)
        for i in range(n):
            p1, p2 = loop[i], loop[(i + 1) % n]
            area += (p2[0] - p1[0]) * (p2[1] + p1[1])
        return abs(area) / 2.0

    areas = [polygon_area(loop) for loop in loops]
    outer_loops, hole_loops = [], []

    for i, loop in enumerate(loops):
        is_inside_any = False
        for j, other_loop in enumerate(loops):
            if i == j:
                continue
            if point_in_polygon(centroids[i], other_loop) and areas[j] > areas[i]:
                is_inside_any = True
                break
        (hole_loops if is_inside_any else outer_loops).append(loop)

    return outer_loops, hole_loops


def create_bowyer_watson_mesh(
    boundary_front,
    sizing_system,
    target_triangle_count: Optional[int] = None,
    max_edge_length: Optional[float] = None,
    smoothing_iterations: int = 0,
    seed: Optional[int] = None,
    holes: Optional[List[np.ndarray]] = None,
    auto_detect_holes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bowyer-Watson 网格生成公共接口。

    参数:
        boundary_front: 边界阵面列表（Front 对象）
        sizing_system: QuadtreeSizing 尺寸场对象
        target_triangle_count: 目标三角形数量（可选）
        max_edge_length: 全局最大边长（可选）
        smoothing_iterations: Laplacian 平滑迭代次数（默认0）
        seed: 随机种子
        holes: 孔洞边界列表
        auto_detect_holes: 是否自动检测孔洞

    返回:
        (points, simplices, boundary_mask)
    """
    from delaunay.bw_core import BowyerWatsonMeshGenerator

    timer = TimeSpan("开始 Bowyer-Watson 网格生成流程...")
    final_holes = holes or []

    if auto_detect_holes:
        verbose("自动检测边界环...")
        outer_loops, hole_loops = _extract_boundary_loops_from_fronts(boundary_front)
        if hole_loops:
            verbose(f"  检测到 {len(hole_loops)} 个顺时针环（孔洞）")
            final_holes = list(final_holes) + hole_loops
        if outer_loops:
            verbose(f"  检测到 {len(outer_loops)} 个逆时针环（外边界）")

    boundary_points = []
    boundary_edges = []
    node_index_map = {}
    current_idx = 0

    for front in boundary_front:
        for node_elem in front.node_elems:
            node_hash = node_elem.hash
            if node_hash not in node_index_map:
                node_index_map[node_hash] = current_idx
                boundary_points.append(node_elem.coords)
                current_idx += 1
        if len(front.node_elems) >= 2:
            idx1 = node_index_map[front.node_elems[0].hash]
            idx2 = node_index_map[front.node_elems[1].hash]
            boundary_edges.append((idx1, idx2))

    boundary_points = np.array(boundary_points)
    verbose(f"边界点数: {len(boundary_points)}")
    verbose(f"边界边数: {len(boundary_edges)}")
    if final_holes:
        verbose(f"孔洞数: {len(final_holes)}")

    generator = BowyerWatsonMeshGenerator(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        sizing_system=sizing_system,
        max_edge_length=max_edge_length,
        smoothing_iterations=smoothing_iterations,
        seed=seed,
        holes=final_holes if final_holes else None,
    )

    points, simplices, boundary_mask = generator.generate_mesh(target_triangle_count=target_triangle_count)
    timer.show_to_console("Bowyer-Watson 网格生成流程完成")
    return points, simplices, boundary_mask

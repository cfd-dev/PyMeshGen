"""Shared front-processing helpers for Delaunay mesh generation."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
try:
    from utils.geom_toolkit import point_in_polygon
except ModuleNotFoundError:
    from geom_toolkit import point_in_polygon


@dataclass(frozen=True)
class BoundaryInput:
    """Boundary representation extracted from front objects."""

    boundary_points: np.ndarray
    boundary_edges: List[Tuple[int, int]]
    holes: List[np.ndarray]
    outer_boundary: Optional[np.ndarray]


def _build_front_graph(fronts) -> Tuple[dict, dict]:
    """Build an undirected graph from boundary fronts."""
    adjacency = {}
    node_coords = {}

    for front in fronts:
        if len(front.node_elems) < 2:
            continue
        n1, n2 = front.node_elems[0], front.node_elems[1]
        h1, h2 = n1.hash, n2.hash
        c1, c2 = n1.coords, n2.coords

        node_coords[h1] = c1
        node_coords[h2] = c2
        adjacency.setdefault(h1, []).append(h2)
        adjacency.setdefault(h2, []).append(h1)

    return adjacency, node_coords


def _trace_boundary_loops(adjacency: dict, node_coords: dict) -> List[np.ndarray]:
    """Trace closed loops from the front adjacency graph."""
    visited_edges = set()
    loops = []

    for start_node in adjacency:
        if start_node in visited_edges:
            continue

        loop = []
        current = start_node
        prev = None

        while True:
            if current in node_coords:
                loop.append(node_coords[current])

            if prev is not None:
                visited_edges.add(tuple(sorted((prev, current))))

            neighbors = adjacency.get(current, [])
            next_node = None
            for neighbor in neighbors:
                edge = tuple(sorted((current, neighbor)))
                if edge not in visited_edges:
                    next_node = neighbor
                    break

            if next_node is None:
                break

            prev = current
            current = next_node

            if current == start_node:
                break

        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=float))

    return loops


def _polygon_area(loop: np.ndarray) -> float:
    """Return the absolute polygon area."""
    area = 0.0
    node_count = len(loop)
    for i in range(node_count):
        p1, p2 = loop[i], loop[(i + 1) % node_count]
        area += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return abs(area) / 2.0


def _classify_boundary_loops(loops: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split traced loops into outer boundaries and hole loops."""
    if not loops:
        return [], []

    centroids = [np.mean(loop, axis=0) for loop in loops]
    areas = [_polygon_area(loop) for loop in loops]
    outer_loops, hole_loops = [], []

    for i, loop in enumerate(loops):
        is_inside = any(
            point_in_polygon(centroids[i], other_loop) and areas[j] > areas[i]
            for j, other_loop in enumerate(loops)
            if i != j
        )
        (hole_loops if is_inside else outer_loops).append(loop)

    return outer_loops, hole_loops


def _extract_boundary_points_and_edges(fronts) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extract unique boundary points and edges from front objects."""
    boundary_points = []
    boundary_edges = []
    node_index_map = {}
    current_idx = 0

    for front in fronts:
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

    return np.asarray(boundary_points, dtype=float), boundary_edges


def _build_boundary_input(
    boundary_front,
    holes: Optional[Sequence[np.ndarray]] = None,
    auto_detect_holes: bool = True,
) -> BoundaryInput:
    """Build the normalized boundary input used by all Delaunay backends."""
    final_holes = list(holes) if holes else []
    outer_boundary = None

    if auto_detect_holes:
        outer_loops, hole_loops = extract_boundary_loops(boundary_front)
        if hole_loops:
            final_holes.extend(hole_loops)
        if outer_loops:
            outer_boundary = outer_loops[0]

    boundary_points, boundary_edges = _extract_boundary_points_and_edges(boundary_front)
    return BoundaryInput(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        holes=final_holes,
        outer_boundary=outer_boundary,
    )


def _normalize_backend_name(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized in {"triangle", "bowyer_watson"}:
        return normalized
    return "bowyer_watson"


def extract_boundary_loops(fronts) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """从 fronts 中提取边界环。

    Args:
        fronts: Front 对象列表
    
    Returns:
        (outer_loops, hole_loops): 外边界和孔洞列表
    """
    adjacency, node_coords = _build_front_graph(fronts)
    loops = _trace_boundary_loops(adjacency, node_coords)
    return _classify_boundary_loops(loops)


def create_bowyer_watson_mesh(
    boundary_front,
    sizing_system,
    target_triangle_count: Optional[int] = None,
    max_edge_length: Optional[float] = None,
    smoothing_iterations: int = 0,
    seed: Optional[int] = None,
    holes: Optional[List[np.ndarray]] = None,
    auto_detect_holes: bool = True,
    backend: str = "bowyer_watson",
    triangle_point_strategy: str = "equilateral",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bowyer-Watson 网格生成公共接口。

    Args:
        boundary_front: 边界阵面列表
        sizing_system: QuadtreeSizing 尺寸场
        target_triangle_count: 目标三角形数
        max_edge_length: 最大边长
        smoothing_iterations: 平滑迭代次数
        seed: 随机种子
        holes: 孔洞边界
        auto_detect_holes: 自动检测孔洞
        backend: Delaunay 后端，"bowyer_watson" 或 "triangle"
        triangle_point_strategy: Triangle 后端内部点生成策略
    
    Returns:
        (points, simplices, boundary_mask)
    """
    boundary_input = _build_boundary_input(
        boundary_front=boundary_front,
        holes=holes,
        auto_detect_holes=auto_detect_holes,
    )
    normalized_backend = _normalize_backend_name(backend)
    if normalized_backend == "triangle":
        from .triangle_backend import create_triangle_mesh

        return create_triangle_mesh(
            boundary_points=boundary_input.boundary_points,
            boundary_edges=boundary_input.boundary_edges,
            sizing_system=sizing_system,
            holes=boundary_input.holes or None,
            outer_boundary=boundary_input.outer_boundary,
            seed=seed,
            point_strategy=triangle_point_strategy,
        )

    from .bw_core_stable import GmshBowyerWatsonMeshGenerator

    generator = GmshBowyerWatsonMeshGenerator(
        boundary_points=boundary_input.boundary_points,
        boundary_edges=boundary_input.boundary_edges,
        sizing_system=sizing_system,
        max_edge_length=max_edge_length,
        smoothing_iterations=smoothing_iterations,
        seed=seed,
        holes=boundary_input.holes or None,
        outer_boundary=boundary_input.outer_boundary,
    )

    return generator.generate_mesh(target_triangle_count=target_triangle_count)

"""
Bowyer-Watson 网格生成器 - 辅助函数

包含：
- 边界环提取
- 公共接口 create_bowyer_watson_mesh
"""

import numpy as np
from typing import List, Tuple, Optional
try:
    from utils.geom_toolkit import point_in_polygon
except ModuleNotFoundError:
    from geom_toolkit import point_in_polygon


def extract_boundary_loops(fronts) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """从 fronts 中提取边界环。

    Args:
        fronts: Front 对象列表
    
    Returns:
        (outer_loops, hole_loops): 外边界和孔洞列表
    """
    # 构建邻接表
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
    
    # 追踪环
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
                if tuple(sorted((current, neighbor))) not in visited_edges:
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
    
    # 区分外边界和孔洞
    centroids = [np.mean(loop, axis=0) for loop in loops]
    
    def polygon_area(loop):
        area = 0.0
        n = len(loop)
        for i in range(n):
            p1, p2 = loop[i], loop[(i+1) % n]
            area += (p2[0] - p1[0]) * (p2[1] + p1[1])
        return abs(area) / 2.0
    
    areas = [polygon_area(loop) for loop in loops]
    outer_loops, hole_loops = [], []
    
    for i, loop in enumerate(loops):
        is_inside = any(
            point_in_polygon(centroids[i], other_loop) and areas[j] > areas[i]
            for j, other_loop in enumerate(loops) if i != j
        )
        (hole_loops if is_inside else outer_loops).append(loop)
    
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
    use_gmsh_implementation: bool = True,
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
        use_gmsh_implementation: 使用 Gmsh 实现
        backend: Delaunay 后端，"bowyer_watson" 或 "triangle"
        triangle_point_strategy: Triangle 后端内部点生成策略
    
    Returns:
        (points, simplices, boundary_mask)
    """
    # 自动检测孔洞
    final_holes = list(holes) if holes else []
    outer_boundary = None
    
    if auto_detect_holes:
        outer_loops, hole_loops = extract_boundary_loops(boundary_front)
        if hole_loops:
            final_holes.extend(hole_loops)
        if outer_loops:
            outer_boundary = outer_loops[0]
    
    # 提取边界点和边
    boundary_points = []
    boundary_edges = []
    node_index_map = {}
    current_idx = 0
    
    for front in boundary_front:
        for node_elem in front.node_elems:
            h = node_elem.hash
            if h not in node_index_map:
                node_index_map[h] = current_idx
                boundary_points.append(node_elem.coords)
                current_idx += 1
        
        if len(front.node_elems) >= 2:
            idx1 = node_index_map[front.node_elems[0].hash]
            idx2 = node_index_map[front.node_elems[1].hash]
            boundary_edges.append((idx1, idx2))
    
    boundary_points = np.array(boundary_points)

    normalized_backend = str(backend).strip().lower()
    if normalized_backend == "triangle":
        from delaunay.triangle_backend import create_triangle_mesh

        return create_triangle_mesh(
            boundary_points=boundary_points,
            boundary_edges=boundary_edges,
            sizing_system=sizing_system,
            holes=final_holes if final_holes else None,
            outer_boundary=outer_boundary,
            seed=seed,
            point_strategy=triangle_point_strategy,
        )

    from delaunay.bw_core_stable import (
        BowyerWatsonMeshGenerator,
        GmshBowyerWatsonMeshGenerator,
    )

    # 选择实现：默认走更成熟的 Gmsh 版本以保证边界恢复与质量稳定
    if use_gmsh_implementation:
        GeneratorClass = GmshBowyerWatsonMeshGenerator
    else:
        GeneratorClass = BowyerWatsonMeshGenerator
    
    # 创建生成器并生成网格
    generator = GeneratorClass(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        sizing_system=sizing_system,
        max_edge_length=max_edge_length,
        smoothing_iterations=smoothing_iterations,
        seed=seed,
        holes=final_holes if final_holes else None,
        outer_boundary=outer_boundary
    )
    
    return generator.generate_mesh(target_triangle_count=target_triangle_count)

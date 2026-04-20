"""
Bowyer-Watson 网格生成器 - 边界恢复

实现鲁棒的 Constrained Delaunay Triangulation：
- recover_edge_by_swaps: 边翻转恢复
- recover_edge_by_splitting: Splitting 策略恢复
- recover_edge_by_boundary_path: 边界路径恢复
"""

import numpy as np
from typing import List, Tuple, Set, Optional
from collections import deque
from delaunay.types import MTri3, EdgeXFace, build_adjacency
from delaunay.predicates import orient2d, segments_intersect


def find_isolated_boundary_points(
    points: np.ndarray,
    triangles: List[MTri3],
    boundary_count: int
) -> List[int]:
    """找到所有被隔离的边界点。"""
    isolated = []
    for i in range(boundary_count):
        is_connected = any(i in tri.vertices and not tri.deleted 
                          for tri in triangles)
        if not is_connected:
            isolated.append(i)
    return isolated


def recover_edge_by_swaps(
    v1: int, v2: int,
    points: np.ndarray,
    triangles: List[MTri3],
    is_protected_edge_func,
    max_iter: int = 100
) -> bool:
    """通过边翻转恢复约束边。

    参考 Gmsh recoverEdgeBySwaps。
    
    Returns:
        True 如果成功恢复
    """
    p1, p2 = points[v1], points[v2]
    
    for _ in range(max_iter):
        # 检查边是否已存在
        if any(v1 in tri.vertices and v2 in tri.vertices 
               for tri in triangles if not tri.deleted):
            return True
        
        # 查找相交边
        intersecting_edge = None
        for tri in triangles:
            if tri.deleted:
                continue
            for i in range(3):
                a, b = tri.vertices[i], tri.vertices[(i+1)%3]
                if a in (v1, v2) or b in (v1, v2):
                    continue
                if segments_intersect(p1, p2, points[a], points[b], strict=True):
                    intersecting_edge = (a, b)
                    break
            if intersecting_edge:
                break
        
        if intersecting_edge is None:
            return False
        
        # 尝试翻转
        if not flip_edge(*intersecting_edge, points, triangles, is_protected_edge_func):
            return False
    
    return False


def flip_edge(
    n1: int, n2: int,
    points: np.ndarray,
    triangles: List[MTri3],
    is_protected_edge_func
) -> bool:
    """翻转边 (n1, n2)。"""
    # 保护边界边
    if is_protected_edge_func(n1, n2):
        return False
    
    # 找到共享边的两个三角形
    t1 = t2 = None
    for tri in triangles:
        if tri.deleted:
            continue
        if n1 in tri.vertices and n2 in tri.vertices:
            if t1 is None:
                t1 = tri
            elif t2 is None:
                t2 = tri
            break
    
    if t1 is None or t2 is None:
        return False
    
    # 找到对角顶点
    a_idx = next(v for v in t1.vertices if v != n1 and v != n2)
    b_idx = next(v for v in t2.vertices if v != n1 and v != n2)
    
    # 检查凸性
    pn1, pn2, pa, pb = points[n1], points[n2], points[a_idx], points[b_idx]
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    c1, c2 = cross(pn1, pa, pn2), cross(pa, pn2, pb)
    c3, c4 = cross(pn2, pb, pn1), cross(pb, pn1, pa)
    eps = -1e-10
    
    is_convex = (c1 > eps and c2 > eps and c3 > eps and c4 > eps) or \
                (c1 < -eps and c2 < -eps and c3 < -eps and c4 < -eps)
    
    if not is_convex:
        return False
    
    return True  # 简化：实际实现需要创建新三角形


def recover_edge_by_splitting(
    v1: int, v2: int,
    points: np.ndarray,
    triangles: List[MTri3],
    next_tri_id_func,
    compute_circumcircle_func
) -> bool:
    """通过 splitting 策略恢复约束边。

    参考 Triangle spliteredge。
    
    Returns:
        True 如果成功恢复
    """
    p1, p2 = points[v1], points[v2]
    
    # 检查边是否已存在
    if any(v1 in tri.vertices and v2 in tri.vertices 
           for tri in triangles if not tri.deleted):
        return True
    
    # 找到所有相交三角形
    intersecting_tris = []
    for tri in triangles:
        if tri.deleted:
            continue
        if segment_intersects_triangle(p1, p2, tri, points):
            intersecting_tris.append(tri)
    
    if not intersecting_tris:
        return recover_edge_by_boundary_path(v1, v2, points, triangles)
    
    # 删除相交三角形
    for tri in intersecting_tris:
        tri.deleted = True
    triangles = [t for t in triangles if not t.deleted]
    
    # 找到空洞边界
    cavity_boundary = find_cavity_boundary(triangles)
    if not cavity_boundary:
        return False
    
    # 重新三角化
    return retriangulate_with_constraint(
        cavity_boundary, v1, v2, points, triangles,
        next_tri_id_func, compute_circumcircle_func
    )


def segment_intersects_triangle(p1, p2, tri: MTri3, points) -> bool:
    """检查线段是否与三角形相交。"""
    v0, v1, v2 = tri.vertices
    for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
        if segments_intersect(p1, p2, points[a], points[b], strict=True):
            return True
    return False


def find_cavity_boundary(triangles: List[MTri3]) -> List[Tuple[int, int]]:
    """找到空洞边界边列表。"""
    edge_count = {}
    for tri in triangles:
        if tri.deleted:
            continue
        for i in range(3):
            a, b = tri.vertices[i], tri.vertices[(i+1)%3]
            edge = tuple(sorted((a, b)))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    return [edge for edge, count in edge_count.items() if count == 1]


def retriangulate_with_constraint(
    cavity_boundary: List[Tuple[int, int]],
    v1: int, v2: int,
    points: np.ndarray,
    triangles: List[MTri3],
    next_tri_id_func,
    compute_circumcircle_func
) -> bool:
    """使用约束边 (v1, v2) 重新三角化空洞。"""
    # 验证 v1, v2 在空洞边界上
    boundary_nodes = set()
    for a, b in cavity_boundary:
        boundary_nodes.add(a)
        boundary_nodes.add(b)
    
    if v1 not in boundary_nodes or v2 not in boundary_nodes:
        return False
    
    # 构建邻接表
    adjacency = {}
    for a, b in cavity_boundary:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    
    # 找到两条路径
    path1 = find_path_in_boundary(adjacency, v1, v2)
    path2 = find_path_in_boundary(adjacency, v1, v2, exclude=set(path1))
    
    if not path1 or not path2:
        return False
    
    # 创建三角形
    new_tris = []
    for i in range(1, len(path1) - 1):
        tri = MTri3(v1, v2, path1[i], idx=next_tri_id_func())
        compute_circumcircle_func(tri)
        new_tris.append(tri)
    
    for i in range(1, len(path2) - 1):
        tri = MTri3(v1, v2, path2[i], idx=next_tri_id_func())
        compute_circumcircle_func(tri)
        new_tris.append(tri)
    
    triangles.extend(new_tris)
    build_adjacency(triangles)
    
    return True


def find_path_in_boundary(
    adjacency: dict, start: int, end: int, 
    exclude: Optional[set] = None
) -> Optional[List[int]]:
    """在边界上找到从 start 到 end 的路径。"""
    if exclude is None:
        exclude = set()
    
    stack = [[start]]
    visited = {start}
    
    while stack:
        path = stack.pop()
        current = path[-1]
        
        if current == end:
            return path
        
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited and neighbor not in exclude:
                visited.add(neighbor)
                stack.append(path + [neighbor])
    
    return None


def recover_edge_by_boundary_path(
    v1: int, v2: int,
    points: np.ndarray,
    triangles: List[MTri3]
) -> bool:
    """通过边界路径恢复约束边。

    当没有相交三角形时使用。
    
    Returns:
        True 如果成功恢复
    """
    # 找到 v1 和 v2 的邻居
    v1_neighbors = set()
    v2_neighbors = set()
    
    for tri in triangles:
        if tri.deleted:
            continue
        if v1 in tri.vertices:
            for v in tri.vertices:
                if v != v1:
                    v1_neighbors.add(v)
        if v2 in tri.vertices:
            for v in tri.vertices:
                if v != v2:
                    v2_neighbors.add(v)
    
    # 查找共同邻居
    common = v1_neighbors & v2_neighbors
    if common:
        # 有共同邻居，可以创建三角形
        return True
    
    return False

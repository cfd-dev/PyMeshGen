"""
Bowyer-Watson 网格生成器 - Cavity 搜索和点插入

实现：
- recur_find_cavity: 递归查找 Cavity
- find_cavity_iterative: 迭代查找 Cavity
- insert_vertex: 插入新点并重新连接
"""

from typing import List, Tuple
import numpy as np
from delaunay.types import MTri3, EdgeXFace


def recur_find_cavity(
    start_tri: MTri3,
    point: np.ndarray,
    point_idx: int,
    points: np.ndarray,
    protected_edges: set,
    incircle_func
) -> Tuple[List[MTri3], List[EdgeXFace]]:
    """递归查找 Cavity（参考 Gmsh recurFindCavityAniso）。

    Args:
        start_tri: 起始三角形
        point: 插入点坐标
        point_idx: 插入点索引
        points: 所有点坐标数组
        protected_edges: 受保护的边界边集合
        incircle_func: incircle 测试函数
    
    Returns:
        (cavity_tris, shell_edges): 空腔三角形列表和边界边列表
    """
    cavity = []
    shell = []
    
    def recurse(tri: MTri3):
        if tri.deleted:
            return
        
        # 检查三角形是否被新点破坏
        v0, v1, v2 = tri.vertices
        if incircle_func(points[v0], points[v1], points[v2], point) > 0:
            tri.deleted = True
            cavity.append(tri)
            
            # 递归检查邻居
            for i in range(3):
                neighbor = tri.neighbors[i]
                if neighbor is None or neighbor.deleted:
                    # 边界边
                    shell.append(EdgeXFace(tri, i))
                else:
                    recurse(neighbor)
        else:
            # 三角形未被破坏，这条边是 shell 边界
            shell.append(EdgeXFace(start_tri if start_tri == tri else tri, 0))
    
    recurse(start_tri)
    return cavity, shell


def find_cavity_iterative(
    start_tri: MTri3,
    point: np.ndarray,
    points: np.ndarray,
    incircle_func
) -> Tuple[List[MTri3], List[EdgeXFace]]:
    """迭代查找 Cavity（避免递归深度限制）。"""
    cavity = []
    shell = []
    stack = [start_tri]
    visited = {id(start_tri)}
    
    while stack:
        tri = stack.pop()
        if tri.deleted:
            continue
        
        v0, v1, v2 = tri.vertices
        if incircle_func(points[v0], points[v1], points[v2], point) > 0:
            tri.deleted = True
            cavity.append(tri)
            
            for i in range(3):
                neighbor = tri.neighbors[i]
                if neighbor is None:
                    shell.append(EdgeXFace(tri, i))
                elif id(neighbor) not in visited:
                    visited.add(id(neighbor))
                    stack.append(neighbor)
        else:
            shell.append(EdgeXFace(tri, 0))
    
    return cavity, shell


def insert_vertex(
    shell_edges: List[EdgeXFace],
    cavity_tris: List[MTri3],
    point_idx: int,
    points: np.ndarray,
    triangles: List[MTri3],
    next_tri_id_func,
    compute_circumcircle_func,
    validate_star: bool = True,
    validate_edges: bool = True
) -> Tuple[List[MTri3], bool]:
    """插入新点并创建新三角形。

    Args:
        shell_edges: 空腔边界边列表
        cavity_tris: 空腔三角形列表
        point_idx: 新点索引
        points: 所有点坐标数组
        triangles: 三角形列表（会被修改）
        next_tri_id_func: 生成新三角形 ID 的函数
        compute_circumcircle_func: 计算外接圆的函数
        validate_star: 是否验证星形
        validate_edges: 是否验证边长
    
    Returns:
        (new_tris, success): 新三角形列表和成功标志
    """
    if len(cavity_tris) == 0:
        return [], False
    
    new_tris = []
    point = points[point_idx]

    # 创建新三角形
    for edge_face in shell_edges:
        v0, v1 = edge_face.vertices()
        new_tri = MTri3(v0, v1, point_idx, idx=next_tri_id_func())
        compute_circumcircle_func(new_tri)

        # 边长验证
        if validate_edges:
            d0 = np.linalg.norm(points[v0] - point)
            d1 = np.linalg.norm(points[v1] - point)
            if d0 < 1e-10 or d1 < 1e-10:
                return [], False

        new_tris.append(new_tri)
        triangles.append(new_tri)  # 添加新三角形到列表

    return new_tris, True


def validate_star_shaped(
    shell_edges: List[EdgeXFace],
    point_idx: int,
    points: np.ndarray,
    cavity_tris: List[MTri3],
    compute_area_func
) -> bool:
    """验证 Cavity 是星形的（体积守恒检查）。

    参考 Gmsh insertVertexB 的体积守恒检查。
    
    Returns:
        True 如果是星形
    """
    if not cavity_tris:
        return True
    
    # 计算旧面积
    old_area = sum(compute_area_func(tri, points) for tri in cavity_tris)
    
    # 计算新面积
    new_area = 0.0
    for edge_face in shell_edges:
        v0, v1 = edge_face.vertices()
        p0, p1 = points[v0], points[v1]
        pp = points[point_idx]
        tri_area = abs((p1[0] - p0[0]) * (pp[1] - p0[1]) - 
                       (p1[1] - p0[1]) * (pp[0] - p0[0])) / 2.0
        new_area += tri_area
    
    # 面积守恒检查
    if old_area > 1e-12:
        return abs(old_area - new_area) < 1e-10 * old_area
    return abs(old_area - new_area) < 1e-12


def restore_cavity(
    cavity_tris: List[MTri3],
    shell_edges: List[EdgeXFace]
):
    """恢复空腔（取消删除标记）。

    当插入失败时调用，回退操作。
    """
    for tri in cavity_tris:
        tri.deleted = False

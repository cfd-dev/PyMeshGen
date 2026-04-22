"""
Bowyer-Watson Delaunay 网格生成器 - Cavity 搜索算法

实现 Gmsh 风格的递归空腔搜索算法：
- recurFindCavityAniso: 递归查找违反 Delaunay 的三角形
- insertVertexB: 插入顶点并重新三角化
- 星形空腔验证

参考: Gmsh meshGFaceDelaunayInsertion.cpp 中的 recurFindCavityAniso 和 insertVertexB
"""

from typing import List, Tuple, Set, Optional
import numpy as np

from .bw_types import MTri3, EdgeXFace, collect_cavity_shell, compute_cavity_volume


# =============================================================================
# 递归空腔搜索（Gmsh recurFindCavityAniso 风格）
# =============================================================================

def recur_find_cavity(
    start_tri: MTri3,
    point: np.ndarray,
    point_idx: int,
    points: np.ndarray,
    protected_edges: Set[frozenset],
    in_circumcircle_func
) -> Tuple[List[MTri3], List[EdgeXFace]]:
    """递归查找空腔（Cavity）。
    
    参考 Gmsh recurFindCavityAniso 算法：
    1. 标记当前三角形为已删除
    2. 将当前三角形加入空腔
    3. 遍历三个邻居：
       a. 如果是保护边，加入 shell，不继续递归
       b. 如果邻居不存在，加入 shell
       c. 如果邻居存在且未删除：
          - 检查邻居是否被新点破坏（inCircumCircle）
          - 如果是，递归处理邻居
          - 如果否，加入 shell
    
    参数:
        start_tri: 起始三角形
        point: 插入点坐标
        point_idx: 插入点索引
        points: 所有点坐标数组
        protected_edges: 受保护的边界边集合
        in_circumcircle_func: 点在圆内测试函数
    
    返回:
        (cavity_triangles, shell_edges): 空腔三角形列表和边界边列表
    
    递归逻辑：
        cavity: 包含所有需要删除的三角形
        shell: 包含空腔的边界边
        根据 Euler 公式: shell.size() == cavity.size() + 2
    """
    cavity_triangles = []
    shell_edges = []
    cavity_set = set()  # 用于快速查找
    
    def _recurse(tri: MTri3):
        """递归函数。"""
        tri_id = id(tri)
        
        # 基线条件：已处理或已删除
        if tri_id in cavity_set:
            return
        if tri.is_deleted():
            return
        
        # 检查是否在圆内
        if not in_circumcircle_func(point, tri):
            return
        
        # 标记为已删除并加入空腔
        tri.set_deleted(True)
        cavity_set.add(tri_id)
        cavity_triangles.append(tri)
        
        # 遍历三个邻居
        for i in range(3):
            edge = tri.get_edge_sorted(i)
            neighbor = tri.get_neighbor(i)
            
            # 检查是否是受保护的边界边
            if frozenset(edge) in protected_edges:
                # 是保护边，加入 shell，不继续递归
                shell_edges.append(EdgeXFace(tri, i))
                continue
            
            if neighbor is None:
                # 边界上的边，加入 shell
                shell_edges.append(EdgeXFace(tri, i))
            elif not neighbor.is_deleted():
                # 邻居存在且未删除，递归检查
                if in_circumcircle_func(point, neighbor):
                    # 邻居也在圆内，递归处理
                    _recurse(neighbor)
                else:
                    # 邻居不在圆内，这条边是 shell 边界
                    shell_edges.append(EdgeXFace(tri, i))
    
    # 开始递归
    _recurse(start_tri)
    
    return cavity_triangles, shell_edges


def find_cavity_iterative(
    start_tri: MTri3,
    point: np.ndarray,
    point_idx: int,
    points: np.ndarray,
    protected_edges: Set[frozenset],
    in_circumcircle_func
) -> Tuple[List[MTri3], List[EdgeXFace]]:
    """迭代版本的空腔搜索（避免递归深度限制）。
    
    使用显式栈模拟递归，适用于大规模网格。
    
    参数:
        同 recur_find_cavity
    
    返回:
        (cavity_triangles, shell_edges)
    """
    cavity_triangles = []
    shell_edges = []
    cavity_set = set()
    
    # 使用栈模拟递归
    stack = [start_tri]
    
    while stack:
        tri = stack.pop()
        tri_id = id(tri)
        
        # 基线条件
        if tri_id in cavity_set:
            continue
        if tri.is_deleted():
            continue
        if not in_circumcircle_func(point, tri):
            continue
        
        # 标记为已删除并加入空腔
        tri.set_deleted(True)
        cavity_set.add(tri_id)
        cavity_triangles.append(tri)
        
        # 遍历三个邻居
        for i in range(3):
            edge = tri.get_edge_sorted(i)
            neighbor = tri.get_neighbor(i)
            
            # 检查是否是受保护的边界边
            if frozenset(edge) in protected_edges:
                shell_edges.append(EdgeXFace(tri, i))
                continue
            
            if neighbor is None:
                shell_edges.append(EdgeXFace(tri, i))
            elif not neighbor.is_deleted():
                if in_circumcircle_func(point, neighbor):
                    stack.append(neighbor)
                else:
                    shell_edges.append(EdgeXFace(tri, i))
    
    return cavity_triangles, shell_edges


# =============================================================================
# 星形空腔验证（Gmsh insertVertexB 风格）
# =============================================================================

def validate_star_shaped(
    shell_edges: List[EdgeXFace],
    new_point_idx: int,
    cavity_triangles: List[MTri3],
    points: np.ndarray,
    tolerance: float = 1e-3
) -> bool:
    """验证空腔是星形的（新点能看到所有 shell 边）。
    
    参考 Gmsh insertVertexB 的体积守恒检查：
    |oldVolume - newVolume| < EPS * oldVolume
    
    这确保了插入点不会产生重叠或空洞。
    
    参数:
        shell_edges: 空腔边界边
        new_point_idx: 新插入点索引
        cavity_triangles: 空腔三角形列表
        points: 所有点坐标数组
        tolerance: 相对容差
    
    返回:
        True 如果空腔是星形的
    """
    if not cavity_triangles:
        return True
    
    # 计算旧空腔面积
    old_area = compute_cavity_volume(cavity_triangles, points)
    
    # 计算新三角形总面积
    new_area = 0.0
    new_point = points[new_point_idx]
    
    for edge_xface in shell_edges:
        v1, v2 = edge_xface.get_edge()
        p1, p2 = points[v1], points[v2]
        
        # 计算新三角形面积
        tri_area = 0.5 * abs(
            (p2[0] - p1[0]) * (new_point[1] - p1[1]) -
            (p2[1] - p1[1]) * (new_point[0] - p1[0])
        )
        new_area += tri_area
    
    # 面积守恒检查（相对误差 < tolerance）
    if old_area > 1e-12:
        return abs(old_area - new_area) < tolerance * old_area
    else:
        return abs(old_area - new_area) < tolerance


def validate_edge_lengths(
    shell_edges: List[EdgeXFace],
    new_point_idx: int,
    points: np.ndarray,
    min_edge_ratio: float = 0.01
) -> bool:
    """验证新三角形的边长不过小。
    
    参考 Gmsh insertVertexB：
    检查新点与 shell 边顶点的距离，避免产生退化三角形。
    
    参数:
        shell_edges: 空腔边界边
        new_point_idx: 新插入点索引
        points: 所有点坐标数组
        min_edge_ratio: 最小边长比例（相对于平均边长）
    
    返回:
        True 如果边长有效
    """
    new_point = points[new_point_idx]
    
    for edge_xface in shell_edges:
        v1, v2 = edge_xface.get_edge()
        p1, p2 = points[v1], points[v2]
        
        # 计算新点到边两端点的距离
        d1 = np.linalg.norm(new_point - p1)
        d2 = np.linalg.norm(new_point - p2)
        
        # 检查最小边长
        edge_length = np.linalg.norm(p2 - p1)
        min_allowed = edge_length * min_edge_ratio
        
        if d1 < min_allowed or d2 < min_allowed:
            return False
    
    return True


# =============================================================================
# 顶点插入与重新连接（Gmsh insertVertexB 风格）
# =============================================================================

def insert_vertex(
    shell_edges: List[EdgeXFace],
    cavity_triangles: List[MTri3],
    new_point_idx: int,
    points: np.ndarray,
    all_triangles: List[MTri3],
    validate_star: bool = True,
    validate_edges: bool = True,
) -> Tuple[List[MTri3], bool]:
    """插入顶点并重新三角化。
    
    参考 Gmsh insertVertexB：
    1. 验证空腔有效性
    2. 创建新三角形
    3. 检查边长和质量
    4. 连接新三角形邻居
    5. 从 all_triangles 中移除空腔三角形
    
    参数:
        shell_edges: 空腔边界边
        cavity_triangles: 空腔三角形列表
        new_point_idx: 新插入点索引
        points: 所有点坐标数组
        all_triangles: 所有三角形列表（会被修改）
        validate_star: 是否验证星形空腔
        validate_edges: 是否验证边长
    
    返回:
        (new_triangles, success): 新三角形列表和成功标志
    """
    # 步骤 1: 验证空腔有效性
    if len(cavity_triangles) == 0:
        return [], False  # 无法插入
    
    if validate_star:
        if not validate_star_shaped(shell_edges, new_point_idx, cavity_triangles, points):
            return [], False  # 非星形空腔
    
    if validate_edges:
        if not validate_edge_lengths(shell_edges, new_point_idx, points):
            return [], False  # 边长过小
    
    # 步骤 2: 创建新三角形
    new_triangles = []
    
    for edge_xface in shell_edges:
        v1, v2 = edge_xface.get_edge()
        p1, p2 = points[v1], points[v2]
        p_new = points[new_point_idx]

        tri_area = 0.5 * abs(
            (p2[0] - p1[0]) * (p_new[1] - p1[1]) -
            (p2[1] - p1[1]) * (p_new[0] - p1[0])
        )
        if tri_area <= 1e-14:
            return [], False
        
        # 创建新三角形
        new_tri = MTri3(v1, v2, new_point_idx)
        new_triangles.append(new_tri)
    
    # 步骤 3: 从 all_triangles 中移除空腔三角形
    cavity_ids = set(id(tri) for tri in cavity_triangles)
    all_triangles[:] = [tri for tri in all_triangles if id(tri) not in cavity_ids]
    
    # 步骤 4: 添加新三角形
    all_triangles.extend(new_triangles)
    
    # 步骤 5: 构建新三角形的邻接关系
    _connect_new_triangles(new_triangles, shell_edges, all_triangles)
    
    return new_triangles, True


def _connect_new_triangles(
    new_triangles: List[MTri3],
    shell_edges: List[EdgeXFace],
    all_triangles: List[MTri3]
):
    """连接新三角形的邻接关系。
    
    参考 Gmsh connectTris：
    1. 新三角形之间的邻接关系
    2. 新三角形与现有三角形的邻接关系
    """
    # 构建边到三角形的映射（仅新三角形）
    edge_to_tri = {}
    
    for tri in new_triangles:
        for i in range(3):
            edge_key = tri.get_edge_sorted(i)
            
            if edge_key in edge_to_tri:
                # 新三角形之间的邻接
                other_tri, other_local_idx = edge_to_tri[edge_key]
                tri.neighbors[i] = other_tri
                other_tri.neighbors[other_local_idx] = tri
            else:
                edge_to_tri[edge_key] = (tri, i)
    
    # 查找与现有三角形的邻接关系
    # 需要遍历所有三角形（包括新三角形）
    full_edge_map = {}
    
    for tri in all_triangles:
        for i in range(3):
            edge_key = tri.get_edge_sorted(i)
            
            if edge_key in full_edge_map:
                other_tri, other_local_idx = full_edge_map[edge_key]
                if other_tri is not tri:  # 避免自引用
                    tri.neighbors[i] = other_tri
                    other_tri.neighbors[other_local_idx] = tri
            else:
                full_edge_map[edge_key] = (tri, i)


def restore_cavity(cavity_triangles: List[MTri3]):
    """恢复空腔标记（插入失败时的回退操作）。
    
    参考 Gmsh insertVertexB 失败恢复：
    恢复所有三角形的删除标记。
    """
    for tri in cavity_triangles:
        tri.set_deleted(False)

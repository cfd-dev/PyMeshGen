"""
_splitting.py - 实现 splitting 策略，用于恢复被穿透的约束边

参考 Triangle 的 spliteredge 算法：
1. 找到所有与线段相交的三角形
2. 删除这些三角形
3. 使用线段重新三角化空洞
"""

from typing import List, Tuple
import numpy as np
from bw_types import MTri3, build_adjacency_from_triangles, find_triangles_intersecting_segment


def recover_edge_by_splitting(
    self,
    v1: int,
    v2: int
) -> bool:
    """通过 splitting 策略恢复约束边。

    参考 Triangle spliteredge 算法：
    1. 找到所有与线段 (v1,v2) 相交的三角形
    2. 删除这些三角形，形成空洞
    3. 在空洞边界上使用 (v1,v2) 重新三角化
    4. 递归处理被分割的子边

    参数:
        v1, v2: 约束边的两个端点
        
    返回:
        True 如果成功恢复
    """
    from utils.geom_toolkit import verbose
    
    p1, p2 = self.points[v1], self.points[v2]
    
    # 步骤 0: 检查边是否已存在
    if self._is_boundary_edge_in_mesh(v1, v2):
        verbose(f"    [信息] 边 ({v1},{v2}) 已存在")
        return True
    
    # 步骤 1: 找到所有相交的三角形
    intersecting_tris = find_triangles_intersecting_segment(
        self.triangles, p1, p2, self.points
    )
    
    if not intersecting_tris:
        # 没有相交的三角形，边可能已经存在
        verbose(f"    [信息] 没有找到与边 ({v1},{v2}) 相交的三角形")
        return self._is_boundary_edge_in_mesh(v1, v2)
    
    verbose(f"    [信息] 找到 {len(intersecting_tris)} 个与边 ({v1},{v2}) 相交的三角形")
    
    # 步骤 2: 删除相交三角形
    for tri in intersecting_tris:
        tri.set_deleted(True)
    
    self.triangles = [t for t in self.triangles if not t.is_deleted()]
    build_adjacency_from_triangles(self.triangles)
    
    # 步骤 3: 找到空洞边界
    cavity_boundary = self._find_cavity_boundary_edges()
    
    if not cavity_boundary:
        verbose(f"    [错误] 无法找到空洞边界")
        return False
    
    # 步骤 4: 使用 (v1,v2) 重新三角化空洞
    return self._retriangulate_cavity_with_edge(cavity_boundary, v1, v2)


def retriangulate_cavity_with_edge(
    self,
    cavity_boundary: List[Tuple[int, int]],
    v1: int,
    v2: int
) -> bool:
    """使用约束边 (v1,v2) 重新三角化空洞。

    参考 Triangle triangulatepolygon 逻辑：
    - 从 v1 开始，沿着空洞边界走到 v2
    - 创建三角形扇 (v1, vi, vi+1)
    - 从 v2 开始，沿着空洞边界回到 v1
    - 创建三角形扇 (v2, vi, vi+1)

    参数:
        cavity_boundary: 空洞边界边列表 [(a,b), ...]
        v1, v2: 约束边的两个端点
        
    返回:
        True 如果成功创建约束边
    """
    from utils.geom_toolkit import verbose
    
    # 验证 v1 和 v2 都在空洞边界上
    boundary_nodes = set()
    for edge in cavity_boundary:
        boundary_nodes.add(edge[0])
        boundary_nodes.add(edge[1])
    
    if v1 not in boundary_nodes or v2 not in boundary_nodes:
        verbose(f"    [警告] v1={v1} 或 v2={v2} 不在空洞边界上")
        return False
    
    # 构建空洞边界的邻接表
    adjacency = {}
    for a, b in cavity_boundary:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    
    # 从 v1 走到 v2，创建三角形扇
    new_triangles = []
    
    # 找到从 v1 到 v2 的两条路径
    path1 = self._find_path_in_boundary(adjacency, v1, v2)
    path2 = self._find_path_in_boundary(adjacency, v1, v2, exclude=path1)
    
    if not path1 or not path2:
        verbose(f"    [警告] 无法找到从 v1 到 v2 的两条路径")
        return False
    
    verbose(f"    [信息] 找到两条路径：path1={len(path1)} 个点，path2={len(path2)} 个点")
    
    # 沿着 path1 创建三角形扇 (v1, vi, vi+1)
    for i in range(len(path1) - 2):
        vi = path1[i + 1]
        vi1 = path1[i + 2]
        new_tri = MTri3(v1, vi, vi1, idx=self._next_tri_id())
        self._compute_circumcircle(new_tri)
        new_triangles.append(new_tri)
    
    # 沿着 path2 创建三角形扇 (v2, vi, vi+1)
    for i in range(len(path2) - 2):
        vi = path2[i + 1]
        vi1 = path2[i + 2]
        new_tri = MTri3(v2, vi, vi1, idx=self._next_tri_id())
        self._compute_circumcircle(new_tri)
        new_triangles.append(new_tri)
    
    self.triangles.extend(new_triangles)
    build_adjacency_from_triangles(self.triangles)
    
    verbose(f"    [信息] 创建了 {len(new_triangles)} 个新三角形")
    
    # 验证约束边是否已创建
    if self._is_boundary_edge_in_mesh(v1, v2):
        verbose(f"    [成功] 约束边 ({v1},{v2}) 已成功创建")
        return True
    else:
        verbose(f"    [警告] 约束边 ({v1},{v2}) 创建失败")
        return False


def find_path_in_boundary(
    self,
    adjacency: dict,
    start: int,
    end: int,
    exclude: List[int] = None
) -> List[int]:
    """在边界上找到从 start 到 end 的路径。
    
    使用简单的 BFS 路径查找。

    参数:
        adjacency: 邻接表 {node: [neighbors]}
        start: 起始节点
        end: 结束节点
        exclude: 排除的节点列表
        
    返回:
        路径节点列表 [start, ..., end]
    """
    from collections import deque
    
    if exclude is None:
        exclude = set()
    
    # 简单的 BFS 路径查找
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        if current == end:
            return path
        
        for neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue
            
            # 跳过被排除的节点
            if neighbor in exclude:
                continue
            
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    
    return None


def segment_intersects_triangle(p1, p2, tri, points: np.ndarray) -> bool:
    """检查线段 p1-p2 是否与三角形 tri 相交。
    
    参数:
        p1, p2: 线段的两个端点
        tri: 三角形
        points: 点坐标数组
        
    返回:
        True 如果相交
    """
    v0, v1, v2 = tri.vertices
    pt0, pt1, pt2 = points[v0], points[v1], points[v2]
    
    # 检查线段是否与三角形的三条边相交
    if segment_intersects_segment(p1, p2, pt0, pt1):
        return True
    if segment_intersects_segment(p1, p2, pt1, pt2):
        return True
    if segment_intersects_segment(p1, p2, pt0, pt2):
        return True
    
    return False


def segment_intersects_segment(p1, p2, v1, v2) -> bool:
    """检查线段 p1-p2 是否与线段 v1-v2 相交。
    
    参数:
        p1, p2: 第一个线段的两个端点
        v1, v2: 第二个线段的两个端点
        
    返回:
        True 如果相交
    """
    def ccw(a, b, c):
        """计算三点是否逆时针排列。"""
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    
    d1 = ccw(p1, p2, v1)
    d2 = ccw(p1, p2, v2)
    d3 = ccw(v1, v2, p1)
    d4 = ccw(v1, v2, p2)
    
    # 严格相交
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    return False


# 将函数绑定到 GmshBowyerWatsonMeshGenerator 类
def patch_bw_core_gmsh():
    """将 splitting 策略绑定到 bw_core_gmsh 模块。"""
    import delaunay.bw_core_gmsh as bw_module
    
    bw_module.GmshBowyerWatsonMeshGenerator._recover_edge_by_splitting = recover_edge_by_splitting
    bw_module.GmshBowyerWatsonMeshGenerator._retriangulate_cavity_with_edge = retriangulate_cavity_with_edge
    bw_module.GmshBowyerWatsonMeshGenerator._find_path_in_boundary = find_path_in_boundary
    bw_module.GmshBowyerWatsonMeshGenerator.segment_intersects_triangle = segment_intersects_triangle


if __name__ == "__main__":
    # 测试
    patch_bw_core_gmsh()
    print("splitting 策略已绑定到 bw_core_gmsh 模块")

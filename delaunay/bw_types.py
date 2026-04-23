"""
Bowyer-Watson Delaunay 网格生成器 - Gmsh 风格数据结构

参考 Gmsh 的核心数据结构：
- MTri3: 三角形包装类，带懒删除和邻接关系
- edgeXface: 边 - 面关系结构
- compareTri3Ptr: 三角形比较器

这些数据结构是 Gmsh Bowyer-Watson 实现的基础。
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np


# =============================================================================
# MTri3 - Gmsh 风格的三角形包装类
# =============================================================================

class MTri3:
    """Gmsh MTri3 风格的三角形包装类。

    参考 Gmsh MTri3 类：
    - 显式存储三个邻居三角形指针
    - 支持懒删除（deleted 标记）
    - 缓存外接圆信息
    - 顶点索引按升序存储

    关键设计：
    - deleted 标志用于懒删除，避免频繁的集合操作
    - neigh[3] 维护三角形邻接关系，支持 O(1) 的邻居访问
    - circum_radius 作为三角形质量度量
    """

    __slots__ = [
        'vertices',        # tuple: 排序后的顶点索引 (v0, v1, v2)
        'neighbors',       # list[MTri3|None]: 三个邻居三角形 [neigh0, neigh1, neigh2]
        'circumcenter',    # np.ndarray: 外接圆心 [x, y]
        'circumradius',    # float: 外接圆半径
        'deleted',         # bool: 懒删除标记
        'idx',             # int: 三角形唯一标识
        'quality',         # float: 质量度量 (0-1)
    ]

    def __init__(self, v1: int, v2: int, v3: int, idx: int = -1):
        """创建三角形，顶点索引自动排序。"""
        self.vertices = tuple(sorted([v1, v2, v3]))
        self.neighbors = [None, None, None]  # neighbors[i] 对应边 (vertices[i], vertices[(i+1)%3])
        self.circumcenter = None
        self.circumradius = None
        self.deleted = False
        self.idx = idx
        self.quality = 0.0

    def __eq__(self, other):
        """三角形相等性基于顶点索引。"""
        return isinstance(other, MTri3) and self.vertices == other.vertices

    def __hash__(self):
        return hash(self.vertices)

    def __repr__(self):
        status = "DELETED" if self.deleted else "ACTIVE"
        radius_str = f"{self.circumradius:.4f}" if self.circumradius is not None else "N/A"
        return f"MTri3({self.vertices}, r={radius_str}, {status})"

    def get_edge(self, i: int) -> Tuple[int, int]:
        """获取第 i 条边的两个顶点索引。"""
        v0, v1, v2 = self.vertices
        edges = [(v0, v1), (v1, v2), (v0, v2)]
        return edges[i]

    def get_edge_sorted(self, i: int) -> Tuple[int, int]:
        """获取第 i 条边（排序后的元组）。"""
        edge = self.get_edge(i)
        return tuple(sorted(edge))

    def get_neighbor_index(self, neighbor: 'MTri3') -> int:
        """获取邻居三角形的局部索引。"""
        for i, n in enumerate(self.neighbors):
            if n is neighbor:
                return i
        return -1

    def set_neighbor(self, i: int, tri: 'MTri3'):
        """设置第 i 个邻居。"""
        self.neighbors[i] = tri

    def get_neighbor(self, i: int) -> Optional['MTri3']:
        """获取第 i 个邻居。"""
        return self.neighbors[i]

    def has_neighbor(self) -> bool:
        """检查是否有非空邻居。"""
        return any(n is not None for n in self.neighbors)

    def is_deleted(self) -> bool:
        """检查三角形是否已删除。"""
        return self.deleted

    def set_deleted(self, val: bool):
        """设置删除标记。"""
        self.deleted = val

    def contains_vertex(self, v: int) -> bool:
        """检查是否包含指定顶点。"""
        return v in self.vertices

    def get_shared_edge(self, other: 'MTri3') -> Optional[Tuple[int, int]]:
        """获取与另一个三角形的共享边。

        返回:
            共享边的顶点元组 (v1, v2)，如果没有共享边则返回 None
        """
        if other is None:
            return None

        # 找到共同的顶点（应该有 2 个）
        common_vertices = set(self.vertices) & set(other.vertices)
        if len(common_vertices) == 2:
            return tuple(sorted(common_vertices))

        return None


# =============================================================================
# edgeXface - Gmsh 风格的边 - 面关系结构
# =============================================================================

class EdgeXFace:
    """Gmsh edgeXface 风格的边 - 面关系结构。

    参考 Gmsh edgeXface 结构和 Triangle 的 subsegment 概念：
    - 用于三角形连接关系的构建和维护
    - 支持空腔边界（Cavity Shell）的表示
    - 支持约束边（constrained edge）标记
    - 用于排序和查找相同的边

    关键设计：
    - 三角形和局部边索引的组合唯一标识一条边
    - is_constrained 标记约束边，防止被删除或翻转
    - neighbor 指针指向相邻的 EdgeXFace
    - 支持比较运算符（用于排序和查找）
    """

    __slots__ = ['triangle', 'local_edge_idx', 'is_constrained', 'neighbor']

    def __init__(self, tri: MTri3, edge_idx: int, is_constrained: bool = False):
        """创建边 - 面对。

        参数:
            tri: 包含该边的三角形
            edge_idx: 边在三角形中的局部索引 (0, 1, 2)
            is_constrained: 是否是约束边（不能被删除或翻转）
        """
        self.triangle = tri
        self.local_edge_idx = edge_idx
        self.is_constrained = is_constrained
        self.neighbor = None  # 相邻的 EdgeXFace（如果存在）

    def get_edge(self) -> Tuple[int, int]:
        """获取边的两个顶点（排序后）。"""
        return self.triangle.get_edge_sorted(self.local_edge_idx)

    def get_vertices(self) -> Tuple[int, int]:
        """获取边的两个顶点（排序后，同上）。"""
        return self.get_edge()

    def __eq__(self, other):
        """两条边相等当且仅当顶点相同。"""
        if not isinstance(other, EdgeXFace):
            return False
        return self.get_edge() == other.get_edge()

    def __hash__(self):
        return hash(self.get_edge())

    def __lt__(self, other):
        """用于排序：按边字典序。"""
        return self.get_edge() < other.get_edge()

    def __repr__(self):
        constrained_str = "CONSTRAINED" if self.is_constrained else ""
        return f"EdgeXFace(tri={self.triangle.idx}, edge={self.local_edge_idx}, vertices={self.get_edge()}, {constrained_str})"


# =============================================================================
# 三角形比较器（Gmsh compareTri3Ptr 风格）
# =============================================================================

def compare_triangles_by_radius(a: MTri3, b: MTri3) -> bool:
    """比较两个三角形的外接圆半径。

    参考 Gmsh compareTri3Ptr：
    - 按外接圆半径降序排列（最差的三角形在前）
    - 用于优先级队列排序

    返回:
        True 如果 a 的半径 > b 的半径
    """
    if a.circumradius is None or b.circumradius is None:
        return False
    return a.circumradius > b.circumradius


class TriangleComparator:
    """Gmsh compareTri3Ptr 风格的比较器。

    用于 std::set<MTri3*, compareTri3Ptr> 优先级队列。
    确保每次迭代都能访问到质量最差的三角形。
    """

    def __call__(self, a: MTri3, b: MTri3) -> bool:
        """比较两个三角形。

        返回 True 如果 a 应该在 b 之前（a 更差）。
        """
        if a.circumradius != b.circumradius:
            return a.circumradius > b.circumradius  # 半径大的在前
        return id(a) < id(b)  # 指针地址作为次级排序键


class TriangulationState:
    """Persistent topology/index cache for a triangle list.

    The active Bowyer-Watson path mutates `self.triangles` frequently. Repeatedly
    rebuilding one-off edge maps and scanning the full list makes those hot paths
    harder to reason about. This cache keeps the common edge/vertex indices
    together and rebuilds adjacency from the same pass.
    """

    __slots__ = [
        'triangles',
        'triangle_by_id',
        'edge_to_tris',
        'vertex_to_tris',
        'vertex_to_neighbors',
        '_source_len',
    ]

    def __init__(self, triangles: Optional[List[MTri3]] = None):
        self.triangles: List[MTri3] = []
        self.triangle_by_id: Dict[int, MTri3] = {}
        self.edge_to_tris: Dict[Tuple[int, int], List[Tuple[MTri3, int]]] = {}
        self.vertex_to_tris: Dict[int, List[MTri3]] = {}
        self.vertex_to_neighbors: Dict[int, set] = {}
        self._source_len = 0
        if triangles is not None:
            self.rebuild(triangles)

    def ensure(self, triangles: List[MTri3]) -> 'TriangulationState':
        if triangles is not self.triangles or len(triangles) != self._source_len:
            self.rebuild(triangles)
        return self

    def rebuild(self, triangles: List[MTri3]) -> 'TriangulationState':
        self.triangles = triangles
        self._source_len = len(triangles)
        self.triangle_by_id = {}
        self.edge_to_tris = defaultdict(list)
        self.vertex_to_tris = defaultdict(list)
        self.vertex_to_neighbors = defaultdict(set)

        for tri in triangles:
            tri.neighbors = [None, None, None]

        for tri in triangles:
            if tri.is_deleted():
                continue

            self.triangle_by_id[tri.idx] = tri

            v0, v1, v2 = tri.vertices
            self.vertex_to_tris[v0].append(tri)
            self.vertex_to_tris[v1].append(tri)
            self.vertex_to_tris[v2].append(tri)

            self.vertex_to_neighbors[v0].update((v1, v2))
            self.vertex_to_neighbors[v1].update((v0, v2))
            self.vertex_to_neighbors[v2].update((v0, v1))

            for i in range(3):
                edge_key = tri.get_edge_sorted(i)
                entries = self.edge_to_tris[edge_key]
                if entries:
                    other_tri, other_local_idx = entries[0]
                    tri.neighbors[i] = other_tri
                    other_tri.neighbors[other_local_idx] = tri
                entries.append((tri, i))

        self.edge_to_tris = dict(self.edge_to_tris)
        self.vertex_to_tris = dict(self.vertex_to_tris)
        self.vertex_to_neighbors = {
            vertex: set(neighbors)
            for vertex, neighbors in self.vertex_to_neighbors.items()
        }
        return self

    def _filter_deleted_entries(
        self,
        edge_key: Tuple[int, int],
    ) -> List[Tuple[MTri3, int]]:
        entries = self.edge_to_tris.get(edge_key, [])
        if not entries:
            return []

        active_entries = [(tri, local_idx) for tri, local_idx in entries if not tri.is_deleted()]
        if len(active_entries) != len(entries):
            if active_entries:
                self.edge_to_tris[edge_key] = active_entries
            else:
                self.edge_to_tris.pop(edge_key, None)
        return active_entries

    def active_triangles(self) -> List[MTri3]:
        return [tri for tri in self.triangles if not tri.is_deleted()]

    def active_triangle_count(self) -> int:
        return sum(1 for tri in self.triangles if not tri.is_deleted())

    def edge_entries(self, v1: int, v2: int) -> List[Tuple[MTri3, int]]:
        edge_key = (min(v1, v2), max(v1, v2))
        return self._filter_deleted_entries(edge_key)

    def incident_triangles(self, v1: int, v2: int) -> List[MTri3]:
        return [tri for tri, _ in self.edge_entries(v1, v2)]

    def edge_use_count(self, v1: int, v2: int) -> int:
        return len(self.edge_entries(v1, v2))

    def has_edge(self, v1: int, v2: int) -> bool:
        return self.edge_use_count(v1, v2) > 0

    def triangles_with_vertex(self, vertex: int) -> List[MTri3]:
        triangles = self.vertex_to_tris.get(vertex, [])
        active_triangles = [tri for tri in triangles if not tri.is_deleted()]
        if len(active_triangles) != len(triangles):
            if active_triangles:
                self.vertex_to_tris[vertex] = active_triangles
            else:
                self.vertex_to_tris.pop(vertex, None)
        return active_triangles

    def vertex_neighbors_for(self, vertex: int) -> set:
        triangles = self.triangles_with_vertex(vertex)
        if not triangles:
            return set()

        neighbors = set()
        for tri in triangles:
            for other_vertex in tri.vertices:
                if other_vertex != vertex:
                    neighbors.add(other_vertex)
        self.vertex_to_neighbors[vertex] = neighbors
        return neighbors

    def active_vertices(self) -> set:
        active = set()
        for tri in self.active_triangles():
            active.update(tri.vertices)
        return active

    def compact_deleted(self) -> List[MTri3]:
        if all(not tri.is_deleted() for tri in self.triangles):
            return self.triangles

        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
        self.rebuild(self.triangles)
        return self.triangles


# =============================================================================
# 辅助函数
# =============================================================================

def build_adjacency_from_triangles(triangles: List[MTri3]):
    """从三角形列表构建邻接关系。

    参考 Gmsh connectTris：通过边匹配建立双向邻接关系。
    时间复杂度：O(n log n) 或 O(n) 使用哈希表

    算法：
    1. 收集所有边 - 面对
    2. 使用哈希表匹配相同的边
    3. 建立双向邻接关系
    """
    # 清空旧邻接关系
    for tri in triangles:
        tri.neighbors = [None, None, None]

    # 构建边到三角形的映射
    edge_to_tri = {}

    for tri in triangles:
        for i in range(3):
            edge_key = tri.get_edge_sorted(i)

            if edge_key in edge_to_tri:
                # 找到相邻三角形，建立双向邻接关系
                other_tri, other_local_idx = edge_to_tri[edge_key]
                tri.neighbors[i] = other_tri
                other_tri.neighbors[other_local_idx] = tri
            else:
                edge_to_tri[edge_key] = (tri, i)


def collect_cavity_shell(cavity_triangles: List[MTri3]) -> List[EdgeXFace]:
    """收集空腔的边界边（Shell）。

    参考 Gmsh recurFindCavityAniso 的 shell 收集逻辑：
    - 遍历空腔中所有三角形的所有边
    - 如果某条边只被一个空腔三角形拥有，则是边界边

    返回:
        空腔边界边的 EdgeXFace 列表
    """
    edge_count = {}
    edge_to_tris = {}

    for tri in cavity_triangles:
        if tri.is_deleted():
            continue

        for i in range(3):
            edge_key = tri.get_edge_sorted(i)

            if edge_key not in edge_count:
                edge_count[edge_key] = 0
                edge_to_tris[edge_key] = []

            edge_count[edge_key] += 1
            edge_to_tris[edge_key].append((tri, i))

    # 边界边是只出现一次的边
    shell = []
    for edge_key, count in edge_count.items():
        if count == 1:
            tri, local_idx = edge_to_tris[edge_key][0]
            shell.append(EdgeXFace(tri, local_idx))

    return shell


def collect_cavity_shell_with_constraints(cavity_triangles: List[MTri3], protected_edges: set) -> List[EdgeXFace]:
    """收集空腔的边界边（Shell），增强版：标记约束边。

    参考 Gmsh recurFindCavityAniso 的 shell 收集逻辑：
    - 遍历空腔中所有三角形的所有边
    - 如果某条边只被一个空腔三角形拥有，则是边界边
    - 如果某条边在 protected_edges 中，标记为约束边

    参数:
        cavity_triangles: 空腔三角形列表
        protected_edges: 保护边集合（frozenset 格式）
        
    返回:
        空腔边界边的 EdgeXFace 列表（约束边已标记）
    """
    edge_count = {}
    edge_to_tris = {}

    for tri in cavity_triangles:
        if tri.is_deleted():
            continue

        for i in range(3):
            edge_key = tri.get_edge_sorted(i)

            if edge_key not in edge_count:
                edge_count[edge_key] = 0
                edge_to_tris[edge_key] = []

            edge_count[edge_key] += 1
            edge_to_tris[edge_key].append((tri, i))

    # 边界边是只出现一次的边
    shell = []
    for edge_key, count in edge_count.items():
        if count == 1:
            tri, local_idx = edge_to_tris[edge_key][0]
            # 检查是否是约束边
            is_constrained = frozenset(edge_key) in protected_edges
            shell.append(EdgeXFace(tri, local_idx, is_constrained=is_constrained))

    return shell


def compute_cavity_volume(cavity_triangles: List[MTri3], points: np.ndarray) -> float:
    """计算空腔的总面积（体积）。

    参考 Gmsh getSurfUV：计算参数空间的三角形面积。
    用于星形空腔验证：|oldVolume - newVolume| < EPS * oldVolume

    返回:
        空腔总面积
    """
    total_area = 0.0

    for tri in cavity_triangles:
        if tri.is_deleted():
            continue

        v0, v1, v2 = tri.vertices
        p0, p1, p2 = points[v0], points[v1], points[v2]

        # 计算三角形面积
        area = 0.5 * abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1]) -
            (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        total_area += area

    return total_area


def segment_intersects_triangle_segment(p1, p2, v1, v2):
    """检查线段 p1-p2 是否与线段 v1-v2 相交。
    
    参考 Triangle 的几何谓词实现。
    
    参数:
        p1, p2: 线段的两个端点
        v1, v2: 三角形的一条边的两个端点
        
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


def find_triangles_intersecting_segment(triangles: List[MTri3], p1, p2, points: np.ndarray) -> List[MTri3]:
    """找到所有与线段 p1-p2 相交的三角形。
    
    参数:
        triangles: 三角形列表
        p1, p2: 线段的两个端点
        points: 点坐标数组
        
    返回:
        相交的三角形列表
    """
    intersecting = []
    
    for tri in triangles:
        if tri.is_deleted():
            continue
        
        # 检查线段是否与三角形的三条边相交
        v0, v1, v2 = tri.vertices
        pt0, pt1, pt2 = points[v0], points[v1], points[v2]
        
        if (segment_intersects_triangle_segment(p1, p2, pt0, pt1) or
            segment_intersects_triangle_segment(p1, p2, pt1, pt2) or
            segment_intersects_triangle_segment(p1, p2, pt0, pt2)):
            intersecting.append(tri)
    
    return intersecting

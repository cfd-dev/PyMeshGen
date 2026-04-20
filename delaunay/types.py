"""
Bowyer-Watson 网格生成器 - 数据类型定义

包含：
- MTri3: Gmsh 风格的三角形包装类
- EdgeXFace: 边 - 面关系结构
"""

from typing import Tuple, Optional, List


class MTri3:
    """Gmsh 风格的三角形包装类。

    特性：
    - 顶点索引按升序存储
    - 显式存储邻接关系
    - 支持懒删除
    - 缓存外接圆和质量信息
    """

    __slots__ = [
        'vertices',      # tuple: 排序后的顶点索引
        'neighbors',     # list: 三个邻居三角形
        'circumcenter',  # np.ndarray: 外接圆心
        'circumradius',  # float: 外接圆半径
        'deleted',       # bool: 懒删除标记
        'idx',           # int: 三角形 ID
        'quality',       # float: 质量度量
    ]

    def __init__(self, v1: int, v2: int, v3: int, idx: int = -1):
        self.vertices = tuple(sorted([v1, v2, v3]))
        self.neighbors = [None, None, None]
        self.circumcenter = None
        self.circumradius = None
        self.deleted = False
        self.idx = idx
        self.quality = 0.0

    def __eq__(self, other):
        return isinstance(other, MTri3) and self.vertices == other.vertices

    def __hash__(self):
        return hash(self.vertices)

    def __repr__(self):
        status = "D" if self.deleted else "."
        r = f"{self.circumradius:.3f}" if self.circumradius else "NaN"
        return f"MTri3{self.vertices}[r={r}]{status}"

    def get_edge(self, i: int) -> Tuple[int, int]:
        """获取第 i 条边。"""
        v0, v1, v2 = self.vertices
        return ((v0, v1), (v1, v2), (v0, v2))[i]

    def is_deleted(self) -> bool:
        return self.deleted

    def set_deleted(self, val: bool = True):
        self.deleted = val


class EdgeXFace:
    """边 - 面关系结构。

    用于表示三角形的一条边及其所属三角形。
    """

    __slots__ = ['tri', 'edge_idx', 'ori']

    def __init__(self, tri: MTri3, edge_idx: int, ori: int = 1):
        self.tri = tri
        self.edge_idx = edge_idx
        self.ori = ori

    def vertices(self) -> Tuple[int, int]:
        return self.tri.get_edge(self.edge_idx)

    def __repr__(self):
        return f"EdgeXFace({self.vertices()}, tri={self.tri.idx})"


def build_adjacency(triangles: List[MTri3]):
    """构建三角形的邻接关系。

    时间复杂度：O(n log n)
    """
    # 清空旧邻接关系
    for tri in triangles:
        tri.neighbors = [None, None, None]

    # 构建边到三角形的映射
    edge_map = {}
    for tri in triangles:
        if tri.deleted:
            continue
        for i in range(3):
            edge = tuple(sorted(tri.get_edge(i)))
            if edge in edge_map:
                other_tri, other_idx = edge_map[edge]
                tri.neighbors[i] = other_tri
                other_tri.neighbors[other_idx] = tri
            else:
                edge_map[edge] = (tri, i)


def collect_shell_edges(cavity_tris: List[MTri3]) -> List[EdgeXFace]:
    """收集空腔边界边。

    Euler 公式：shell.size() == cavity.size() + 2
    """
    edge_count = {}
    for tri in cavity_tris:
        for i in range(3):
            edge = tuple(sorted(tri.get_edge(i)))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    shell = []
    for tri in cavity_tris:
        for i in range(3):
            edge = tuple(sorted(tri.get_edge(i)))
            if edge_count.get(edge, 0) == 1:
                shell.append(EdgeXFace(tri, i))

    return shell


def compute_cavity_area(triangles: List[MTri3], points) -> float:
    """计算空腔的总面积。"""
    total = 0.0
    for tri in triangles:
        v0, v1, v2 = tri.vertices
        p0, p1, p2 = points[v0], points[v1], points[v2]
        area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - 
                   (p1[1] - p0[1]) * (p2[0] - p0[0])) / 2.0
        total += area
    return total

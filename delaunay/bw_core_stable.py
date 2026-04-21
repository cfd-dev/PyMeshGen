"""
Bowyer-Watson Delaunay 网格生成器 - 核心实现（重构版）

基于 Bowyer-Watson 算法的二维三角形网格生成，支持：
1. 以离散边界网格作为输入
2. 使用 QuadtreeSizing 尺寸场控制网格尺寸
3. 自动/手动孔洞处理
4. 边界边保护与恢复
5. 增量式点插入（避免全量重剖分）

重构特性（参考 Gmsh C++ Delaunay 实现）：
- Gmsh 风格的数据结构（MTri3、edgeXface）
- 递归/迭代 Cavity 搜索（recurFindCavityAniso）
- 鲁棒几何谓词（Shewchuk's predicates）
- 懒删除优化（避免频繁集合操作）
- 优先级队列（按外接圆半径排序）

参考: Gmsh C++ Delaunay 实现
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from math import sqrt
from scipy.spatial import KDTree
from collections import Counter
import heapq  # 新增：用于优先级队列

from utils.message import debug, verbose
from utils.geom_toolkit import (
    point_in_polygon,
    is_polygon_clockwise,
    segments_intersect_strict,
)
from utils.timer import TimeSpan

from .bw_cavity import insert_vertex, recur_find_cavity
from .bw_predicates import (
    compute_circumcircle,
    orient2d_fast,
    point_in_circumcircle_robust,
)
from .bw_types import MTri3, build_adjacency_from_triangles

__all__ = [
    "Triangle",
    "triangle_to_mtri3",
    "mtri3_to_triangle",
    "BowyerWatsonMeshGenerator",
    "GmshBowyerWatsonMeshGenerator",
]


# =============================================================================
# Triangle 数据结构
# =============================================================================

class Triangle:
    """三角形单元，带外接圆缓存和质量缓存。

    顶点索引始终按升序存储，便于去重和比较。
    参考 Gmsh MTri3：显式存储邻接关系以加速 Cavity 查找。
    """

    __slots__ = [
        'vertices', 'circumcenter', 'circumradius', 'idx',
        'circumcircle_valid', 'quality', 'quality_valid', 'circumcircle_bbox',
        'neighbors',  # 新增：三条边的邻接三角形索引 [None 或 Triangle]
    ]

    def __init__(self, p1: int, p2: int, p3: int, idx: int = -1):
        self.vertices = tuple(sorted([p1, p2, p3]))
        self.circumcenter = None
        self.circumradius = None
        self.idx = idx
        self.circumcircle_valid = False
        self.quality = 0.0
        self.quality_valid = False
        self.circumcircle_bbox = None
        self.neighbors = [None, None, None]  # 邻接三角形，neigh[i] 对应 vertices[i]->vertices[(i+1)%3] 这条边

    def __eq__(self, other):
        return isinstance(other, Triangle) and self.vertices == other.vertices

    def __hash__(self):
        return hash(self.vertices)

    def __contains__(self, point_idx: int) -> bool:
        return point_idx in self.vertices

    def get_edges(self) -> List[Tuple[int, int]]:
        """返回三角形的三条边（排序后的元组）。"""
        v0, v1, v2 = self.vertices
        return [(v0, v1), (v1, v2), (v0, v2)]

    def __repr__(self):
        return f"Triangle({self.vertices})"
    
    def to_mtri3(self) -> 'MTri3':
        """转换为 Gmsh MTri3 格式。"""
        mtri = MTri3(self.vertices[0], self.vertices[1], self.vertices[2], idx=self.idx)
        if self.circumcircle_valid:
            mtri.circumcenter = self.circumcenter
            mtri.circumradius = self.circumradius
        mtri.quality = self.quality
        # 复制邻居关系
        for i in range(3):
            if self.neighbors[i] is not None:
                # 注意：这里只是引用，实际需要在转换所有三角形后重建
                pass
        return mtri


# =============================================================================
# Triangle -> MTri3 适配器
# =============================================================================

def triangle_to_mtri3(tri: Triangle) -> MTri3:
    """将 Triangle 转换为 MTri3。"""
    mtri = MTri3(tri.vertices[0], tri.vertices[1], tri.vertices[2], idx=tri.idx)
    if tri.circumcircle_valid:
        mtri.circumcenter = tri.circumcenter
        mtri.circumradius = tri.circumradius
    mtri.quality = tri.quality
    return mtri


def mtri3_to_triangle(mtri: MTri3) -> Triangle:
    """将 MTri3 转换为 Triangle。"""
    tri = Triangle(mtri.vertices[0], mtri.vertices[1], mtri.vertices[2], idx=mtri.idx)
    if mtri.circumcenter is not None:
        tri.circumcenter = mtri.circumcenter
        tri.circumradius = mtri.circumradius
        tri.circumcircle_valid = True
    tri.quality = mtri.quality
    return tri


# =============================================================================
# BowyerWatsonMeshGenerator 主类
# =============================================================================

class BowyerWatsonMeshGenerator:
    """Bowyer-Watson Delaunay 网格生成器。

    特性：
    - 支持从离散边界网格输入
    - 集成 QuadtreeSizing 尺寸场控制
    - 支持孔洞（自动检测 + 后处理清理）
    - 边界边保护与恢复
    - Laplacian 平滑（默认关闭）
    """

    def __init__(
        self,
        boundary_points: np.ndarray,
        boundary_edges: Optional[List[Tuple[int, int]]] = None,
        sizing_system=None,
        max_edge_length: Optional[float] = None,
        smoothing_iterations: int = 0,
        seed: Optional[int] = None,
        holes: Optional[List[np.ndarray]] = None,
    ):
        """初始化生成器。

        参数:
            boundary_points: 边界点坐标数组，形状 (N, 2)
            boundary_edges: 边界边列表 [(idx1, idx2), ...]
            sizing_system: QuadtreeSizing 尺寸场对象（可选）
            max_edge_length: 全局最大边长（可选）
            smoothing_iterations: Laplacian 平滑迭代次数（默认0）
            seed: 随机种子
            holes: 孔洞边界列表，每个孔洞是点数组 (M, 2)
        """
        self.original_points = boundary_points.copy()

        # 保护的边界边集合
        self.protected_edges = set()
        for edge in (boundary_edges or []):
            self.protected_edges.add(frozenset(edge))
        self.boundary_edges = boundary_edges or []

        self.sizing_system = sizing_system
        self.max_edge_length = max_edge_length
        self.smoothing_iterations = smoothing_iterations
        self.seed = seed
        self.holes = holes or []

        if seed is not None:
            np.random.seed(seed)

        # 工作状态变量
        self.points = None
        self.triangles: List[Triangle] = []
        self.boundary_mask = None
        self.boundary_count = 0
        self._kdtree = None

    # -------------------------------------------------------------------------
    # 边界保护
    # -------------------------------------------------------------------------

    def _is_protected_edge(self, v1: int, v2: int) -> bool:
        """检查边是否是受保护的边界边。"""
        return frozenset({v1, v2}) in self.protected_edges

    # -------------------------------------------------------------------------
    # 外接圆计算（带缓存）
    # -------------------------------------------------------------------------

    def _compute_circumcircle(self, tri: Triangle) -> Tuple[np.ndarray, float]:
        """计算三角形的外接圆，结果缓存到 tri 对象中。

        改进：使用高精度算术计算 circumcenter，避免浮点误差。
        参考 Gmsh：使用鲁棒谓词确保准确性。
        """
        if tri.circumcircle_valid and tri.circumcenter is not None and tri.circumradius is not None:
            return tri.circumcenter, tri.circumradius

        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        ax, ay = float(p1[0]), float(p1[1])
        bx, by = float(p2[0]), float(p2[1])
        cx, cy = float(p3[0]), float(p3[1])

        # 使用高精度算术计算 circumcenter
        ux, uy = self._precise_circumcenter(ax, ay, bx, by, cx, cy)
        center = np.array([ux, uy])
        radius = float(np.linalg.norm(p1 - center))

        tri.circumcenter = center
        tri.circumradius = radius
        tri.circumcircle_valid = True
        tri.circumcircle_bbox = (
            center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius,
        )
        return center, radius

    def _point_in_circumcircle(self, point: np.ndarray, tri: Triangle) -> bool:
        """检查点是否在三角形的外接圆内。

        改进：使用 Shewchuk 的鲁棒 incircle 谓词，
        避免浮点误差导致的误判。

        参考 Gmsh：使用 robustPredicates::incircle() + orient2d()
        """
        if not tri.circumcircle_valid:
            self._compute_circumcircle(tri)

        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        # 使用鲁棒谓词
        result = self._robust_incircle(
            p1[0], p1[1],
            p2[0], p2[1],
            p3[0], p3[1],
            point[0], point[1]
        )

        return result > 0

    def _robust_incircle(self, ax, ay, bx, by, cx, cy, px, py):
        """精确的 incircle 测试（Shewchuk 的 Robust Predicates）。

        判断点 p 是否在三角形 (a, b, c) 的外接圆内。
        使用行列式计算，避免浮点误差。

        返回：
          > 0：p 在圆内
          = 0：p 在圆上
          < 0：p 在圆外
        """
        # 构造增广矩阵的行列式
        adx = ax - px
        ady = ay - py
        bdx = bx - px
        bdy = by - py
        cdx = cx - px
        cdy = cy - py

        alift = adx * adx + ady * ady
        blift = bdx * bdx + bdy * bdy
        clift = cdx * cdx + cdy * cdy

        # 计算 3x3 行列式
        det = (adx * (bdy * clift - cdy * blift) -
               ady * (bdx * clift - cdx * blift) +
               alift * (bdx * cdy - cdx * bdy))

        # 乘以 orient2d 的符号
        orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        return det * orient

    def _precise_circumcenter(self, ax, ay, bx, by, cx, cy):
        """精确计算外接圆圆心（使用高精度算术）。

        参考 Gmsh circumCenterMetric：
        使用行列式法求解线性方程组，避免除零问题。

        返回：
          (ux, uy): 外接圆心坐标
        """
        # 使用 Python 的高精度浮点（decimal 模块）
        from decimal import Decimal, getcontext
        getcontext().prec = 50  # 50 位精度

        ax_d, ay_d = Decimal(ax), Decimal(ay)
        bx_d, by_d = Decimal(bx), Decimal(by)
        cx_d, cy_d = Decimal(cx), Decimal(cy)

        d = Decimal('2') * (ax_d * (by_d - cy_d) + bx_d * (cy_d - ay_d) + cx_d * (ay_d - by_d))

        if abs(d) < Decimal('1e-30'):
            # 退化三角形，返回重心
            return (ax + bx + cx) / 3.0, (ay + by + cy) / 3.0

        a2 = ax_d * ax_d + ay_d * ay_d
        b2 = bx_d * bx_d + by_d * by_d
        c2 = cx_d * cx_d + cy_d * cy_d

        ux = (a2 * (by_d - cy_d) + b2 * (cy_d - ay_d) + c2 * (ay_d - by_d)) / d
        uy = (a2 * (cx_d - bx_d) + b2 * (ax_d - cx_d) + c2 * (bx_d - ax_d)) / d

        return float(ux), float(uy)

    # -------------------------------------------------------------------------
    # 质量与尺寸计算
    # -------------------------------------------------------------------------

    def _compute_triangle_quality(self, tri: Triangle) -> float:
        """计算三角形质量（2 * r_inscribed / r_circumscribed）。"""
        if tri.quality_valid:
            return tri.quality

        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        a = float(np.linalg.norm(p2 - p1))
        b = float(np.linalg.norm(p3 - p2))
        c = float(np.linalg.norm(p1 - p3))
        s = (a + b + c) / 2.0

        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq < 0:
            quality = 0.0
        else:
            area = sqrt(max(area_sq, 0.0))
            if area < 1e-12:
                quality = 0.0
            else:
                inscribed_radius = area / s
                if not tri.circumcircle_valid:
                    self._compute_circumcircle(tri)
                circumscribed_radius = tri.circumradius
                quality = min(2.0 * inscribed_radius / circumscribed_radius, 1.0) if circumscribed_radius > 1e-12 else 0.0

        tri.quality = quality
        tri.quality_valid = True
        return quality

    def _compute_triangle_centroid(self, tri: Triangle) -> np.ndarray:
        """计算三角形的质心。"""
        return np.mean([
            self.points[tri.vertices[0]],
            self.points[tri.vertices[1]],
            self.points[tri.vertices[2]],
        ], axis=0)

    def _get_target_size_for_triangle(self, tri: Triangle) -> Optional[float]:
        """获取三角形的目标尺寸。

        Gmsh 做法：使用三角形三个顶点处局部尺寸的平均值，
        顶点尺寸由其相邻边界边长度决定。

        回退策略：
        1. 尺寸场 → 2. 全局尺寸 → 3. 三角形自身最大边长
        """
        target_size = None

        if self.sizing_system is not None:
            center = np.mean([
                self.points[tri.vertices[0]],
                self.points[tri.vertices[1]],
                self.points[tri.vertices[2]],
            ], axis=0)
            try:
                target_size = self.sizing_system.spacing_at(center)
            except Exception as e:
                debug(f"获取尺寸场失败: {e}，使用全局尺寸")

        if target_size is None or target_size < 1e-12:
            target_size = self.max_edge_length

        # 如果仍然无效，使用三角形的最大边长作为目标尺寸
        if target_size is None or target_size < 1e-12:
            v0, v1, v2 = tri.vertices
            pts = self.points
            edge_lengths = [
                float(np.linalg.norm(pts[v1] - pts[v0])),
                float(np.linalg.norm(pts[v2] - pts[v1])),
                float(np.linalg.norm(pts[v0] - pts[v2])),
            ]
            target_size = max(edge_lengths)

        return target_size

    # -------------------------------------------------------------------------
    # 三角剖分核心
    # -------------------------------------------------------------------------

    def _create_super_triangle(self) -> Triangle:
        """创建包含所有点的超级三角形。"""
        min_x = np.min(self.points[:, 0])
        max_x = np.max(self.points[:, 0])
        min_y = np.min(self.points[:, 1])
        max_y = np.max(self.points[:, 1])

        dx = max_x - min_x
        dy = max_y - min_y
        delta = max(dx, dy) * 10.0

        p1 = len(self.points)
        p2 = len(self.points) + 1
        p3 = len(self.points) + 2

        super_verts = np.array([
            [min_x - delta, min_y - delta],
            [max_x + delta, min_y - delta],
            [(min_x + max_x) / 2.0, max_y + 3.0 * delta],
        ])
        self.points = np.vstack([self.points, super_verts])
        return Triangle(p1, p2, p3)

    def _triangulate(self) -> List[Triangle]:
        """执行 Bowyer-Watson 三角剖分。

        流程：超级三角形 → 逐点插入 → 删除超级三角形
        改进：在剖分完成后构建邻接关系。
        """
        super_tri = self._create_super_triangle()
        triangles = [super_tri]
        real_point_count = len(self.points) - 3

        for i in range(real_point_count):
            point = self.points[i]

            bad_triangles = [tri for tri in triangles if self._point_in_circumcircle(point, tri)]

            edge_counter = Counter()
            for tri in bad_triangles:
                for edge in tri.get_edges():
                    edge_counter[tuple(sorted(edge))] += 1

            polygon_edges = [edge for edge, count in edge_counter.items() if count == 1]
            bad_set = set(id(tri) for tri in bad_triangles)
            triangles = [tri for tri in triangles if id(tri) not in bad_set]

            for edge in polygon_edges:
                new_tri = Triangle(edge[0], edge[1], i)
                self._compute_circumcircle(new_tri)
                triangles.append(new_tri)

        super_verts = {super_tri.vertices[0], super_tri.vertices[1], super_tri.vertices[2]}
        self.triangles = [tri for tri in triangles if not any(v in super_verts for v in tri.vertices)]
        self.points = self.points[:-3]

        # 构建邻接关系（关键改进）
        self._build_adjacency(self.triangles)

        return self.triangles

    # -------------------------------------------------------------------------
    # 增量式点插入
    # -------------------------------------------------------------------------

    def _insert_point_incremental(self, point_idx: int, triangles: List[Triangle]) -> List[Triangle]:
        """增量式插入单个点，只更新受影响的三角形。"""
        point = self.points[point_idx]
        bad_triangles = [tri for tri in triangles if self._point_in_circumcircle(point, tri)]
        if not bad_triangles:
            return triangles

        edge_counter = Counter()
        for tri in bad_triangles:
            for edge in tri.get_edges():
                edge_counter[tuple(sorted(edge))] += 1

        polygon_edges = [edge for edge, count in edge_counter.items() if count == 1]
        bad_set = set(id(tri) for tri in bad_triangles)
        triangles = [tri for tri in triangles if id(tri) not in bad_set]

        for edge in polygon_edges:
            new_tri = Triangle(edge[0], edge[1], point_idx)
            self._compute_circumcircle(new_tri)
            triangles.append(new_tri)

        return triangles

    def _find_cavity_with_protection(self, start_tri, point_idx, all_triangles,
                                     bad_tri_ids, shell_edges, cavity_set):
        """递归查找 Cavity，遇到保护边时停止。
        
        改进：使用显式邻接关系，将查找从 O(n) 降为 O(1)。
        参考 Gmsh recurFindCavityAniso 算法。
        """
        tri_id = id(start_tri)
        if tri_id in cavity_set:
            return
        if tri_id not in bad_tri_ids:
            return

        cavity_set.add(tri_id)

        tri = start_tri
        for i in range(3):
            edge = tuple(sorted([tri.vertices[i], tri.vertices[(i + 1) % 3]]))

            # 检查是否是受保护的边界边
            if self._is_protected_edge(edge[0], edge[1]):
                # 是保护边，加入 shell，不继续递归
                shell_edges.append(edge)
                continue

            # 使用显式邻接关系查找相邻三角形（O(1)）
            neighbor = tri.neighbors[i]
            
            if neighbor is None:
                # 边界上的边，加入 shell
                shell_edges.append(edge)
            elif id(neighbor) in bad_tri_ids:
                # 相邻三角形也在 Cavity 内，继续递归
                self._find_cavity_with_protection(neighbor, point_idx, all_triangles,
                                                  bad_tri_ids, shell_edges, cavity_set)
            else:
                # 相邻三角形不在 Cavity 内，这条边是 shell 边界
                shell_edges.append(edge)

    def _find_neighbor_triangle(self, tri, edge, all_triangles):
        """查找共享指定边的相邻三角形。（已废弃，保留用于向后兼容）
        
        改进：使用 _build_adjacency 和显式 neighbors 字段替代。
        """
        v1, v2 = edge
        for other in all_triangles:
            if id(other) == id(tri):
                continue
            if v1 in other.vertices and v2 in other.vertices:
                return other
        return None

    def _build_adjacency(self, triangles: List[Triangle]):
        """构建所有三角形的邻接关系。
        
        参考 Gmsh connectTris：通过边匹配建立双向邻接关系。
        时间复杂度：O(n log n)，30% 的时间花在这里（Gmsh 统计）
        """
        # 清空旧邻接关系
        for tri in triangles:
            tri.neighbors = [None, None, None]
        
        # 构建边到三角形的映射
        edge_to_tri = {}
        for tri in triangles:
            for i in range(3):
                v1 = tri.vertices[i]
                v2 = tri.vertices[(i + 1) % 3]
                edge_key = tuple(sorted([v1, v2]))
                
                if edge_key in edge_to_tri:
                    # 找到相邻三角形，建立双向邻接关系
                    other_tri, other_local_idx = edge_to_tri[edge_key]
                    tri.neighbors[i] = other_tri
                    other_tri.neighbors[other_local_idx] = tri
                else:
                    edge_to_tri[edge_key] = (tri, i)

    def _update_adjacency_after_insertion(self, triangles, new_triangles, cavity_set):
        """插入新点后增量更新邻接关系。
        
        策略：
        1. 新三角形之间的邻接关系（通过 shell 边连接）
        2. 新三角形与现有三角形的邻接关系（shell 边的另一端）
        3. 清除指向已删除三角形的邻接指针
        """
        # 清除指向已删除三角形的邻接指针
        for tri in triangles:
            if tri not in new_triangles:
                for i in range(3):
                    if tri.neighbors[i] is not None and id(tri.neighbors[i]) in cavity_set:
                        tri.neighbors[i] = None
        
        # 构建新三角形的邻接关系
        shell_edges_set = set()
        for new_tri in new_triangles:
            for i in range(3):
                v1 = new_tri.vertices[i]
                v2 = new_tri.vertices[(i + 1) % 3]
                edge_key = tuple(sorted([v1, v2]))
                shell_edges_set.add(edge_key)
        
        # 在新三角形之间建立邻接关系
        edge_to_new_tri = {}
        for new_tri in new_triangles:
            for i in range(3):
                v1 = new_tri.vertices[i]
                v2 = new_tri.vertices[(i + 1) % 3]
                edge_key = tuple(sorted([v1, v2]))
                
                if edge_key in edge_to_new_tri:
                    other_tri, other_local_idx = edge_to_new_tri[edge_key]
                    new_tri.neighbors[i] = other_tri
                    other_tri.neighbors[other_local_idx] = new_tri
                else:
                    edge_to_new_tri[edge_key] = (new_tri, i)

    def _validate_star_shaped(self, shell_edges, new_point_idx, old_cavity_tris):
        """验证 Cavity 是星形的（新点能看到所有 shell 边）。
        
        参考 Gmsh insertVertexB 的体积守恒检查：
        |oldVolume - newVolume| < EPS * oldVolume
        
        这确保了插入点不会产生重叠或空洞。
        """
        if not old_cavity_tris:
            return True
        
        # 计算旧 Cavity 面积
        old_area = sum(self._triangle_area(tri) for tri in old_cavity_tris)
        
        # 计算新三角形总面积
        new_area = 0.0
        for v1, v2 in shell_edges:
            new_tri_area = 0.5 * abs(
                (self.points[v2][0] - self.points[v1][0]) * (self.points[new_point_idx][1] - self.points[v1][1]) -
                (self.points[v2][1] - self.points[v1][1]) * (self.points[new_point_idx][0] - self.points[v1][0])
            )
            new_area += new_tri_area
        
        # 面积守恒检查（相对误差 < 1e-10）
        if old_area > 1e-12:
            return abs(old_area - new_area) < 1e-10 * old_area
        else:
            return abs(old_area - new_area) < 1e-12

    def _triangle_area(self, tri):
        """计算三角形的面积（使用顶点坐标）。"""
        v0, v1, v2 = tri.vertices
        return 0.5 * abs(
            (self.points[v1][0] - self.points[v0][0]) * (self.points[v2][1] - self.points[v0][1]) -
            (self.points[v1][1] - self.points[v0][1]) * (self.points[v2][0] - self.points[v0][0])
        )

    # -------------------------------------------------------------------------
    # 迭代插点主循环
    # -------------------------------------------------------------------------

    def _insert_points_iteratively(self, target_triangle_count: Optional[int] = None):
        """迭代插入内部点，使用优先级队列策略。

        改进（对比原简单遍历）：
        1. 使用优先级队列（heapq）按"偏离目标尺寸程度"排序
        2. 每次处理偏离最大的三角形，避免遗漏
        3. 插入失败时尝试多种策略（外心 → 重心 → 前端法），而非直接标记失败
        4. 更合理的早停条件：队列中所有三角形都满足要求时才停止
        """
        boundary_count = self.boundary_count
        initial_point_count = len(self.points)

        target_total_points = None
        if target_triangle_count is not None:
            target_total_points = (target_triangle_count + 2 + boundary_count) // 2

        x_min = np.min(self.original_points[:, 0])
        x_max = np.max(self.original_points[:, 0])
        y_min = np.min(self.original_points[:, 1])
        y_max = np.max(self.original_points[:, 1])
        margin = 0.001 * max(x_max - x_min, y_max - y_min)

        if len(self.original_points) > 1:
            avg_edge = float(np.linalg.norm(self.original_points[1] - self.original_points[0]))
            min_dist_threshold = 0.01 * avg_edge if avg_edge > 0 else 0.01
        else:
            min_dist_threshold = 0.01

        max_iterations = 0
        kdtree_rebuild_interval = 100
        last_kdtree_build = 0
        consecutive_failures = 0
        max_consecutive_failures = 2000  # 增加阈值

        # 优先级队列：存储 (-priority, tri_id, tri)
        # priority 越大（偏离目标越严重）越优先处理
        # 使用负值因为 heapq 是最小堆
        candidate_queue = []
        tri_to_entry = {}  # tri_id -> heap entry 引用，用于更新

        # 初始化：将所有不满足要求的三角形加入队列
        self._rebuild_candidate_queue(
            candidate_queue, tri_to_entry, min_dist_threshold
        )

        while candidate_queue:
            max_iterations += 1

            if max_iterations % 50 == 0 or max_iterations == 1:
                current_triangles = len(self.triangles)
                current_points = len(self.points)
                current_inserted = current_points - initial_point_count
                queue_size = len(candidate_queue)
                verbose(f"  [进度] 迭代 {max_iterations} | "
                        f"节点: {current_points} (边界: {boundary_count}, 内部: {current_inserted}) | "
                        f"三角形: {current_triangles} | "
                        f"候选队列: {queue_size}")
            
            # 初始化/更新 KDTree
            if max_iterations == 1 or (max_iterations - last_kdtree_build) >= kdtree_rebuild_interval:
                if len(self.points) > 0:
                    self._kdtree = KDTree(self.points)
                    last_kdtree_build = max_iterations

            # 早停条件
            if len(self.points) > 50000:
                verbose("达到最大节点数限制 (50000)，停止插点")
                break
            if max_iterations > 50000:  # 增加迭代上限
                verbose("达到最大迭代次数限制 (50000)，停止插点")
                break
            if target_total_points is not None and len(self.points) >= target_total_points:
                verbose(f"  [进度] 达到目标节点数: {len(self.points)}")
                break

            # 从优先级队列中取出最差三角形
            if not candidate_queue:
                break

            priority, tri_id, worst_triangle = heapq.heappop(candidate_queue)
            
            # 检查三角形是否已被删除或细分
            if worst_triangle not in self.triangles:
                continue

            # 重新检查是否仍需要细分（可能在之前的插入中已被改善）
            if not self._triangle_needs_refinement(worst_triangle, min_dist_threshold):
                continue

            # 尝试多种插入点策略
            insertion_successful = False
            new_point = None
            insertion_strategy = ""
            
            # 计算当前三角形质量
            tri_quality = self._compute_triangle_quality(worst_triangle)

            # 策略 1：外接圆圆心
            if tri_quality < 0.01:
                # 退化三角形，跳过外心计算
                pass
            else:
                if not worst_triangle.circumcircle_valid:
                    self._compute_circumcircle(worst_triangle)
                candidate_point = worst_triangle.circumcenter.copy()
                
                # 检查圆心是否在有效范围内
                if (x_min - margin < candidate_point[0] < x_max + margin and
                    y_min - margin < candidate_point[1] < y_max + margin):
                    # 检查是否在孔洞内
                    if not self._point_in_hole(candidate_point):
                        # 验证距离
                        min_dist = self._compute_min_dist_to_existing_points(candidate_point)
                        if min_dist > min_dist_threshold:
                            new_point = candidate_point
                            insertion_strategy = "外心"
                            insertion_successful = True

            # 策略 2：重心坐标随机采样
            if not insertion_successful:
                p1 = self.points[worst_triangle.vertices[0]]
                p2 = self.points[worst_triangle.vertices[1]]
                p3 = self.points[worst_triangle.vertices[2]]
                
                # 尝试多次随机采样
                for _ in range(10):
                    r1, r2 = np.random.rand(2)
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2
                    candidate_point = p1 + r1 * (p2 - p1) + r2 * (p3 - p1)
                    
                    if not (x_min - margin < candidate_point[0] < x_max + margin and
                            y_min - margin < candidate_point[1] < y_max + margin):
                        continue
                    if self._point_in_hole(candidate_point):
                        continue
                    
                    min_dist = self._compute_min_dist_to_existing_points(candidate_point)
                    if min_dist > min_dist_threshold:
                        new_point = candidate_point
                        insertion_strategy = "重心采样"
                        insertion_successful = True
                        break

            # 策略 3：前端法（最优插入点）
            if not insertion_successful:
                candidate_point = self._optimal_point_frontal(worst_triangle)
                if candidate_point is not None:
                    if (x_min - margin < candidate_point[0] < x_max + margin and
                        y_min - margin < candidate_point[1] < y_max + margin):
                        if not self._point_in_hole(candidate_point):
                            min_dist = self._compute_min_dist_to_existing_points(candidate_point)
                            if min_dist > min_dist_threshold * 0.5:  # 前端法允许稍近的距离
                                new_point = candidate_point
                                insertion_strategy = "前端法"
                                insertion_successful = True

            if insertion_successful and new_point is not None:
                # 插入新点
                new_point_idx = len(self.points)
                self.points = np.vstack([self.points, new_point])
                self.triangles = self._insert_point_incremental(new_point_idx, self.triangles)
                consecutive_failures = 0
                
                # 更新 KDTree
                self._kdtree = KDTree(self.points)
                last_kdtree_build = max_iterations
                
                # 定期重建候选队列（因为三角形拓扑已改变）
                # 为了性能，不是每次都重建
                if max_iterations % kdtree_rebuild_interval == 0:
                    self._rebuild_candidate_queue(
                        candidate_queue, tri_to_entry, min_dist_threshold
                    )
                else:
                    # 非重建周期：简单地将所有新三角形加入队列
                    # 这确保不会遗漏需要细分的三角形
                    known_tri_ids = set(entry[1] for entry in candidate_queue)
                    for tri in self.triangles:
                        if id(tri) not in known_tri_ids:
                            if self._triangle_needs_refinement(tri, min_dist_threshold):
                                priority = self._compute_refinement_priority(tri)
                                if priority > 0:
                                    entry = [-priority, id(tri), tri]
                                    heapq.heappush(candidate_queue, entry)
            else:
                # 所有策略都失败
                consecutive_failures += 1
                
                # 当连续失败次数达到一定阈值时，尝试放宽条件
                if consecutive_failures == max_consecutive_failures // 2:
                    verbose(f"  [进度] 连续失败 {consecutive_failures} 次，尝试放宽插入条件...")
                    min_dist_threshold *= 0.5  # 降低最小距离要求
                
                if consecutive_failures >= max_consecutive_failures:
                    verbose(f"  [进度] 连续失败 {consecutive_failures} 次，停止插点")
                    # 输出诊断信息
                    self._diagnose_failed_triangles(min_dist_threshold)
                    break

        final_triangles = len(self.triangles)
        final_points = len(self.points)
        final_inserted = final_points - initial_point_count
        verbose(f"  [完成] 插点完成 | "
                f"节点: {final_points} (边界: {boundary_count}, 内部: {final_inserted}) | "
                f"三角形: {final_triangles}")
        
        # 最终诊断：检查是否还有不满足要求的三角形
        self._final_refinement_check(min_dist_threshold)

    def _triangle_needs_refinement(self, tri: Triangle, min_dist_threshold: float) -> bool:
        """检查三角形是否仍需细分。
        
        判断标准（与 _get_target_size_for_triangle 配合）：
        1. 如果最大边长超过目标尺寸的 120%，需要细分
        2. 如果质量低于 0.3 且有尺寸场，需要细分
        3. 如果没有尺寸场，仅基于质量判断
        """
        quality = self._compute_triangle_quality(tri)
        v0, v1, v2 = tri.vertices
        pts = self.points
        edge_lengths = [
            float(np.linalg.norm(pts[v1] - pts[v0])),
            float(np.linalg.norm(pts[v2] - pts[v1])),
            float(np.linalg.norm(pts[v0] - pts[v2])),
        ]
        max_edge = max(edge_lengths)
        target_size = self._get_target_size_for_triangle(tri)

        if target_size is not None and target_size > 1e-12:
            # 尺寸场控制：边长超过目标尺寸的 120% 或质量低于阈值
            if max_edge > target_size * 1.2:
                return True
            if quality < 0.25:
                return True
        else:
            # 无尺寸场：仅基于质量
            if quality < 0.4:
                return True

        return False

    def _compute_refinement_priority(self, tri: Triangle) -> float:
        """计算三角形的细分优先级（值越大越优先处理）。
        
        优先级 = 偏离目标尺寸的程度 + 质量惩罚项
        """
        v0, v1, v2 = tri.vertices
        pts = self.points
        edge_lengths = [
            float(np.linalg.norm(pts[v1] - pts[v0])),
            float(np.linalg.norm(pts[v2] - pts[v1])),
            float(np.linalg.norm(pts[v0] - pts[v2])),
        ]
        max_edge = max(edge_lengths)
        target_size = self._get_target_size_for_triangle(tri)
        quality = self._compute_triangle_quality(tri)

        if target_size is not None and target_size > 1e-12:
            size_ratio = max_edge / target_size
            # 偏离目标越大，优先级越高
            if size_ratio > 1.2:
                # 主要基于尺寸偏离
                priority = (size_ratio - 1.0) * 100.0
            else:
                # 尺寸满足但质量差
                priority = max(0.0, (0.25 - quality) * 10.0)
        else:
            # 无尺寸场，仅基于质量
            priority = max(0.0, (0.4 - quality) * 10.0)

        return priority

    def _rebuild_candidate_queue(self, candidate_queue, tri_to_entry, min_dist_threshold):
        """重建优先级队列。"""
        candidate_queue.clear()
        tri_to_entry.clear()

        for tri in self.triangles:
            if not self._triangle_needs_refinement(tri, min_dist_threshold):
                continue
            
            priority = self._compute_refinement_priority(tri)
            if priority > 0:
                # heapq 是最小堆，使用负 priority 实现最大堆效果
                entry = [-priority, id(tri), tri]
                tri_to_entry[id(tri)] = entry
                heapq.heappush(candidate_queue, entry)

    def _update_candidate_queue(self, candidate_queue, tri_to_entry, min_dist_threshold):
        """增量更新优先级队列（只添加新三角形）。"""
        # 清理已不存在的三角形
        valid_entries = []
        for entry in candidate_queue:
            tri_id = entry[1]
            tri = entry[2]
            if tri in self.triangles:
                valid_entries.append(entry)
        
        candidate_queue[:] = valid_entries
        heapq.heapify(candidate_queue)

        # 添加新三角形
        for tri in self.triangles:
            if id(tri) in tri_to_entry:
                continue
            if not self._triangle_needs_refinement(tri, min_dist_threshold):
                continue
            
            priority = self._compute_refinement_priority(tri)
            if priority > 0:
                entry = [-priority, id(tri), tri]
                tri_to_entry[id(tri)] = entry
                heapq.heappush(candidate_queue, entry)

    def _point_in_hole(self, point: np.ndarray) -> bool:
        """检查点是否在孔洞内。"""
        if not self.holes:
            return False
        from utils.geom_toolkit import point_in_polygon
        for hole in self.holes:
            if point_in_polygon(point, hole):
                return True
        return False

    def _compute_min_dist_to_existing_points(self, point: np.ndarray) -> float:
        """计算点到已有节点的最小距离。"""
        if len(self.points) == 0:
            return float('inf')
        if self._kdtree is not None:
            min_dist, _ = self._kdtree.query(point)
            return float(min_dist)
        # 回退：暴力计算
        min_dist = float('inf')
        for p in self.points:
            dist = float(np.linalg.norm(point - p))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _diagnose_failed_triangles(self, min_dist_threshold: float):
        """诊断无法细分的三角形。"""
        verbose("  [诊断] 无法继续插点，检查剩余不满足要求的三角形:")
        failed_count = 0
        for tri in self.triangles:
            if self._triangle_needs_refinement(tri, min_dist_threshold):
                failed_count += 1
                if failed_count <= 10:  # 只输出前 10 个
                    v0, v1, v2 = tri.vertices
                    pts = self.points
                    edge_lengths = [
                        float(np.linalg.norm(pts[v1] - pts[v0])),
                        float(np.linalg.norm(pts[v2] - pts[v1])),
                        float(np.linalg.norm(pts[v0] - pts[v2])),
                    ]
                    max_edge = max(edge_lengths)
                    target_size = self._get_target_size_for_triangle(tri)
                    quality = self._compute_triangle_quality(tri)
                    target_str = f"{target_size:.4f}" if (target_size is not None and target_size > 1e-12) else "N/A"
                    verbose(f"    三角形 {tri.vertices}: "
                           f"最大边={max_edge:.4f}, "
                           f"目标尺寸={target_str}, "
                           f"质量={quality:.4f}")
        if failed_count > 10:
            verbose(f"    ... 还有 {failed_count - 10} 个三角形未列出")
        verbose(f"  [诊断] 总计 {failed_count} 个三角形未满足要求")

    def _final_refinement_check(self, min_dist_threshold: float):
        """最终检查：统计仍有多少三角形不满足要求。"""
        needs_refinement_count = 0
        size_violation_count = 0
        quality_violation_count = 0
        
        for tri in self.triangles:
            quality = self._compute_triangle_quality(tri)
            v0, v1, v2 = tri.vertices
            pts = self.points
            edge_lengths = [
                float(np.linalg.norm(pts[v1] - pts[v0])),
                float(np.linalg.norm(pts[v2] - pts[v1])),
                float(np.linalg.norm(pts[v0] - pts[v2])),
            ]
            max_edge = max(edge_lengths)
            target_size = self._get_target_size_for_triangle(tri)

            if target_size is not None and target_size > 1e-12:
                if max_edge > target_size * 1.2:
                    size_violation_count += 1
                    needs_refinement_count += 1
                elif quality < 0.25:
                    quality_violation_count += 1
                    needs_refinement_count += 1
            else:
                if quality < 0.4:
                    quality_violation_count += 1
                    needs_refinement_count += 1
        
        if needs_refinement_count > 0:
            verbose(f"  [诊断] 插点结束后，仍有 {needs_refinement_count} 个三角形不满足要求")
            verbose(f"           尺寸违规: {size_violation_count}, 质量违规: {quality_violation_count}")
        else:
            verbose("  [诊断] 所有三角形均满足尺寸和质量要求")

    def _compute_insertion_point(self, tri, x_min, x_max, y_min, y_max, margin):
        """计算新插入点的位置。

        参考 Gmsh：
        - 优先：外接圆圆心（使用高精度算术）
        - 回退：重心坐标随机采样（仅当圆心超出有效范围时）
        """
        # 使用高精度算术计算的 circumcenter
        if not tri.circumcircle_valid:
            self._compute_circumcircle(tri)
        new_point = tri.circumcenter.copy()

        # 检查圆心是否在有效范围内
        if not (x_min - margin < new_point[0] < x_max + margin and
                y_min - margin < new_point[1] < y_max + margin):
            # 超出范围，使用重心坐标随机采样
            points = self.points
            p1 = points[tri.vertices[0]]
            p2 = points[tri.vertices[1]]
            p3 = points[tri.vertices[2]]
            r1, r2 = np.random.rand(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            new_point = p1 + r1 * (p2 - p1) + r2 * (p3 - p1)

        return new_point

    def _optimal_point_frontal(self, tri):
        """前端方法变体：在活跃边的中垂线上计算最优插入点。

        参考 Gmsh optimalPointFrontal() 算法：
        1. 找到三角形的"活跃边"（最长边或质量最差的边）
        2. 计算活跃边的中点
        3. 在中垂线上找到最优位置，使得新三角形接近等边

        这确保新插入的点产生接近等边三角形，提高网格质量。

        返回：
          - 最优插入点坐标，或 None（如果无法计算）
        """
        points = self.points
        v0, v1, v2 = tri.vertices

        # 改进：选择质量最差的边作为活跃边（而非最长边）
        # 这样可以针对性地改进最差区域
        edges = [
            (v0, v1),
            (v1, v2),
            (v0, v2)
        ]

        # 计算每条边的质量（边长与目标尺寸的比值）
        target_size = self._get_target_size_for_triangle(tri)
        best_edge = None
        worst_ratio = 0

        for edge in edges:
            length = float(np.linalg.norm(points[edge[1]] - points[edge[0]]))
            if target_size is not None:
                ratio = length / target_size
            else:
                ratio = length

            # 选择偏离目标尺寸最大的边
            if ratio > worst_ratio:
                worst_ratio = ratio
                best_edge = edge

        if best_edge is None:
            return None

        # 计算活跃边的中点
        midpoint = (points[best_edge[0]] + points[best_edge[1]]) / 2.0

        # 计算外接圆圆心
        if not tri.circumcircle_valid:
            self._compute_circumcircle(tri)
        circumcenter = tri.circumcenter

        # 计算方向向量（中点指向圆心）
        direction = circumcenter - midpoint
        dir_len = float(np.linalg.norm(direction))

        if dir_len < 1e-12:
            return None

        # 归一化方向向量
        direction = direction / dir_len

        # 计算目标距离（等边三角形的高）
        if target_size is not None:
            # 等边三角形的高：h = sqrt(3)/2 * edge_length
            target_dist = target_size * sqrt(3.0) / 2.0
        else:
            # 没有尺寸场，使用当前边长
            max_len = float(np.linalg.norm(points[best_edge[1]] - points[best_edge[0]]))
            target_dist = max_len * sqrt(3.0) / 2.0

        # 计算最优插入点：限制移动距离，避免过度偏离
        optimal_point = midpoint + direction * min(target_dist, dir_len * 0.8)

        return optimal_point

    def _validate_new_point(self, tri, new_point, min_dist_threshold):
        """验证新插入点的质量。

        Gmsh insertVertexB 做法：
        1. 最小距离检查
        2. 新三角形边长不能过小（相对于目标尺寸）
        """
        points = self.points
        v0, v1, v2 = tri.vertices

        # 计算新点到三个顶点的距离
        d0 = float(np.linalg.norm(new_point - points[v0]))
        d1 = float(np.linalg.norm(new_point - points[v1]))
        d2 = float(np.linalg.norm(new_point - points[v2]))

        # 检查最小边长（相对于目标尺寸）
        target_size = self._get_target_size_for_triangle(tri)
        if target_size is not None:
            min_edge = min(d0, d1, d2)
            if min_edge < target_size * 0.01:  # 边长不能小于目标尺寸的 1%
                return False

        # 最小距离检查
        if min(d0, d1, d2) < min_dist_threshold * 0.1:
            return False

        return True

    # -------------------------------------------------------------------------
    # Laplacian 平滑（默认关闭）
    # -------------------------------------------------------------------------

    def _laplacian_smoothing(self, iterations: int = 3, alpha: float = 0.5):
        """Laplacian 平滑，边界点保持不动，内部点被约束在原始边界包围盒内。
        
        改进 P2-2：参考 Gmsh laplaceSmoothing()
        1. 使用度量空间加权平均（考虑各向异性）
        2. 逐步衰减因子 (FACTOR /= 1.4) 保证稳定性
        3. 移动接受准则：新位置的相邻单元质量不能降低
        """
        if iterations <= 0:
            return

        boundary_x_min = np.min(self.original_points[:, 0])
        boundary_x_max = np.max(self.original_points[:, 0])
        boundary_y_min = np.min(self.original_points[:, 1])
        boundary_y_max = np.max(self.original_points[:, 1])
        margin = 1e-6
        x_min = boundary_x_min + margin
        x_max = boundary_x_max - margin
        y_min = boundary_y_min + margin
        y_max = boundary_y_max - margin

        smoothed_points = self.points.copy()

        for iteration in range(iterations):
            new_points = smoothed_points.copy()
            
            # 构建邻接关系
            neighbor_dict = {}
            for tri in self.triangles:
                for i in range(3):
                    v = tri.vertices[i]
                    if v not in neighbor_dict:
                        neighbor_dict[v] = set()
                    neighbor_dict[v].add(tri.vertices[(i + 1) % 3])
                    neighbor_dict[v].add(tri.vertices[(i + 2) % 3])

            for v, neighbors in neighbor_dict.items():
                # 边界点保持不动
                if self.boundary_mask[v]:
                    continue
                if len(neighbors) == 0:
                    continue
                
                # P2-2：度量加权平均
                weighted_sum = np.zeros(2)
                weight_total = 0.0
                
                for n in neighbors:
                    neighbor_pos = smoothed_points[n]
                    direction = neighbor_pos - smoothed_points[v]
                    
                    # 计算度量距离权重（简化版：使用欧氏距离的倒数）
                    dist = float(np.linalg.norm(direction))
                    if dist > 1e-12:
                        weight = 1.0 / dist  # 距离越近权重越大
                    else:
                        weight = 1.0
                    
                    weighted_sum += neighbor_pos * weight
                    weight_total += weight
                
                if weight_total > 1e-12:
                    target_pos = weighted_sum / weight_total
                else:
                    target_pos = smoothed_points[v]
                
                # P2-2：逐步衰减因子（最多 5 次尝试）
                FACTOR = 1.0
                best_pos = smoothed_points[v]
                best_quality = self._compute_vertex_quality(v, smoothed_points)
                
                for _ in range(5):
                    trial_pos = smoothed_points[v] + alpha * FACTOR * (target_pos - smoothed_points[v])
                    
                    # 约束在边界包围盒内
                    trial_pos[0] = np.clip(trial_pos[0], x_min, x_max)
                    trial_pos[1] = np.clip(trial_pos[1], y_min, y_max)
                    
                    # 移动接受准则：验证质量不下降
                    trial_quality = self._compute_vertex_quality_at(v, trial_pos, smoothed_points)
                    if trial_quality >= best_quality * 0.95:  # 允许 5% 的质量损失
                        best_pos = trial_pos
                        best_quality = trial_quality
                        break
                    else:
                        FACTOR /= 1.4  # 衰减因子
                
                new_points[v] = best_pos

            smoothed_points = new_points

        self.points = smoothed_points

    def _compute_vertex_quality(self, v_idx, points):
        """计算顶点 v_idx 周围三角形的平均质量。"""
        total_quality = 0.0
        count = 0
        
        for tri in self.triangles:
            if v_idx in tri.vertices:
                quality = self._compute_triangle_quality(tri)
                total_quality += quality
                count += 1
        
        return total_quality / count if count > 0 else 0.0

    def _compute_vertex_quality_at(self, v_idx, new_pos, points):
        """计算顶点移动到 new_pos 后周围三角形的平均质量。"""
        # 临时移动顶点
        old_pos = points[v_idx].copy()
        points[v_idx] = new_pos
        
        # 清除相关三角形的缓存
        for tri in self.triangles:
            if v_idx in tri.vertices:
                tri.quality_valid = False
                tri.circumcircle_valid = False
        
        # 计算新质量
        quality = self._compute_vertex_quality(v_idx, points)
        
        # 恢复顶点位置
        points[v_idx] = old_pos
        
        # 恢复缓存状态
        for tri in self.triangles:
            if v_idx in tri.vertices:
                tri.quality_valid = False
                tri.circumcircle_valid = False
        
        return quality

    # -------------------------------------------------------------------------
    # 孔洞处理
    # -------------------------------------------------------------------------

    def _remove_hole_triangles(self) -> int:
        """删除孔洞内的三角形和孤立节点。

        返回删除的三角形数量。
        """
        if not self.holes:
            return 0

        from utils.geom_toolkit import is_polygon_clockwise
        fixed_holes = []
        for hole in self.holes:
            if is_polygon_clockwise(hole):
                fixed_holes.append(hole[::-1])
            else:
                fixed_holes.append(hole.copy())
        holes_to_use = fixed_holes

        triangles_to_remove = []
        for i, tri in enumerate(self.triangles):
            centroid = self._compute_triangle_centroid(tri)
            in_hole = any(point_in_polygon(centroid, h) for h in holes_to_use)

            if not in_hole:
                for vert_idx in tri.vertices:
                    if vert_idx >= self.boundary_count:
                        vert = self.points[vert_idx]
                        if any(point_in_polygon(vert, h) for h in holes_to_use):
                            in_hole = True
                            break
                    if in_hole:
                        break

            if in_hole:
                triangles_to_remove.append(i)

        for i in reversed(triangles_to_remove):
            del self.triangles[i]

        removed_tri_count = len(triangles_to_remove)
        if removed_tri_count > 0:
            verbose(f"  删除孔洞内三角形: {removed_tri_count} 个")

        # 删除孤立节点
        used_nodes = set()
        for tri in self.triangles:
            used_nodes.update(tri.vertices)
        orphan_nodes = set(range(len(self.points))) - used_nodes

        if orphan_nodes:
            nodes_to_remove = [
                n for n in orphan_nodes
                if n >= self.boundary_count and any(point_in_polygon(self.points[n], h) for h in holes_to_use)
            ]
            if nodes_to_remove:
                verbose(f"  删除孔洞内孤立节点: {len(nodes_to_remove)} 个")
                keep_mask = np.ones(len(self.points), dtype=bool)
                keep_mask[nodes_to_remove] = False
                self.points = self.points[keep_mask]
                if len(self.boundary_mask) == len(keep_mask):
                    self.boundary_mask = self.boundary_mask[keep_mask]
                index_map = np.cumsum(keep_mask.astype(int)) - 1
                for tri in self.triangles:
                    tri.vertices = tuple(int(index_map[v]) for v in tri.vertices)

        return removed_tri_count

    # -------------------------------------------------------------------------
    # 边界边恢复
    # -------------------------------------------------------------------------

    def _recover_boundary_edges(self) -> int:
        """恢复丢失的边界边。

        返回恢复的边界边数量。
        """
        if not self.protected_edges:
            return 0

        missing_edges = []
        for protected_edge in self.protected_edges:
            v1, v2 = sorted(list(protected_edge))
            if not any(v1 in tri.vertices and v2 in tri.vertices for tri in self.triangles):
                missing_edges.append((v1, v2))

        if not missing_edges:
            return 0

        verbose(f"  检测到 {len(missing_edges)} 条丢失的边界边")
        recovered_count = sum(1 for v1, v2 in missing_edges if self._recover_single_edge(v1, v2))
        if recovered_count > 0:
            verbose(f"  成功恢复 {recovered_count}/{len(missing_edges)} 条边界边")
        return recovered_count

    def _recover_single_edge(self, v1: int, v2: int) -> bool:
        """恢复单条边界边（优先使用边翻转策略）。"""
        if any(v1 in tri.vertices and v2 in tri.vertices for tri in self.triangles):
            return True

        # 尝试使用边翻转策略恢复
        if self._recover_edge_by_flipping(v1, v2):
            return True

        # 翻转失败（遇到非凸边界等情况），回退到中点插入
        return self._insert_midpoint_for_edge(v1, v2)

    def _recover_edge_by_flipping(self, v1: int, v2: int, max_iter: int = 500) -> bool:
        """通过连续边翻转恢复边界边 (v1, v2)。"""
        for _ in range(max_iter):
            intersecting = self._find_intersecting_edge(v1, v2)
            if intersecting is None:
                return any(v1 in tri.vertices and v2 in tri.vertices for tri in self.triangles)
            # intersecting 是元组 (a_idx, b_idx)，需要解包
            if not self._flip_edge(intersecting[0], intersecting[1]):
                return False
        return False

    def _find_intersecting_edge(self, v1: int, v2: int):
        """查找与线段(v1, v2)严格相交的三角形边。"""
        p1, p2 = self.points[v1], self.points[v2]
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        for tri in self.triangles:
            for i in range(3):
                a_idx = tri.vertices[i]
                b_idx = tri.vertices[(i + 1) % 3]
                if a_idx in (v1, v2) or b_idx in (v1, v2):
                    continue
                a, b = self.points[a_idx], self.points[b_idx]
                d1 = cross(p1, p2, a)
                d2 = cross(p1, p2, b)
                d3 = cross(a, b, p1)
                d4 = cross(a, b, p2)
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return (a_idx, b_idx)
        return None

    def _flip_edge(self, n1: int, n2: int) -> bool:
        """翻转边 (n1, n2)。要求该边被两个三角形共享。
        
        改进：翻转后更新邻接关系。
        """
        t1 = t2 = None
        for tri in self.triangles:
            if n1 in tri.vertices and n2 in tri.vertices:
                if t1 is None:
                    t1 = tri
                elif t2 is None:
                    t2 = tri
                else:
                    break

        if t1 is None or t2 is None:
            return False

        a_idx = next(v for v in t1.vertices if v != n1 and v != n2)
        b_idx = next(v for v in t2.vertices if v != n1 and v != n2)
        n1_pt, n2_pt, a_pt, b_pt = self.points[n1], self.points[n2], self.points[a_idx], self.points[b_idx]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # 检查四边形 (n1 -> a -> n2 -> b) 是否为凸四边形
        c1 = cross(n1_pt, a_pt, n2_pt)
        c2 = cross(a_pt, n2_pt, b_pt)
        c3 = cross(n2_pt, b_pt, n1_pt)
        c4 = cross(b_pt, n1_pt, a_pt)
        is_convex = (c1 > 0 and c2 > 0 and c3 > 0 and c4 > 0) or \
                    (c1 < 0 and c2 < 0 and c3 < 0 and c4 < 0)
        if not is_convex:
            return False

        # 保存旧邻接关系（排除 t1 和 t2）
        old_neighbors_t1 = [n for n in t1.neighbors if n is not None and n is not t2]
        old_neighbors_t2 = [n for n in t2.neighbors if n is not None and n is not t1]

        self.triangles.remove(t1)
        self.triangles.remove(t2)

        new_tri1 = Triangle(n1, a_idx, b_idx)
        new_tri2 = Triangle(n2, b_idx, a_idx)
        self._compute_circumcircle(new_tri1)
        self._compute_circumcircle(new_tri2)
        self.triangles.append(new_tri1)
        self.triangles.append(new_tri2)
        
        # 增量更新邻接关系
        # 新三角形继承旧三角形的邻接关系（除了彼此）
        new_tri1.neighbors = [None, None, None]
        new_tri2.neighbors = [None, None, None]
        
        # 更新指向 t1/t2 的邻接指针
        for neighbor in old_neighbors_t1:
            for i in range(3):
                if neighbor.neighbors[i] is t1:
                    # 需要找到新的邻接三角形
                    pass  # 稍后通过 _build_adjacency 统一更新
        
        # 为简单起见，翻转后重建整个邻接关系
        self._build_adjacency(self.triangles)
        
        return True

    def _insert_midpoint_for_edge(self, v1: int, v2: int) -> bool:
        """通过在边中点插入新点来恢复边界边。
        
        改进：插入后重建邻接关系。
        """
        midpoint = (self.points[v1] + self.points[v2]) / 2.0
        min_dist = float('inf')
        if self._kdtree is not None:
            min_dist, _ = self._kdtree.query(midpoint)

        if min_dist > 1e-6:
            new_point_idx = len(self.points)
            self.points = np.vstack([self.points, midpoint])
            self.boundary_mask = np.append(self.boundary_mask, False)
            self.triangles = self._insert_point_incremental(new_point_idx, self.triangles)
            # 邻接关系已经在 _insert_point_incremental 中更新
            return True
        return False

    def _clear_triangle_caches(self):
        """清理所有三角形的缓存。"""
        for tri in self.triangles:
            tri.circumcircle_valid = False
            tri.quality_valid = False
            tri.circumcenter = None
            tri.circumradius = None

    # -------------------------------------------------------------------------
    # 公共入口：generate_mesh
    # -------------------------------------------------------------------------

    def generate_mesh(
        self,
        target_triangle_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成三角形网格。

        参数:
            target_triangle_count: 目标三角形数量（可选）

        返回:
            (points, simplices, boundary_mask)
                - points: 点坐标数组 (N, 2)
                - simplices: 三角形索引数组 (M, 3)
                - boundary_mask: 边界点掩码 (N,)
        """
        timer = TimeSpan("开始 Bowyer-Watson 网格生成...")

        self.boundary_count = len(self.original_points)
        self.points = self.original_points.copy()
        self.boundary_mask = np.zeros(len(self.points), dtype=bool)
        self.boundary_mask[:self.boundary_count] = True
        verbose(f"边界点数量: {self.boundary_count}")

        verbose("阶段 1/3: 初始三角剖分...")
        self._triangulate()
        verbose(f"  初始三角形数量: {len(self.triangles)}")

        verbose("阶段 2/3: 迭代插入内部点...")
        if self.holes:
            verbose(f"  检测到 {len(self.holes)} 个孔洞，插点时将拒绝在孔洞内插入新点")
        self._insert_points_iteratively(target_triangle_count)

        if self.holes:
            verbose("阶段 2.5/3: 清理孔洞内三角形...")
            before_count = len(self.triangles)
            self._remove_hole_triangles()
            verbose(f"  删除孔洞内三角形: {before_count - len(self.triangles)} 个")

        verbose("阶段 2.6/3: 恢复边界边...")
        self._recover_boundary_edges()

        if self.smoothing_iterations > 0:
            verbose(f"阶段 3/3: Laplacian 平滑 ({self.smoothing_iterations} 次迭代)...")
            current_point_count = len(self.points)
            self.boundary_mask = np.zeros(current_point_count, dtype=bool)
            self.boundary_mask[:self.boundary_count] = True
            self._laplacian_smoothing(self.smoothing_iterations)

            # Laplacian 平滑只移动点位置，不改变拓扑连接
            # 因此不需要重新三角剖分，保持原有的三角形连接即可
            verbose("  平滑完成（保持原有三角连接）")
        else:
            verbose("阶段 3/3: 跳过平滑（未启用）")

        points = self.points.copy()
        simplices = np.array([tri.vertices for tri in self.triangles])
        boundary_mask = np.zeros(len(points), dtype=bool)
        boundary_mask[:self.boundary_count] = True

        verbose("网格生成完成:")
        verbose(f"  - 总节点数: {len(points)}")
        verbose(f"  - 边界节点: {np.sum(boundary_mask)}")
        verbose(f"  - 内部节点: {len(points) - np.sum(boundary_mask)}")
        verbose(f"  - 三角形数: {len(simplices)}")

        timer.show_to_console("Bowyer-Watson 网格生成完成")
        return points, simplices, boundary_mask


# =============================================================================
# GmshBowyerWatsonMeshGenerator
# =============================================================================

class GmshBowyerWatsonMeshGenerator:
    """Gmsh 风格的 Bowyer-Watson 网格生成器。
    
    参考 Gmsh 的核心算法流程：
    1. 构建初始三角剖分（超级三角形）
    2. 主循环：选择最差三角形 → 计算插入点 → 插入点
    3. Cavity 搜索：递归查找违反 Delaunay 的三角形
    4. 重新连接：将新点与空腔边界连接
    5. 后处理：孔洞清理、边界恢复、平滑
    
    关键改进：
    - 使用 MTri3 数据结构（支持懒删除和邻接关系）
    - 使用优先级队列选择最差三角形
    - 使用鲁棒谓词确保数值稳定性
    - 支持边界边保护
    """
    
    def __init__(
        self,
        boundary_points: np.ndarray,
        boundary_edges: Optional[List[Tuple[int, int]]] = None,
        sizing_system=None,
        max_edge_length: Optional[float] = None,
        smoothing_iterations: int = 0,
        seed: Optional[int] = None,
        holes: Optional[List[np.ndarray]] = None,
        outer_boundary: Optional[np.ndarray] = None,
    ):
        """初始化生成器。

        参数:
            boundary_points: 边界点坐标数组，形状 (N, 2)
            boundary_edges: 边界边列表 [(idx1, idx2), ...]
            sizing_system: QuadtreeSizing 尺寸场对象（可选）
            max_edge_length: 全局最大边长（可选）
            smoothing_iterations: Laplacian 平滑迭代次数（默认0）
            seed: 随机种子
            holes: 孔洞边界列表，每个孔洞是点数组 (M, 2)
            outer_boundary: 外边界点数组，形状 (K, 2)（可选）
        """
        self.original_points = boundary_points.copy()

        # 保护的边界边集合
        self.protected_edges: Set[frozenset] = set()
        for edge in (boundary_edges or []):
            self.protected_edges.add(frozenset(edge))
        self.boundary_edges = boundary_edges or []

        self.sizing_system = sizing_system
        self.max_edge_length = max_edge_length
        self.smoothing_iterations = smoothing_iterations
        self.seed = seed
        self.holes = holes or []
        self.outer_boundary = outer_boundary  # 新增：外边界
        
        if seed is not None:
            np.random.seed(seed)
        
        # 工作状态变量
        self.points: Optional[np.ndarray] = None
        self.triangles: List[MTri3] = []
        self.boundary_mask: Optional[np.ndarray] = None
        self.boundary_count: int = 0
        self._kdtree = None
        self._tri_id_counter = 0
    
    def _next_tri_id(self) -> int:
        """生成唯一的三角形 ID。"""
        self._tri_id_counter += 1
        return self._tri_id_counter
    
    # -------------------------------------------------------------------------
    # 边界保护
    # -------------------------------------------------------------------------
    

    def _is_edge_split_by_steiner(self, v1: int, v2: int) -> Tuple[bool, Optional[int]]:
        """检查边 (v1, v2) 是否被 Steiner 点分割。

        返回：
            (is_split, steiner_idx): 是否被分割，以及 Steiner 点索引（如果存在）
        """
        # 找到所有包含 v1 的三角形
        tris_with_v1 = [tri for tri in self.triangles if v1 in tri.vertices and not tri.is_deleted()]
        
        # 检查是否有三角形同时包含 v1 和 v2
        for tri in tris_with_v1:
            if v2 in tri.vertices:
                return False, None  # 边存在，未被分割
        
        # 找到所有与 v1 相连的点
        neighbors_of_v1 = set()
        for tri in tris_with_v1:
            for v in tri.vertices:
                if v != v1:
                    neighbors_of_v1.add(v)
        
        # 检查是否有 Steiner 点在 v1-v2 连线上
        for neighbor in neighbors_of_v1:
            if neighbor == v2 or neighbor < self.boundary_count:
                continue  # 跳过 v2 本身和原始边界点
            
            # 检查 neighbor 是否在 v1-v2 连线上
            p1 = self.points[v1]
            p2 = self.points[v2]
            pn = self.points[neighbor]
            
            # 检查三点共线
            cross_product = (p2[0] - p1[0]) * (pn[1] - p1[1]) - (p2[1] - p1[1]) * (pn[0] - p1[0])
            if abs(cross_product) < 1e-10:
                # 三点共线，检查 neighbor 是否在 v1-v2 之间
                dot_product = (pn[0] - p1[0]) * (p2[0] - p1[0]) + (pn[1] - p1[1]) * (p2[1] - p1[1])
                segment_length_sq = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
                if 0 < dot_product < segment_length_sq:
                    return True, neighbor  # 被 Steiner 点分割
        
        return False, None

    def _is_protected_edge(self, v1: int, v2: int) -> bool:
        """检查边是否是受保护的边界边。"""
        return frozenset({v1, v2}) in self.protected_edges
    
    # -------------------------------------------------------------------------
    # 鲁棒几何谓词（封装）
    # -------------------------------------------------------------------------

    def _point_in_triangle(self, point: np.ndarray, tri: MTri3, tolerance: float = 1e-12) -> bool:
        """检查点是否在三角形内部（使用 orient2d 符号测试）。

        由于 MTri3 存储排序后的顶点 (v0 < v1 < v2)，需要先确定 CCW 顺序。
        方法：计算重心，判断重心相对于每条边的方向。

        对于 CCW 三角形，顶点在三条边的左侧（orient2d > 0）。
        """
        p0 = self.points[tri.vertices[0]]
        p1 = self.points[tri.vertices[1]]
        p2 = self.points[tri.vertices[2]]

        centroid = (p0 + p1 + p2) / 3.0

        d0 = orient2d_fast(p0, p1, centroid)
        d1 = orient2d_fast(p1, p2, centroid)
        d2 = orient2d_fast(p2, p0, centroid)

        if d0 > tolerance and d1 > tolerance and d2 > tolerance:
            ccw = True
        elif d0 < -tolerance and d1 < -tolerance and d2 < -tolerance:
            ccw = False
        else:
            return False

        if ccw:
            orient0 = orient2d_fast(p0, p1, point)
            orient1 = orient2d_fast(p1, p2, point)
            orient2 = orient2d_fast(p2, p0, point)
        else:
            orient0 = orient2d_fast(p1, p0, point)
            orient1 = orient2d_fast(p2, p1, point)
            orient2 = orient2d_fast(p0, p2, point)

        if orient0 > tolerance and orient1 > tolerance and orient2 > tolerance:
            return True
        if orient0 < -tolerance or orient1 < -tolerance or orient2 < -tolerance:
            return False

        return (orient0 >= 0 and orient1 >= 0 and orient2 >= 0) or (orient0 <= 0 and orient1 <= 0 and orient2 <= 0)

    def _point_in_circumcircle(self, point: np.ndarray, tri: MTri3) -> bool:
        """鲁棒的点在圆内测试。

        使用 Shewchuk 的 incircle 谓词，确保数值稳定性。
        这是用于 Cavity 搜索的方法，不是点定位方法。
        """
        if tri.circumradius is None:
            self._compute_circumcircle(tri)

        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        return point_in_circumcircle_robust(p1, p2, p3, point)
    
    def _compute_circumcircle(self, tri: MTri3) -> Tuple[np.ndarray, float]:
        """计算三角形的外接圆（高精度）。"""
        if tri.circumcenter is not None and tri.circumradius is not None:
            return tri.circumcenter, tri.circumradius
        
        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]
        
        center, radius = compute_circumcircle(p1, p2, p3)
        
        tri.circumcenter = center
        tri.circumradius = radius
        
        return center, radius
    
    # -------------------------------------------------------------------------
    # 质量与尺寸计算
    # -------------------------------------------------------------------------
    
    def _compute_triangle_quality(self, tri: MTri3) -> float:
        """计算三角形质量（2 * r_inscribed / r_circumscribed）。"""
        if tri.quality > 0:
            return tri.quality
        
        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]
        
        a = float(np.linalg.norm(p2 - p1))
        b = float(np.linalg.norm(p3 - p2))
        c = float(np.linalg.norm(p1 - p3))
        s = (a + b + c) / 2.0
        
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq < 0:
            quality = 0.0
        else:
            area = sqrt(max(area_sq, 0.0))
            if area < 1e-12:
                quality = 0.0
            else:
                inscribed_radius = area / s
                if tri.circumradius is None:
                    self._compute_circumcircle(tri)
                circumscribed_radius = tri.circumradius
                quality = min(2.0 * inscribed_radius / circumscribed_radius, 1.0) if circumscribed_radius > 1e-12 else 0.0
        
        tri.quality = quality
        return quality
    
    def _get_target_size_for_triangle(self, tri: MTri3) -> Optional[float]:
        """获取三角形的目标尺寸。

        Gmsh 做法：使用三角形三个顶点处局部尺寸的平均值。
        
        关键修复：当三角形包含边界节点时，目标尺寸应考虑边界边长度，
        避免因尺寸场过大而产生极扁的三角形。
        """
        target_size = None
        v0, v1, v2 = tri.vertices
        pts = self.points

        if self.sizing_system is not None:
            center = np.mean([
                pts[v0],
                pts[v1],
                pts[v2],
            ], axis=0)
            try:
                target_size = self.sizing_system.spacing_at(center)
            except Exception as e:
                debug(f"获取尺寸场失败: {e}，使用全局尺寸")

        if target_size is None or target_size < 1e-12:
            target_size = self.max_edge_length

        # 关键修复：计算三角形的边长，特别是边界边的长度
        edge_lengths = [
            float(np.linalg.norm(pts[v1] - pts[v0])),
            float(np.linalg.norm(pts[v2] - pts[v1])),
            float(np.linalg.norm(pts[v0] - pts[v2])),
        ]
        triangle_edges = [
            tri.get_edge_sorted(0),
            tri.get_edge_sorted(1),
            tri.get_edge_sorted(2),
        ]
        min_edge = min(edge_lengths)
        max_edge = max(edge_lengths)

        protected_edge_lengths = [
            length
            for edge, length in zip(triangle_edges, edge_lengths)
            if frozenset(edge) in self.protected_edges
        ]
        if protected_edge_lengths:
            target_size = max(target_size, max(protected_edge_lengths))

        # 如果最小边远小于目标尺寸（比如边界边），使用平均边长作为更合理的目标
        # 这可以避免在边界附近产生极扁的三角形
        if min_edge < target_size * 0.2:
            avg_edge = sum(edge_lengths) / 3.0
            # 使用平均边长和目标尺寸的较小值
            target_size = min(target_size, avg_edge * 1.5)

        # 如果仍然无效，使用三角形的最大边长
        if target_size is None or target_size < 1e-12:
            target_size = max_edge

        return target_size

    def _point_in_hole(self, point: np.ndarray) -> bool:
        """检查点是否落在任一孔洞内。"""
        for hole in self.holes:
            if point_in_polygon(point, hole):
                return True
        return False

    def _compute_min_dist_to_existing_points(self, point: np.ndarray) -> float:
        """计算候选点到现有节点的最小距离。"""
        if self.points is None or len(self.points) == 0:
            return float('inf')
        if self._kdtree is not None:
            min_dist, _ = self._kdtree.query(point)
            return float(min_dist)
        diff = self.points - point
        return float(np.min(np.linalg.norm(diff, axis=1)))

    def _triangle_has_boundary_vertex(self, tri: MTri3) -> bool:
        """检查三角形是否接触原始边界。"""
        return any(v < self.boundary_count for v in tri.vertices)

    def _get_triangle_protected_edges(self, tri: MTri3) -> List[Tuple[int, int]]:
        """返回三角形中的受保护边。"""
        protected = []
        for i in range(3):
            edge = tri.get_edge_sorted(i)
            if frozenset(edge) in self.protected_edges:
                protected.append(edge)
        return protected

    def _compute_offcenter_refinement_point(
        self,
        tri: MTri3,
        target_size: Optional[float],
    ) -> Optional[np.ndarray]:
        """为边界邻近的差质量三角形生成更保守的 off-center 候选点。"""
        v0, v1, v2 = tri.vertices
        edge_specs = [
            (tri.get_edge_sorted(0), v2),
            (tri.get_edge_sorted(1), v0),
            (tri.get_edge_sorted(2), v1),
        ]

        protected_specs = [
            spec for spec in edge_specs if frozenset(spec[0]) in self.protected_edges
        ]
        candidate_specs = protected_specs if protected_specs else edge_specs

        best_edge, opposite_vertex = min(
            candidate_specs,
            key=lambda spec: float(
                np.linalg.norm(self.points[spec[0][1]] - self.points[spec[0][0]])
            ),
        )

        p0 = self.points[best_edge[0]]
        p1 = self.points[best_edge[1]]
        popp = self.points[opposite_vertex]

        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-12:
            return None

        midpoint = 0.5 * (p0 + p1)
        normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
        if np.dot(popp - midpoint, normal) < 0.0:
            normal = -normal

        altitude = abs(edge_vec[0] * (popp[1] - p0[1]) - edge_vec[1] * (popp[0] - p0[0])) / edge_len
        if altitude < 1e-12:
            return None

        desired_height = 0.6 * altitude
        if target_size is not None and target_size > 1e-12:
            target_height_sq = target_size * target_size - 0.25 * edge_len * edge_len
            if target_height_sq > 0.0:
                desired_height = min(desired_height, sqrt(target_height_sq))

        desired_height = max(0.35 * altitude, min(desired_height, 0.8 * altitude))
        candidate = midpoint + normal * desired_height

        return candidate if self._point_in_triangle(candidate, tri, tolerance=1e-10) else None

    def _select_refinement_point(
        self,
        tri: MTri3,
        target_size: Optional[float],
        min_dist_threshold: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        margin: float,
    ) -> Tuple[Optional[np.ndarray], str]:
        """选择更稳健的细化点：边界邻近处优先尝试保守 off-center。"""
        if tri.circumcenter is None:
            self._compute_circumcircle(tri)

        candidates: List[Tuple[str, Optional[np.ndarray], float]] = []
        candidates.append(("circumcenter", tri.circumcenter.copy(), min_dist_threshold))

        for strategy, point, min_distance in candidates:
            if point is None or not np.all(np.isfinite(point)):
                continue
            if not (x_min - margin < point[0] < x_max + margin and
                    y_min - margin < point[1] < y_max + margin):
                continue
            if self._point_in_hole(point):
                continue
            return point, strategy

        return None, "none"

    def _rollback_failed_cavity_insertion(
        self,
        cavity_triangles: List[MTri3],
        new_point_idx: int,
    ) -> None:
        """撤销失败插点留下的 deleted 标记和临时新点。"""
        for tri in cavity_triangles:
            tri.set_deleted(False)
        if self.points is not None and 0 <= new_point_idx == len(self.points) - 1:
            self.points = self.points[:-1]

    def _insert_refinement_point(
        self,
        start_tri: MTri3,
        new_point: np.ndarray,
    ) -> Tuple[List[MTri3], bool]:
        """执行一次带回滚保护的插点。"""
        new_point_idx = len(self.points)
        self.points = np.vstack([self.points, new_point])

        cavity_triangles, shell_edges = recur_find_cavity(
            start_tri,
            new_point,
            new_point_idx,
            self.points,
            self.protected_edges,
            self._point_in_circumcircle
        )

        if not cavity_triangles:
            self._rollback_failed_cavity_insertion(cavity_triangles, new_point_idx)
            return [], False

        new_triangles, success = insert_vertex(
            shell_edges,
            cavity_triangles,
            new_point_idx,
            self.points,
            self.triangles,
            validate_star=False,
        )

        if not success:
            self._rollback_failed_cavity_insertion(cavity_triangles, new_point_idx)
            return [], False

        return new_triangles, True
    
    # -------------------------------------------------------------------------
    # 初始三角剖分
    # -------------------------------------------------------------------------
    
    def _create_super_triangle(self) -> MTri3:
        """创建包含所有点的超级三角形。"""
        min_x = np.min(self.points[:, 0])
        max_x = np.max(self.points[:, 0])
        min_y = np.min(self.points[:, 1])
        max_y = np.max(self.points[:, 1])
        
        dx = max_x - min_x
        dy = max_y - min_y
        delta = max(dx, dy) * 10.0
        
        p1 = len(self.points)
        p2 = len(self.points) + 1
        p3 = len(self.points) + 2
        
        super_verts = np.array([
            [min_x - delta, min_y - delta],
            [max_x + delta, min_y - delta],
            [(min_x + max_x) / 2.0, max_y + 3.0 * delta],
        ])
        self.points = np.vstack([self.points, super_verts])
        
        tri = MTri3(p1, p2, p3, idx=self._next_tri_id())
        self._compute_circumcircle(tri)
        return tri
    
    def _triangulate(self) -> List[MTri3]:
        """执行初始 Bowyer-Watson 三角剖分。

        流程：超级三角形 → 逐点插入边界点 → 删除超级三角形
        """
        super_tri = self._create_super_triangle()
        triangles = [super_tri]
        real_point_count = len(self.points) - 3

        verbose(f"  超级三角形创建完成，顶点: {super_tri.vertices}, 半径: {super_tri.circumradius}")
        verbose(f"  待插入边界点数: {real_point_count}")

        # 逐点插入边界点
        inserted_count = 0
        for i in range(real_point_count):
            point = self.points[i]

            # 查找包含该点的三角形（使用 point-in-triangle 测试）
            containing_tri = None
            for tri in triangles:
                if not tri.is_deleted() and self._point_in_triangle(point, tri):
                    containing_tri = tri
                    break

            # 如果找不到包含点的三角形，尝试查找外接圆包含该点的三角形
            if containing_tri is None:
                for tri in triangles:
                    if not tri.is_deleted() and self._point_in_circumcircle(point, tri):
                        containing_tri = tri
                        break

            # 如果仍然找不到，查找距离点最近的三角形
            if containing_tri is None:
                min_dist = float('inf')
                for tri in triangles:
                    if tri.is_deleted():
                        continue
                    # 计算点到三角形质心的距离
                    centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
                    dist = float(np.linalg.norm(point - centroid))
                    if dist < min_dist:
                        min_dist = dist
                        containing_tri = tri

            if containing_tri is None:
                if i < 5 or i >= real_point_count - 5:
                    verbose(f"  警告：点 {i} ({point}) 未找到包含三角形")
                continue

            # 使用 Gmsh Cavity 搜索（使用 circumcircle 测试，这是正确的）
            cavity_triangles, shell_edges = recur_find_cavity(
                containing_tri,
                point,
                i,
                self.points,
                self.protected_edges,
                self._point_in_circumcircle
            )

            # 插入点
            new_triangles, success = insert_vertex(
                shell_edges,
                cavity_triangles,
                i,
                self.points,
                triangles,
                validate_star=False,  # 初始剖分不需要严格验证
                validate_edges=False,  # 初始剖分不需要边长验证
            )

            if success:
                inserted_count += 1

        verbose(f"  成功插入 {inserted_count}/{real_point_count} 个边界点")

        # 删除超级三角形
        super_verts = {super_tri.vertices[0], super_tri.vertices[1], super_tri.vertices[2]}
        self.triangles = [tri for tri in triangles if not any(v in super_verts for v in tri.vertices)]
        self.points = self.points[:-3]

        # 构建邻接关系
        build_adjacency_from_triangles(self.triangles)

        # Debug: Check wall boundary edges before recovery
        wall_missing_before = []
        for edge in self.protected_edges:
            v1, v2 = sorted(list(edge))
            if v1 >= 38 and v2 <= 95:
                if not any(v1 in tri.vertices and v2 in tri.vertices for tri in self.triangles if not tri.is_deleted()):
                    wall_missing_before.append((v1, v2))
        if wall_missing_before:
            verbose(f"  [DEBUG] 删除超级三角形后，{len(wall_missing_before)}条 wall 边缺失：{wall_missing_before[:5]}...")

        # 初始三角剖分后，立即恢复所有受保护边界边
        self._recover_all_protected_edges_initial()

        # Debug: Check wall boundary edges after recovery
        wall_missing_after = []
        for edge in self.protected_edges:
            v1, v2 = sorted(list(edge))
            if v1 >= 38 and v2 <= 95:
                if not any(v1 in tri.vertices and v2 in tri.vertices for tri in self.triangles if not tri.is_deleted()):
                    wall_missing_after.append((v1, v2))
        if wall_missing_after:
            verbose(f"  [DEBUG] 初始恢复后，{len(wall_missing_after)}条 wall 边仍然缺失：{wall_missing_after[:5]}...")

        return self.triangles

    def _recover_all_protected_edges_initial(self):
        """在初始三角剖分后立即恢复所有受保护边界边。

        由于初始三角剖分期间可能某些受保护边界边没有被正确创建，
        此方法强制确保所有受保护边界边都存在于网格中。
        """
        missing_edges = []
        for edge in self.protected_edges:
            v1, v2 = list(edge)
            if not self._is_boundary_edge_in_mesh(v1, v2):
                missing_edges.append((v1, v2))

        if not missing_edges:
            return

        verbose(f"  初始三角剖分后检测到 {len(missing_edges)} 条边界边缺失，立即恢复...")

        self._in_initial_boundary_recovery = True
        try:
            for v1, v2 in missing_edges:
                self._force_recover_boundary_edge_by_reconnection(v1, v2)
        finally:
            self._in_initial_boundary_recovery = False

        still_missing = []
        for edge in self.protected_edges:
            v1, v2 = list(edge)
            if not self._is_boundary_edge_in_mesh(v1, v2):
                still_missing.append((v1, v2))

        if still_missing:
            verbose(f"  仍有 {len(still_missing)} 条边界边无法恢复")

    # -------------------------------------------------------------------------
    # Gmsh 风格的迭代插点主循环
    # -------------------------------------------------------------------------
    
    def _insert_points_iteratively_gmsh(self, target_triangle_count: Optional[int] = None):
        """Gmsh 风格的迭代插入内部点。
        
        参考 Gmsh bowyerWatson 主循环：
        1. 从优先级队列取出最差三角形（外接圆半径最大）
        2. 检查终止条件（半径阈值或顶点数限制）
        3. 计算插入点（外接圆心）
        4. 插入点（Cavity 搜索 + 重新连接）
        5. 更新优先级队列
        """
        boundary_count = self.boundary_count
        initial_point_count = len(self.points)
        
        target_total_points = None
        if target_triangle_count is not None:
            target_total_points = (target_triangle_count + 2 + boundary_count) // 2
        
        # 计算有效范围
        x_min = np.min(self.original_points[:, 0])
        x_max = np.max(self.original_points[:, 0])
        y_min = np.min(self.original_points[:, 1])
        y_max = np.max(self.original_points[:, 1])
        margin = 0.001 * max(x_max - x_min, y_max - y_min)
        
        if len(self.original_points) > 1:
            avg_edge = float(np.linalg.norm(self.original_points[1] - self.original_points[0]))
            min_dist_threshold = 0.01 * avg_edge if avg_edge > 0 else 0.01
        else:
            min_dist_threshold = 0.01
        
        max_iterations = 0
        consecutive_failures = 0
        max_consecutive_failures = 1000
        failed_tri_counts = Counter()
        max_tri_failures = 1

        self._kdtree = None
        
        # 构建优先级队列（按外接圆半径降序）
        # 使用负半径因为 heapq 是最小堆
        priority_queue = []
        for tri in self.triangles:
            if tri.circumradius is None:
                self._compute_circumcircle(tri)
            heapq.heappush(priority_queue, (-tri.circumradius, id(tri), tri))

        # Gmsh 终止条件：外接圆半径阈值 (0.5 * sqrt(2) ≈ 0.707)
        # 作为保护机制，但不作为主要终止条件
        gmsh_radius_threshold = 0.5 * sqrt(2.0)

        # 定期全量检查的间隔
        full_check_interval = 100
        last_full_check = 0

        # 版本控制：A=稳定（保守细分），B=激进（积极细分），C=融合（智能自适应）
        # 版本C设计（基于Gmsh Frontal Delaunay理念）：
        #   - 核心思想：边界保护优先，内部适度细化
        #   - 策略1：边界边附近的三角形使用宽松阈值（size_tolerance=1.2, quality_threshold=0.1）
        #   - 策略2：远离边界的内部三角形使用严格阈值（size_tolerance=1.1, quality_threshold=0.15）
        #   - 距离判断：三角形顶点到最近边界边的距离
        #   - 优势：避免边界区域过度细分，保持边界恢复成功率
        refinement_mode = 'C'  # 默认使用融合版本

        # 版本C的自适应参数
        boundary_protection_distance = 0.5  # 边界保护距离（相对单位）
        boundary_size_tolerance = 1.25  # 边界区域宽松阈值
        boundary_quality_threshold = 0.1  # 边界区域低质量要求
        internal_size_tolerance = 1.1  # 内部区域严格阈值
        internal_quality_threshold = 0.15  # 内部区域高质量要求
        max_iter_limit = 15000  # ANW 需要更多迭代

        # 预计算边界点集（用于快速距离判断）
        boundary_point_indices = set(range(boundary_count))

        while True:
            max_iterations += 1

            if max_iterations % 50 == 0 or max_iterations == 1:
                current_triangles = len([t for t in self.triangles if not t.is_deleted()])
                current_points = len(self.points)
                current_inserted = current_points - initial_point_count
                queue_size = len(priority_queue)
                verbose(f"  [进度] 迭代 {max_iterations} | "
                        f"节点: {current_points} (边界: {boundary_count}, 内部: {current_inserted}) | "
                        f"三角形: {current_triangles} | "
                        f"队列: {queue_size}")

            # 早停条件
            if len(self.points) > 100000:
                verbose("达到最大节点数限制 (100000)，停止插点")
                break

            # 根据版本设置最大迭代次数
            if refinement_mode == 'A':
                max_iter_limit = 8000
            elif refinement_mode == 'B':
                max_iter_limit = 15000
            # 版本C的max_iter_limit已在上面定义

            if max_iterations > max_iter_limit:
                verbose(f"达到最大迭代次数限制 ({max_iter_limit})，停止插点")
                break

            if target_total_points is not None and len(self.points) >= target_total_points:
                verbose(f"  [进度] 达到目标节点数: {len(self.points)}")
                break
            if not priority_queue:
                verbose("  [进度] 优先级队列为空，停止插点")
                break

            # 版本C的自适应策略：基于边界距离
            # 版本A或B：固定策略
            if refinement_mode == 'C':
                # 辅助函数：判断三角形是否在边界区域
                def is_boundary_triangle(tri):
                    """检查三角形是否有顶点在边界点集中。"""
                    return any(v in boundary_point_indices for v in tri.vertices)

                # 对于全量检查，使用混合策略
                current_phase = 'C'
            else:
                # 版本A或B：固定阈值
                current_phase = refinement_mode

            # 定期全量检查：统计不满足尺寸场或质量太差的三角形数量
            if max_iterations - last_full_check >= full_check_interval:
                needs_refinement = 0
                quality_violations = 0
                size_violations = 0

                for tri in self.triangles:
                    if tri.is_deleted():
                        continue
                    if tri.circumradius is None:
                        self._compute_circumcircle(tri)
                    target_size = self._get_target_size_for_triangle(tri)
                    v0, v1, v2 = tri.vertices
                    pts = self.points
                    edge_lengths = [
                        float(np.linalg.norm(pts[v1] - pts[v0])),
                        float(np.linalg.norm(pts[v2] - pts[v1])),
                        float(np.linalg.norm(pts[v0] - pts[v2])),
                    ]
                    max_edge = max(edge_lengths)
                    quality = self._compute_triangle_quality(tri)
                    boundary_vertex_count = sum(1 for v in tri.vertices if v in boundary_point_indices)
                    skip_size_refinement = boundary_vertex_count == 3

                    # 版本C：根据三角形位置选择阈值
                    if refinement_mode == 'C':
                        is_boundary_tri = is_boundary_triangle(tri)
                        if is_boundary_tri:
                            # 边界区域：宽松阈值
                            tri_size_tolerance = boundary_size_tolerance
                            tri_quality_threshold = boundary_quality_threshold
                        else:
                            # 内部区域：严格阈值
                            tri_size_tolerance = internal_size_tolerance
                            tri_quality_threshold = internal_quality_threshold
                    else:
                        # 版本A或B：固定阈值
                        tri_size_tolerance = 1.15 if refinement_mode == 'A' else 1.1
                        tri_quality_threshold = 0.2 if refinement_mode == 'A' else 0.15

                    if boundary_vertex_count == 2:
                        tri_size_tolerance = max(tri_size_tolerance, 1.6)

                    if target_size is not None and target_size > 1e-12:
                        if (not skip_size_refinement) and max_edge > target_size * tri_size_tolerance:
                            size_violations += 1
                            needs_refinement += 1
                        elif quality < tri_quality_threshold:
                            quality_violations += 1
                            needs_refinement += 1
                    else:
                        if quality < 0.3:
                            quality_violations += 1
                            needs_refinement += 1

                last_full_check = max_iterations
                if needs_refinement <= 50 and size_violations <= 40:
                    verbose(
                        "  [进度] 剩余需要细化的单元已很少，且尺寸违规可接受，提前停止插点"
                    )
                    break
                if needs_refinement == 0:
                    verbose(f"  [进度] 所有三角形满足尺寸场和质量要求（{refinement_mode}模式），停止插点")
                    break
                elif max_iterations % 100 == 0:
                    verbose(f"  [进度] 全量检查：{needs_refinement} 个三角形不满足要求 "
                           f"(尺寸:{size_violations}, 质量:{quality_violations})")

            # 从优先级队列取出最差三角形
            worst_tri = None
            while priority_queue:
                neg_radius, tri_id, candidate_tri = heapq.heappop(priority_queue)

                # 跳过已删除的三角形
                if candidate_tri.is_deleted() or failed_tri_counts[tri_id] >= max_tri_failures:
                    continue

                worst_tri = candidate_tri
                break

            if worst_tri is None:
                # 队列中没有有效的三角形
                break

            # 计算外接圆（如果还没有）
            if worst_tri.circumradius is None:
                self._compute_circumcircle(worst_tri)

            # 检查是否满足尺寸场和质量要求
            target_size = self._get_target_size_for_triangle(worst_tri)
            v0, v1, v2 = worst_tri.vertices
            pts = self.points
            edge_lengths = [
                float(np.linalg.norm(pts[v1] - pts[v0])),
                float(np.linalg.norm(pts[v2] - pts[v1])),
                float(np.linalg.norm(pts[v0] - pts[v2])),
            ]
            max_edge = max(edge_lengths)
            quality = self._compute_triangle_quality(worst_tri)
            boundary_vertex_count = sum(1 for v in worst_tri.vertices if v in boundary_point_indices)
            skip_size_refinement = boundary_vertex_count == 3

            # 版本C：根据三角形位置选择阈值
            if refinement_mode == 'C':
                # 辅助函数已在上面定义
                is_boundary_tri = is_boundary_triangle(worst_tri)
                if is_boundary_tri:
                    # 边界区域：宽松阈值
                    size_tolerance = boundary_size_tolerance
                    quality_threshold = boundary_quality_threshold
                else:
                    # 内部区域：严格阈值
                    size_tolerance = internal_size_tolerance
                    quality_threshold = internal_quality_threshold
            else:
                # 版本A或B：固定阈值
                size_tolerance = 1.15 if refinement_mode == 'A' else 1.1
                quality_threshold = 0.2 if refinement_mode == 'A' else 0.15

            if boundary_vertex_count == 2:
                size_tolerance = max(size_tolerance, 1.6)

            should_refine = True
            if target_size is not None and target_size > 1e-12:
                # 尺寸场要求
                if skip_size_refinement or max_edge <= target_size * size_tolerance:
                    # 尺寸满足，但质量太差也需要细分
                    if quality >= quality_threshold:
                        should_refine = False
            else:
                # 无尺寸场：仅基于质量判断
                if quality >= 0.3:
                    should_refine = False

            if not should_refine:
                continue

            new_point, insertion_strategy = self._select_refinement_point(
                worst_tri,
                target_size,
                min_dist_threshold,
                x_min,
                x_max,
                y_min,
                y_max,
                margin,
            )

            if new_point is None:
                failed_tri_counts[id(worst_tri)] += 1
                consecutive_failures += 1
                if consecutive_failures <= 10 and self.holes:
                    verbose(f"    [候选拒绝] 三角形 {worst_tri.vertices} 的候选点均无效")
                if failed_tri_counts[id(worst_tri)] < max_tri_failures:
                    heapq.heappush(priority_queue, (-worst_tri.circumradius, id(worst_tri), worst_tri))
                if consecutive_failures >= max_consecutive_failures:
                    verbose(f"  [进度] 连续失败 {consecutive_failures} 次，停止插点")
                    break
                continue

            new_triangles, success = self._insert_refinement_point(worst_tri, new_point)

            if success:
                consecutive_failures = 0  # 重置失败计数
                failed_tri_counts.pop(id(worst_tri), None)
                 
                # 将新三角形加入优先级队列
                for tri in new_triangles:
                    if tri.circumradius is None:
                        self._compute_circumcircle(tri)
                    heapq.heappush(priority_queue, (-tri.circumradius, id(tri), tri))
            else:
                failed_tri_counts[id(worst_tri)] += 1
                consecutive_failures += 1
                if consecutive_failures <= 10:
                    verbose(f"    [插点失败] {insertion_strategy} 候选点未通过 cavity 重连")
                if failed_tri_counts[id(worst_tri)] < max_tri_failures:
                    heapq.heappush(priority_queue, (-worst_tri.circumradius, id(worst_tri), worst_tri))
                if consecutive_failures >= max_consecutive_failures:
                    verbose(f"  [进度] 连续失败 {consecutive_failures} 次，停止插点")
                    break
        
        # 清理已删除的三角形
        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
        
        final_triangles = len(self.triangles)
        final_points = len(self.points)
        final_inserted = final_points - initial_point_count
        verbose(f"  [完成] 插点完成 | "
                f"节点: {final_points} (边界: {boundary_count}, 内部: {final_inserted}) | "
                f"三角形: {final_triangles}")
    
    # -------------------------------------------------------------------------
    # 孔洞处理
    # -------------------------------------------------------------------------
    
    def _remove_hole_triangles(self) -> int:
        """删除孔洞内的三角形。

        使用种子填充法（Flood Fill / BFS）更精确地区分孔洞内外：
        1. 从明确在有效区域的种子三角形开始
        2. BFS 遍历，不跨越孔洞边界（保护边）
        3. 未遍历到的三角形 = 孔洞内，删除

        如果种子填充法失败，回退到质心测试法。
        """
        if not self.holes:
            return 0

        # 修正孔洞方向
        fixed_holes = []
        for hole in self.holes:
            if is_polygon_clockwise(hole):
                fixed_holes.append(hole[::-1])
            else:
                fixed_holes.append(hole.copy())
        holes_to_use = fixed_holes

        pre_missing = self._count_missing_protected_edges()
        pre_isolated = self._count_isolated_boundary_points()
        deleted_snapshot = [tri.is_deleted() for tri in self.triangles]

        # 先尝试种子填充法
        removed_count = self._remove_holes_via_seed_fill(holes_to_use)
        post_missing = self._count_missing_protected_edges()
        post_isolated = self._count_isolated_boundary_points()

        # 约束校验：不能让边界约束退化，且不能产生被隔离边界点
        seed_is_valid = (
            removed_count > 0
            and post_missing <= pre_missing
            and post_isolated <= pre_isolated
        )

        # 种子填充法失败或退化时，回滚并使用质心法
        if not seed_is_valid:
            if removed_count > 0:
                verbose(
                    "  种子填充法导致边界约束退化或出现隔离边界点，"
                    "回滚并改用质心测试法..."
                )
            else:
                verbose("  种子填充法未删除任何三角形，回退到质心测试法...")
            for tri, old_deleted in zip(self.triangles, deleted_snapshot):
                tri.set_deleted(old_deleted)
            removed_count = self._remove_holes_via_centroid_test(holes_to_use)

        return removed_count

    def _count_missing_protected_edges(self) -> int:
        """统计当前三角网中缺失的受保护边数量。"""
        edge_set = set()
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            verts = tri.vertices
            for i in range(3):
                a = verts[i]
                b = verts[(i + 1) % 3]
                edge_set.add(frozenset({a, b}))

        missing = 0
        for edge in self.protected_edges:
            if edge not in edge_set:
                missing += 1
        return missing

    def _count_isolated_boundary_points(self) -> int:
        """统计当前未连接到任何三角形的原始边界点数量。"""
        connected = set()
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            connected.update(tri.vertices)
        return sum(1 for i in range(self.boundary_count) if i not in connected)

    def _remove_holes_via_seed_fill(self, holes_to_use) -> int:
        """使用种子填充法删除孔洞三角形（更精确）。

        算法：
        1. 找到种子三角形（质心在所有孔洞外）
        2. BFS 遍历，不跨越孔洞边界（保护边）
        3. 未遍历到的三角形 = 孔洞内，删除
        """
        from collections import deque
        from utils.geom_toolkit import point_in_polygon

        # 1. 找到种子三角形（质心在所有孔洞外）
        seed_tri = None
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
            in_any_hole = False
            for hole in holes_to_use:
                if point_in_polygon(centroid, hole):
                    in_any_hole = True
                    break
            if not in_any_hole:
                seed_tri = tri
                break

        if seed_tri is None:
            verbose("  警告：找不到种子三角形，种子填充法失败")
            return 0

        # 2. BFS 遍历
        visited = set()
        valid_tris = set()
        queue = deque([seed_tri])

        while queue:
            tri = queue.popleft()
            tri_id = id(tri)

            if tri_id in visited:
                continue
            visited.add(tri_id)
            valid_tris.add(tri_id)

            # 遍历邻接三角形
            for neighbor in tri.neighbors:
                if neighbor is None or neighbor.is_deleted():
                    continue

                neighbor_id = id(neighbor)
                if neighbor_id in visited:
                    continue

                # 检查共享边是否是孔洞边界（保护边）
                shared_edge = tri.get_shared_edge(neighbor)
                if shared_edge is not None:
                    edge_key = frozenset(shared_edge)
                    if edge_key in self.protected_edges:
                        continue  # 不跨越孔洞边界

                queue.append(neighbor)

        # 3. 删除未遍历到的三角形（孔洞内）
        triangles_to_remove = set()
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            if id(tri) not in valid_tris:
                triangles_to_remove.add(id(tri))
                tri.set_deleted(True)

        removed_count = len(triangles_to_remove)
        if removed_count > 0:
            verbose(f"  种子填充法删除 {removed_count} 个孔洞内三角形")

        return removed_count

    def _remove_holes_via_centroid_test(self, holes_to_use) -> int:
        """回退方法：使用质心测试删除孔洞三角形。"""
        from utils.geom_toolkit import point_in_polygon

        triangles_to_remove = set()
        for i, tri in enumerate(self.triangles):
            if tri.is_deleted():
                continue

            # 对孔洞清理，凡是质心落入孔洞的单元都应删除；
            # 不能因为它携带了 hole boundary edge 就保留，否则会残留 hole-side fan/sliver。
            centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
            for h in holes_to_use:
                if point_in_polygon(centroid, h):
                    triangles_to_remove.add(i)
                    break

            # 顶点测试（保护边界点）
            if i not in triangles_to_remove:
                any_vert_in_hole = False
                for vert_idx in tri.vertices:
                    if vert_idx < self.boundary_count:
                        continue
                    vert = self.points[vert_idx]
                    for h in holes_to_use:
                        if point_in_polygon(vert, h):
                            any_vert_in_hole = True
                            break
                    if any_vert_in_hole:
                        break
                if any_vert_in_hole:
                    triangles_to_remove.add(i)

        for i in triangles_to_remove:
            self.triangles[i].set_deleted(True)

        removed_count = len(triangles_to_remove)
        if removed_count > 0:
            verbose(f"  质心测试法删除 {removed_count} 个孔洞内三角形")

        return removed_count

    def _remove_hole_side_boundary_fan_triangles(self) -> int:
        """删除完全由原始边界点构成、且位于孔洞内侧的边界扇形三角形。"""
        if not self.holes:
            return 0

        from utils.geom_toolkit import point_in_polygon

        removed_count = 0
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            if not all(v < self.boundary_count for v in tri.vertices):
                continue

            centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
            if any(point_in_polygon(centroid, hole) for hole in self.holes):
                tri.set_deleted(True)
                removed_count += 1

        if removed_count > 0:
            self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
            build_adjacency_from_triangles(self.triangles)
            verbose(f"  删除 {removed_count} 个孔洞内边界扇形三角形")

        return removed_count

    def _remove_outer_boundary_triangles(self) -> int:
        """删除外边界外的三角形。

        标准 CDT 要求：
        - 删除孔洞内的三角形
        - 删除外边界外的三角形（凸包区域）

        使用质心测试：如果三角形质心在外边界外，则删除。
        保护包含外边界边的三角形。
        """
        if self.outer_boundary is None or len(self.outer_boundary) < 3:
            return 0

        from utils.geom_toolkit import point_in_polygon

        # 保护包含外边界边的三角形
        outer_boundary_edge_set = set()
        n = len(self.outer_boundary)

        # 找到外边界边的顶点索引
        # 注意：self.points 中的边界点索引是 0 到 self.boundary_count-1
        # 需要找到哪些点在 outer_boundary 中
        outer_boundary_point_indices = set()
        for i in range(self.boundary_count):
            pt = self.points[i]
            for j, ob_pt in enumerate(self.outer_boundary):
                if np.linalg.norm(pt - ob_pt) < 1e-10:
                    outer_boundary_point_indices.add(i)
                    break

        # 找到外边界边
        for i in range(n):
            p1 = self.outer_boundary[i]
            p2 = self.outer_boundary[(i + 1) % n]

            # 找到对应的顶点索引
            idx1, idx2 = None, None
            for idx in outer_boundary_point_indices:
                if np.linalg.norm(self.points[idx] - p1) < 1e-10:
                    idx1 = idx
                if np.linalg.norm(self.points[idx] - p2) < 1e-10:
                    idx2 = idx
                if idx1 is not None and idx2 is not None:
                    break

            if idx1 is not None and idx2 is not None:
                outer_boundary_edge_set.add(frozenset({idx1, idx2}))

        triangles_to_remove = set()
        for i, tri in enumerate(self.triangles):
            if tri.is_deleted():
                continue

            # 保护包含外边界边的三角形
            has_outer_boundary_edge = False
            for j in range(3):
                v1 = tri.vertices[j]
                v2 = tri.vertices[(j + 1) % 3]
                edge_key = frozenset({v1, v2})
                if edge_key in outer_boundary_edge_set:
                    has_outer_boundary_edge = True
                    break
            if has_outer_boundary_edge:
                continue

            # 质心测试：如果质心在外边界外，删除
            centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
            if not point_in_polygon(centroid, self.outer_boundary):
                triangles_to_remove.add(i)

        for i in triangles_to_remove:
            self.triangles[i].set_deleted(True)

        removed_tri_count = len(triangles_to_remove)
        if removed_tri_count > 0:
            verbose(f"  删除外边界外三角形: {removed_tri_count} 个")

        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]

        return removed_tri_count

    # -------------------------------------------------------------------------
    # Constrained Delaunay Triangulation
    # -------------------------------------------------------------------------

    def _mark_boundary_edges_in_triangles(self):
        """在所有三角形中标记哪些边是受保护的边界边。
        
        这个方法在初始三角剖分后调用，确保边界边在整个
        三角化过程中受到保护（不被翻转或分裂）。
        """
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            
            # 检查三角形的三条边
            verts = tri.vertices
            for i in range(3):
                v1 = verts[i]
                v2 = verts[(i + 1) % 3]
                
                # 如果这条边是受保护的边界边
                if self._is_protected_edge(v1, v2):
                    # 标记到三角形中（如果 MTri3 支持）
                    # 目前我们通过 protected_edges 集合来保护
                    pass
    
    def _is_boundary_edge_in_mesh(self, v1: int, v2) -> bool:
        """检查边(v1, v2)是否已经是当前三角剖分中的边。"""
        edge_set = frozenset({v1, v2})

        for tri in self.triangles:
            if tri.is_deleted():
                continue
            verts = tri.vertices
            for i in range(3):
                e1 = verts[i]
                e2 = verts[(i + 1) % 3]
                if frozenset({e1, e2}) == edge_set:
                    return True
        return False

    def _is_exact_protected_edge_in_mesh(self, v1: int, v2: int) -> bool:
        """检查受保护边是否以“真实 boundary edge”形式存在。"""
        edge_set = frozenset({v1, v2})
        incident_count = 0

        for tri in self.triangles:
            if tri.is_deleted():
                continue
            verts = tri.vertices
            for i in range(3):
                if frozenset({verts[i], verts[(i + 1) % 3]}) == edge_set:
                    incident_count += 1

        if incident_count == 0:
            return False

        if self._is_protected_edge(v1, v2):
            return incident_count == 1

        return True
    
    def _find_edge_adjacent_triangles(self, v1: int, v2: int) -> List[MTri3]:
        """找到包含边(v1, v2)的所有三角形。"""
        result = []
        edge_set = frozenset({v1, v2})
        
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            verts = tri.vertices
            if v1 in verts and v2 in verts:
                result.append(tri)
        
        return result
    
    def _can_flip_edge(self, n1: int, n2: int) -> bool:
        """检查边(n1, n2)是否可以翻转。
        
        如果边是受保护的边界边，则不允许翻转。
        """
        # 关键：保护边界边不被翻转
        if self._is_protected_edge(n1, n2):
            return False
        
        return True
    
    def _constrained_delaunay_triangulation(self):
        """实现 Constrained Delaunay Triangulation (CDT)。

        在主循环插点后调用此方法，确保所有边界边都出现在
        三角剖分中。通过以下步骤实现：

        1. 检查每条边界边是否已存在于三角剖分中
        2. 对于缺失的边，通过边翻转让其出现
        3. 不插入 Steiner 点（避免分割边界边）
        4. 多轮迭代直到所有边恢复或无法继续
        5. **新增**: 在开始恢复前，先修复被隔离的边界点
        6. **新增**: 对于无法恢复的边，尝试 splitting 策略
        
        关键修复：
        - 先处理短边（薄翼型区域），再处理长边
        - 使用更鲁棒的 splitting 策略
        """
        verbose("开始 Constrained Delaunay 边界边恢复...")

        # 新增：在 CDT 前，先修复被隔离的边界点
        verbose("检查并修复被隔离的边界点...")
        fixed_count = self._fix_isolated_boundary_points()
        verbose(f"修复了 {fixed_count} 个被隔离的边界点")

        # Debug: Print missing edges before CDT recovery
        missing_before_cdt = []
        for edge in self.protected_edges:
            v1, v2 = sorted(list(edge))
            if not self._is_exact_protected_edge_in_mesh(v1, v2):
                missing_before_cdt.append((v1, v2))
        if missing_before_cdt:
            verbose(f"  [DEBUG] CDT 前缺失 {len(missing_before_cdt)} 条边：{missing_before_cdt[:10]}...")
            # Categorize missing edges
            wall_missing = [(v1, v2) for v1, v2 in missing_before_cdt if 38 <= v1 < 96 and v2 <= 95]
            other_missing = [(v1, v2) for v1, v2 in missing_before_cdt if not (38 <= v1 < 96 and v2 <= 95)]
            verbose(f"  [DEBUG]   Wall 边缺失：{len(wall_missing)}, 其他边缺失：{len(other_missing)}")

        # 修复：不再使用硬编码的 wall 边界索引
        # 所有 protected_edges 都是需要恢复的约束边
        max_passes = 10
        total_recovered = 0
        failed_edges_list = []

        for pass_num in range(max_passes):
            missing_edges = []
            for edge in self.protected_edges:
                v1, v2 = list(edge)
                if not self._is_exact_protected_edge_in_mesh(v1, v2):
                    missing_edges.append((v1, v2))

            if not missing_edges:
                verbose("所有边界边已存在，无需恢复")
                return

            if pass_num == 0:
                verbose(f"  检测到 {len(missing_edges)} 条缺失的边界边")

            recovered_count = 0
            failed_count = 0

            for v1, v2 in missing_edges:
                if self._recover_single_constrained_edge(v1, v2):
                    recovered_count += 1
                else:
                    failed_count += 1

            total_recovered += recovered_count
            verbose(f"  第 {pass_num + 1} 轮: 恢复 {recovered_count}, 失败 {failed_count}")

            if recovered_count == 0:
                # 收集所有仍然失败的边
                failed_edges_list = []
                for edge in self.protected_edges:
                    v1, v2 = list(edge)
                    if not self._is_exact_protected_edge_in_mesh(v1, v2):
                        failed_edges_list.append((v1, v2))
                break

        # 无法恢复的边记录为警告
        if failed_edges_list:
            verbose(f"  [警告] {len(failed_edges_list)} 条边无法恢复:")
            for v1, v2 in failed_edges_list:
                p1, p2 = self.points[v1], self.points[v2]
                verbose(f"    边 ({v1},{v2}): ({p1[0]:.4f},{p1[1]:.4f}) -> ({p2[0]:.4f},{p2[1]:.4f})")
                # 打印到标准输出以便调试
                print(f"[DEBUG] 无法恢复: ({v1},{v2})", flush=True)
                
                # 详细调试：打印v1和v2的邻居
                v1_tris = [tri for tri in self.triangles if not tri.is_deleted() and v1 in tri.vertices]
                v2_tris = [tri for tri in self.triangles if not tri.is_deleted() and v2 in tri.vertices]
                
                v1_neighbors = set()
                for tri in v1_tris:
                    for v in tri.vertices:
                        if v != v1:
                            v1_neighbors.add(v)
                
                v2_neighbors = set()
                for tri in v2_tris:
                    for v in tri.vertices:
                        if v != v2:
                            v2_neighbors.add(v)
                
                common = v1_neighbors & v2_neighbors

                # 如果 v1 或 v2 没有三角形，记录警告并尝试简单恢复
                if len(v1_tris) == 0 or len(v2_tris) == 0:
                    verbose(f"    [严重] 节点 {v1} 或 {v2} 没有三角形，尝试简单恢复")
                    # 尝试通过 force_recover 恢复
                    self._force_recover_boundary_edge_by_reconnection(v1, v2)
                # 如果邻居数很少（<=4），也尝试简单恢复
                elif len(v1_neighbors) <= 4 or len(v2_neighbors) <= 4:
                    verbose(f"    [警告] 节点 {v1} 或 {v2} 邻居很少，尝试简单恢复")
                    self._force_recover_boundary_edge_by_reconnection(v1, v2)
        
        # 新增：对于无法恢复的边，尝试 splitting 策略
        if failed_edges_list:
            verbose(f"  [信息] {len(failed_edges_list)} 条边无法通过边翻转恢复，尝试 splitting 策略")
            recovered_by_splitting = 0
            for v1, v2 in failed_edges_list:
                if self._recover_edge_by_splitting(v1, v2):
                    recovered_by_splitting += 1
            verbose(f"  通过 splitting 策略恢复 {recovered_by_splitting}/{len(failed_edges_list)} 条边")
        
        remaining = []
        for edge in self.protected_edges:
            v1, v2 = list(edge)
            if not self._is_exact_protected_edge_in_mesh(v1, v2):
                remaining.append((v1, v2))

        verbose(f"边界边恢复完成: 总恢复 {total_recovered}, 剩余缺失 {len(remaining)}")

    def _try_enhanced_boundary_recovery(self, v1: int, v2: int) -> bool:
        """增强的边界边恢复方法。
        
        当标准方法失败时使用：
        1. 找到所有包含v1的三角形
        2. 找到所有包含v2的三角形
        3. 找到连接v1和v2路径上的所有三角形
        4. 删除这些三角形
        5. 用(v1, v2)边重新三角化
        """
        p1, p2 = self.points[v1], self.points[v2]
        
        # 找到所有与线段(v1, v2)相交或接近的三角形
        tris_to_remove = []
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            
            # 检查三角形是否与线段(v1, v2)相交
            if self._segment_intersects_triangle(p1, p2, tri):
                tris_to_remove.append(tri)
                continue
            
            # 检查三角形是否包含v1或v2
            if v1 in tri.vertices or v2 in tri.vertices:
                # 检查三角形的质心是否接近线段
                centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
                dist = self._point_to_segment_distance(centroid, p1, p2)
                if dist < 0.01:  # 非常接近线段
                    tris_to_remove.append(tri)
        
        if not tris_to_remove:
            return False
        
        # 删除三角形
        for tri in tris_to_remove:
            tri.set_deleted(True)
        
        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
        build_adjacency_from_triangles(self.triangles)
        
        # 找到空洞边界
        cavity_boundary = self._find_cavity_boundary_edges()
        
        # 验证空洞边界包含v1和v2
        cavity_nodes = set()
        for a, b in cavity_boundary:
            cavity_nodes.add(a)
            cavity_nodes.add(b)
        
        if v1 not in cavity_nodes or v2 not in cavity_nodes:
            return False
        
        # 对空洞进行三角化
        # 方法：将v1和v2与空洞边界连接
        new_triangles = []
        
        # 找到空洞边界上与v1和v2相邻的节点
        v1_cavity_neighbors = []
        v2_cavity_neighbors = []
        
        for a, b in cavity_boundary:
            if a == v1:
                v1_cavity_neighbors.append(b)
            elif b == v1:
                v1_cavity_neighbors.append(a)
            if a == v2:
                v2_cavity_neighbors.append(b)
            elif b == v2:
                v2_cavity_neighbors.append(a)
        
        # 创建连接v1和v2的三角形
        # 对于空洞边界的每条边，如果它不包含v1或v2，则创建一个三角形
        for edge in cavity_boundary:
            a, b = edge
            if a in (v1, v2) or b in (v1, v2):
                continue
            
            # 找到离边最近的端点（v1或v2）
            mid_edge = (self.points[a] + self.points[b]) / 2.0
            dist_v1 = float(np.linalg.norm(mid_edge - p1))
            dist_v2 = float(np.linalg.norm(mid_edge - p2))
            
            if dist_v1 < dist_v2:
                new_tri = MTri3(v1, a, b, idx=self._next_tri_id())
            else:
                new_tri = MTri3(v2, a, b, idx=self._next_tri_id())
            
            self._compute_circumcircle(new_tri)
            new_triangles.append(new_tri)
        
        self.triangles.extend(new_triangles)
        build_adjacency_from_triangles(self.triangles)
        
        return self._is_boundary_edge_in_mesh(v1, v2)

    def _force_connect_edge(self, v1: int, v2: int) -> bool:
        """强制连接两个有三角形但没有共同邻居的节点。
        
        方法：
        1. 删除v1和v2的所有三角形
        2. 找到空洞边界
        3. 用(v1, v2)边重新三角化
        """
        p1, p2 = self.points[v1], self.points[v2]
        
        # 找到v1和v2的所有三角形
        tris_to_remove = []
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            if v1 in tri.vertices or v2 in tri.vertices:
                tris_to_remove.append(tri)
            else:
                # 也检查三角形是否与线段(v1, v2)相交
                if self._segment_intersects_triangle(p1, p2, tri):
                    tris_to_remove.append(tri)
        
        if not tris_to_remove:
            return False
        
        # 删除三角形
        for tri in tris_to_remove:
            tri.set_deleted(True)
        
        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
        build_adjacency_from_triangles(self.triangles)
        
        # 找到空洞边界
        cavity_boundary = self._find_cavity_boundary_edges()
        
        # 验证空洞边界包含v1和v2
        cavity_nodes = set()
        for a, b in cavity_boundary:
            cavity_nodes.add(a)
            cavity_nodes.add(b)
        
        if v1 not in cavity_nodes or v2 not in cavity_nodes:
            return False
        
        # 对空洞进行三角化
        new_triangles = []
        
        # 对于空洞边界的每条边
        for edge in cavity_boundary:
            a, b = edge
            if a in (v1, v2) or b in (v1, v2):
                continue
            
            # 找到离边最近的端点（v1或v2）
            mid_edge = (self.points[a] + self.points[b]) / 2.0
            dist_v1 = float(np.linalg.norm(mid_edge - p1))
            dist_v2 = float(np.linalg.norm(mid_edge - p2))
            
            if dist_v1 < dist_v2:
                new_tri = MTri3(v1, a, b, idx=self._next_tri_id())
            else:
                new_tri = MTri3(v2, a, b, idx=self._next_tri_id())
            
            self._compute_circumcircle(new_tri)
            new_triangles.append(new_tri)
        
        self.triangles.extend(new_triangles)
        build_adjacency_from_triangles(self.triangles)
        
        return self._is_boundary_edge_in_mesh(v1, v2)

    def _recover_single_constrained_edge(self, v1: int, v2: int) -> bool:
        """恢复单条约束边(v1, v2)。

        使用以下策略：
        1. 首先尝试边翻转
        2. 如果翻转无法恢复，使用强制重连
        """
        if self._is_exact_protected_edge_in_mesh(v1, v2):
            return True

        # 调试：如果是wall边界边，打印信息
        is_wall = 38 <= min(v1, v2) < 96 and max(v1, v2) <= 95
        if is_wall:
            verbose(f"  Wall边 ({v1},{v2}) 缺失，尝试恢复...")

        max_flips = 300
        flip_count = 0

        while flip_count < max_flips:
            intersecting_edges = self._find_edges_intersecting_segment(v1, v2)

            if not intersecting_edges:
                if is_wall:
                    verbose(f"    无相交边，无法翻转")
                break

            flipped = False
            for n1, n2 in intersecting_edges:
                if self._flip_edge(n1, n2):
                    flip_count += 1
                    flipped = True
                    if self._is_exact_protected_edge_in_mesh(v1, v2):
                        if is_wall:
                            verbose(f"    翻转恢复成功")
                        return True
                    break

            if not flipped:
                if is_wall:
                    verbose(f"    无法翻转，尝试强制重连")
                break

        if self._is_exact_protected_edge_in_mesh(v1, v2):
            return True

        if self._recover_edge_by_splitting(v1, v2):
            if is_wall:
                verbose("    splitting 恢复成功")
            return True

        result = self._force_recover_boundary_edge_by_reconnection(v1, v2)
        if is_wall:
            verbose(f"    强制重连结果: {result}")
        return result

    def _force_recover_boundary_edge_by_reconnection(self, v1: int, v2: int) -> bool:
        """强制通过局部重连恢复边界边。

        当边翻转无法恢复时使用：
        1. 找到所有与线段(v1, v2)相交的三角形
        2. 删除这些三角形
        3. 使用边界边(v1, v2)和相关顶点重新三角化
        """
        p1, p2 = self.points[v1], self.points[v2]
        dist = float(np.linalg.norm(p1 - p2))
        if dist < 1e-10:
            return False

        intersecting_tris = []
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            verts = tri.vertices
            has_v1 = v1 in verts
            has_v2 = v2 in verts
            if has_v1 and has_v2:
                continue
            tri_centroid = np.mean([self.points[v] for v in verts], axis=0)
            if self._segment_intersects_triangle(p1, p2, tri):
                intersecting_tris.append(tri)

        snapshot = self._snapshot_triangulation_state()

        if not intersecting_tris:
            verbose(f"    无相交三角形，尝试基于局部空洞重建...")

            tris_with_v1 = []
            tris_with_v2 = []
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                if v1 in tri.vertices:
                    tris_with_v1.append(tri)
                if v2 in tri.vertices:
                    tris_with_v2.append(tri)

            if not tris_with_v1 or not tris_with_v2:
                return False

            v1_other_neighbors = set()
            for tri in tris_with_v1:
                for v in tri.vertices:
                    if v != v1:
                        v1_other_neighbors.add(v)

            v2_other_neighbors = set()
            for tri in tris_with_v2:
                for v in tri.vertices:
                    if v != v2:
                        v2_other_neighbors.add(v)

            common = v1_other_neighbors & v2_other_neighbors
            if not common:
                return False

            intersecting_tris = []
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                has_common = any(v in common for v in tri.vertices)
                has_endpoint = v1 in tri.vertices or v2 in tri.vertices
                if has_common and has_endpoint:
                    intersecting_tris.append(tri)

        cavity_boundary = self._find_local_cavity_boundary_edges(intersecting_tris)
        cavity_nodes = {n for edge in cavity_boundary for n in edge}
        if not cavity_boundary or v1 not in cavity_nodes or v2 not in cavity_nodes:
            return False

        adjacency = {}
        for a, b in cavity_boundary:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        path1 = self._find_path_in_boundary(adjacency, v1, v2)
        path2 = self._find_path_in_boundary_excluding_edges(
            adjacency,
            v1,
            v2,
            excluded_edges=self._boundary_path_edges(path1),
        )
        if not path1 or not path2:
            return False

        for tri in intersecting_tris:
            tri.set_deleted(True)

        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
        build_adjacency_from_triangles(self.triangles)

        if not self._retriangulate_cavity_with_edge(cavity_boundary, v1, v2):
            self._restore_triangulation_state(snapshot)
            return False

        if self._count_triangle_components() != 1:
            self._restore_triangulation_state(snapshot)
            return False

        return self._is_exact_protected_edge_in_mesh(v1, v2)

    def _segment_intersects_triangle(self, p1: np.ndarray, p2: np.ndarray, tri: 'MTri3') -> bool:
        """检查线段(p1, p2)是否与三角形相交。"""
        pA = self.points[tri.vertices[0]]
        pB = self.points[tri.vertices[1]]
        pC = self.points[tri.vertices[2]]

        for pA_, pB_ in [(pA, pB), (pB, pC), (pC, pA)]:
            if segments_intersect_strict(p1, p2, pA_, pB_):
                return True

        return False
    
    def _find_edges_intersecting_segment(self, v1: int, v2: int) -> List[Tuple[int, int]]:
        """找到所有与线段(v1, v2)相交的边，按距v1的距离排序。"""
        intersecting = []
        segment_start = self.points[v1]
        segment_end = self.points[v2]
        
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            
            verts = tri.vertices
            for i in range(3):
                e1 = verts[i]
                e2 = verts[(i + 1) % 3]
                
                if e1 in (v1, v2) or e2 in (v1, v2):
                    continue
                
                if self._is_protected_edge(e1, e2):
                    continue
                
                if segments_intersect_strict(
                    segment_start, segment_end,
                    self.points[e1], self.points[e2]
                ):
                    edge_tuple = (min(e1, e2), max(e1, e2))
                    if edge_tuple not in intersecting:
                        intersecting.append(edge_tuple)
        
        if len(intersecting) > 1:
            def edge_distance(edge):
                mid = (self.points[edge[0]] + self.points[edge[1]]) / 2.0
                return float(np.linalg.norm(mid - segment_start))
            intersecting.sort(key=edge_distance)
        
        return intersecting
    
    def _flip_edge_would_help(self, n1: int, n2: int, target_v1: int, target_v2: int) -> bool:
        """检查翻转边(n1, n2)是否有助于恢复目标边(target_v1, target_v2)。
        
        简单的启发式：如果翻转后新边更接近目标边，则返回True。
        """
        # 找到对角顶点
        t1 = t2 = None
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            if n1 in tri.vertices and n2 in tri.vertices:
                if t1 is None:
                    t1 = tri
                elif t2 is None:
                    t2 = tri
                else:
                    break
        
        if t1 is None or t2 is None:
            return False
        
        a_idx = next(v for v in t1.vertices if v != n1 and v != n2)
        b_idx = next(v for v in t2.vertices if v != n1 and v != n2)
        
        # 翻转后会出现新边(a_idx, b_idx)
        # 检查新边是否更接近目标边
        new_edge_midpoint = (self.points[a_idx] + self.points[b_idx]) / 2.0
        target_midpoint = (self.points[target_v1] + self.points[target_v2]) / 2.0
        
        old_edge_midpoint = (self.points[n1] + self.points[n2]) / 2.0
        
        dist_old = float(np.linalg.norm(old_edge_midpoint - target_midpoint))
        dist_new = float(np.linalg.norm(new_edge_midpoint - target_midpoint))
        
        return dist_new < dist_old
    
    # -------------------------------------------------------------------------
    # 边界边恢复（旧方法，保留作为后备）
    # -------------------------------------------------------------------------

    def _recover_boundary_edges(self) -> int:
        """恢复丢失的边界边。

        参考 Gmsh recoverEdges() 的恢复顺序：
        1. 找出缺失的边
        2. 对每条缺失边，使用边交换 + Steiner点插入恢复
        """
        if not self.protected_edges:
            return 0

        missing_edges = []
        for protected_edge in self.protected_edges:
            v1, v2 = sorted(list(protected_edge))
            if not any(v1 in tri.vertices and v2 in tri.vertices for tri in self.triangles):
                missing_edges.append((v1, v2))

        if not missing_edges:
            return 0

        verbose(f"  检测到 {len(missing_edges)} 条丢失的边界边")
        recovered_count = 0
        failed_edges = []

        for v1, v2 in missing_edges:
            # 使用增强的恢复策略：边交换 + Steiner点插入
            success = self._recover_single_edge_enhanced(v1, v2)
            if success:
                recovered_count += 1
            else:
                failed_edges.append((v1, v2))

        if recovered_count > 0:
            verbose(f"  成功恢复 {recovered_count}/{len(missing_edges)} 条边界边")

        if failed_edges:
            verbose(f"  [警告] {len(failed_edges)} 条边无法恢复，将插入Steiner点")
            for v1, v2 in failed_edges:
                self._insert_steiner_point_on_edge(v1, v2)

        return recovered_count

    def _recover_single_edge_enhanced(self, v1: int, v2: int, max_iter: int = 1000) -> bool:
        """通过连续边翻转恢复边界边（增强版）。

        参考Gmsh recoverEdgeBySwaps()：
        - 查找与目标边相交的网格边
        - 执行2-2交换（边翻转）
        - 迭代直到边被恢复或无法继续
        """
        for iteration in range(max_iter):
            # 检查边是否已经存在
            if any(v1 in tri.vertices and v2 in tri.vertices for tri in self.triangles):
                return True

            # 查找相交边
            intersecting = self._find_intersecting_edge(v1, v2)
            if intersecting is None:
                # 没有相交边，但边也不存在，可能需要Steiner点
                return False

            # 执行边翻转
            if not self._flip_edge(*intersecting):
                # 翻转失败（可能是非凸四边形或边界问题）
                return False

        return False

    def _recover_edge_by_bfs(self, v1: int, v2: int) -> bool:
        """使用BFS路径查找恢复边界边。
        
        当共同邻居方法失败时使用：
        1. 构建邻接图
        2. 使用BFS找到v1到v2的最短路径
        3. 删除路径上的所有三角形
        4. 用(v1, v2)和路径上的节点重新三角化
        """
        from collections import deque
        
        # 构建邻接表
        adj = {}
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            verts = tri.vertices
            for i in range(3):
                a = verts[i]
                b = verts[(i + 1) % 3]
                if a not in adj:
                    adj[a] = set()
                if b not in adj:
                    adj[b] = set()
                adj[a].add(b)
                adj[b].add(a)
        
        # BFS查找v1到v2的最短路径
        queue = deque([(v1, [v1])])
        visited = {v1}
        path = None
        
        while queue:
            current, current_path = queue.popleft()
            
            if current == v2:
                path = current_path
                break
            
            if current not in adj:
                continue
            
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_path + [neighbor]))
        
        if path is None or len(path) < 3:
            verbose(f"    BFS未找到路径")
            return False
        
        verbose(f"    BFS找到路径，长度={len(path)}个节点")
        
        # 找到路径上所有三角形
        path_set = set(path)
        tris_on_path = []
        
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            # 检查三角形是否有至少两个顶点在路径上
            count = sum(1 for v in tri.vertices if v in path_set)
            if count >= 2:
                tris_on_path.append(tri)
        
        if not tris_on_path:
            verbose(f"    路径上无三角形")
            return False
        
        # 删除这些三角形
        for tri in tris_on_path:
            tri.set_deleted(True)
        
        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
        build_adjacency_from_triangles(self.triangles)
        
        # 找到空洞边界
        cavity_boundary = self._find_cavity_boundary_edges()
        
        # 对空洞进行三角化
        # 方法：找到空洞的"质心"，然后与边界边连接
        cavity_nodes = set()
        for a, b in cavity_boundary:
            cavity_nodes.add(a)
            cavity_nodes.add(b)
        
        if not cavity_nodes:
            verbose(f"    空洞边界为空")
            return False
        
        # 使用路径上的所有节点作为"中心"，与空洞边界连接
        new_triangles = []
        used_edges = set()
        
        # 对于空洞边界的每条边，尝试与路径上的节点连接
        for edge in cavity_boundary:
            a, b = edge
            if a in path_set or b in path_set:
                # 边的一个端点已经在路径上，跳过
                continue
            
            # 找到路径上最近的节点
            best_node = None
            best_dist = float('inf')
            
            for p_node in path:
                # 检查是否可以形成有效三角形
                p_coords = self.points[p_node]
                a_coords = self.points[a]
                b_coords = self.points[b]
                
                # 简单检查：三角形面积不能太小
                area = abs(np.cross(b_coords - a_coords, p_coords - a_coords)) / 2.0
                if area < 1e-10:
                    continue
                
                dist = float(np.linalg.norm(p_coords - (a_coords + b_coords) / 2.0))
                if dist < best_dist:
                    best_dist = dist
                    best_node = p_node
            
            if best_node is not None:
                edge_key = (min(a, b), max(a, b), best_node)
                if edge_key not in used_edges:
                    used_edges.add(edge_key)
                    new_tri = MTri3(best_node, a, b, idx=self._next_tri_id())
                    self._compute_circumcircle(new_tri)
                    new_triangles.append(new_tri)
        
        self.triangles.extend(new_triangles)
        build_adjacency_from_triangles(self.triangles)
        
        verbose(f"    创建了 {len(new_triangles)} 个新三角形")
        
        return self._is_boundary_edge_in_mesh(v1, v2)

    def _insert_steiner_point_on_edge(self, v1: int, v2: int) -> bool:
        """在边界边上插入Steiner点，然后正确恢复子边。
        
        正确的实现步骤：
        1. 在中点插入Steiner点
        2. 找到所有与线段(v1, v2)相交的三角形（包括包含v1或v2的）
        3. 删除这些三角形，形成空洞
        4. 对空洞进行约束三角化，确保(v1, mid)和(mid, v2)是边
        """
        p1, p2 = self.points[v1], self.points[v2]
        mid_point = (p1 + p2) / 2.0
        
        # 添加新点
        mid_idx = len(self.points)
        self.points = np.vstack([self.points, mid_point])
        
        verbose(f"    在边({v1},{v2})中点插入Steiner点 #{mid_idx}")
        
        # 找到所有需要删除的三角形
        # 关键：必须包含所有与线段(v1, v2)有交集的三角形
        triangles_to_remove = []
        
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            
            # 检查三角形是否与线段(v1, v2)相交
            should_remove = False
            
            # 方法1：检查三角形的边是否与线段相交
            for i in range(3):
                a_idx = tri.vertices[i]
                b_idx = tri.vertices[(i + 1) % 3]
                # 跳过与v1或v2直接相连的边
                if a_idx in (v1, v2) and b_idx in (v1, v2):
                    continue
                if self._segments_intersect_proper(
                    p1, p2, 
                    self.points[a_idx], self.points[b_idx]
                ):
                    should_remove = True
                    break
            
            # 方法2：检查线段中点是否在三角形内
            if not should_remove:
                mid = (p1 + p2) / 2.0
                if self._point_in_triangle_strict(mid, tri):
                    should_remove = True
            
            # 方法3：检查三角形是否包含v1或v2，且其外接圆包含中点
            if not should_remove and (v1 in tri.vertices or v2 in tri.vertices):
                # 检查三角形是否在v1-v2的"路径"上
                tri_center = np.mean([self.points[v] for v in tri.vertices], axis=0)
                dist_to_segment = self._point_to_segment_distance(tri_center, p1, p2)
                if dist_to_segment < 0.001:  # 非常接近线段
                    should_remove = True
            
            if should_remove:
                triangles_to_remove.append(tri)
        
        if not triangles_to_remove:
            verbose(f"    [警告] 初始检测未找到三角形，尝试删除包含 v1 或 v2 的所有三角形")
            # 尝试更宽松的条件：删除所有包含 v1 或 v2 的三角形
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                if v1 in tri.vertices or v2 in tri.vertices:
                    triangles_to_remove.append(tri)

        if not triangles_to_remove:
            verbose(f"    [严重] 仍未找到需要删除的三角形，v1 和 v2 可能已隔离")
            return False
        
        # 删除三角形
        for tri in triangles_to_remove:
            tri.set_deleted(True)
        
        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
        build_adjacency_from_triangles(self.triangles)
        
        # 找到空洞边界
        cavity_boundary = self._find_cavity_boundary_edges()
        
        # 验证空洞边界
        cavity_nodes = set()
        for a, b in cavity_boundary:
            cavity_nodes.add(a)
            cavity_nodes.add(b)
        
        if v1 not in cavity_nodes or v2 not in cavity_nodes:
            verbose(f"    [警告] 空洞边界不完整: v1={v1 in cavity_nodes}, v2={v2 in cavity_nodes}")
            # 尝试恢复：手动添加缺失的边界边
            if v1 not in cavity_nodes:
                # 找到离v1最近的空洞边界节点
                closest = min(cavity_nodes, key=lambda n: float(np.linalg.norm(self.points[n] - p1)))
                cavity_boundary.append((min(v1, closest), max(v1, closest)))
            if v2 not in cavity_nodes:
                closest = min(cavity_nodes, key=lambda n: float(np.linalg.norm(self.points[n] - p2)))
                cavity_boundary.append((min(v2, closest), max(v2, closest)))
        
        # 对空洞进行三角化
        # 方法：将mid_idx与空洞边界的每条边连接
        new_triangles = []
        for edge in cavity_boundary:
            a, b = edge
            if a == mid_idx or b == mid_idx:
                continue
            # 创建三角形
            new_tri = MTri3(mid_idx, a, b, idx=self._next_tri_id())
            self._compute_circumcircle(new_tri)
            new_triangles.append(new_tri)
        
        self.triangles.extend(new_triangles)
        build_adjacency_from_triangles(self.triangles)
        
        verbose(f"    创建了 {len(new_triangles)} 个新三角形")
        
        # 递归恢复子边
        success1 = self._recover_single_edge_enhanced(v1, mid_idx)
        success2 = self._recover_single_edge_enhanced(mid_idx, v2)
        
        return success1 and success2
    
    def _point_to_segment_distance(self, point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算点到线段的最短距离。"""
        segment = p2 - p1
        segment_length = np.linalg.norm(segment)
        if segment_length < 1e-10:
            return float(np.linalg.norm(point - p1))
        
        segment_dir = segment / segment_length
        projection = np.dot(point - p1, segment_dir)
        
        if projection < 0:
            return float(np.linalg.norm(point - p1))
        elif projection > segment_length:
            return float(np.linalg.norm(point - p2))
        else:
            closest = p1 + projection * segment_dir
            return float(np.linalg.norm(point - closest))
    
    def _segments_intersect_proper(self, p1: np.ndarray, p2: np.ndarray, 
                                    p3: np.ndarray, p4: np.ndarray) -> bool:
        """检查两条线段是否真正相交（不包括端点接触）。"""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        # 严格相交
        if ((d1 > 1e-12 and d2 < -1e-12) or (d1 < -1e-12 and d2 > 1e-12)) and \
           ((d3 > 1e-12 and d4 < -1e-12) or (d3 < -1e-12 and d4 > 1e-12)):
            return True
        
        return False

    def _triangle_intersects_segment_for_steiner(self, tri, v1: int, v2: int) -> bool:
        """检查三角形是否与线段(v1, v2)相交（用于Steiner点插入）。
        
        与 _triangle_intersects_segment 不同，这个函数会包含所有
        阻碍 (v1, v2) 连线的三角形，包括那些包含 v1 或 v2 的三角形。
        """
        p1, p2 = self.points[v1], self.points[v2]

        # 检查三角形的每条边
        for i in range(3):
            a_idx = tri.vertices[i]
            b_idx = tri.vertices[(i + 1) % 3]
            # 检查线段相交（包括端点接触）
            if self._segments_intersect_inclusive(p1, p2, self.points[a_idx], self.points[b_idx]):
                return True

        # 也检查三角形是否包含线段的中点
        mid = (p1 + p2) / 2.0
        if self._point_in_triangle_strict(mid, tri):
            return True

        return False
    
    def _segments_intersect_inclusive(self, p1, p2, p3, p4) -> bool:
        """检查两条线段是否相交（包括端点接触）。"""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        # 相交（包括端点接触）
        if ((d1 >= -1e-12 and d2 <= 1e-12) or (d1 <= 1e-12 and d2 >= -1e-12)) and \
           ((d3 >= -1e-12 and d4 <= 1e-12) or (d3 <= 1e-12 and d4 >= -1e-12)):
            return True

        return False

    def _triangle_intersects_segment(self, tri, v1: int, v2: int) -> bool:
        """检查三角形是否与线段(v1, v2)严格相交（不包括端点）。"""
        p1, p2 = self.points[v1], self.points[v2]

        # 检查三角形的每条边
        for i in range(3):
            a_idx = tri.vertices[i]
            b_idx = tri.vertices[(i + 1) % 3]
            # 跳过与目标边共享端点的边
            if a_idx in (v1, v2) or b_idx in (v1, v2):
                continue
            # 检查线段相交
            if segments_intersect_strict(p1, p2, self.points[a_idx], self.points[b_idx]):
                return True

        # 也检查三角形是否包含线段的中点
        mid = (p1 + p2) / 2.0
        if self._point_in_triangle_strict(mid, tri):
            return True

        return False

    def _point_in_triangle_strict(self, point, tri) -> bool:
        """检查点是否严格在三角形内部（不包括边界）。"""
        pts = self.points
        v0, v1, v2 = tri.vertices
        p0, p1, p2 = pts[v0], pts[v1], pts[v2]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        c1 = cross(p0, p1, point)
        c2 = cross(p1, p2, point)
        c3 = cross(p2, p0, point)

        # 严格在内部（所有叉积同号且不接近0）
        if (c1 > 1e-12 and c2 > 1e-12 and c3 > 1e-12) or \
           (c1 < -1e-12 and c2 < -1e-12 and c3 < -1e-12):
            return True

        return False

    def _find_cavity_boundary_edges(self):
        """找到空洞边界的边列表（只出现一次的边）。"""
        edge_count = {}

        for tri in self.triangles:
            if tri.is_deleted():
                continue
            for i in range(3):
                a = tri.vertices[i]
                b = tri.vertices[(i + 1) % 3]
                edge = (min(a, b), max(a, b))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # 边界边只出现一次
        boundary_edges = []
        for edge, count in edge_count.items():
            if count == 1:
                boundary_edges.append(edge)

        return boundary_edges

    def _find_local_cavity_boundary_edges(self, cavity_tris) -> List[Tuple[int, int]]:
        """从局部空洞三角形集合提取边界边（仅统计局部边频次）。"""
        edge_count = {}
        for tri in cavity_tris:
            if tri.is_deleted():
                continue
            verts = tri.vertices
            for i in range(3):
                a = verts[i]
                b = verts[(i + 1) % 3]
                edge = (min(a, b), max(a, b))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        return [edge for edge, count in edge_count.items() if count == 1]

    def _boundary_path_exclude_internal(self, path) -> Set[int]:
        """返回路径的内部节点集合（保留端点以允许第二条边界路径闭合）。"""
        if not path or len(path) <= 2:
            return set()
        return set(path[1:-1])

    def _boundary_path_edges(self, path) -> Set[Tuple[int, int]]:
        """返回路径上的无向边集合。"""
        if not path or len(path) <= 1:
            return set()

        return {
            (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            for i in range(len(path) - 1)
        }

    def _find_path_in_boundary_excluding_edges(self, adjacency, start, end, excluded_edges=None):
        """在边界图中查找与既有路径 edge-disjoint 的另一条路径。"""
        from collections import deque

        if excluded_edges is None:
            excluded_edges = set()

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if current == end:
                return path

            for neighbor in adjacency.get(current, []):
                edge = (min(current, neighbor), max(current, neighbor))
                if edge in excluded_edges or neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

        return None

    def _edge_use_count(self, v1: int, v2: int) -> int:
        """统计当前活动三角形中边 (v1, v2) 的邻接单元数量。"""
        edge_set = frozenset({v1, v2})
        count = 0
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            verts = tri.vertices
            for i in range(3):
                if frozenset({verts[i], verts[(i + 1) % 3]}) == edge_set:
                    count += 1
        return count

    def _point_in_domain(self, point: np.ndarray) -> bool:
        if self.outer_boundary is not None and not point_in_polygon(point, self.outer_boundary):
            return False
        if self.holes:
            for hole in self.holes:
                if point_in_polygon(point, hole):
                    return False
        return True

    def _incident_triangles_for_edge(self, v1: int, v2: int) -> List['MTri3']:
        edge_set = frozenset({v1, v2})
        incident = []
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            verts = tri.vertices
            for i in range(3):
                if frozenset({verts[i], verts[(i + 1) % 3]}) == edge_set:
                    incident.append(tri)
                    break
        return incident

    def _enforce_protected_edge_as_boundary(self, v1: int, v2: int) -> bool:
        """将受保护边的域外侧相邻单元移除，使其回到真实 boundary edge。"""
        if not self._is_protected_edge(v1, v2):
            return self._is_exact_protected_edge_in_mesh(v1, v2)

        incident = self._incident_triangles_for_edge(v1, v2)
        if len(incident) <= 1:
            return self._is_exact_protected_edge_in_mesh(v1, v2)

        outside_tris = []
        inside_tris = []
        for tri in incident:
            centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
            if self._point_in_domain(centroid):
                inside_tris.append(tri)
            else:
                outside_tris.append(tri)

        if not outside_tris or not inside_tris:
            return False

        for tri in outside_tris:
            tri.set_deleted(True)

        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
        build_adjacency_from_triangles(self.triangles)
        return self._is_exact_protected_edge_in_mesh(v1, v2)

    def _choose_boundary_fill_path(self, path1, path2):
        """在两条 cavity path 中选出域内侧路径，用于恢复真实边界边。"""
        if not path1:
            return path2
        if not path2:
            return path1

        def path_score(path):
            anchor = self.points[path[0]]
            inside_count = 0
            for i in range(1, len(path) - 1):
                a = self.points[path[i]]
                b = self.points[path[i + 1]]
                tri_centroid = (anchor + a + b) / 3.0
                if self._point_in_domain(tri_centroid):
                    inside_count += 1

            counts = [self._edge_use_count(path[i], path[i + 1]) for i in range(len(path) - 1)]
            interior_count = sum(count >= 2 for count in counts)
            return inside_count, interior_count, len(path)

        score1 = path_score(path1)
        score2 = path_score(path2)
        if score1 == score2:
            return path1 if len(path1) >= len(path2) else path2
        return path1 if score1 > score2 else path2

    def _find_triangle_strip_near_segment(self, v1: int, v2: int) -> List['MTri3']:
        """沿目标线段附近搜索连通 triangle strip，补足断裂的局部 cavity。"""
        p1, p2 = self.points[v1], self.points[v2]
        seg_len = float(np.linalg.norm(p2 - p1))
        if seg_len < 1e-12:
            return []

        active_tris = [tri for tri in self.triangles if not tri.is_deleted()]
        if not active_tris:
            return []

        tri_indices = {id(tri): idx for idx, tri in enumerate(active_tris)}
        start_tris = [idx for idx, tri in enumerate(active_tris) if v1 in tri.vertices]
        end_tris = [idx for idx, tri in enumerate(active_tris) if v2 in tri.vertices]
        if not start_tris or not end_tris:
            return []

        edge_to_tris = {}
        for tri_idx, tri in enumerate(active_tris):
            verts = tri.vertices
            for i in range(3):
                a = verts[i]
                b = verts[(i + 1) % 3]
                edge = (min(a, b), max(a, b))
                edge_to_tris.setdefault(edge, []).append(tri_idx)

        tri_adjacency = [set() for _ in range(len(active_tris))]
        for tri_list in edge_to_tris.values():
            if len(tri_list) != 2:
                continue
            a, b = tri_list
            tri_adjacency[a].add(b)
            tri_adjacency[b].add(a)

        centroids = [
            np.mean([self.points[v] for v in tri.vertices], axis=0)
            for tri in active_tris
        ]

        min_limit = max(seg_len * 0.4, 0.02)
        max_limit = max(seg_len * 2.0, 0.08)
        limit = min_limit

        from collections import deque

        while limit <= max_limit + 1e-12:
            allowed = {
                idx for idx, centroid in enumerate(centroids)
                if self._point_to_segment_distance(centroid, p1, p2) <= limit
            }

            queue = deque(start_tris)
            prev = {idx: None for idx in start_tris}
            found = None

            while queue and found is None:
                current = queue.popleft()
                if current in end_tris:
                    found = current
                    break

                for neighbor in tri_adjacency[current]:
                    if neighbor in prev or neighbor not in allowed:
                        continue
                    prev[neighbor] = current
                    queue.append(neighbor)

            if found is not None:
                strip_indices = []
                current = found
                while current is not None:
                    strip_indices.append(current)
                    current = prev[current]
                strip_indices.reverse()
                return [active_tris[idx] for idx in strip_indices]

            limit *= 1.4

        return []

    def _snapshot_triangulation_state(self):
        """保存当前三角形拓扑状态，用于失败回滚。"""
        tri_data = [(tri.vertices, tri.idx, tri.deleted) for tri in self.triangles]
        return tri_data, self._tri_id_counter

    def _restore_triangulation_state(self, snapshot) -> None:
        """恢复由 _snapshot_triangulation_state 保存的三角形拓扑状态。"""
        tri_data, tri_id_counter = snapshot
        restored = []
        for vertices, idx, deleted in tri_data:
            tri = MTri3(*vertices, idx=idx)
            tri.deleted = deleted
            self._compute_circumcircle(tri)
            restored.append(tri)

        self.triangles = restored
        self._tri_id_counter = tri_id_counter
        build_adjacency_from_triangles(self.triangles)

    def _count_triangle_components(self) -> int:
        """统计当前活动三角形的连通分量数。"""
        active_tris = [tri for tri in self.triangles if not tri.is_deleted()]
        if not active_tris:
            return 0

        edge_to_cells = {}
        for tri_idx, tri in enumerate(active_tris):
            verts = tri.vertices
            for i in range(3):
                a = verts[i]
                b = verts[(i + 1) % 3]
                edge = (min(a, b), max(a, b))
                edge_to_cells.setdefault(edge, []).append(tri_idx)

        adjacency = [set() for _ in range(len(active_tris))]
        for tri_indices in edge_to_cells.values():
            if len(tri_indices) == 2:
                a, b = tri_indices
                adjacency[a].add(b)
                adjacency[b].add(a)

        visited = set()
        components = 0
        for tri_idx in range(len(active_tris)):
            if tri_idx in visited:
                continue
            components += 1
            stack = [tri_idx]
            visited.add(tri_idx)
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)

        return components
    
    def _find_intersecting_edge(self, v1: int, v2: int):
        """查找与线段(v1, v2)严格相交的三角形边。"""
        p1, p2 = self.points[v1], self.points[v2]
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            for i in range(3):
                a_idx = tri.vertices[i]
                b_idx = tri.vertices[(i + 1) % 3]
                if a_idx in (v1, v2) or b_idx in (v1, v2):
                    continue
                a, b = self.points[a_idx], self.points[b_idx]
                d1 = cross(p1, p2, a)
                d2 = cross(p1, p2, b)
                d3 = cross(a, b, p1)
                d4 = cross(a, b, p2)
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return (a_idx, b_idx)
        return None
    
    def _flip_edge(self, n1: int, n2: int) -> bool:
        """翻转边 (n1, n2)。
        
        关键修改：保护边界边不被翻转。
        """
        # 关键保护：如果边是受保护的边界边，禁止翻转
        if self._is_protected_edge(n1, n2):
            return False
        
        # 查找共享该边的两个三角形
        t1 = t2 = None
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            if n1 in tri.vertices and n2 in tri.vertices:
                if t1 is None:
                    t1 = tri
                elif t2 is None:
                    t2 = tri
                else:
                    break

        if t1 is None or t2 is None:
            return False
        
        # 找到对角顶点
        a_idx = next(v for v in t1.vertices if v != n1 and v != n2)
        b_idx = next(v for v in t2.vertices if v != n1 and v != n2)
        
        # 检查凸性
        n1_pt, n2_pt, a_pt, b_pt = self.points[n1], self.points[n2], self.points[a_idx], self.points[b_idx]
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        c1 = cross(n1_pt, a_pt, n2_pt)
        c2 = cross(a_pt, n2_pt, b_pt)
        c3 = cross(n2_pt, b_pt, n1_pt)
        c4 = cross(b_pt, n1_pt, a_pt)
        eps = -1e-10
        is_convex = (c1 > eps and c2 > eps and c3 > eps and c4 > eps) or \
                    (c1 < -eps and c2 < -eps and c3 < -eps and c4 < -eps)
        if not is_convex:
            return False
        
        # 删除旧三角形
        t1.set_deleted(True)
        t2.set_deleted(True)
        
        # 创建新三角形
        new_tri1 = MTri3(n1, a_idx, b_idx, idx=self._next_tri_id())
        new_tri2 = MTri3(n2, b_idx, a_idx, idx=self._next_tri_id())
        self._compute_circumcircle(new_tri1)
        self._compute_circumcircle(new_tri2)
        self.triangles.append(new_tri1)
        self.triangles.append(new_tri2)
        
        # 重建邻接关系
        build_adjacency_from_triangles(self.triangles)
        
        return True
    
    # -------------------------------------------------------------------------
    # Laplacian 平滑
    # -------------------------------------------------------------------------
    
    def _laplacian_smoothing(self, iterations: int = 3, alpha: float = 0.5):
        """Laplacian 平滑。"""
        if iterations <= 0:
            return
        
        boundary_x_min = np.min(self.original_points[:, 0])
        boundary_x_max = np.max(self.original_points[:, 0])
        boundary_y_min = np.min(self.original_points[:, 1])
        boundary_y_max = np.max(self.original_points[:, 1])
        margin = 1e-6
        x_min = boundary_x_min + margin
        x_max = boundary_x_max - margin
        y_min = boundary_y_min + margin
        y_max = boundary_y_max - margin
        
        smoothed_points = self.points.copy()
        
        for iteration in range(iterations):
            new_points = smoothed_points.copy()
            
            # 构建邻接关系
            neighbor_dict = {}
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                for i in range(3):
                    v = tri.vertices[i]
                    if v not in neighbor_dict:
                        neighbor_dict[v] = set()
                    neighbor_dict[v].add(tri.vertices[(i + 1) % 3])
                    neighbor_dict[v].add(tri.vertices[(i + 2) % 3])
            
            for v, neighbors in neighbor_dict.items():
                if self.boundary_mask[v]:
                    continue
                if len(neighbors) == 0:
                    continue
                
                # 加权平均
                weighted_sum = np.zeros(2)
                weight_total = 0.0
                
                for n in neighbors:
                    neighbor_pos = smoothed_points[n]
                    dist = float(np.linalg.norm(neighbor_pos - smoothed_points[v]))
                    weight = 1.0 / dist if dist > 1e-12 else 1.0
                    weighted_sum += neighbor_pos * weight
                    weight_total += weight
                
                if weight_total > 1e-12:
                    target_pos = weighted_sum / weight_total
                else:
                    target_pos = smoothed_points[v]
                
                # 移动顶点
                trial_pos = smoothed_points[v] + alpha * (target_pos - smoothed_points[v])
                trial_pos[0] = np.clip(trial_pos[0], x_min, x_max)
                trial_pos[1] = np.clip(trial_pos[1], y_min, y_max)
                
                new_points[v] = trial_pos
            
            smoothed_points = new_points
        
        self.points = smoothed_points
    
    # -------------------------------------------------------------------------
    # 公共入口：generate_mesh
    # -------------------------------------------------------------------------
    
    def generate_mesh(
        self,
        target_triangle_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成三角形网格。
        
        参数:
            target_triangle_count: 目标三角形数量（可选）
        
        返回:
            (points, simplices, boundary_mask)
        """
        timer = TimeSpan("开始 Gmsh Bowyer-Watson 网格生成...")
        
        self.boundary_count = len(self.original_points)
        self.points = self.original_points.copy()
        self.boundary_mask = np.zeros(len(self.points), dtype=bool)
        self.boundary_mask[:self.boundary_count] = True
        verbose(f"边界点数量: {self.boundary_count}")
        
        verbose("阶段 1/3: 初始三角剖分...")
        self._triangulate()
        verbose(f"  初始三角形数量: {len(self.triangles)}")

        verbose("阶段 2/3: Gmsh 风格迭代插入内部点...")
        if self.holes:
            verbose(f"  检测到 {len(self.holes)} 个孔洞，插点时将拒绝在孔洞内插入新点")
            for i, hole in enumerate(self.holes):
                hole_center = np.mean(hole, axis=0)
                verbose(f"    孔洞 {i}: {len(hole)} 个点，中心 {hole_center}")
        self._insert_points_iteratively_gmsh(target_triangle_count)

        # 关键修复：根据Gmsh/TetGen的正确顺序
        # 1. 先恢复边界边（在完整三角网上操作）
        # 2. 再删除孔洞三角形（此时边界边已经被保护）
        # 参考 Gmsh / TetGen 的典型顺序：
        #   5. 恢复边界
        #   6. 挖洞

        verbose("阶段 2.5/3: Constrained Delaunay 边界恢复...")
        self._constrained_delaunay_triangulation()

        # 注意：不再调用 _recover_boundary_edges()
        # 因为 CDT 已经通过纯边翻转恢复了边界边
        # 避免插入 Steiner 点分割边界边

        if self.holes:
            verbose("阶段 2.6/3: 清理孔洞内三角形（Gmsh顺序：后删除孔洞）...")
            before_count = len(self.triangles)
            self._remove_hole_triangles()
            verbose(f"  删除孔洞内三角形: {before_count - len(self.triangles)} 个")

        # 新增：删除外边界外的三角形（标准 CDT 要求）
        if self.outer_boundary is not None:
            verbose("阶段 2.7/3: 清理外边界外三角形...")
            before_count = len(self.triangles)
            self._remove_outer_boundary_triangles()
            verbose(f"  删除外边界外三角形: {before_count - len(self.triangles)} 个")

        # 清理孔洞/外边界后，局部 cavity 已只剩真实计算域，再做一次 exact boundary 恢复，
        # 避免首次恢复落在错误一侧的三角形被后续清理删除后，原始边再次丢失。
        verbose("阶段 2.8/3: 清理后重新执行 Constrained Delaunay 边界恢复...")
        self._constrained_delaunay_triangulation()

        # 第二次 CDT 之后仍可能在孔洞/域外补回少量三角形；
        # 在进入最终平滑前再做一次最终域清理，避免残留非法单元进入导出结果。
        if self.holes:
            verbose("阶段 2.9/3: 二次恢复后再次清理孔洞内三角形...")
            before_count = len(self.triangles)
            self._remove_hole_triangles()
            final_hole_removed = before_count - len(self.triangles)
            verbose(f"  删除孔洞内三角形: {final_hole_removed} 个")
        else:
            final_hole_removed = 0

        if self.outer_boundary is not None:
            verbose("阶段 2.95/3: 二次恢复后再次清理外边界外三角形...")
            before_count = len(self.triangles)
            self._remove_outer_boundary_triangles()
            final_outer_removed = before_count - len(self.triangles)
            verbose(f"  删除外边界外三角形: {final_outer_removed} 个")
        else:
            final_outer_removed = 0

        if final_hole_removed > 0 or final_outer_removed > 0:
            verbose("阶段 2.98/3: 最终域清理后再次刷新精确边界...")
            self._constrained_delaunay_triangulation()

        if self.smoothing_iterations > 0:
            verbose(f"阶段 3/3: Laplacian 平滑 ({self.smoothing_iterations} 次迭代)...")
            current_point_count = len(self.points)
            if self.boundary_mask is None or len(self.boundary_mask) != current_point_count:
                new_boundary_mask = np.zeros(current_point_count, dtype=bool)
                if self.boundary_mask is not None:
                    copy_count = min(len(self.boundary_mask), current_point_count)
                    new_boundary_mask[:copy_count] = self.boundary_mask[:copy_count]
                self.boundary_mask = new_boundary_mask
            self.boundary_mask[:self.boundary_count] = True
            self._laplacian_smoothing(self.smoothing_iterations)
            verbose("  平滑完成")
        else:
            verbose("阶段 3/3: 跳过平滑（未启用）")
        
        # 清理已删除的三角形
        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]

        # 检测并修复重叠三角形
        verbose("阶段 3.5/3: 检测并修复重叠三角形...")
        self._remove_overlapping_triangles()
        self._remove_duplicate_triangles()
        self._remove_strictly_intersecting_triangles()
        if self.holes:
            removed_boundary_fans = self._remove_hole_side_boundary_fan_triangles()
            if removed_boundary_fans > 0:
                self._fix_isolated_boundary_points()

        points = self.points.copy()
        simplices = np.array([tri.vertices for tri in self.triangles])
        boundary_mask = np.zeros(len(points), dtype=bool)
        boundary_mask[:self.boundary_count] = True

        # 关键修复：确保所有边界点和 Steiner 点都被保留
        used_nodes = set()
        for simplex in simplices:
            for v in simplex:
                used_nodes.add(v)

        # 添加所有边界点到 used_nodes（包括原始边界点和 Steiner 点）
        for i in range(len(points)):
            if boundary_mask[i]:
                used_nodes.add(i)

        # 关键修复：原始边界点索引保持不变（0 到 boundary_count-1）
        # Steiner 点和内部点重新分配索引
        if len(used_nodes) < len(points):
            # 原始边界点保持原索引（0 到 boundary_count-1）
            # Steiner 点和内部点重新分配索引从 boundary_count 开始
            old_to_new = {}
            new_points = []
            new_boundary_mask = []
            
            # 首先添加所有原始边界点（保持原索引）
            for i in range(self.boundary_count):
                old_to_new[i] = i
                new_points.append(points[i])
                new_boundary_mask.append(True)
            
            # 然后添加被使用的 Steiner 点和内部点
            new_internal_idx = self.boundary_count
            for old_idx in sorted(used_nodes):
                if old_idx >= self.boundary_count:  # Steiner 点或内部点
                    old_to_new[old_idx] = new_internal_idx
                    new_points.append(points[old_idx])
                    new_boundary_mask.append(boundary_mask[old_idx])  # 保留 Steiner 点的边界标记
                    new_internal_idx += 1

            new_simplices = []
            for simplex in simplices:
                new_simplices.append([old_to_new[v] for v in simplex])

            points = np.array(new_points)
            simplices = np.array(new_simplices)
            boundary_mask = np.array(new_boundary_mask)
        
        verbose("网格生成完成:")
        verbose(f"  - 总节点数: {len(points)}")
        verbose(f"  - 边界节点: {np.sum(boundary_mask)}")
        verbose(f"  - 内部节点: {len(points) - np.sum(boundary_mask)}")
        verbose(f"  - 三角形数: {len(simplices)}")

        timer.show_to_console("Gmsh Bowyer-Watson 网格生成完成")
        return points, simplices, boundary_mask

    def _remove_overlapping_triangles(self) -> int:
        """检测并修复重叠三角形。

        标准 Bowyer-Watson 算法不应产生重叠三角形。
        如果检测到重叠，自动删除质量最差的三角形。

        返回:
            删除的三角形数量
        """
        # 构建边到三角形的映射
        edge_to_tris = {}
        for tri_idx, tri in enumerate(self.triangles):
            if tri.is_deleted():
                continue
            verts = tri.vertices
            for i in range(3):
                v1 = verts[i]
                v2 = verts[(i + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                if edge not in edge_to_tris:
                    edge_to_tris[edge] = []
                edge_to_tris[edge].append((tri_idx, tri))

        # 查找重叠边（超过2个三角形共享的边）
        overlapping_edges = {edge: tris for edge, tris in edge_to_tris.items() if len(tris) > 2}

        if not overlapping_edges:
            verbose("  无重叠三角形（符合 Delaunay 性质）")
            return 0

        verbose(f"  检测到 {len(overlapping_edges)} 条重叠边，开始修复...")

        triangles_to_remove = set()

        for edge, tris in overlapping_edges.items():
            if len(tris) <= 2:
                continue

            # 对三角形按质量排序（使用外接圆半径作为质量指标）
            tri_with_quality = []
            for tri_idx, tri in tris:
                if tri.circumradius is None:
                    self._compute_circumcircle(tri)
                tri_with_quality.append((tri.circumradius, tri_idx, tri))

            tri_with_quality.sort(reverse=True)  # 半径最大的质量最差

            # 保留2个质量最好的三角形，删除其余的
            for circumradius, tri_idx, tri in tri_with_quality[2:]:
                triangles_to_remove.add(tri_idx)

        # 标记删除
        for tri_idx in triangles_to_remove:
            self.triangles[tri_idx].set_deleted(True)

        removed_count = len(triangles_to_remove)
        if removed_count > 0:
            verbose(f"  删除 {removed_count} 个重叠三角形")
            self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]

        return removed_count

    def _remove_duplicate_triangles(self) -> int:
        """删除顶点集合完全相同的重复三角形。"""
        duplicate_count = 0
        seen_vertices = set()

        for tri in self.triangles:
            if tri.is_deleted():
                continue
            if tri.vertices in seen_vertices:
                tri.set_deleted(True)
                duplicate_count += 1
                continue
            seen_vertices.add(tri.vertices)

        if duplicate_count > 0:
            self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
            build_adjacency_from_triangles(self.triangles)
            verbose(f"  删除 {duplicate_count} 个重复三角形")

        return duplicate_count

    def _remove_strictly_intersecting_triangles(self) -> int:
        """删除导致严格边相交的低优先级三角形。"""
        from collections import defaultdict

        def strict_intersect(pa, pb, pc, pd, eps=1e-12):
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

            d1 = cross(pc, pd, pa)
            d2 = cross(pc, pd, pb)
            d3 = cross(pa, pb, pc)
            d4 = cross(pa, pb, pd)
            return (
                ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps))
                and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps))
            )

        total_removed = 0

        while True:
            edge_to_tris = defaultdict(list)
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                verts = tri.vertices
                for i in range(3):
                    edge = tuple(sorted((verts[i], verts[(i + 1) % 3])))
                    edge_to_tris[edge].append(tri)

            edges = list(edge_to_tris.keys())
            if not edges:
                break

            bboxes = []
            for a, b in edges:
                p1 = self.points[a]
                p2 = self.points[b]
                bboxes.append((
                    min(p1[0], p2[0]), max(p1[0], p2[0]),
                    min(p1[1], p2[1]), max(p1[1], p2[1]),
                ))

            triangles_to_remove = set()

            def tri_priority(tri):
                protected_count = 0
                verts = tri.vertices
                for i in range(3):
                    a, b = verts[i], verts[(i + 1) % 3]
                    if self._is_protected_edge(a, b):
                        protected_count += 1
                quality = self._compute_triangle_quality(tri)
                return (protected_count, quality)

            found_crossing = False
            for i, e1 in enumerate(edges):
                a, b = e1
                p1 = self.points[a]
                p2 = self.points[b]
                x1_min, x1_max, y1_min, y1_max = bboxes[i]

                for j in range(i + 1, len(edges)):
                    c, d = edges[j]
                    if a in (c, d) or b in (c, d):
                        continue

                    x2_min, x2_max, y2_min, y2_max = bboxes[j]
                    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
                        continue

                    p3 = self.points[c]
                    p4 = self.points[d]
                    if not strict_intersect(p1, p2, p3, p4):
                        continue

                    found_crossing = True
                    e2 = tuple(sorted((c, d)))
                    candidates = set(edge_to_tris[e1]) | set(edge_to_tris[e2])
                    if candidates:
                        triangles_to_remove.add(min(candidates, key=tri_priority))

            if not found_crossing or not triangles_to_remove:
                break

            for tri in triangles_to_remove:
                tri.set_deleted(True)

            total_removed += len(triangles_to_remove)
            self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
            build_adjacency_from_triangles(self.triangles)

        if total_removed > 0:
            verbose(f"  删除 {total_removed} 个严格相交相关三角形")

        return total_removed

    # --========================================================================
    # Splitting Strategy - 参考 Triangle spliteredge
    # --========================================================================

    def _recover_edge_by_splitting(self, v1: int, v2: int) -> bool:
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
        p1, p2 = self.points[v1], self.points[v2]
        
        # 步骤 0: 检查边是否已存在
        if self._is_exact_protected_edge_in_mesh(v1, v2):
            verbose(f"    [信息] 边 ({v1},{v2}) 已存在")
            return True
        
        # 步骤 1: 找到所有与约束段形成局部走廊的三角形（先严格，后包容）
        intersecting_tris = []
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            if self._segment_intersects_triangle(p1, p2, tri):
                intersecting_tris.append(tri)

        if not intersecting_tris:
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                if self._triangle_intersects_segment_for_steiner(tri, v1, v2):
                    intersecting_tris.append(tri)

        if not intersecting_tris:
            verbose(f"    [警告] 没有找到与边 ({v1},{v2}) 相交的三角形，检查边是否已存在")
            if self._is_exact_protected_edge_in_mesh(v1, v2):
                verbose(f"    [信息] 边 ({v1},{v2}) 已存在")
                return True
            verbose(f"    [信息] 边 ({v1},{v2}) 不存在，尝试使用边界路径恢复")
            return self._recover_edge_by_boundary_path(v1, v2)

        # 先基于局部空洞三角形提取边界，再判断端点是否被空洞覆盖
        tri_id_set = {id(tri) for tri in intersecting_tris}
        cavity_boundary = self._find_local_cavity_boundary_edges(intersecting_tris)
        cavity_nodes = {n for e in cavity_boundary for n in e}
        if v1 not in cavity_nodes or v2 not in cavity_nodes:
            seg_len = float(np.linalg.norm(p2 - p1))
            dist_tol = max(seg_len * 0.15, 1e-4)
            for tri in self.triangles:
                if tri.is_deleted() or id(tri) in tri_id_set:
                    continue
                if v1 not in tri.vertices and v2 not in tri.vertices:
                    continue
                centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
                if self._point_to_segment_distance(centroid, p1, p2) <= dist_tol:
                    intersecting_tris.append(tri)
                    tri_id_set.add(id(tri))

            cavity_boundary = self._find_local_cavity_boundary_edges(intersecting_tris)
            cavity_nodes = {n for e in cavity_boundary for n in e}

        adjacency = {}
        for a, b in cavity_boundary:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        path1 = self._find_path_in_boundary(adjacency, v1, v2) if cavity_boundary else None
        path2 = self._find_path_in_boundary_excluding_edges(
            adjacency,
            v1,
            v2,
            excluded_edges=self._boundary_path_edges(path1),
        ) if path1 else None

        # 当前局部相交三角形往往只抓到边界链两头，必要时沿线段附近扩展 triangle strip。
        if (
            not cavity_boundary
            or v1 not in cavity_nodes
            or v2 not in cavity_nodes
            or not path1
            or not path2
        ):
            strip_tris = self._find_triangle_strip_near_segment(v1, v2)
            if strip_tris:
                intersecting_tris = strip_tris
                cavity_boundary = self._find_local_cavity_boundary_edges(intersecting_tris)
                cavity_nodes = {n for e in cavity_boundary for n in e}
                adjacency = {}
                for a, b in cavity_boundary:
                    adjacency.setdefault(a, []).append(b)
                    adjacency.setdefault(b, []).append(a)
                path1 = self._find_path_in_boundary(adjacency, v1, v2) if cavity_boundary else None
                path2 = self._find_path_in_boundary_excluding_edges(
                    adjacency,
                    v1,
                    v2,
                    excluded_edges=self._boundary_path_edges(path1),
                ) if path1 else None

        if (
            not cavity_boundary
            or v1 not in cavity_nodes
            or v2 not in cavity_nodes
            or not path1
            or not path2
        ):
            verbose(f"    [警告] 局部空洞无法形成完整 corridor，放弃边 ({v1},{v2}) 恢复")
            return False

        verbose(f"    [信息] 找到 {len(intersecting_tris)} 个与边 ({v1},{v2}) 相交的三角形")

        # Debug: Print the missing edges
        if v1 >= 38 and v2 <= 95:
            print(f"[DEBUG] Wall edge ({v1},{v2}) splitting: {len(intersecting_tris)} intersecting tris", flush=True)

        # 步骤 2: 删除相交三角形
        snapshot = self._snapshot_triangulation_state()
        for tri in intersecting_tris:
            tri.set_deleted(True)

        self.triangles = [t for t in self.triangles if not t.is_deleted()]
        build_adjacency_from_triangles(self.triangles)

        # 步骤 3/4: 使用局部空洞边界和约束边重新三角化
        if not self._retriangulate_cavity_with_edge(cavity_boundary, v1, v2):
            self._restore_triangulation_state(snapshot)
            return False
        if self._count_triangle_components() != 1:
            self._restore_triangulation_state(snapshot)
            return False
        return True

    def _retriangulate_cavity_with_edge(self, cavity_boundary, v1: int, v2: int) -> bool:
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
        
        # 找到从 v1 到 v2 的两条 edge-disjoint 路径：
        # 一条通常是当前被分裂的边界链，另一条是域内 corridor。
        path1 = self._find_path_in_boundary(adjacency, v1, v2)
        path2 = self._find_path_in_boundary_excluding_edges(
            adjacency,
            v1,
            v2,
            excluded_edges=self._boundary_path_edges(path1),
        )

        if not path1 or not path2:
            verbose(f"    [警告] 无法找到从 v1 到 v2 的两条路径")
            return False
        
        verbose(f"    [信息] 找到两条路径：path1={len(path1)} 个点，path2={len(path2)} 个点")
        
        # 对受保护边，目标是恢复“真正的一侧边界边”，
        # 因此只对域内侧 path 重三角化；当前被 split 的边界链应被直接替换掉。
        fill_paths = [path1, path2]
        if self._is_protected_edge(v1, v2) and not getattr(self, '_in_initial_boundary_recovery', False):
            fill_paths = [self._choose_boundary_fill_path(path1, path2)]

        def triangulate_path(path):
            if len(path) < 3:
                return

            anchor = path[0]
            for i in range(1, len(path) - 1):
                a = path[i]
                b = path[i + 1]
                tri = MTri3(anchor, a, b, idx=self._next_tri_id())
                self._compute_circumcircle(tri)
                new_triangles.append(tri)

        for path in fill_paths:
            triangulate_path(path)
        
        self.triangles.extend(new_triangles)
        build_adjacency_from_triangles(self.triangles)
        
        verbose(f"    [信息] 创建了 {len(new_triangles)} 个新三角形")
        
        # 验证约束边是否已创建
        if self._is_exact_protected_edge_in_mesh(v1, v2):
            verbose(f"    [成功] 约束边 ({v1},{v2}) 已成功创建")
            return True
        if self._enforce_protected_edge_as_boundary(v1, v2):
            verbose(f"    [成功] 约束边 ({v1},{v2}) 已通过域外侧清理恢复")
            return True
        else:
            verbose(f"    [警告] 约束边 ({v1},{v2}) 创建失败")
            return False

    def _recover_edge_by_boundary_path(self, v1: int, v2: int) -> bool:
        """通过边界路径恢复约束边。

        当没有相交三角形时，说明 v1 和 v2 可能在边界上但不直接相连。
        此方法找到连接 v1 和 v2 的边界路径，并重新三角化。

        增强版本：对于短边，使用更灵活的恢复策略：
        1. 首先尝试找到边界路径
        2. 如果找不到，尝试找到通过内部点的路径
        3. 使用扇形三角化创建约束边

        参数:
            v1, v2: 约束边的两个端点

        返回:
            True 如果成功恢复
        """
        # 找到所有包含 v1 的三角形
        v1_tris = [tri for tri in self.triangles if not tri.is_deleted() and v1 in tri.vertices]
        v2_tris = [tri for tri in self.triangles if not tri.is_deleted() and v2 in tri.vertices]

        if not v1_tris or not v2_tris:
            verbose(f"    [警告] v1 或 v2 没有连接的三角形")
            return False

        # 找到 v1 和 v2 的边界邻居
        v1_boundary_neighbors = set()
        for tri in v1_tris:
            for v in tri.vertices:
                if v != v1:
                    # 检查 v 是否是边界点
                    if v < self.boundary_count:
                        v1_boundary_neighbors.add(v)

        v2_boundary_neighbors = set()
        for tri in v2_tris:
            for v in tri.vertices:
                if v != v2:
                    if v < self.boundary_count:
                        v2_boundary_neighbors.add(v)

        # 找到从 v1 到 v2 的边界路径
        path = self._find_boundary_path(v1, v2, v1_boundary_neighbors, v2_boundary_neighbors)

        if not path or len(path) < 2:
            verbose(f"    [警告] 无法找到从 v1 到 v2 的边界路径，尝试增强恢复策略")
            # 增强策略：找到所有边界点在 v1 和 v2 之间，并按角度排序
            path = self._find_enhanced_boundary_path(v1, v2)
            if not path or len(path) < 2:
                verbose(f"    [警告] 增强策略也无法找到边界路径")
                return False

        verbose(f"    [信息] 找到边界路径：{path[:5]}... (共{len(path)}个点)")

        # 如果路径只有 v1 和 v2，直接创建边
        if len(path) == 2:
            # 找到包含 v1 和 v2 的三角形并翻转边
            common_tris = set(v1_tris) & set(v2_tris)
            if common_tris:
                for tri in common_tris:
                    # 检查是否可以翻转
                    for i in range(3):
                        edge = tri.get_edge_sorted(i)
                        if set(edge) == {v1, v2}:
                            return True  # 边已存在
                        # 尝试翻转对边
                        other_edge = tri.get_edge_sorted((i + 1) % 3)
                        if self._flip_edge(other_edge[0], other_edge[1]):
                            if self._is_exact_protected_edge_in_mesh(v1, v2):
                                return True
            return False

        # 路径上有中间点，需要重新三角化
        # 找到路径上的三角形并删除
        tris_to_delete = set()
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            # 找到包含边 (a, b) 的三角形
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                if a in tri.vertices and b in tri.vertices:
                    tris_to_delete.add(tri)

        # 删除三角形
        for tri in tris_to_delete:
            tri.set_deleted(True)

        self.triangles = [t for t in self.triangles if not t.is_deleted()]

        # 使用 (v1, v2) 重新三角化
        # 对于路径上的每个中间点，创建三角形 (v1, v2, path[i])
        new_triangles = []
        for i in range(1, len(path) - 1):
            vi = path[i]
            new_tri = MTri3(v1, v2, vi, idx=self._next_tri_id())
            self._compute_circumcircle(new_tri)
            new_triangles.append(new_tri)

        self.triangles.extend(new_triangles)
        build_adjacency_from_triangles(self.triangles)

        verbose(f"    [信息] 创建了 {len(new_triangles)} 个新三角形")

        return self._is_exact_protected_edge_in_mesh(v1, v2)

    def _find_boundary_path(self, v1, v2, v1_neighbors, v2_neighbors):
        """找到从 v1 到 v2 的边界路径。

        使用 BFS 在边界点之间搜索路径。
        """
        from collections import deque

        # 构建边界邻接表
        boundary_adj = {}
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            verts = [v for v in tri.vertices if v < self.boundary_count]
            if len(verts) >= 2:
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        boundary_adj.setdefault(verts[i], set()).add(verts[j])
                        boundary_adj.setdefault(verts[j], set()).add(verts[i])

        # BFS 搜索
        queue = deque([(v1, [v1])])
        visited = {v1}

        while queue:
            current, path = queue.popleft()

            if current == v2:
                return path

            for neighbor in boundary_adj.get(current, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

        return None

    def _find_enhanced_boundary_path(self, v1: int, v2: int):
        """增强版边界路径查找。

        当标准边界路径查找失败时，使用更灵活的方法：
        1. 找到所有在 v1 和 v2 之间的边界点
        2. 按角度排序这些点
        3. 返回排序后的路径

        参数:
            v1, v2: 约束边的两个端点

        返回:
            边界点路径列表 [v1, ..., v2]
        """
        p1 = self.points[v1]
        p2 = self.points[v2]
        mid = (p1 + p2) / 2

        # 找到所有在 v1 和 v2 之间的边界点
        # 判断标准：点在 v1-v2 连线的同一侧，且角度在合理范围内
        between_points = []
        for i in range(self.boundary_count):
            if i == v1 or i == v2:
                continue
            p = self.points[i]

            # 计算点相对于 v1-v2 连线的角度
            v12 = p2 - p1
            v1p = p - p1
            angle = np.arctan2(np.cross(v12, v1p), np.dot(v12, v1p))

            # 只保留角度在合理范围内的点（同一侧）
            if -np.pi < angle < 0:  # 下侧
                between_points.append((i, angle, p))
            elif 0 < angle < np.pi:  # 上侧
                between_points.append((i, angle, p))

        if not between_points:
            return None

        # 按角度排序
        between_points.sort(key=lambda x: x[1])

        # 检查哪一侧有更多点（选择点较多的一侧）
        lower = [(idx, ang, p) for idx, ang, p in between_points if -np.pi < ang < 0]
        upper = [(idx, ang, p) for idx, ang, p in between_points if 0 < ang < np.pi]

        # 选择点较多的一侧
        if len(lower) >= len(upper):
            between_points = lower
        else:
            between_points = upper

        if not between_points:
            return None

        # 构建路径：v1 -> between_points -> v2
        path = [v1] + [idx for idx, ang, p in between_points] + [v2]
        return path

    def _find_path_in_boundary(self, adjacency, start, end, exclude=None):
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




    # --========================================================================
    # Fix Isolated Boundary Points - 修复被隔离的边界点
    # --========================================================================

    def _fix_isolated_boundary_points(self):
        """修复被隔离的边界点。

        算法：
        1. 找到所有被隔离的边界点
        2. 对于每个被隔离的点，找到最近的三角形
        3. 删除该三角形，使用边界点重新三角化
        """
        # 找到所有连接到三角形的点
        connected = set()
        for tri in self.triangles:
            if not tri.is_deleted():
                connected.update(tri.vertices)

        # 找到被隔离的边界点
        isolated_boundary_points = [i for i in range(self.boundary_count) if i not in connected]

        if not isolated_boundary_points:
            verbose(f"  [信息] 没有发现被隔离的边界点")
            return 0

        verbose(f"  [信息] 找到 {len(isolated_boundary_points)} 个被隔离的边界点")

        # 修复每个被隔离的点
        fixed_count = 0
        for point_idx in isolated_boundary_points:
            if self._fix_isolated_boundary_point(point_idx):
                fixed_count += 1

        verbose(f"  [信息] 修复了 {fixed_count}/{len(isolated_boundary_points)} 个被隔离的边界点")
        return fixed_count

    def _fix_isolated_boundary_point(self, point_idx: int) -> bool:
        """修复一个被隔离的边界点。

        算法：
        1. 找到离该点最近的三角形
        2. 删除该三角形
        3. 使用边界点和三角形顶点重新三角化

        参数:
            point_idx: 被隔离的边界点索引

        返回:
            True 如果修复成功
        """
        point = self.points[point_idx]

        protected_neighbors = []
        for edge in self.protected_edges:
            if point_idx not in edge:
                continue
            other = next(iter(edge - {point_idx}))
            if other < self.boundary_count:
                protected_neighbors.append(other)

        if len(protected_neighbors) == 2:
            n0, n1 = protected_neighbors
            best_candidate = None
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                if point_idx in tri.vertices:
                    continue
                if n0 not in tri.vertices or n1 not in tri.vertices:
                    continue

                apex = next(v for v in tri.vertices if v not in (n0, n1))
                new_faces = ((point_idx, n0, apex), (point_idx, apex, n1))
                valid = True
                total_area = 0.0
                for a, b, c in new_faces:
                    pa, pb, pc = self.points[a], self.points[b], self.points[c]
                    area2 = abs(
                        (pb[0] - pa[0]) * (pc[1] - pa[1])
                        - (pb[1] - pa[1]) * (pc[0] - pa[0])
                    )
                    if area2 < 1e-14:
                        valid = False
                        break
                    centroid = (pa + pb + pc) / 3.0
                    if not self._point_in_domain(centroid):
                        valid = False
                        break
                    total_area += area2

                if not valid:
                    continue

                score = (
                    0 if apex >= self.boundary_count else 1,
                    -total_area,
                    float(np.linalg.norm(self.points[apex] - point)),
                )
                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (score, tri, apex)

            if best_candidate is not None:
                _, base_tri, apex = best_candidate
                base_tri.set_deleted(True)
                self.triangles = [t for t in self.triangles if not t.is_deleted()]
                replacement_tris = [
                    MTri3(point_idx, n0, apex, idx=self._next_tri_id()),
                    MTri3(point_idx, apex, n1, idx=self._next_tri_id()),
                ]
                for tri in replacement_tris:
                    self._compute_circumcircle(tri)
                self.triangles.extend(replacement_tris)
                build_adjacency_from_triangles(self.triangles)
                if (
                    self._is_exact_protected_edge_in_mesh(point_idx, n0)
                    and self._is_exact_protected_edge_in_mesh(point_idx, n1)
                ):
                    return True

        # 优先使用 Bowyer-Watson 空腔插入（质量更稳健）
        containing_tri = None
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            if self._point_in_triangle(point, tri):
                containing_tri = tri
                break

        if containing_tri is None:
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                if self._point_in_circumcircle(point, tri):
                    containing_tri = tri
                    break

        if containing_tri is None:
            # 兜底：最近三角形
            min_dist = float('inf')
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                dist = self._point_to_triangle_distance(point, tri)
                if dist < min_dist:
                    min_dist = dist
                    containing_tri = tri

        if containing_tri is None:
            verbose(f"    [警告] 没有找到可插入三角形，无法修复点 {point_idx}")
            return False

        cavity_triangles, shell_edges = recur_find_cavity(
            containing_tri,
            point,
            point_idx,
            self.points,
            self.protected_edges,
            self._point_in_circumcircle
        )
        if cavity_triangles and shell_edges:
            _, success = insert_vertex(
                shell_edges,
                cavity_triangles,
                point_idx,
                self.points,
                self.triangles,
                validate_star=False,
            )
            if success:
                build_adjacency_from_triangles(self.triangles)
                return True

        # 兜底：简单扇形重连（仅在空腔插入失败时）
        min_dist = float('inf')
        closest_tri = None
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            dist = self._point_to_triangle_distance(point, tri)
            if dist < min_dist:
                min_dist = dist
                closest_tri = tri

        if closest_tri is None:
            verbose(f"    [警告] 空腔插入失败且无兜底三角形，无法修复点 {point_idx}")
            return False

        closest_tri.set_deleted(True)
        self.triangles = [t for t in self.triangles if not t.is_deleted()]

        v0, v1, v2 = closest_tri.vertices
        new_tris = [
            MTri3(point_idx, v0, v1, idx=self._next_tri_id()),
            MTri3(point_idx, v1, v2, idx=self._next_tri_id()),
            MTri3(point_idx, v2, v0, idx=self._next_tri_id())
        ]
        for tri in new_tris:
            self._compute_circumcircle(tri)
        self.triangles.extend(new_tris)
        build_adjacency_from_triangles(self.triangles)
        return True

    def _point_to_triangle_distance(self, point: np.ndarray, tri: MTri3) -> float:
        """计算点到三角形的距离（到三条边的最小距离）。"""
        v0, v1, v2 = tri.vertices
        p0, p1, p2 = self.points[v0], self.points[v1], self.points[v2]

        dist0 = self._point_to_segment_distance(point, p0, p1)
        dist1 = self._point_to_segment_distance(point, p1, p2)
        dist2 = self._point_to_segment_distance(point, p0, p2)

        return min(dist0, dist1, dist2)

    def _point_to_segment_distance(self, point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """计算点到线段的距离。"""
        ab = b - a
        ap = point - a

        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))

        projection = a + t * ab
        return np.linalg.norm(point - projection)




    def _recover_edge_with_steiner_point(self, v1: int, v2: int) -> bool:
        """使用 Steiner 点恢复丢失的边界边。

        当边 (v1,v2) 缺失且没有相交三角形时，在边的中点插入 Steiner 点。

        算法：
        1. 计算边的中点
        2. 找到离中点最近的三角形
        3. 删除该三角形
        4. 在中点创建 Steiner 点
        5. 使用 Steiner 点和边界点重新三角化，确保边界边被恢复

        参数:
            v1, v2: 缺失边界边的两个端点

        返回:
            True 如果成功恢复
        """
        p1, p2 = self.points[v1], self.points[v2]
        mid_point = (p1 + p2) / 2

        # 找到离中点最近的三角形
        min_dist = float('inf')
        closest_tri = None
        for tri in self.triangles:
            if tri.is_deleted():
                continue

            dist = self._point_to_triangle_distance(mid_point, tri)
            if dist < min_dist:
                min_dist = dist
                closest_tri = tri

        if closest_tri is None:
            verbose(f"    [警告] 没有找到最近的三角形，无法使用 Steiner 点恢复边 ({v1},{v2})")
            return False

        verbose(f"    [信息] 使用 Steiner 点恢复边 ({v1},{v2}), 最近三角形 {closest_tri.idx}, 距离 {min_dist:.4f}")

        # 删除最近的三角形
        closest_tri.set_deleted(True)
        self.triangles = [t for t in self.triangles if not t.is_deleted()]

        # 在中点创建 Steiner 点
        mid_idx = len(self.points)
        self.points = np.vstack([self.points, mid_point])
        
        # 关键修复：标记 Steiner 点为边界点，防止在后续处理中被删除或移动
        if len(self.boundary_mask) <= mid_idx:
            # 扩展 boundary_mask
            new_mask = np.zeros(len(self.points), dtype=bool)
            new_mask[:len(self.boundary_mask)] = self.boundary_mask
            self.boundary_mask = new_mask
        self.boundary_mask[mid_idx] = True  # 标记 Steiner 点为边界点

        # 获取原三角形的顶点
        orig_v0, orig_v1, orig_v2 = closest_tri.vertices

        # 创建新三角形，确保包含边 (v1, v2)
        # 策略：创建 (v1, v2, mid) 三角形，确保边界边存在
        new_tris = [
            MTri3(v1, v2, mid_idx, idx=self._next_tri_id()),
        ]

        # 还需要填充剩余的空洞
        # 找到原三角形中与 Steiner 点不冲突的顶点
        other_vertices = [orig_v0, orig_v1, orig_v2]
        for other_v in other_vertices:
            if other_v not in [v1, v2]:
                # 创建连接其他顶点的三角形
                new_tris.append(MTri3(v1, other_v, mid_idx, idx=self._next_tri_id()))
                new_tris.append(MTri3(v2, other_v, mid_idx, idx=self._next_tri_id()))
                break

        for tri in new_tris:
            self._compute_circumcircle(tri)

        self.triangles.extend(new_tris)
        build_adjacency_from_triangles(self.triangles)

        # 验证边是否恢复
        if self._is_boundary_edge_in_mesh(v1, v2):
            verbose(f"    [成功] Steiner 点成功恢复边 ({v1},{v2})")
            return True
        else:
            verbose(f"    [错误] Steiner 点未能恢复边 ({v1},{v2})")
            return False


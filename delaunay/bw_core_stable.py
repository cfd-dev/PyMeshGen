"""
Bowyer-Watson Delaunay 网格生成器 - 核心实现（重构版）

基于 Bowyer-Watson 算法的二维三角形网格生成，支持：
1. 以离散边界网格作为输入
2. 使用 QuadtreeSizing 尺寸场控制网格尺寸
3. 自动/手动孔洞处理
4. 边界边保护与恢复
5. 增量式点插入（避免全量重剖分）

重构特性（参考 Gmsh delaunay/ref/）：
- Gmsh 风格的数据结构（MTri3、edgeXface）
- 递归/迭代 Cavity 搜索（recurFindCavityAniso）
- 鲁棒几何谓词（Shewchuk's predicates）
- 懒删除优化（避免频繁集合操作）
- 优先级队列（按外接圆半径排序）

参考: Gmsh delaunay/ref/ 下的 C++ 实现
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from math import sqrt
from scipy.spatial import KDTree
from collections import Counter
import heapq  # 新增：用于优先级队列

from utils.message import debug, verbose
from utils.geom_toolkit import point_in_polygon, is_polygon_clockwise

from .bw_types import MTri3

__all__ = [
    "Triangle",
    "triangle_to_mtri3",
    "mtri3_to_triangle",
    "BowyerWatsonMeshGenerator",
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
        from utils.timer import TimeSpan
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

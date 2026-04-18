"""
Bowyer-Watson Delaunay 网格生成器 - Gmsh 风格重构实现

这是一个全新的实现，参考 Gmsh 的 C++ 代码，使用以下改进：
1. Gmsh 风格的数据结构（MTri3、edgeXface）
2. 递归/迭代 Cavity 搜索（recurFindCavityAniso）
3. 鲁棒几何谓词（Shewchuk's predicates）
4. 懒删除优化（避免频繁集合操作）
5. 优先级队列（按外接圆半径排序）

参考: Gmsh delaunay/ref/ 下的 C++ 实现
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from math import sqrt
from scipy.spatial import KDTree
import heapq

from utils.message import debug, verbose
from utils.geom_toolkit import point_in_polygon, is_polygon_clockwise
from utils.timer import TimeSpan

# Gmsh 风格的数据结构和算法
from delaunay.bw_types import (
    MTri3,
    EdgeXFace,
    TriangleComparator,
    build_adjacency_from_triangles,
    collect_cavity_shell,
    compute_cavity_volume,
)
from delaunay.bw_cavity import (
    recur_find_cavity,
    find_cavity_iterative,
    validate_star_shaped,
    validate_edge_lengths,
    insert_vertex,
    restore_cavity,
)
from delaunay.bw_predicates import (
    orient2d_fast,
    incircle,
    circumcenter_precise,
    compute_circumcircle,
    point_in_circumcircle_robust,
)


# =============================================================================
# GmshBowyerWatsonMeshGenerator - 全新实现
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
        min_edge = min(edge_lengths)
        max_edge = max(edge_lengths)

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

        # 初始三角剖分后，立即恢复所有受保护边界边
        self._recover_all_protected_edges_initial()

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

        for v1, v2 in missing_edges:
            self._force_recover_boundary_edge_by_reconnection(v1, v2)

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
        failed_tri_ids = set()
        
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

                    if target_size is not None and target_size > 1e-12:
                        if max_edge > target_size * tri_size_tolerance:
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
                if candidate_tri.is_deleted() or tri_id in failed_tri_ids:
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

            should_refine = True
            if target_size is not None and target_size > 1e-12:
                # 尺寸场要求
                if max_edge <= target_size * size_tolerance:
                    # 尺寸满足，但质量太差也需要细分
                    if quality >= quality_threshold:
                        should_refine = False
            else:
                # 无尺寸场：仅基于质量判断
                if quality >= 0.3:
                    should_refine = False

            if not should_refine:
                continue

            # 计算插入点（外接圆心）
            if worst_tri.circumcenter is None:
                self._compute_circumcircle(worst_tri)
            
            new_point = worst_tri.circumcenter.copy()
            
            # 检查圆心是否在有效范围内
            if not (x_min - margin < new_point[0] < x_max + margin and
                    y_min - margin < new_point[1] < y_max + margin):
                # 超出范围，使用重心
                p1 = self.points[worst_tri.vertices[0]]
                p2 = self.points[worst_tri.vertices[1]]
                p3 = self.points[worst_tri.vertices[2]]
                new_point = (p1 + p2 + p3) / 3.0
            
            # 检查插入点是否在孔洞内（增强版）
            in_hole = False
            hole_debug_info = ""
            if self.holes:
                for hole_idx, hole in enumerate(self.holes):
                    if point_in_polygon(new_point, hole):
                        in_hole = True
                        hole_debug_info = f"孔洞 {hole_idx}"
                        break

            if in_hole:
                # 记录失败统计
                failed_tri_ids.add(id(worst_tri))
                worst_tri.set_deleted(True)
                consecutive_failures += 1
                
                # 调试信息：前 10 次孔洞拒绝插入
                if consecutive_failures <= 10:
                    verbose(f"    [孔洞拒绝] 点 {new_point} 在 {hole_debug_info} 内，拒绝插入")
                
                if consecutive_failures >= max_consecutive_failures:
                    verbose(f"  [进度] 连续失败 {consecutive_failures} 次，停止插点")
                    break
                continue
            
            # 添加新点
            new_point_idx = len(self.points)
            self.points = np.vstack([self.points, new_point])
            
            # Gmsh Cavity 搜索
            cavity_triangles, shell_edges = recur_find_cavity(
                worst_tri,
                new_point,
                new_point_idx,
                self.points,
                self.protected_edges,
                self._point_in_circumcircle
            )
            
            if not cavity_triangles:
                failed_tri_ids.add(id(worst_tri))
                consecutive_failures += 1
                continue

            # 插入顶点（跳过星形空腔验证，因为 recur_find_cavity 已经确保空腔有效）
            new_triangles, success = insert_vertex(
                shell_edges,
                cavity_triangles,
                new_point_idx,
                self.points,
                self.triangles,
                validate_star=False,
            )
            
            if success:
                consecutive_failures = 0  # 重置失败计数
                
                # 将新三角形加入优先级队列
                for tri in new_triangles:
                    if tri.circumradius is None:
                        self._compute_circumcircle(tri)
                    heapq.heappush(priority_queue, (-tri.circumradius, id(tri), tri))
            else:
                failed_tri_ids.add(id(worst_tri))
                consecutive_failures += 1
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

        from utils.geom_toolkit import is_polygon_clockwise, point_in_polygon

        # 修正孔洞方向
        fixed_holes = []
        for hole in self.holes:
            if is_polygon_clockwise(hole):
                fixed_holes.append(hole[::-1])
            else:
                fixed_holes.append(hole.copy())
        holes_to_use = fixed_holes

        # 尝试使用种子填充法
        removed_count = self._remove_holes_via_seed_fill(holes_to_use)

        # 如果种子填充法没有删除任何三角形（可能失败），回退到质心测试
        if removed_count == 0:
            verbose("  种子填充法未删除任何三角形，回退到质心测试法...")
            removed_count = self._remove_holes_via_centroid_test(holes_to_use)

        return removed_count

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

        # 保护包含孔洞边界边的三角形
        boundary_edge_set = set()
        for edge in self.protected_edges:
            v1, v2 = list(edge)
            p1, p2 = self.points[v1], self.points[v2]

            # 检查边的中点是否在孔洞内
            centroid = (p1 + p2) / 2.0
            for hole in holes_to_use:
                if point_in_polygon(centroid, hole):
                    boundary_edge_set.add(frozenset({v1, v2}))
                    break

        triangles_to_remove = set()
        for i, tri in enumerate(self.triangles):
            if tri.is_deleted():
                continue

            # 保护包含边界边的三角形
            has_any_boundary_edge = False
            for j in range(3):
                v1 = tri.vertices[j]
                v2 = tri.vertices[(j + 1) % 3]
                edge_key = frozenset({v1, v2})
                if edge_key in boundary_edge_set:
                    has_any_boundary_edge = True
                    break
            if has_any_boundary_edge:
                continue

            # 质心测试
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
        """
        verbose("开始 Constrained Delaunay 边界边恢复...")

        # 新增：在 CDT 前，先修复被隔离的边界点
        verbose("检查并修复被隔离的边界点...")
        fixed_count = self._fix_isolated_boundary_points()
        verbose(f"修复了 {fixed_count} 个被隔离的边界点")

        # 修复：不再使用硬编码的 wall 边界索引
        # 所有 protected_edges 都是需要恢复的约束边
        max_passes = 10
        total_recovered = 0
        failed_edges_list = []

        for pass_num in range(max_passes):
            missing_edges = []
            for edge in self.protected_edges:
                v1, v2 = list(edge)
                if not self._is_boundary_edge_in_mesh(v1, v2):
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
                    if not self._is_boundary_edge_in_mesh(v1, v2):
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
            if not self._is_boundary_edge_in_mesh(v1, v2):
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
        if self._is_boundary_edge_in_mesh(v1, v2):
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
                    if self._is_boundary_edge_in_mesh(v1, v2):
                        if is_wall:
                            verbose(f"    翻转恢复成功")
                        return True
                    break

            if not flipped:
                if is_wall:
                    verbose(f"    无法翻转，尝试强制重连")
                break

        if self._is_boundary_edge_in_mesh(v1, v2):
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

        if not intersecting_tris:
            # 没有相交的三角形，说明(v1, v2)的路径是"畅通"的
            # 但这可能意味着：
            # 1. v1和v2已经在同一个三角形中（但边不存在）- 这不应该发生
            # 2. v1和v2之间没有三角形，需要创建新的三角形
            # 3. v1和v2被空洞分隔
            
            # 尝试找到包含v1或v2的三角形，并重建三角剖分
            verbose(f"    无相交三角形，尝试查找并重建...")
            
            # 找到所有包含v1或v2的三角形
            tris_with_v1 = []
            tris_with_v2 = []
            
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                if v1 in tri.vertices:
                    tris_with_v1.append(tri)
                if v2 in tri.vertices:
                    tris_with_v2.append(tri)
            
            # 如果两者都有三角形，尝试找到连接路径
            if tris_with_v1 and tris_with_v2:
                # 找到v1的其他邻居（除了v2）
                v1_other_neighbors = set()
                for tri in tris_with_v1:
                    for v in tri.vertices:
                        if v != v1:
                            v1_other_neighbors.add(v)

                # 找到v2的其他邻居（除了v1）
                v2_other_neighbors = set()
                for tri in tris_with_v2:
                    for v in tri.vertices:
                        if v != v2:
                            v2_other_neighbors.add(v)

                # 检查是否有共同的邻居
                common = v1_other_neighbors & v2_other_neighbors

                if common:
                    verbose(f"    找到 {len(common)} 个共同邻居，重建三角剖分")
                    # 删除包含共同邻居和v1或v2的三角形
                    tris_to_delete = []
                    for tri in self.triangles:
                        if tri.is_deleted():
                            continue
                        has_common = any(v in common for v in tri.vertices)
                        has_endpoint = v1 in tri.vertices or v2 in tri.vertices
                        if has_common and has_endpoint:
                            tris_to_delete.append(tri)

                    for tri in tris_to_delete:
                        tri.set_deleted(True)

                    self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]
                    build_adjacency_from_triangles(self.triangles)

                    # 用(v1, v2)和共同邻居创建新三角形
                    for v_common in sorted(list(common)):
                        new_tri = MTri3(v1, v2, v_common, idx=self._next_tri_id())
                        self._compute_circumcircle(new_tri)
                        self.triangles.append(new_tri)

                    build_adjacency_from_triangles(self.triangles)
                    return self._is_boundary_edge_in_mesh(v1, v2)
                else:
                    # 没有共同邻居，恢复失败
                    return False
            else:
                return False

        verts_to_keep = set()
        for tri in intersecting_tris:
            for v in tri.vertices:
                if v != v1 and v != v2:
                    verts_to_keep.add(v)

        boundary_edge_key = frozenset({v1, v2})

        for tri in intersecting_tris:
            tri.set_deleted(True)

        build_adjacency_from_triangles(self.triangles)

        if len(verts_to_keep) == 0:
            return False

        if len(verts_to_keep) == 1:
            v_other = list(verts_to_keep)[0]
            new_tri = MTri3(v1, v2, v_other, idx=self._next_tri_id())
            self._compute_circumcircle(new_tri)
            self.triangles.append(new_tri)
        else:
            verts_list = list(verts_to_keep)
            verts_list.sort()

            fan_tris = []
            for i in range(len(verts_list)):
                v_a = verts_list[i]
                v_b = verts_list[(i + 1) % len(verts_list)]
                tri1 = MTri3(v1, v_a, v_b, idx=self._next_tri_id())
                tri2 = MTri3(v2, v_b, v_a, idx=self._next_tri_id())
                self._compute_circumcircle(tri1)
                self._compute_circumcircle(tri2)
                fan_tris.extend([tri1, tri2])

            self.triangles.extend(fan_tris)

        build_adjacency_from_triangles(self.triangles)
        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]

        return self._is_boundary_edge_in_mesh(v1, v2)

    def _segment_intersects_triangle(self, p1: np.ndarray, p2: np.ndarray, tri: 'MTri3') -> bool:
        """检查线段(p1, p2)是否与三角形相交。"""
        pA = self.points[tri.vertices[0]]
        pB = self.points[tri.vertices[1]]
        pC = self.points[tri.vertices[2]]

        for pA_, pB_ in [(pA, pB), (pB, pC), (pC, pA)]:
            if self._segments_intersect_strict(p1, p2, pA_, pB_):
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
                
                if self._segments_intersect_strict(
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

        参考Gmsh recoverEdges()实现（BOWYER_WATSON_GMSH_DESIGN.md 第724-750行）：
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
            if self._segments_intersect_strict(p1, p2, self.points[a_idx], self.points[b_idx]):
                return True

        # 也检查三角形是否包含线段的中点
        mid = (p1 + p2) / 2.0
        if self._point_in_triangle_strict(mid, tri):
            return True

        return False

    def _segments_intersect_strict(self, p1, p2, p3, p4) -> bool:
        """检查两条线段是否严格相交（不包括端点接触）。"""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        # 严格相交（不包括端点）
        if ((d1 > 1e-12 and d2 < -1e-12) or (d1 < -1e-12 and d2 > 1e-12)) and \
           ((d3 > 1e-12 and d4 < -1e-12) or (d3 < -1e-12 and d4 > 1e-12)):
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
        # 参考：BOWYER_WATSON_GMSH_DESIGN.md 第706-717行
        #   // 5. 恢复边界: recoverboundary(t)
        #   // 6. 挖洞: carveholes()

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

        if self.smoothing_iterations > 0:
            verbose(f"阶段 3/3: Laplacian 平滑 ({self.smoothing_iterations} 次迭代)...")
            current_point_count = len(self.points)
            self.boundary_mask = np.zeros(current_point_count, dtype=bool)
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
        if self._is_boundary_edge_in_mesh(v1, v2):
            verbose(f"    [信息] 边 ({v1},{v2}) 已存在")
            return True
        
        # 步骤 1: 找到所有相交的三角形
        intersecting_tris = []
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            if self._segment_intersects_triangle(p1, p2, tri):
                intersecting_tris.append(tri)
        
        if not intersecting_tris:
            # 没有相交的三角形，边可能已经存在
            verbose(f"    [警告] 没有找到与边 ({v1},{v2}) 相交的三角形，检查边是否已存在")
            if self._is_boundary_edge_in_mesh(v1, v2):
                verbose(f"    [信息] 边 ({v1},{v2}) 已存在")
                return True
            else:
                # 边不存在且没有相交三角形，尝试使用 Steiner 点
                verbose(f"    [信息] 边 ({v1},{v2}) 不存在，尝试使用 Steiner 点恢复")
                return self._recover_edge_with_steiner_point(v1, v2)
        
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

        # 找到最近的三角形
        min_dist = float('inf')
        closest_tri = None
        for tri in self.triangles:
            if tri.is_deleted():
                continue

            # 计算点到三角形的距离
            dist = self._point_to_triangle_distance(point, tri)
            if dist < min_dist:
                min_dist = dist
                closest_tri = tri

        if closest_tri is None:
            verbose(f"    [警告] 没有找到最近的三角形，无法修复点 {point_idx}")
            return False

        verbose(f"    [信息] 修复点 {point_idx}, 最近三角形 {closest_tri.idx}, 距离 {min_dist:.4f}")

        # 删除最近的三角形
        closest_tri.set_deleted(True)
        self.triangles = [t for t in self.triangles if not t.is_deleted()]

        # 使用边界点和三角形顶点重新三角化
        v0, v1, v2 = closest_tri.vertices

        # 创建 3 个新三角形：(point, v0, v1), (point, v1, v2), (point, v2, v0)
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


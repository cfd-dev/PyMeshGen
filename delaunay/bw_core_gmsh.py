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
        self.protected_edges: Set[frozenset] = set()
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
                set(),  # 初始时没有保护边
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

        return self.triangles
    
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
        max_iter_limit = 7000  # 介于A(5000)和B(10000)之间

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
            if len(self.points) > 50000:
                verbose("达到最大节点数限制 (50000)，停止插点")
                break

            # 根据版本设置最大迭代次数
            if refinement_mode == 'A':
                max_iter_limit = 5000
            elif refinement_mode == 'B':
                max_iter_limit = 10000
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
            
            # 检查插入点是否在孔洞内
            in_hole = False
            if self.holes:
                for hole in self.holes:
                    if point_in_polygon(new_point, hole):
                        in_hole = True
                        break
            
            if in_hole:
                failed_tri_ids.add(id(worst_tri))
                worst_tri.set_deleted(True)
                consecutive_failures += 1
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

        关键修复：
        - 不删除任何包含边界节点的三角形
        - 不删除任何包含 protected_edges 的三角形（增强保护）
        - 只删除所有顶点都在孔洞内部的三角形
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
            if tri.is_deleted():
                continue

            # 关键修复1：如果三角形有任何边界节点，跳过
            # 这些三角形在孔洞边界上，需要保留
            has_boundary_vertex = any(v < self.boundary_count for v in tri.vertices)
            if has_boundary_vertex:
                continue

            # 关键修复2：如果三角形包含 protected_edges 的任何边，跳过
            # 这是更严格的保护机制，确保边界边不被误删
            has_protected_edge = False
            for j in range(3):
                v1 = tri.vertices[j]
                v2 = tri.vertices[(j + 1) % 3]
                if self._is_protected_edge(v1, v2):
                    has_protected_edge = True
                    break
            if has_protected_edge:
                continue

            # 所有顶点都是内部节点，且没有保护边，检查是否完全在孔洞内
            # 需要质心和所有顶点都在孔洞内才删除
            centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
            centroid_in_hole = False
            for h in holes_to_use:
                if point_in_polygon(centroid, h):
                    centroid_in_hole = True
                    break

            if not centroid_in_hole:
                continue

            # 质心在孔洞内，检查顶点（至少一个顶点在孔洞内才删除）
            any_vert_in_hole = False
            for vert_idx in tri.vertices:
                vert = self.points[vert_idx]
                for h in holes_to_use:
                    if point_in_polygon(vert, h):
                        any_vert_in_hole = True
                        break
                if any_vert_in_hole:
                    break

            if any_vert_in_hole:
                triangles_to_remove.append(i)

        for i in reversed(triangles_to_remove):
            self.triangles[i].set_deleted(True)

        removed_tri_count = len(triangles_to_remove)
        if removed_tri_count > 0:
            verbose(f"  删除孔洞内三角形: {removed_tri_count} 个")

        # 清理已删除的三角形
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
        
        这个方法应该在主循环插点之后调用。
        """
        verbose("开始 Constrained Delaunay 边界边恢复...")
        
        recovered_count = 0
        failed_count = 0
        missing_count = 0
        
        # 先统计缺失的边界边
        missing_edges = []
        for edge in self.protected_edges:
            v1, v2 = list(edge)
            if not self._is_boundary_edge_in_mesh(v1, v2):
                missing_edges.append((v1, v2))
                missing_count += 1
        
        if missing_count == 0:
            verbose("所有边界边已存在，无需恢复")
            return
        
        verbose(f"  检测到 {missing_count} 条缺失的边界边")
        
        # 尝试恢复每条缺失的边
        for v1, v2 in missing_edges:
            verbose(f"  尝试恢复边界边 ({v1}, {v2})...")
            
            # 尝试通过翻转恢复
            if self._recover_single_constrained_edge(v1, v2):
                recovered_count += 1
                verbose(f"    ✓ 成功恢复边界边 ({v1}, {v2})")
            else:
                failed_count += 1
                verbose(f"    ✗ 无法恢复边界边 ({v1}, {v2})")
        
        verbose(f"边界边恢复完成: 成功 {recovered_count}/{missing_count}, 失败 {failed_count}")
    
    def _recover_single_constrained_edge(self, v1: int, v2: int) -> bool:
        """恢复单条约束边(v1, v2)。
        
        使用经典的 CDT 边翻转策略（不插入 Steiner 点）：
        1. 找到与线段(v1, v2)相交的所有边
        2. 依次翻转这些边
        3. 直到(v1, v2)成为三角形的边
        
        返回 True 如果成功恢复，False 如果无法恢复。
        """
        # 检查边是否已经存在
        if self._is_boundary_edge_in_mesh(v1, v2):
            return True
        
        segment_start = self.points[v1]
        segment_end = self.points[v2]
        
        max_flips = 200  # 防止无限循环
        flip_count = 0
        
        while flip_count < max_flips:
            # 找到所有与线段(v1, v2)相交的边
            intersecting_edge = None
            
            for tri in self.triangles:
                if tri.is_deleted():
                    continue
                
                verts = tri.vertices
                for i in range(3):
                    e1 = verts[i]
                    e2 = verts[(i + 1) % 3]
                    
                    # 跳过v1和v2本身
                    if e1 == v1 or e1 == v2 or e2 == v1 or e2 == v2:
                        continue
                    
                    # 跳过受保护的边界边
                    if self._is_protected_edge(e1, e2):
                        continue
                    
                    # 检查边(e1, e2)是否与线段(v1, v2)相交
                    if self._segments_intersect_strict(
                        segment_start, segment_end,
                        self.points[e1], self.points[e2]
                    ):
                        intersecting_edge = (e1, e2)
                        break
                
                if intersecting_edge:
                    break
            
            if intersecting_edge is None:
                # 没有相交边，检查是否已恢复
                return self._is_boundary_edge_in_mesh(v1, v2)
            
            n1, n2 = intersecting_edge
            
            # 尝试翻转这条边
            if self._flip_edge(n1, n2):
                flip_count += 1
                
                # 检查边(v1, v2)是否已经出现
                if self._is_boundary_edge_in_mesh(v1, v2):
                    return True
            else:
                # 翻转失败（可能是非凸四边形或保护边）
                break
        
        # 检查最终是否恢复
        return self._is_boundary_edge_in_mesh(v1, v2)
    
    def _find_edges_intersecting_segment(self, v1: int, v2: int) -> List[Tuple[int, int]]:
        """找到所有与线段(v1, v2)相交的边。"""
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
                
                # 跳过受保护的边
                if self._is_protected_edge(e1, e2):
                    continue
                
                # 检查边(e1, e2)是否与线段(v1, v2)相交
                if self._segments_intersect_strict(
                    segment_start, segment_end,
                    self.points[e1], self.points[e2]
                ):
                    edge_tuple = (e1, e2) if e1 < e2 else (e2, e1)
                    if edge_tuple not in intersecting:
                        intersecting.append(edge_tuple)
        
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

    def _insert_steiner_point_on_edge(self, v1: int, v2: int):
        """在边界边上插入Steiner点（中点），然后尝试恢复。

        这是Gmsh/TetGen中边恢复的最后手段：
        1. 在边的中点插入新顶点
        2. 删除与原始边(v1, v2)相交的所有三角形
        3. 重新三角化空洞区域
        4. 递归恢复两条子边
        """
        # 计算中点
        p1, p2 = self.points[v1], self.points[v2]
        mid_point = (p1 + p2) / 2.0

        # 添加新点
        mid_idx = len(self.points)
        self.points = np.vstack([self.points, mid_point])

        verbose(f"    在边({v1},{v2})中点插入Steiner点 #{mid_idx}")

        # 查找所有与线段(v1, v2)相交的三角形
        triangles_to_remove = []
        for tri in self.triangles:
            if tri.is_deleted():
                continue
            # 检查三角形是否与线段(v1, v2)相交
            if self._triangle_intersects_segment(tri, v1, v2):
                triangles_to_remove.append(tri)

        # 删除相交三角形
        for tri in triangles_to_remove:
            tri.set_deleted(True)

        # 清理已删除三角形
        self.triangles = [tri for tri in self.triangles if not tri.is_deleted()]

        # 找到空洞边界（只出现一次的边）
        cavity_boundary = self._find_cavity_boundary_edges()

        # 创建新三角形：将中点与空洞边界的每条边连接
        new_triangles = []
        for edge in cavity_boundary:
            a, b = edge
            # 跳过已经包含中点的边
            if a == mid_idx or b == mid_idx:
                continue
            # 创建新三角形
            new_tri = MTri3(mid_idx, a, b, idx=self._next_tri_id())
            self._compute_circumcircle(new_tri)
            new_triangles.append(new_tri)

        self.triangles.extend(new_triangles)

        # 重建邻接关系
        from delaunay.bw_types import build_adjacency_from_triangles
        build_adjacency_from_triangles(self.triangles)

        verbose(f"    创建了 {len(new_triangles)} 个新三角形")

        # 递归恢复两条子边
        self._recover_single_edge_enhanced(v1, mid_idx)
        self._recover_single_edge_enhanced(mid_idx, v2)

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
        is_convex = (c1 > 0 and c2 > 0 and c3 > 0 and c4 > 0) or \
                    (c1 < 0 and c2 < 0 and c3 < 0 and c4 < 0)
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
        
        points = self.points.copy()
        simplices = np.array([tri.vertices for tri in self.triangles])
        boundary_mask = np.zeros(len(points), dtype=bool)
        boundary_mask[:self.boundary_count] = True
        
        verbose("网格生成完成:")
        verbose(f"  - 总节点数: {len(points)}")
        verbose(f"  - 边界节点: {np.sum(boundary_mask)}")
        verbose(f"  - 内部节点: {len(points) - np.sum(boundary_mask)}")
        verbose(f"  - 三角形数: {len(simplices)}")
        
        timer.show_to_console("Gmsh Bowyer-Watson 网格生成完成")
        return points, simplices, boundary_mask

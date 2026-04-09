"""
Bowyer-Watson Delaunay 网格生成器

基于 Bowyer-Watson 算法的二维三角形网格生成，支持：
1. 以离散边界网格作为输入
2. 使用 QuadtreeSizing 尺寸场控制网格尺寸
3. 自动生成高质量三角形网格

优化特性：
- 增量式点插入（避免全量重剖分）
- 空间索引加速最近邻搜索
- 缓存三角形质量和尺寸计算
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from math import sqrt
from scipy.spatial import KDTree
from collections import Counter

from utils.message import info, debug, warning, verbose
from utils.timer import TimeSpan
from utils.geom_toolkit import point_in_polygon, is_polygon_clockwise, point_to_segment_distance


class Triangle:
    """三角形单元类"""

    __slots__ = ['vertices', 'circumcenter', 'circumradius', 'idx', 'circumcircle_valid', 'quality', 'quality_valid', 'circumcircle_bbox']

    def __init__(self, p1: int, p2: int, p3: int, idx: int = -1):
        # 顶点索引排序以便于去重和比较
        self.vertices = tuple(sorted([p1, p2, p3]))
        self.circumcenter = None  # 外接圆圆心
        self.circumradius = None  # 外接圆半径
        self.idx = idx  # 三角形索引
        self.circumcircle_valid = False  # 外接圆是否已计算并有效
        self.quality = 0.0  # 三角形质量缓存
        self.quality_valid = False  # 质量是否已计算并有效
        self.circumcircle_bbox = None  # 外接圆的包围盒（用于空间索引）

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return False
        return self.vertices == other.vertices

    def __hash__(self):
        return hash(self.vertices)

    def __contains__(self, point_idx: int) -> bool:
        return point_idx in self.vertices

    def get_edges(self) -> List[Tuple[int, int]]:
        """返回三角形的三条边（排序后的元组）"""
        v0, v1, v2 = self.vertices
        return [
            (v0, v1),
            (v1, v2),
            (v0, v2),
        ]

    def __repr__(self):
        return f"Triangle({self.vertices})"


class BowyerWatsonMeshGenerator:
    """
    Bowyer-Watson Delaunay 网格生成器
    
    特性：
    - 支持从离散边界网格输入
    - 集成 QuadtreeSizing 尺寸场控制
    - 支持尺寸场和全局尺寸控制
    - Laplacian 平滑优化
    """

    def __init__(
        self,
        boundary_points: np.ndarray,
        boundary_edges: Optional[List[Tuple[int, int]]] = None,
        sizing_system=None,
        max_edge_length: Optional[float] = None,
        smoothing_iterations: int = 3,
        seed: Optional[int] = None,
        holes: Optional[List[np.ndarray]] = None,
    ):
        """
        初始化 Bowyer-Watson 网格生成器

        参数:
            boundary_points: 边界点坐标数组，形状为 (N, 2)
            boundary_edges: 边界边列表，每个元素为 (idx1, idx2)
            sizing_system: QuadtreeSizing 尺寸场对象（可选）
            max_edge_length: 全局最大边长（可选，优先使用 sizing_system）
            smoothing_iterations: Laplacian 平滑迭代次数
            seed: 随机种子
            holes: 孔洞边界列表，每个孔洞是一个点数组，形状为 (M, 2)
        """
        self.original_points = boundary_points.copy()
        # 保护的边界边集合（使用 frozenset 以便快速查找）
        self.protected_edges = set()  # Set[frozenset({idx1, idx2})]
        for edge in (boundary_edges or []):
            self.protected_edges.add(frozenset(edge))
        self.boundary_edges = boundary_edges or []  # 保留用于调试
        self.sizing_system = sizing_system
        self.max_edge_length = max_edge_length
        self.smoothing_iterations = smoothing_iterations
        self.seed = seed
        self.holes = holes or []  # 孔洞边界列表

        if seed is not None:
            np.random.seed(seed)

        # 工作状态变量
        self.points = None  # 所有点（边界+内部）
        self.triangles = []  # 三角形列表
        self.boundary_mask = None  # 边界点掩码
        self.boundary_count = 0  # 边界点数量
        self._kdtree = None  # KD-tree 用于加速最近邻搜索

    def _is_protected_edge(self, v1: int, v2: int) -> bool:
        """
        检查边是否是受保护的边界边

        参数:
            v1, v2: 边的两个顶点索引

        返回:
            True 如果边是受保护的边界边
        """
        return frozenset({v1, v2}) in self.protected_edges

    def _compute_circumcircle(self, tri: Triangle) -> Tuple[np.ndarray, float]:
        """
        计算三角形的外接圆（带缓存）

        返回:
            (center, radius): 外接圆圆心和半径
        """
        # 如果已计算且有效，直接返回缓存值
        if tri.circumcircle_valid and tri.circumcenter is not None and tri.circumradius is not None:
            return tri.circumcenter, tri.circumradius

        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        ax, ay = float(p1[0]), float(p1[1])
        bx, by = float(p2[0]), float(p2[1])
        cx, cy = float(p3[0]), float(p3[1])

        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

        if abs(d) < 1e-12:
            # 三点共线或非常接近，返回重心
            center = np.array([(ax + bx + cx) / 3.0, (ay + by + cy) / 3.0])
            radius = float(np.linalg.norm(p1 - center))
        else:
            ux = (
                (ax**2 + ay**2) * (by - cy)
                + (bx**2 + by**2) * (cy - ay)
                + (cx**2 + cy**2) * (ay - by)
            ) / d
            uy = (
                (ax**2 + ay**2) * (cx - bx)
                + (bx**2 + by**2) * (ax - cx)
                + (cx**2 + cy**2) * (bx - ax)
            ) / d

            center = np.array([ux, uy])
            radius = float(np.linalg.norm(p1 - center))

        # 缓存结果
        tri.circumcenter = center
        tri.circumradius = radius
        tri.circumcircle_valid = True
        # 计算包围盒用于空间索引
        tri.circumcircle_bbox = (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        )

        return center, radius

    def _point_in_circumcircle_bbox(self, point: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
        """
        快速检查点是否可能在包围盒内（初步筛选）
        """
        return (bbox[0] <= point[0] <= bbox[2]) and (bbox[1] <= point[1] <= bbox[3])

    def _is_point_in_valid_region(self, point: np.ndarray) -> bool:
        """
        检查点是否在有效区域内（外边界内且不在任何孔洞内）

        参数:
            point: 待检查的点，形状为 (2,)

        返回:
            True 如果点在有效区域内，False 否则
        """
        # 如果没有孔洞，只需检查点是否在外边界内
        if not self.holes:
            return True  # 假设外边界包围盒检查已通过

        # 检查点是否在任何孔洞内
        for hole in self.holes:
            if point_in_polygon(point, hole):
                return False  # 点在孔洞内，无效

        return True

    def _point_in_circumcircle(self, point: np.ndarray, tri: Triangle) -> bool:
        """
        检查点是否在三角形的外接圆内（含边界）
        """
        # 确保外接圆已计算
        if not tri.circumcircle_valid:
            self._compute_circumcircle(tri)

        distance = float(np.linalg.norm(point - tri.circumcenter))
        # 使用小的容差值避免浮点误差
        return distance < tri.circumradius * (1.0 + 1e-10)

    def _create_super_triangle(self) -> Triangle:
        """
        创建包含所有点的超级三角形
        """
        min_x = np.min(self.points[:, 0])
        max_x = np.max(self.points[:, 0])
        min_y = np.min(self.points[:, 1])
        max_y = np.max(self.points[:, 1])

        dx = max_x - min_x
        dy = max_y - min_y
        delta = max(dx, dy) * 10.0

        # 超级三角形的三个顶点索引
        p1 = len(self.points)
        p2 = len(self.points) + 1
        p3 = len(self.points) + 2

        # 添加超级三角形顶点
        super_verts = np.array([
            [min_x - delta, min_y - delta],
            [max_x + delta, min_y - delta],
            [(min_x + max_x) / 2.0, max_y + 3.0 * delta],
        ])
        self.points = np.vstack([self.points, super_verts])

        return Triangle(p1, p2, p3)

    def _triangulate(self) -> List[Triangle]:
        """
        执行 Bowyer-Watson 三角剖分算法

        算法流程：
        1. 创建超级三角形包含所有点
        2. 逐点插入，删除外接圆包含该点的三角形
        3. 重新连接形成新的三角形
        4. 删除包含超级三角形顶点的三角形

        返回:
            三角形列表
        """
        # 创建超级三角形
        super_tri = self._create_super_triangle()
        triangles = [super_tri]

        # 获取真实点数量（不包括超级三角形顶点）
        real_point_count = len(self.points) - 3

        # 逐点插入
        for i in range(real_point_count):
            point = self.points[i]

            # 找到所有外接圆包含当前点的三角形
            bad_triangles = []
            for tri in triangles:
                if self._point_in_circumcircle(point, tri):
                    bad_triangles.append(tri)

            # 找到坏三角形的边界边（不共享的边）- 使用 Counter 优化
            edge_counter = Counter()
            for tri in bad_triangles:
                for edge in tri.get_edges():
                    edge_key = tuple(sorted(edge))
                    edge_counter[edge_key] += 1

            # 只出现一次的边是边界边
            polygon_edges = [
                edge for edge, count in edge_counter.items() if count == 1
            ]

            # 删除坏三角形
            bad_set = set(id(tri) for tri in bad_triangles)
            triangles = [tri for tri in triangles if id(tri) not in bad_set]

            # 创建新三角形连接边界边和新点
            for edge in polygon_edges:
                new_tri = Triangle(edge[0], edge[1], i)
                # 预计算外接圆
                self._compute_circumcircle(new_tri)
                triangles.append(new_tri)

        # 移除包含超级三角形顶点的三角形
        super_verts = {super_tri.vertices[0], super_tri.vertices[1], super_tri.vertices[2]}
        final_triangles = [
            tri for tri in triangles
            if not any(v in super_verts for v in tri.vertices)
        ]

        # 移除超级三角形顶点
        self.points = self.points[:-3]
        self.triangles = final_triangles

        return final_triangles

    def _compute_triangle_quality(self, tri: Triangle) -> float:
        """
        计算三角形质量（基于纵横比）（带缓存）

        质量定义：quality = 2 * r_inscribed / r_circumscribed
        值越接近 1，三角形质量越好（等边三角形质量为 1）
        """
        # 如果质量已计算且有效，直接返回缓存值
        if tri.quality_valid:
            return tri.quality

        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        a = float(np.linalg.norm(p2 - p1))
        b = float(np.linalg.norm(p3 - p2))
        c = float(np.linalg.norm(p1 - p3))

        # 半周长
        s = (a + b + c) / 2.0

        # 面积（海伦公式）
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq < 0:
            quality = 0.0
        else:
            area = sqrt(max(area_sq, 0.0))

            if area < 1e-12:
                quality = 0.0
            else:
                # 内切圆半径
                inscribed_radius = area / s

                # 外接圆半径
                if not tri.circumcircle_valid:
                    self._compute_circumcircle(tri)
                circumscribed_radius = tri.circumradius

                if circumscribed_radius < 1e-12:
                    quality = 0.0
                else:
                    # 质量 = 2 * r_inscribed / r_circumscribed
                    quality = 2.0 * inscribed_radius / circumscribed_radius
                    quality = min(quality, 1.0)

        # 缓存结果
        tri.quality = quality
        tri.quality_valid = True

        return quality

    def _compute_triangle_centroid(self, tri: Triangle) -> np.ndarray:
        """
        计算三角形的质心
        """
        return np.mean([
            self.points[tri.vertices[0]],
            self.points[tri.vertices[1]],
            self.points[tri.vertices[2]],
        ], axis=0)

    def _get_target_size_for_triangle(self, tri: Triangle) -> Optional[float]:
        """
        获取三角形的目标尺寸

        优先级：
        1. sizing_system（尺寸场）
        2. max_edge_length（全局尺寸）
        3. None（不限制）
        """
        if self.sizing_system is not None:
            # 使用尺寸场：取三角形中心处的尺寸
            center = np.mean([
                self.points[tri.vertices[0]],
                self.points[tri.vertices[1]],
                self.points[tri.vertices[2]],
            ], axis=0)
            try:
                return self.sizing_system.spacing_at(center)
            except Exception as e:
                debug(f"获取尺寸场失败: {e}，使用全局尺寸")
                return self.max_edge_length
        else:
            return self.max_edge_length

    def _insert_point_incremental(self, point_idx: int, triangles: List[Triangle]) -> List[Triangle]:
        """
        增量式插入单个点到现有三角剖分中

        这是优化的关键：只更新受影响的三角形，而不是重新剖分所有点

        参数:
            point_idx: 新插入点的索引
            triangles: 当前三角形列表

        返回:
            更新后的三角形列表
        """
        point = self.points[point_idx]

        # 找到所有外接圆包含新点的三角形（bad triangles）
        bad_triangles = []
        for tri in triangles:
            if self._point_in_circumcircle(point, tri):
                bad_triangles.append(tri)

        if not bad_triangles:
            return triangles

        # 找到 bad triangles 的边界边（不共享的边）- 使用 Counter 优化
        edge_counter = Counter()
        for tri in bad_triangles:
            for edge in tri.get_edges():
                edge_key = tuple(sorted(edge))
                edge_counter[edge_key] += 1

        # 只出现一次的边是边界边
        polygon_edges = [
            edge for edge, count in edge_counter.items() if count == 1
        ]

        # 删除 bad triangles
        bad_set = set(id(tri) for tri in bad_triangles)
        triangles = [tri for tri in triangles if id(tri) not in bad_set]

        # 创建新三角形连接边界边和新点
        for edge in polygon_edges:
            new_tri = Triangle(edge[0], edge[1], point_idx)
            # 预计算外接圆
            self._compute_circumcircle(new_tri)
            triangles.append(new_tri)

        return triangles

    def _insert_point_with_boundary_protection(self, point_idx: int, triangles: List[Triangle]) -> Tuple[List[Triangle], bool]:
        """
        增量式插入单个点，同时保护边界边

        参考C++的recurFindCavityAniso和insertVertexB逻辑：
        - 在查找空腔时，不跨越受保护的边界边
        - 如果点插入会破坏边界边，则拒绝插入

        参数:
            point_idx: 新插入点的索引
            triangles: 当前三角形列表

        返回:
            (更新后的三角形列表, 是否成功插入)
        """
        point = self.points[point_idx]

        # 找到所有外接圆包含新点的三角形（bad triangles）
        bad_triangles = []
        for tri in triangles:
            if self._point_in_circumcircle(point, tri):
                bad_triangles.append(tri)

        if not bad_triangles:
            return triangles, False

        # 找到 bad triangles 的边界边（不共享的边）
        edge_counter = Counter()
        for tri in bad_triangles:
            for edge in tri.get_edges():
                edge_key = tuple(sorted(edge))
                edge_counter[edge_key] += 1

        # 只出现一次的边是空腔边界边
        cavity_boundary_edges = [
            edge for edge, count in edge_counter.items() if count == 1
        ]

        # 检查是否有受保护的边界边会被破坏
        # 受保护的边界边是指在bad_triangles内部但在cavity_boundary_edges中不存在的边
        protected_edges_in_cavity = []
        for tri in bad_triangles:
            for edge in tri.get_edges():
                edge_key = tuple(sorted(edge))
                # 如果这条边是受保护的边界边
                if self._is_protected_edge(edge[0], edge[1]):
                    # 检查它是否在空腔边界上
                    if edge_key not in cavity_boundary_edges:
                        # 这条边会被破坏，需要保护
                        protected_edges_in_cavity.append(edge_key)

        # 如果有受保护的边界边会被破坏，采用保守策略：
        # 仍然插入点，但需要在插入后恢复这些边界边
        # 删除 bad triangles
        bad_set = set(id(tri) for tri in bad_triangles)
        triangles = [tri for tri in triangles if id(tri) not in bad_set]

        # 创建新三角形连接边界边和新点
        for edge in cavity_boundary_edges:
            new_tri = Triangle(edge[0], edge[1], point_idx)
            # 预计算外接圆
            self._compute_circumcircle(new_tri)
            triangles.append(new_tri)

        # 返回受影响的受保护边列表，用于后续恢复
        return triangles, len(protected_edges_in_cavity) > 0

    def _insert_points_iteratively(self, target_triangle_count: Optional[int] = None):
        """
        迭代插入内部点，使用外接圆圆心策略

        优化策略：
        - 使用增量式插入（避免全量重剖分）
        - 使用 KD-tree 加速最近邻搜索
        - 批量处理候选三角形
        - 缓存三角形质量计算结果
        - 预计算边界范围

        终止条件（满足任一即停止）：
        1. 达到目标三角形数量
        2. 所有三角形都满足尺寸和质量要求
        3. 达到最大节点数限制
        """
        boundary_count = self.boundary_count
        initial_point_count = len(self.points)
        inserted_points = 0  # 已插入的内部点计数器

        # 根据 Euler 公式计算目标节点数（如果指定了目标三角形数）
        target_total_points = None
        if target_triangle_count is not None:
            target_total_points = (target_triangle_count + 2 + boundary_count) // 2

        # 预计算边界范围和最小距离阈值
        x_min = np.min(self.original_points[:, 0])
        x_max = np.max(self.original_points[:, 0])
        y_min = np.min(self.original_points[:, 1])
        y_max = np.max(self.original_points[:, 1])
        margin = 0.001 * max(x_max - x_min, y_max - y_min)

        # 预计算最小距离阈值
        if len(self.original_points) > 1:
            avg_edge = np.mean([
                float(np.linalg.norm(self.original_points[1] - self.original_points[0]))
            ])
            min_dist_threshold = 0.01 * avg_edge if avg_edge > 0 else 0.01
        else:
            min_dist_threshold = 0.01

        max_iterations = 0
        kdtree_rebuild_interval = 100  # 增加 KD-tree 重建间隔
        last_kdtree_build = 0
        consecutive_failures = 0  # 连续插入失败次数
        max_consecutive_failures = 500  # 最大连续失败次数
        failed_triangles = set()  # 记录失败的三角形（用顶点元组标识）

        while True:
            max_iterations += 1

            # 定期输出进度（每 10 次迭代输出一次）
            if max_iterations % 10 == 0 or max_iterations == 1:
                current_triangles = len(self.triangles)
                current_points = len(self.points)
                current_inserted = current_points - initial_point_count
                verbose(f"  [进度] 迭代 {max_iterations} | "
                       f"节点: {current_points} (边界: {boundary_count}, 内部: {current_inserted}) | "
                       f"三角形: {current_triangles}")

            # 安全检查：防止无限循环
            if len(self.points) > 100000:
                warning(f"达到最大节点数限制 (100000)，停止插点")
                break

            if max_iterations > 50000:
                warning(f"达到最大迭代次数限制 (50000)，停止插点")
                break

            # 检查终止条件 1: 达到目标三角形数量
            if target_total_points is not None and len(self.points) >= target_total_points:
                verbose(f"  [进度] 达到目标节点数: {len(self.points)}")
                break

            if len(self.triangles) == 0:
                break

            # 寻找需要细分的三角形 - 优化：使用缓存
            worst_quality = float('inf')
            worst_triangle = None
            needs_refinement = False

            # 缓存点坐标访问
            points = self.points

            for tri in self.triangles:
                # 跳过已经标记为失败的三角形
                if tri.vertices in failed_triangles:
                    continue
                
                # 计算三角形质量（使用缓存）
                quality = self._compute_triangle_quality(tri)

                # 计算最大边长（优化：向量化计算）
                v0, v1, v2 = tri.vertices
                edge_lengths = [
                    float(np.linalg.norm(points[v1] - points[v0])),
                    float(np.linalg.norm(points[v2] - points[v1])),
                    float(np.linalg.norm(points[v0] - points[v2])),
                ]
                max_edge = max(edge_lengths)

                # 获取目标尺寸
                target_size = self._get_target_size_for_triangle(tri)

                # 检查是否需要细分
                # 逻辑：需要同时满足尺寸要求和质量要求
                should_split = False
                if target_size is not None:
                    # 有尺寸场时：检查边长是否满足尺寸场要求
                    if max_edge > target_size * 1.1:
                        should_split = True
                    # 即使尺寸满足，质量太差也需要细分（保证内部单元质量）
                    elif quality < 0.3:
                        should_split = True
                else:
                    # 无尺寸场时：只检查质量
                    if quality < 0.5:
                        should_split = True

                if should_split:
                    needs_refinement = True
                    if quality < worst_quality:
                        worst_quality = quality
                        worst_triangle = tri

            # 终止条件 2: 所有三角形都满足要求
            if not needs_refinement:
                verbose(f"  [进度] 所有三角形满足尺寸和质量要求")
                break

            if worst_triangle is None:
                break

            # 确保外接圆已计算
            if not worst_triangle.circumcircle_valid:
                self._compute_circumcircle(worst_triangle)

            new_point = worst_triangle.circumcenter.copy()

            # 检查新点是否在边界范围内
            if not (x_min - margin < new_point[0] < x_max + margin and
                    y_min - margin < new_point[1] < y_max + margin):
                # 如果外接圆中心在边界外，尝试在三角形内部随机生成点
                p1 = points[worst_triangle.vertices[0]]
                p2 = points[worst_triangle.vertices[1]]
                p3 = points[worst_triangle.vertices[2]]

                r1, r2 = np.random.rand(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                new_point = p1 + r1 * (p2 - p1) + r2 * (p3 - p1)

            # 优化：插点阶段不检查孔洞，允许在孔洞内插点
            # 孔洞内的点和单元将在后处理阶段一次性删除
            # （这样可以避免大量迭代浪费在尝试插入孔洞内的点）

            # 使用 KD-tree 加速最近邻搜索（优化重建频率）
            if len(self.points) > 0:
                # 构建 KD-tree（优化频率）
                if max_iterations == 1 or (max_iterations - last_kdtree_build) >= kdtree_rebuild_interval:
                    self._kdtree = KDTree(self.points)
                    last_kdtree_build = max_iterations

                min_dist, _ = self._kdtree.query(new_point)
            else:
                min_dist = float('inf')

            # 如果距离足够远，则添加新点
            if min_dist > min_dist_threshold:
                # 添加新点
                new_point_idx = len(self.points)
                self.points = np.vstack([self.points, new_point])
                inserted_points += 1

                # 增量式插入（关键优化：避免全量重剖分）
                self.triangles = self._insert_point_incremental(new_point_idx, self.triangles)
            else:
                # 点距离过近，标记并跳过当前三角形
                failed_triangles.add(worst_triangle.vertices)
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    verbose(f"  [进度] 连续失败 {consecutive_failures} 次，停止插点")
                    break
                continue  # 继续下一轮循环，尝试其他三角形
        
        # 最终进度输出
        final_triangles = len(self.triangles)
        final_points = len(self.points)
        final_inserted = final_points - initial_point_count
        verbose(f"  [完成] 插点完成 | "
               f"节点: {final_points} (边界: {boundary_count}, 内部: {final_inserted}) | "
               f"三角形: {final_triangles}")

    def _laplacian_smoothing(self, iterations: int = 3, alpha: float = 0.5):
        """
        Laplacian 平滑优化网格

        通过迭代调整内部点位置来改善网格质量
        边界点保持不动

        参数:
            iterations: 迭代次数
            alpha: 平滑系数 (0-1)
        """
        if iterations <= 0:
            return

        smoothed_points = self.points.copy()

        for iteration in range(iterations):
            new_points = smoothed_points.copy()

            # 构建邻接表
            neighbor_dict = {}
            for tri in self.triangles:
                for i in range(3):
                    v = tri.vertices[i]
                    if v not in neighbor_dict:
                        neighbor_dict[v] = set()
                    neighbor_dict[v].add(tri.vertices[(i + 1) % 3])
                    neighbor_dict[v].add(tri.vertices[(i + 2) % 3])

            # 更新内部点位置（边界点不动）
            for v, neighbors in neighbor_dict.items():
                # 边界点不移动
                if self.boundary_mask[v]:
                    continue

                if len(neighbors) > 0:
                    neighbor_center = np.mean(
                        [smoothed_points[n] for n in neighbors], axis=0
                    )
                    new_points[v] = smoothed_points[v] + alpha * (
                        neighbor_center - smoothed_points[v]
                    )

            smoothed_points = new_points

        self.points = smoothed_points

    def _remove_hole_triangles(self) -> int:
        """
        删除孔洞内的三角形和孤立节点

        处理流程：
        1. 删除质心在孔洞内的三角形，或任何顶点在孔洞内的三角形
        2. 删除孔洞内的孤立节点（不被任何三角形引用的节点）
        3. 保持孔洞边界点（原始边界点不会被删除）

        返回:
            删除的三角形数量
        """
        if not self.holes:
            verbose(f"  _remove_hole_triangles: 没有孔洞，提前返回")
            return 0
        
        verbose(f"  _remove_hole_triangles: 当前三角形数={len(self.triangles)}, 孔洞数={len(self.holes)}, 边界点数={self.boundary_count}")
        
        # 重要：确保孔洞多边形是逆时针方向（point_in_polygon要求）
        # 检查并修复孔洞方向
        from utils.geom_toolkit import is_polygon_clockwise
        fixed_holes = []
        for hole_idx, hole in enumerate(self.holes):
            if is_polygon_clockwise(hole):
                # 如果是顺时针，反转
                verbose(f"    [HOLE-{hole_idx}] 检测到顺时针，反转为逆时针")
                fixed_holes.append(hole[::-1])
            else:
                fixed_holes.append(hole.copy())
        
        # 使用修复后的孔洞
        holes_to_use = fixed_holes

        # === 第 1 步：删除孔洞内的三角形 ===
        triangles_to_remove = []

        for i, tri in enumerate(self.triangles):
            # 计算三角形质心
            centroid = self._compute_triangle_centroid(tri)

            # 检查质心是否在任何孔洞内
            in_hole = False
            for hole in holes_to_use:
                if point_in_polygon(centroid, hole):
                    in_hole = True
                    break
            
            # 如果质心不在孔洞内，检查是否有任何顶点在孔洞内
            # 这样可以捕获那些部分在孔洞内的三角形
            if not in_hole:
                for vert_idx in tri.vertices:
                    vert = self.points[vert_idx]
                    for hole in holes_to_use:
                        if point_in_polygon(vert, hole):
                            # 检查这个顶点是否严格在孔洞内部（不在边界上）
                            # 如果是孔洞边界点，不应该删除
                            if vert_idx >= self.boundary_count:
                                in_hole = True
                                break
                    if in_hole:
                        break

            if in_hole:
                triangles_to_remove.append(i)

        # 从后往前删除，避免索引变化
        for i in reversed(triangles_to_remove):
            del self.triangles[i]

        removed_tri_count = len(triangles_to_remove)
        if removed_tri_count > 0:
            verbose(f"  删除孔洞内三角形: {removed_tri_count} 个")

        # === 第 2 步：删除孔洞内的孤立节点 ===
        # 构建剩余三角形中所有被引用的节点集合
        used_nodes = set()
        for tri in self.triangles:
            used_nodes.update(tri.vertices)

        # 找到未被引用的节点（孤立节点）
        all_nodes = set(range(len(self.points)))
        orphan_nodes = all_nodes - used_nodes

        if orphan_nodes:
            # 检查孤立节点是否在孔洞内
            nodes_to_remove = []
            for node_idx in orphan_nodes:
                # 跳过原始边界点（它们可能是孔洞边界）
                if node_idx < self.boundary_count:
                    continue

                point = self.points[node_idx]
                in_hole = False
                for hole in self.holes:
                    if point_in_polygon(point, hole):
                        in_hole = True
                        break

                # 只删除在孔洞内的孤立节点
                if in_hole:
                    nodes_to_remove.append(node_idx)

            if nodes_to_remove:
                verbose(f"  删除孔洞内孤立节点: {len(nodes_to_remove)} 个")
                # 从 points 数组中删除
                keep_mask = np.ones(len(self.points), dtype=bool)
                keep_mask[nodes_to_remove] = False
                self.points = self.points[keep_mask]

                # 更新 boundary_mask（如果已存在且长度匹配）
                if len(self.boundary_mask) == len(keep_mask):
                    self.boundary_mask = self.boundary_mask[keep_mask]

                # 更新所有三角形的顶点索引（需要重新映射）
                index_map = np.cumsum(keep_mask.astype(int)) - 1
                for tri in self.triangles:
                    new_vertices = tuple(int(index_map[v]) for v in tri.vertices)
                    tri.vertices = new_vertices
                
                # 重要：boundary_count不受影响，因为我们只删除了非边界点
                # boundary_count保持原值不变

        return removed_tri_count

    def _recover_boundary_edges(self) -> int:
        """
        恢复丢失的边界边

        参考C++的recoverboundary逻辑：
        1. 检查所有受保护的边界边是否存在于当前三角剖分中
        2. 对于缺失的边，通过局部重剖分恢复
        3. 使用边翻转（edge flip）和点插入策略

        返回:
            恢复的边界边数量
        """
        if not self.protected_edges:
            return 0

        recovered_count = 0
        missing_edges = []

        # 第1步：检查哪些边界边丢失了
        for protected_edge in self.protected_edges:
            v1, v2 = sorted(list(protected_edge))
            # 检查这条边是否在任何三角形中
            edge_exists = False
            for tri in self.triangles:
                if v1 in tri.vertices and v2 in tri.vertices:
                    edge_exists = True
                    break

            if not edge_exists:
                missing_edges.append((v1, v2))

        if not missing_edges:
            return 0

        verbose(f"  检测到 {len(missing_edges)} 条丢失的边界边")

        # 第2步：恢复每条丢失的边
        for v1, v2 in missing_edges:
            if self._recover_single_edge(v1, v2):
                recovered_count += 1

        if recovered_count > 0:
            verbose(f"  成功恢复 {recovered_count}/{len(missing_edges)} 条边界边")

        return recovered_count

    def _recover_single_edge(self, v1: int, v2: int) -> bool:
        """
        恢复单条边界边

        策略：
        1. 找到所有与v1和v2相连的三角形
        2. 检查是否可以通过边翻转恢复
        3. 如果不能，在边的中点附近插入新点

        参数:
            v1, v2: 边的两个顶点

        返回:
            True 如果边被成功恢复
        """
        p1 = self.points[v1]
        p2 = self.points[v2]

        # 找到所有包含v1或v2的三角形
        tris_with_v1 = []
        tris_with_v2 = []

        for tri in self.triangles:
            if v1 in tri.vertices:
                tris_with_v1.append(tri)
            if v2 in tri.vertices:
                tris_with_v2.append(tri)

        # 找到同时包含v1和v2的三角形（如果边已存在）
        common_tris = [tri for tri in tris_with_v1 if v2 in tri.vertices]

        if len(common_tris) > 0:
            # 边已经存在
            return True

        # 找到" crossing"三角形的集合
        # 这些三角形形成了一个从v1到v2的路径，需要被重剖分
        crossing_tris = self._find_crossing_triangles(v1, v2, tris_with_v1, tris_with_v2)

        if not crossing_tris:
            # 找不到跨越的三角形，尝试插入中点
            return self._insert_midpoint_for_edge(v1, v2)

        # 尝试通过边翻转恢复
        # 简化策略：删除跨越三角形，重新三角剖分该区域
        return self._retriangulate_for_edge_recovery(v1, v2, crossing_tris)

    def _find_crossing_triangles(self, v1: int, v2: int, 
                                  tris_with_v1: List[Triangle],
                                  tris_with_v2: List[Triangle]) -> List[Triangle]:
        """
        找到与边(v1,v2)相交的三角形集合

        参数:
            v1, v2: 目标边的顶点
            tris_with_v1: 包含v1的三角形列表
            tris_with_v2: 包含v2的三角形列表

        返回:
            跨越的三角形列表
        """
        p1 = self.points[v1]
        p2 = self.points[v2]

        crossing_tris = []

        for tri in self.triangles:
            # 获取三角形的三个顶点
            verts = tri.vertices
            tri_points = [self.points[verts[0]], self.points[verts[1]], self.points[verts[2]]]

            # 检查边(p1, p2)是否与三角形相交
            if self._edge_intersects_triangle(p1, p2, tri_points):
                crossing_tris.append(tri)

        return crossing_tris

    def _edge_intersects_triangle(self, p1: np.ndarray, p2: np.ndarray, 
                                   tri_points: List[np.ndarray]) -> bool:
        """
        检查线段(p1,p2)是否与三角形相交

        参数:
            p1, p2: 线段的两个端点
            tri_points: 三角形的三个顶点

        返回:
            True 如果线段与三角形内部相交（不包括端点）
        """
        # 简化检查：线段的中心是否在三角形内
        midpoint = (p1 + p2) / 2.0

        # 使用重心坐标检查
        v0 = tri_points[2] - tri_points[0]
        v1 = tri_points[1] - tri_points[0]
        v2 = midpoint - tri_points[0]

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # 检查点是否在三角形内（有容差）
        tolerance = 1e-10
        return (u >= -tolerance) and (v >= -tolerance) and (u + v <= 1.0 + tolerance)

    def _retriangulate_for_edge_recovery(self, v1: int, v2: int, 
                                          crossing_tris: List[Triangle]) -> bool:
        """
        通过局部重剖分恢复边界边

        参数:
            v1, v2: 目标边的顶点
            crossing_tris: 需要重剖分的三角形列表

        返回:
            True 如果重剖分成功
        """
        if len(crossing_tris) == 0:
            return False

        # 收集所有涉及的顶点
        all_vertices = set()
        for tri in crossing_tris:
            all_vertices.update(tri.vertices)

        all_vertices.discard(v1)
        all_vertices.discard(v2)
        all_vertices.add(v1)
        all_vertices.add(v2)

        # 删除跨越三角形
        bad_set = set(id(tri) for tri in crossing_tris)
        self.triangles = [tri for tri in self.triangles if id(tri) not in bad_set]

        # 创建新的三角形，确保(v1, v2)成为边
        # 策略：将多边形区域三角剖分，(v1, v2)作为约束边
        vertex_list = sorted(list(all_vertices))

        # 简化策略：使用fan triangulation从v1出发
        # 这样可以确保(v1, v2)成为边
        for i in range(len(vertex_list) - 1):
            va = vertex_list[i]
            vb = vertex_list[i + 1]
            if va != v1 and vb != v1 and va != v2 and vb != v2:
                new_tri = Triangle(v1, va, vb)
                self._compute_circumcircle(new_tri)
                self.triangles.append(new_tri)

        return True

    def _insert_midpoint_for_edge(self, v1: int, v2: int) -> bool:
        """
        通过在边中点插入新点来恢复边界边

        参数:
            v1, v2: 边的两个顶点

        返回:
            True 如果点插入成功
        """
        p1 = self.points[v1]
        p2 = self.points[v2]

        # 计算中点
        midpoint = (p1 + p2) / 2.0

        # 检查中点是否已经有点
        min_dist = float('inf')
        if self._kdtree is not None:
            min_dist, _ = self._kdtree.query(midpoint)

        # 如果距离足够远，插入新点
        if min_dist > 1e-6:
            new_point_idx = len(self.points)
            self.points = np.vstack([self.points, midpoint])
            self.boundary_mask = np.append(self.boundary_mask, False)
            self.triangles = self._insert_point_incremental(new_point_idx, self.triangles)
            return True

        return False

    def _clear_triangle_caches(self):
        """
        清理所有三角形的缓存（在平滑后重新剖分前调用）
        """
        for tri in self.triangles:
            tri.circumcircle_valid = False
            tri.quality_valid = False
            tri.circumcenter = None
            tri.circumradius = None

    def generate_mesh(
        self,
        target_triangle_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成三角形网格

        参数:
            target_triangle_count: 目标三角形数量（可选）

        返回:
            (points, simplices, boundary_mask):
                - points: 点坐标数组，形状为 (N, 2)
                - simplices: 三角形索引数组，形状为 (M, 3)
                - boundary_mask: 边界点掩码，形状为 (N,)，True 表示边界点
        """
        timer = TimeSpan("开始 Bowyer-Watson 网格生成...")

        # 初始化
        self.boundary_count = len(self.original_points)
        self.points = self.original_points.copy()
        self.boundary_mask = np.zeros(len(self.points), dtype=bool)
        self.boundary_mask[:self.boundary_count] = True

        verbose(f"边界点数量: {self.boundary_count}")

        # 第一次三角剖分
        verbose(f"阶段 1/3: 初始三角剖分...")
        self._triangulate()
        verbose(f"  初始三角形数量: {len(self.triangles)}")

        # 迭代插入内部点
        verbose(f"阶段 2/3: 迭代插入内部点...")
        if self.holes:
            verbose(f"  检测到 {len(self.holes)} 个孔洞，将拒绝在孔洞内插点")
        self._insert_points_iteratively(target_triangle_count)

        # 删除孔洞内的三角形（后处理）
        if self.holes:
            verbose(f"阶段 2.5/3: 清理孔洞内三角形...")
            self._remove_hole_triangles()

        # 恢复丢失的边界边
        verbose(f"阶段 2.6/3: 恢复边界边...")
        self._recover_boundary_edges()

        # Laplacian 平滑
        if self.smoothing_iterations > 0:
            verbose(f"阶段 3/3: Laplacian 平滑 ({self.smoothing_iterations} 次迭代)...")
            # 更新 boundary_mask 以匹配当前点数
            current_point_count = len(self.points)
            self.boundary_mask = np.zeros(current_point_count, dtype=bool)
            self.boundary_mask[:self.boundary_count] = True

            self._laplacian_smoothing(self.smoothing_iterations)

            # 平滑后需要重新三角剖分
            verbose(f"  平滑后重新三角剖分...")
            self._clear_triangle_caches()
            self.triangles = []
            self._triangulate()

            # 平滑后重新三角剖分后再次清理孔洞内三角形
            if self.holes:
                self._remove_hole_triangles()
            
            # 平滑后重新三角剖分后再次恢复边界边
            self._recover_boundary_edges()
            
            # 边界恢复可能会创建孔洞内的三角形，需要再次清理
            if self.holes:
                self._remove_hole_triangles()
        else:
            verbose(f"阶段 3/3: 跳过平滑（未启用）")
            # 即使不启用平滑，边界恢复后也需要清理孔洞
            if self.holes:
                self._remove_hole_triangles()

        # 准备输出前，最后一次清理孔洞（确保没有孔洞内的三角形）
        if self.holes:
            verbose(f"最终清理孔洞内三角形...")
            
            # 简单策略：删除所有三个顶点都在孔洞边界上的三角形
            # 首先收集所有孔洞边界点的索引
            hole_boundary_points = set()
            for hole in self.holes:
                # 对于每个孔洞，找到其边界点在self.points中的索引
                for hole_pt in hole:
                    # 找到最接近的点索引
                    for pt_idx, pt in enumerate(self.points):
                        if np.linalg.norm(pt[:2] - hole_pt[:2]) < 1e-6:
                            hole_boundary_points.add(pt_idx)
                            break
            
            verbose(f"  孔洞边界点数: {len(hole_boundary_points)}")
            
            # 删除所有三个顶点都在孔洞边界上的三角形
            triangles_to_keep = []
            removed_by_vertex_check = 0
            removed_by_centroid_check = 0
            
            for tri in self.triangles:
                # 检查1：所有三个顶点都在孔洞边界上
                all_vertices_on_hole_boundary = all(v in hole_boundary_points for v in tri.vertices)
                
                if all_vertices_on_hole_boundary:
                    # 这个三角形完全在孔洞边界上，删除
                    removed_by_vertex_check += 1
                    continue
                
                # 检查2：质心在任何孔洞内
                centroid = self._compute_triangle_centroid(tri)
                in_hole = False
                for hole in self.holes:
                    if point_in_polygon(centroid, hole):
                        in_hole = True
                        break
                
                if in_hole:
                    removed_by_centroid_check += 1
                    continue
                
                triangles_to_keep.append(tri)
            
            total_removed = removed_by_vertex_check + removed_by_centroid_check
            if total_removed > 0:
                verbose(f"  最终清理删除了 {total_removed} 个孔洞内三角形 (顶点检查: {removed_by_vertex_check}, 质心检查: {removed_by_centroid_check})")
            else:
                verbose(f"  最终清理未删除任何三角形")
            self.triangles = triangles_to_keep
        
        # 准备输出
        points = self.points.copy()
        simplices = np.array([tri.vertices for tri in self.triangles])
        # 确保 boundary_mask 是最新的
        boundary_mask = np.zeros(len(points), dtype=bool)
        boundary_mask[:self.boundary_count] = True

        # 统计信息
        num_points = len(points)
        num_triangles = len(simplices)
        num_boundary = np.sum(boundary_mask)
        num_internal = num_points - num_boundary

        verbose(f"网格生成完成:")
        verbose(f"  - 总节点数: {num_points}")
        verbose(f"  - 边界节点: {num_boundary}")
        verbose(f"  - 内部节点: {num_internal}")
        verbose(f"  - 三角形数: {num_triangles}")

        timer.show_to_console("Bowyer-Watson 网格生成完成")

        return points, simplices, boundary_mask


def _extract_boundary_loops_from_fronts(boundary_front) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    从 boundary_front 中提取边界环（loop）
    
    算法流程：
    1. 将所有 Front 边组织为邻接表
    2. 追踪连续的边形成环
    3. 使用包含关系区分外边界和孔洞：
       - 外边界：不被任何其他环包含
       - 孔洞：被某个外边界包含
    
    参数:
        boundary_front: Front 对象列表
        
    返回:
        (outer_loops, hole_loops): 外边界环列表和孔洞环列表
    """
    # 1. 构建邻接表：node_hash -> [(node_hash, coords), ...]
    adjacency = {}
    node_coords = {}  # node_hash -> coords
    
    for front in boundary_front:
        if len(front.node_elems) < 2:
            continue
        
        node1 = front.node_elems[0]
        node2 = front.node_elems[1]
        
        hash1, hash2 = node1.hash, node2.hash
        coord1, coord2 = node1.coords, node2.coords
        
        node_coords[hash1] = coord1
        node_coords[hash2] = coord2
        
        if hash1 not in adjacency:
            adjacency[hash1] = []
        if hash2 not in adjacency:
            adjacency[hash2] = []
        
        adjacency[hash1].append(hash2)
        adjacency[hash2].append(hash1)
    
    # 2. 追踪环
    visited_edges = set()
    visited_nodes = set()  # 新增：追踪已访问的节点
    loops = []
    
    for start_node in adjacency:
        if start_node in visited_nodes:  # 修改：检查节点而非边
            continue
        
        # 从当前节点开始追踪一个环
        loop = []
        current = start_node
        prev = None
        
        while True:
            if current in node_coords:
                loop.append(node_coords[current])
                visited_nodes.add(current)  # 标记节点已访问
            
            if prev is not None:
                edge_key = tuple(sorted([prev, current]))
                visited_edges.add(edge_key)
            
            # 找下一个节点
            neighbors = adjacency.get(current, [])
            next_node = None
            
            for neighbor in neighbors:
                edge_key = tuple(sorted([current, neighbor]))
                if edge_key not in visited_edges:
                    next_node = neighbor
                    break
            
            if next_node is None:
                break
            
            prev = current
            current = next_node
            
            # 如果回到起点，环完成
            if current == start_node:
                break
        
        if len(loop) >= 3:
            loops.append(np.array(loop))
    
    if not loops:
        return [], []
    
    # 3. 使用包含关系和面积大小区分外边界和孔洞
    # 计算每个环的形心和面积
    centroids = [np.mean(loop, axis=0) for loop in loops]
    
    def polygon_area(loop):
        """计算多边形面积（无符号）"""
        area = 0.0
        n = len(loop)
        for i in range(n):
            p1 = loop[i]
            p2 = loop[(i + 1) % n]
            area += (p2[0] - p1[0]) * (p2[1] + p1[1])
        return abs(area) / 2.0
    
    areas = [polygon_area(loop) for loop in loops]
    
    # 对于每个环，检查它是否在任何其他环内
    outer_loops = []
    hole_loops = []
    
    for i, loop in enumerate(loops):
        is_inside_any = False
        
        for j, other_loop in enumerate(loops):
            if i == j:
                continue
            
            # 检查环 i 的形心是否在环 j 内
            if point_in_polygon(centroids[i], other_loop):
                # 如果环 i 在环 j 内，且环 j 面积更大，则环 i 是孔洞
                if areas[j] > areas[i]:
                    is_inside_any = True
                    break
        
        if is_inside_any:
            hole_loops.append(loop)
        else:
            outer_loops.append(loop)
    
    return outer_loops, hole_loops


def create_bowyer_watson_mesh(
    boundary_front,
    sizing_system,
    target_triangle_count: Optional[int] = None,
    max_edge_length: Optional[float] = None,
    smoothing_iterations: int = 3,
    seed: Optional[int] = None,
    holes: Optional[List[np.ndarray]] = None,
    auto_detect_holes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bowyer-Watson 网格生成接口函数

    参数:
        boundary_front: 边界阵面列表（Front 对象），参考 Adfront2 的初始输入
        sizing_system: QuadtreeSizing 尺寸场对象
        target_triangle_count: 目标三角形数量（可选）
        max_edge_length: 全局最大边长（可选，优先使用 sizing_system）
        smoothing_iterations: Laplacian 平滑迭代次数
        seed: 随机种子
        holes: 孔洞边界列表，每个孔洞是一个点数组，形状为 (M, 2)
        auto_detect_holes: 是否自动从 boundary_front 中检测孔洞（基于顺时针方向）

    返回:
        (points, simplices, boundary_mask):
            - points: 点坐标数组，形状为 (N, 2)
            - simplices: 三角形索引数组，形状为 (M, 3)
            - boundary_mask: 边界点掩码，形状为 (N,)，True 表示边界点
    """
    timer = TimeSpan("开始 Bowyer-Watson 网格生成流程...")

    # 0. 自动检测孔洞（如果启用）
    final_holes = holes or []
    
    if auto_detect_holes:
        verbose("自动检测边界环...")
        outer_loops, hole_loops = _extract_boundary_loops_from_fronts(boundary_front)
        
        if hole_loops:
            verbose(f"  检测到 {len(hole_loops)} 个顺时针环（孔洞）")
            final_holes = list(final_holes) + hole_loops
        
        if outer_loops:
            verbose(f"  检测到 {len(outer_loops)} 个逆时针环（外边界）")

    # 1. 从边界阵面提取边界点和边界边
    boundary_points = []
    boundary_edges = []
    node_index_map = {}  # Front 节点 hash -> 索引映射
    current_idx = 0

    for front in boundary_front:
        for node_elem in front.node_elems:
            node_hash = node_elem.hash
            if node_hash not in node_index_map:
                node_index_map[node_hash] = current_idx
                boundary_points.append(node_elem.coords)
                current_idx += 1

        # 提取边界边
        if len(front.node_elems) >= 2:
            idx1 = node_index_map[front.node_elems[0].hash]
            idx2 = node_index_map[front.node_elems[1].hash]
            boundary_edges.append((idx1, idx2))

    boundary_points = np.array(boundary_points)
    verbose(f"边界点数: {len(boundary_points)}")
    verbose(f"边界边数: {len(boundary_edges)}")
    
    if final_holes:
        verbose(f"孔洞数: {len(final_holes)}")

    # 2. 创建网格生成器
    generator = BowyerWatsonMeshGenerator(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        sizing_system=sizing_system,
        max_edge_length=max_edge_length,
        smoothing_iterations=smoothing_iterations,
        seed=seed,
        holes=final_holes if final_holes else None,
    )

    # 3. 生成网格
    points, simplices, boundary_mask = generator.generate_mesh(
        target_triangle_count=target_triangle_count
    )

    timer.show_to_console("Bowyer-Watson 网格生成流程完成")

    return points, simplices, boundary_mask

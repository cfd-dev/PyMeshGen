"""
Bowyer-Watson Delaunay 网格生成器 - 核心实现

基于 Bowyer-Watson 算法的二维三角形网格生成，支持：
1. 以离散边界网格作为输入
2. 使用 QuadtreeSizing 尺寸场控制网格尺寸
3. 自动/手动孔洞处理
4. 边界边保护与恢复
5. 增量式点插入（避免全量重剖分）

参考: Gmsh delaunay/ref/ 下的 C++ 实现
"""

import numpy as np
from typing import List, Tuple, Optional
from math import sqrt
from scipy.spatial import KDTree
from collections import Counter

from utils.message import debug, verbose
from utils.geom_toolkit import point_in_polygon, is_polygon_clockwise


# =============================================================================
# Triangle 数据结构
# =============================================================================

class Triangle:
    """三角形单元，带外接圆缓存和质量缓存。

    顶点索引始终按升序存储，便于去重和比较。
    """

    __slots__ = [
        'vertices', 'circumcenter', 'circumradius', 'idx',
        'circumcircle_valid', 'quality', 'quality_valid', 'circumcircle_bbox',
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
        """计算三角形的外接圆，结果缓存到 tri 对象中。"""
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
            center = np.array([(ax + bx + cx) / 3.0, (ay + by + cy) / 3.0])
            radius = float(np.linalg.norm(p1 - center))
        else:
            ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
            uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
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
        """检查点是否在三角形的外接圆内（含容差）。"""
        if not tri.circumcircle_valid:
            self._compute_circumcircle(tri)
        distance = float(np.linalg.norm(point - tri.circumcenter))
        return distance < tri.circumradius * (1.0 + 1e-10)

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
        """获取三角形的目标尺寸（尺寸场 > 全局尺寸 > None）。"""
        if self.sizing_system is not None:
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
        return self.max_edge_length

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

    # -------------------------------------------------------------------------
    # 迭代插点主循环
    # -------------------------------------------------------------------------

    def _insert_points_iteratively(self, target_triangle_count: Optional[int] = None):
        """迭代插入内部点，使用外接圆圆心策略。

        终止条件（满足任一即停止）：
        1. 达到目标三角形数量
        2. 所有三角形都满足尺寸和质量要求
        3. 达到最大节点数限制
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
        max_consecutive_failures = 500
        failed_triangles = set()

        while True:
            max_iterations += 1

            if max_iterations % 10 == 0 or max_iterations == 1:
                current_triangles = len(self.triangles)
                current_points = len(self.points)
                current_inserted = current_points - initial_point_count
                verbose(f"  [进度] 迭代 {max_iterations} | "
                        f"节点: {current_points} (边界: {boundary_count}, 内部: {current_inserted}) | "
                        f"三角形: {current_triangles}")

            if len(self.points) > 100000:
                verbose("达到最大节点数限制 (100000)，停止插点")
                break
            if max_iterations > 50000:
                verbose("达到最大迭代次数限制 (50000)，停止插点")
                break
            if target_total_points is not None and len(self.points) >= target_total_points:
                verbose(f"  [进度] 达到目标节点数: {len(self.points)}")
                break
            if len(self.triangles) == 0:
                break

            worst_quality = float('inf')
            worst_triangle = None
            needs_refinement = False
            points = self.points

            for tri in self.triangles:
                if tri.vertices in failed_triangles:
                    continue

                quality = self._compute_triangle_quality(tri)
                v0, v1, v2 = tri.vertices
                edge_lengths = [
                    float(np.linalg.norm(points[v1] - points[v0])),
                    float(np.linalg.norm(points[v2] - points[v1])),
                    float(np.linalg.norm(points[v0] - points[v2])),
                ]
                max_edge = max(edge_lengths)
                target_size = self._get_target_size_for_triangle(tri)

                should_split = False
                if target_size is not None:
                    if max_edge > target_size * 1.1:
                        should_split = True
                    elif quality < 0.3:
                        should_split = True
                else:
                    if quality < 0.5:
                        should_split = True

                if should_split:
                    needs_refinement = True
                    if quality < worst_quality:
                        worst_quality = quality
                        worst_triangle = tri

            if not needs_refinement:
                verbose("  [进度] 所有三角形满足尺寸和质量要求")
                break
            if worst_triangle is None:
                break

            if not worst_triangle.circumcircle_valid:
                self._compute_circumcircle(worst_triangle)

            new_point = worst_triangle.circumcenter.copy()

            if not (x_min - margin < new_point[0] < x_max + margin and
                    y_min - margin < new_point[1] < y_max + margin):
                p1 = points[worst_triangle.vertices[0]]
                p2 = points[worst_triangle.vertices[1]]
                p3 = points[worst_triangle.vertices[2]]
                r1, r2 = np.random.rand(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                new_point = p1 + r1 * (p2 - p1) + r2 * (p3 - p1)

            if len(self.points) > 0:
                if max_iterations == 1 or (max_iterations - last_kdtree_build) >= kdtree_rebuild_interval:
                    self._kdtree = KDTree(self.points)
                    last_kdtree_build = max_iterations
                min_dist, _ = self._kdtree.query(new_point)
            else:
                min_dist = float('inf')

            if min_dist > min_dist_threshold:
                new_point_idx = len(self.points)
                self.points = np.vstack([self.points, new_point])
                self.triangles = self._insert_point_incremental(new_point_idx, self.triangles)
            else:
                failed_triangles.add(worst_triangle.vertices)
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    verbose(f"  [进度] 连续失败 {consecutive_failures} 次，停止插点")
                    break
                continue

        final_triangles = len(self.triangles)
        final_points = len(self.points)
        final_inserted = final_points - initial_point_count
        verbose(f"  [完成] 插点完成 | "
                f"节点: {final_points} (边界: {boundary_count}, 内部: {final_inserted}) | "
                f"三角形: {final_triangles}")

    # -------------------------------------------------------------------------
    # Laplacian 平滑（默认关闭）
    # -------------------------------------------------------------------------

    def _laplacian_smoothing(self, iterations: int = 3, alpha: float = 0.5):
        """Laplacian 平滑，边界点保持不动，内部点被约束在原始边界包围盒内。"""
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

        for _ in range(iterations):
            new_points = smoothed_points.copy()
            neighbor_dict = {}
            for tri in self.triangles:
                for i in range(3):
                    v = tri.vertices[i]
                    if v not in neighbor_dict:
                        neighbor_dict[v] = set()
                    neighbor_dict[v].add(tri.vertices[(i + 1) % 3])
                    neighbor_dict[v].add(tri.vertices[(i + 2) % 3])

            for v, neighbors in neighbor_dict.items():
                if self.boundary_mask[v]:
                    continue
                if len(neighbors) > 0:
                    neighbor_center = np.mean([smoothed_points[n] for n in neighbors], axis=0)
                    new_points[v] = smoothed_points[v] + alpha * (neighbor_center - smoothed_points[v])
                    new_points[v, 0] = np.clip(new_points[v, 0], x_min, x_max)
                    new_points[v, 1] = np.clip(new_points[v, 1], y_min, y_max)

            smoothed_points = new_points

        self.points = smoothed_points

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
        """恢复单条边界边。"""
        # 检查边是否已经存在
        common_tris = [tri for tri in self.triangles if v1 in tri.vertices and v2 in tri.vertices]
        if common_tris:
            return True

        # 查找与这条边相交的三角形
        crossing_tris = [
            tri for tri in self.triangles
            if self._edge_intersects_triangle(
                self.points[v1], self.points[v2],
                [self.points[tri.vertices[i]] for i in range(3)]
            )
        ]

        if crossing_tris:
            # 有相交三角形，使用重剖分
            return self._retriangulate_for_edge_recovery(v1, v2, crossing_tris)

        # 没有相交三角形，说明这条边在"空洞"中
        # 尝试找到可以形成三角形的第三个顶点
        return self._force_create_edge(v1, v2)
    
    def _force_create_edge(self, v1: int, v2: int) -> bool:
        """强制创建边界边（当边在空洞中时）。
        
        算法：
        1. 找到与 v1 和 v2 都相邻的节点
        2. 使用这些节点创建包含 (v1, v2) 的三角形
        3. 如果没有共同邻居，尝试找到最合适的第三个节点
        """
        # 找到与 v1 相邻的节点
        v1_neighbors = set()
        for tri in self.triangles:
            if v1 in tri.vertices:
                for v in tri.vertices:
                    if v != v1:
                        v1_neighbors.add(v)
        
        # 找到与 v2 相邻的节点
        v2_neighbors = set()
        for tri in self.triangles:
            if v2 in tri.vertices:
                for v in tri.vertices:
                    if v != v2:
                        v2_neighbors.add(v)
        
        # 找到共同邻居
        common_neighbors = v1_neighbors & v2_neighbors
        
        if common_neighbors:
            # 使用共同邻居创建三角形
            created = False
            for v3 in common_neighbors:
                # 检查三角形是否已存在
                exists = any(
                    set(tri.vertices) == {v1, v2, v3}
                    for tri in self.triangles
                )
                if not exists:
                    new_tri = Triangle(v1, v2, v3)
                    self._compute_circumcircle(new_tri)
                    self.triangles.append(new_tri)
                    created = True
            return created
        
        # 没有共同邻居，尝试找到最合适的第三个节点
        # 策略：找到与 v1 或 v2 相邻，且与边 (v1,v2) 形成合理角度的节点
        edge_vec = self.points[v2] - self.points[v1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-10:
            return self._insert_midpoint_for_edge(v1, v2)
        
        # 收集所有候选节点
        candidate_nodes = v1_neighbors | v2_neighbors
        
        best_v3 = None
        best_angle = -1
        
        for v3 in candidate_nodes:
            # 检查三角形是否已存在
            if any(set(tri.vertices) == {v1, v2, v3} for tri in self.triangles):
                continue
            
            # 计算角度（避免太尖锐的三角形）
            v3_v1 = self.points[v1] - self.points[v3]
            v3_v2 = self.points[v2] - self.points[v3]
            
            dot = np.dot(v3_v1, v3_v2)
            norm1 = np.linalg.norm(v3_v1)
            norm2 = np.linalg.norm(v3_v2)
            
            if norm1 < 1e-10 or norm2 < 1e-10:
                continue
            
            cos_angle = dot / (norm1 * norm2)
            # 选择角度最大的（最接近 180 度的三角形最不好）
            if cos_angle > best_angle:
                best_angle = cos_angle
                best_v3 = v3
        
        if best_v3 is not None:
            new_tri = Triangle(v1, v2, best_v3)
            self._compute_circumcircle(new_tri)
            self.triangles.append(new_tri)
            return True
        
        # 最后手段：中点插入
        return self._insert_midpoint_for_edge(v1, v2)

    def _edge_intersects_triangle(self, p1: np.ndarray, p2: np.ndarray,
                                   tri_points: List[np.ndarray]) -> bool:
        """检查线段(p1,p2)是否与三角形内部相交。"""
        midpoint = (p1 + p2) / 2.0
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

        tolerance = 1e-10
        return (u >= -tolerance) and (v >= -tolerance) and (u + v <= 1.0 + tolerance)

    def _retriangulate_for_edge_recovery(self, v1: int, v2: int,
                                          crossing_tris: List[Triangle]) -> bool:
        """通过局部重剖分恢复边界边。
        
        算法：
        1. 收集所有与边(v1,v2)相交的三角形
        2. 删除这些三角形
        3. 收集这些三角形的所有顶点（除了v1和v2）
        4. 将这些顶点按照与边的位置关系分为两侧
        5. 在每一侧创建三角形，确保(v1,v2)成为边界边
        """
        if not crossing_tris:
            return False

        # 收集所有顶点
        all_vertices = set()
        for tri in crossing_tris:
            all_vertices.update(tri.vertices)
        all_vertices.discard(v1)
        all_vertices.discard(v2)
        
        # 删除相交的三角形
        bad_set = set(id(tri) for tri in crossing_tris)
        self.triangles = [tri for tri in self.triangles if id(tri) not in bad_set]
        
        # 计算边的方向
        edge_vec = self.points[v2] - self.points[v1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-10:
            return False
        edge_dir = edge_vec / edge_len
        
        # 将顶点分为两侧（基于边的法线方向）
        normal = np.array([-edge_dir[1], edge_dir[0]])  # 法线方向
        side_a = []  # 法线正方向
        side_b = []  # 法线负方向
        
        for v in all_vertices:
            vec_to_v = self.points[v] - self.points[v1]
            projection = np.dot(vec_to_v, normal)
            if projection > 0:
                side_a.append(v)
            else:
                side_b.append(v)
        
        # 对两侧的顶点按照沿边方向的位置排序
        def sort_along_edge(vertices):
            return sorted(vertices, key=lambda v: np.dot(self.points[v] - self.points[v1], edge_dir))
        
        side_a = sort_along_edge(side_a)
        side_b = sort_along_edge(side_b)
        
        # 在每一侧创建三角形扇
        # 侧 A: v1 -> side_a[0] -> side_a[1] -> ... -> v2
        prev = v1
        for v in side_a:
            new_tri = Triangle(prev, v, v2)
            self._compute_circumcircle(new_tri)
            self.triangles.append(new_tri)
            prev = v
        
        # 侧 B: v1 -> side_b[0] -> side_b[1] -> ... -> v2
        prev = v1
        for v in side_b:
            new_tri = Triangle(prev, v, v2)
            self._compute_circumcircle(new_tri)
            self.triangles.append(new_tri)
            prev = v

        return True

    def _insert_midpoint_for_edge(self, v1: int, v2: int) -> bool:
        """通过在边中点插入新点来恢复边界边。"""
        midpoint = (self.points[v1] + self.points[v2]) / 2.0
        min_dist = float('inf')
        if self._kdtree is not None:
            min_dist, _ = self._kdtree.query(midpoint)

        if min_dist > 1e-6:
            new_point_idx = len(self.points)
            self.points = np.vstack([self.points, midpoint])
            self.boundary_mask = np.append(self.boundary_mask, False)
            self.triangles = self._insert_point_incremental(new_point_idx, self.triangles)
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
            verbose(f"  检测到 {len(self.holes)} 个孔洞，将拒绝在孔洞内插点")
        self._insert_points_iteratively(target_triangle_count)

        if self.holes:
            verbose("阶段 2.5/3: 清理孔洞内三角形...")
            self._remove_hole_triangles()

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

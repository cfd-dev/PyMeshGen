"""
Bowyer-Watson 网格生成器 - 核心实现

参考 Gmsh C++ 代码设计，包含：
- BowyerWatsonMeshGenerator: 基础实现
- GmshBowyerWatsonMeshGenerator: Gmsh 风格实现
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from math import sqrt
from scipy.spatial import KDTree
import heapq

try:
    from utils.message import verbose, debug
except ModuleNotFoundError:
    from message import verbose, debug

try:
    from utils.geom_toolkit import point_in_polygon, is_polygon_clockwise
except ModuleNotFoundError:
    from geom_toolkit import point_in_polygon, is_polygon_clockwise
from delaunay.types import MTri3, EdgeXFace, build_adjacency
from delaunay.predicates import orient2d, incircle, circumcenter_precise, point_in_triangle
from delaunay.cavity import recur_find_cavity, insert_vertex
from delaunay.boundary import (
    recover_edge_by_swaps,
    recover_edge_by_splitting,
    find_isolated_boundary_points,
    segment_intersects_triangle,
    find_cavity_boundary,
    retriangulate_with_constraint,
    find_path_in_boundary
)


class BowyerWatsonMeshGenerator:
    """Bowyer-Watson Delaunay 网格生成器（基础版本）。"""

    def __init__(
        self,
        boundary_points: np.ndarray,
        boundary_edges: Optional[List[Tuple[int, int]]] = None,
        sizing_system=None,
        max_edge_length: Optional[float] = None,
        smoothing_iterations: int = 0,
        seed: Optional[int] = None,
        holes: Optional[List[np.ndarray]] = None,
        outer_boundary: Optional[np.ndarray] = None
    ):
        self.original_points = boundary_points.copy()
        self.protected_edges: Set[frozenset] = set(frozenset(e) for e in (boundary_edges or []))
        self.boundary_edges = boundary_edges or []
        self.sizing_system = sizing_system
        self.max_edge_length = max_edge_length
        self.smoothing_iterations = smoothing_iterations
        self.seed = seed
        self.holes = holes or []
        self.outer_boundary = outer_boundary

        if seed is not None:
            np.random.seed(seed)

        self.points = None
        self.triangles: List[MTri3] = []
        self.boundary_mask = None
        self.boundary_count = 0
        self._kdtree = None
        self._tri_id_counter = 0

    def _next_tri_id(self) -> int:
        self._tri_id_counter += 1
        return self._tri_id_counter

    def _is_protected_edge(self, v1: int, v2: int) -> bool:
        return frozenset({v1, v2}) in self.protected_edges

    def _compute_circumcircle(self, tri: MTri3):
        """计算三角形外接圆。"""
        if tri.circumradius is not None:
            return

        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        cx, cy = circumcenter_precise(p1, p2, p3)
        tri.circumcenter = np.array([cx, cy])
        tri.circumradius = float(np.linalg.norm(p1 - tri.circumcenter))

    def _point_in_circumcircle(self, p1, p2, p3, point) -> bool:
        """检查点是否在外接圆内。"""
        return incircle(p1, p2, p3, point) > 0

    def _create_super_triangle(self) -> MTri3:
        """创建超级三角形。"""
        min_x, max_x = np.min(self.points[:, 0]), np.max(self.points[:, 0])
        min_y, max_y = np.min(self.points[:, 1]), np.max(self.points[:, 1])

        dx = max_x - min_x
        dy = max_y - min_y
        delta = max(dx, dy) * 10.0

        p1, p2, p3 = len(self.points), len(self.points) + 1, len(self.points) + 2

        super_verts = np.array([
            [min_x - delta, min_y - delta],
            [max_x + delta, min_y - delta],
            [(min_x + max_x) / 2, max_y + 3 * delta]
        ])
        self.points = np.vstack([self.points, super_verts])

        tri = MTri3(p1, p2, p3, idx=self._next_tri_id())
        self._compute_circumcircle(tri)
        return tri

    def _triangulate(self) -> List[MTri3]:
        """初始三角剖分。"""
        real_count = len(self.points)
        
        if real_count < 3:
            verbose(f"  Not enough points for triangulation: {real_count}")
            return []

        verbose(f"  Points to triangulate: {real_count}")

        # 使用增量法构建三角剖分
        triangles = []
        
        # 使用前 3 个点创建第一个三角形
        if real_count >= 3:
            tri = MTri3(0, 1, 2, idx=self._next_tri_id())
            self._compute_circumcircle(tri)
            triangles.append(tri)
            verbose(f"  Created initial triangle: {tri.vertices}")

        # 插入剩余点
        inserted_count = 3
        for i in range(3, real_count):
            point = self.points[i]

            # 查找包含点的三角形
            containing_tri = None
            for tri in triangles:
                if not tri.deleted and point_in_triangle(point, tri.vertices, self.points):
                    containing_tri = tri
                    break

            # 如果找不到包含点的三角形，找到外接圆包含该点的三角形
            if containing_tri is None:
                for tri in triangles:
                    if not tri.deleted and self._point_in_circumcircle(
                        self.points[tri.vertices[0]], self.points[tri.vertices[1]],
                        self.points[tri.vertices[2]], point):
                        containing_tri = tri
                        break

            # 如果仍然找不到，找到距离点最近的三角形
            if containing_tri is None:
                min_dist = float('inf')
                for tri in triangles:
                    if tri.deleted:
                        continue
                    centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
                    dist = float(np.linalg.norm(point - centroid))
                    if dist < min_dist:
                        min_dist = dist
                        containing_tri = tri

            if containing_tri is None:
                verbose(f"  Warning: point {i} ({point}) not found in any triangle")
                continue

            # Cavity 搜索
            cavity_tris, shell_edges = recur_find_cavity(
                containing_tri, point, i, self.points,
                self.protected_edges,
                lambda p1, p2, p3, pt: self._point_in_circumcircle(p1, p2, p3, pt)
            )

            # 插入点
            new_tris, success = insert_vertex(
                shell_edges, cavity_tris, i, self.points, triangles,
                self._next_tri_id, self._compute_circumcircle,
                validate_star=False, validate_edges=False
            )
            if success:
                inserted_count += 1

        # 清理已删除的三角形
        self.triangles = [tri for tri in triangles if not tri.deleted]
        
        verbose(f"  Inserted {inserted_count}/{real_count} points")
        verbose(f"  After triangulation: {len(self.triangles)} triangles")

        # 构建邻接关系
        build_adjacency(self.triangles)

        return self.triangles

    def _is_boundary_edge_in_mesh(self, v1: int, v2: int) -> bool:
        """检查边界边是否在网格中。"""
        for tri in self.triangles:
            if tri.deleted:
                continue
            if v1 in tri.vertices and v2 in tri.vertices:
                return True
        return False

    def _force_recover_boundary_edge_by_reconnection(self, v1: int, v2: int):
        """强制通过重连接恢复边界边。"""
        p1, p2 = self.points[v1], self.points[v2]

        # 找到所有与约束边相交的三角形
        intersecting_tris = []
        for tri in self.triangles:
            if tri.deleted:
                continue
            if segment_intersects_triangle(p1, p2, tri, self.points):
                intersecting_tris.append(tri)

        if not intersecting_tris:
            # 检查边是否已存在
            if self._is_boundary_edge_in_mesh(v1, v2):
                return True
            return False

        # 删除相交三角形
        for tri in intersecting_tris:
            tri.deleted = True

        # 找到空洞边界
        cavity_boundary = find_cavity_boundary(self.triangles)
        if not cavity_boundary:
            return False

        # 重新三角化
        return retriangulate_with_constraint(
            cavity_boundary, v1, v2, self.points, self.triangles,
            self._next_tri_id, self._compute_circumcircle
        )

    def _get_target_size(self, tri: MTri3) -> Optional[float]:
        """获取目标尺寸。"""
        if self.sizing_system is not None:
            center = np.mean([self.points[v] for v in tri.vertices], axis=0)
            try:
                size = self.sizing_system.spacing_at(center)
                if size is not None and size > 1e-12:
                    return size
            except:
                pass

        return self.max_edge_length

    def _compute_quality(self, tri: MTri3) -> float:
        """计算三角形质量。"""
        if tri.quality > 0:
            return tri.quality

        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        s = (a + b + c) / 2.0

        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq < 0:
            tri.quality = 0.0
            return 0.0

        area = sqrt(max(area_sq, 0.0))
        if area < 1e-12:
            tri.quality = 0.0
            return 0.0

        r_in = area / s
        if tri.circumradius is None:
            self._compute_circumcircle(tri)
        r_out = tri.circumradius

        quality = min(2.0 * r_in / r_out, 1.0) if r_out > 1e-12 else 0.0
        tri.quality = quality
        return quality

    def _insert_points_iteratively(self, target_count: Optional[int] = None):
        """迭代插入内部点。"""
        boundary_count = self.boundary_count
        initial_points = len(self.points)

        # 优先级队列（按外接圆半径排序）
        priority_queue = []
        for tri in self.triangles:
            if tri.circumradius is None:
                self._compute_circumcircle(tri)
            heapq.heappush(priority_queue, (-tri.circumradius, id(tri), tri))

        max_iterations = 10000
        iteration = 0

        while priority_queue and iteration < max_iterations:
            iteration += 1

            if iteration % 100 == 0:
                tri_count = len([t for t in self.triangles if not t.deleted])
                point_count = len(self.points)
                verbose(f"  [Progress] Iteration {iteration} | Points: {point_count} | Triangles: {tri_count}")

            # 早停
            if len(self.points) > 50000:
                break

            # 获取最差三角形
            worst_tri = None
            while priority_queue:
                _, tri_id, tri = heapq.heappop(priority_queue)
                if not tri.deleted:
                    worst_tri = tri
                    break

            if worst_tri is None:
                break

            # 检查是否需要细化
            target_size = self._get_target_size(worst_tri)
            quality = self._compute_quality(worst_tri)

            edge_lengths = [
                np.linalg.norm(self.points[worst_tri.vertices[1]] - self.points[worst_tri.vertices[0]]),
                np.linalg.norm(self.points[worst_tri.vertices[2]] - self.points[worst_tri.vertices[1]]),
                np.linalg.norm(self.points[worst_tri.vertices[0]] - self.points[worst_tri.vertices[2]])
            ]
            max_edge = max(edge_lengths)

            needs_refinement = False
            if target_size is not None and target_size > 1e-12:
                if max_edge > target_size * 1.2 or quality < 0.2:
                    needs_refinement = True
            elif quality < 0.3:
                needs_refinement = True

            if not needs_refinement:
                continue

            # 计算插入点（外接圆心）
            if worst_tri.circumcenter is None:
                self._compute_circumcircle(worst_tri)
            new_point = worst_tri.circumcenter.copy()

            # 检查是否在孔洞内
            if self.holes and any(point_in_polygon(new_point, hole) for hole in self.holes):
                continue

            # 插入新点
            new_idx = len(self.points)
            self.points = np.vstack([self.points, new_point])

            # Cavity 搜索
            cavity_tris, shell_edges = recur_find_cavity(
                worst_tri, new_point, new_idx, self.points,
                self.protected_edges,
                lambda p1, p2, p3, pt: self._point_in_circumcircle(p1, p2, p3, pt)
            )

            # 插入点
            new_tris, success = insert_vertex(
                shell_edges, cavity_tris, new_idx, self.points, self.triangles,
                self._next_tri_id, self._compute_circumcircle,
                validate_star=False, validate_edges=False
            )

            # 将新三角形加入队列
            for new_tri in new_tris:
                heapq.heappush(priority_queue, (-new_tri.circumradius, id(new_tri), new_tri))

        # 清理已删除的三角形
        self.triangles = [tri for tri in self.triangles if not tri.deleted]

        verbose(f"  [Done] Insert points completed | Points: {len(self.points)} | Triangles: {len(self.triangles)}")

    def _constrained_delaunay_triangulation(self):
        """Constrained Delaunay Triangulation。"""
        verbose("  Starting Constrained Delaunay boundary recovery...")

        recovered = 0
        failed = 0

        for edge in self.protected_edges:
            v1, v2 = list(edge)

            if self._is_boundary_edge_in_mesh(v1, v2):
                recovered += 1
                continue

            # 尝试恢复
            if recover_edge_by_splitting(
                v1, v2, self.points, self.triangles,
                self._next_tri_id, self._compute_circumcircle
            ):
                recovered += 1
            else:
                failed += 1

        verbose(f"  Boundary recovery completed | Success: {recovered} | Failed: {failed}")

    def _remove_hole_triangles(self) -> int:
        """删除孔洞内的三角形。"""
        if not self.holes:
            return 0

        # 修正孔洞方向
        fixed_holes = []
        for hole in self.holes:
            if is_polygon_clockwise(hole):
                fixed_holes.append(hole[::-1])
            else:
                fixed_holes.append(hole.copy())

        removed = 0
        for tri in self.triangles:
            if tri.deleted:
                continue
            centroid = np.mean([self.points[v] for v in tri.vertices], axis=0)
            for hole in fixed_holes:
                if point_in_polygon(centroid, hole):
                    tri.deleted = True
                    removed += 1
                    break

        self.triangles = [tri for tri in self.triangles if not tri.deleted]
        build_adjacency(self.triangles)

        verbose(f"  Hole cleanup: removed {removed} triangles")
        return removed

    def _compute_boundary_mask(self) -> np.ndarray:
        """计算边界点掩码。"""
        mask = np.zeros(len(self.points), dtype=bool)
        edge_count = {}

        for tri in self.triangles:
            if tri.deleted:
                continue
            for i in range(3):
                edge = tuple(sorted(tri.get_edge(i)))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        for edge, count in edge_count.items():
            if count == 1:
                mask[edge[0]] = True
                mask[edge[1]] = True

        self.boundary_mask = mask
        self.boundary_count = int(np.sum(mask))
        return mask

    def generate_mesh(self, target_triangle_count: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成网格的主入口。

        Args:
            target_triangle_count: 目标三角形数量

        Returns:
            (points, simplices, boundary_mask)
        """
        # Initialize point set
        self.points = self.original_points.copy()
        self.boundary_count = len(self.original_points)

        verbose(f"Starting Bowyer-Watson mesh generation")
        verbose(f"  Boundary points: {self.boundary_count}")
        verbose(f"  Holes: {len(self.holes)}")

        # Initial triangulation
        verbose("  Performing initial triangulation...")
        self._triangulate()

        verbose(f"  Initial triangulation completed | Points: {len(self.points)} | Triangles: {len(self.triangles)}")

        # Iterative point insertion
        verbose("  Performing iterative point insertion...")
        self._insert_points_iteratively(target_triangle_count)

        # Remove triangles in holes
        self._remove_hole_triangles()

        # Constrained Delaunay
        self._constrained_delaunay_triangulation()

        # Compute boundary mask
        boundary_mask = self._compute_boundary_mask()

        # Build simplices array
        simplices = np.array([[tri.vertices[0], tri.vertices[1], tri.vertices[2]]
                             for tri in self.triangles if not tri.deleted])

        verbose(f"Mesh generation completed | Points: {len(self.points)} | Triangles: {len(simplices)}")

        return self.points, simplices, boundary_mask


class GmshBowyerWatsonMeshGenerator(BowyerWatsonMeshGenerator):
    """Gmsh 风格的 Bowyer-Watson 网格生成器。

    增强特性：
    - 更精确的尺寸场控制
    - 改进的边界恢复算法
    - 优化的插点策略
    """

    def _get_target_size(self, tri: MTri3) -> Optional[float]:
        """获取目标尺寸（考虑边界边长度）。"""
        target_size = super()._get_target_size(tri)

        # 计算三角形的边长
        v0, v1, v2 = tri.vertices
        pts = self.points
        edge_lengths = [
            np.linalg.norm(pts[v1] - pts[v0]),
            np.linalg.norm(pts[v2] - pts[v1]),
            np.linalg.norm(pts[v0] - pts[v2]),
        ]
        min_edge = min(edge_lengths)

        # 如果最小边远小于目标尺寸，使用更合理的目标
        if target_size is not None and target_size > 1e-12:
            if min_edge < target_size * 0.2:
                avg_edge = sum(edge_lengths) / 3.0
                target_size = min(target_size, avg_edge * 1.5)

        return target_size

    def _insert_points_iteratively(self, target_count: Optional[int] = None):
        """改进的迭代插点（Gmsh 风格）。"""
        boundary_count = self.boundary_count
        initial_points = len(self.points)

        # 优先级队列
        priority_queue = []
        for tri in self.triangles:
            if tri.circumradius is None:
                self._compute_circumcircle(tri)
            heapq.heappush(priority_queue, (-tri.circumradius, id(tri), tri))

        max_iterations = 15000
        iteration = 0
        consecutive_failures = 0
        max_consecutive_failures = 1000
        failed_tri_ids = set()

        while priority_queue and iteration < max_iterations:
            iteration += 1

            if iteration % 50 == 0:
                tri_count = len([t for t in self.triangles if not t.deleted])
                point_count = len(self.points)
                verbose(f"  [进度] 迭代 {iteration} | 节点：{point_count} | 三角形：{tri_count}")

            if len(self.points) > 100000:
                break

            # 获取最差三角形
            worst_tri = None
            while priority_queue:
                _, tri_id, tri = heapq.heappop(priority_queue)
                if not tri.deleted and tri_id not in failed_tri_ids:
                    worst_tri = tri
                    break

            if worst_tri is None:
                break

            # 检查是否需要细化
            target_size = self._get_target_size(worst_tri)
            quality = self._compute_quality(worst_tri)

            edge_lengths = [
                np.linalg.norm(self.points[worst_tri.vertices[1]] - self.points[worst_tri.vertices[0]]),
                np.linalg.norm(self.points[worst_tri.vertices[2]] - self.points[worst_tri.vertices[1]]),
                np.linalg.norm(self.points[worst_tri.vertices[0]] - self.points[worst_tri.vertices[2]])
            ]
            max_edge = max(edge_lengths)

            # 根据三角形是否在边界附近调整阈值
            is_boundary_tri = any(v < boundary_count for v in worst_tri.vertices)
            if is_boundary_tri:
                size_tolerance = 1.25
                quality_threshold = 0.1
            else:
                size_tolerance = 1.1
                quality_threshold = 0.15

            needs_refinement = False
            if target_size is not None and target_size > 1e-12:
                if max_edge > target_size * size_tolerance or quality < quality_threshold:
                    needs_refinement = True
            elif quality < 0.3:
                needs_refinement = True

            if not needs_refinement:
                continue

            # 计算插入点
            if worst_tri.circumcenter is None:
                self._compute_circumcircle(worst_tri)
            new_point = worst_tri.circumcenter.copy()

            # 检查是否在有效范围内
            x_min, x_max = np.min(self.original_points[:, 0]), np.max(self.original_points[:, 0])
            y_min, y_max = np.min(self.original_points[:, 1]), np.max(self.original_points[:, 1])
            margin = 0.001 * max(x_max - x_min, y_max - y_min)

            if not (x_min - margin < new_point[0] < x_max + margin and
                    y_min - margin < new_point[1] < y_max + margin):
                p1 = self.points[worst_tri.vertices[0]]
                p2 = self.points[worst_tri.vertices[1]]
                p3 = self.points[worst_tri.vertices[2]]
                new_point = (p1 + p2 + p3) / 3.0

            # 检查是否在孔洞内
            if self.holes and any(point_in_polygon(new_point, hole) for hole in self.holes):
                failed_tri_ids.add(id(worst_tri))
                worst_tri.deleted = True
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                continue

            # 插入新点
            new_idx = len(self.points)
            self.points = np.vstack([self.points, new_point])

            # Cavity 搜索
            cavity_tris, shell_edges = recur_find_cavity(
                worst_tri, new_point, new_idx, self.points,
                self.protected_edges, self._point_in_circumcircle
            )

            if not cavity_tris:
                failed_tri_ids.add(id(worst_tri))
                consecutive_failures += 1
                continue

            # 插入点
            new_tris, success = insert_vertex(
                shell_edges, cavity_tris, new_idx, self.points, self.triangles,
                self._next_tri_id, self._compute_circumcircle,
                validate_star=False
            )

            if success:
                consecutive_failures = 0
                for new_tri in new_tris:
                    heapq.heappush(priority_queue, (-new_tri.circumradius, id(new_tri), new_tri))
            else:
                failed_tri_ids.add(id(worst_tri))
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break

        # 清理
        self.triangles = [tri for tri in self.triangles if not tri.deleted]

        verbose(f"  [完成] 插点完成 | 节点：{len(self.points)} | 三角形：{len(self.triangles)}")

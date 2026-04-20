"""
Bowyer-Watson 网格生成器 - 核心实现

包含：
- BowyerWatsonMeshGenerator: 基础实现
- GmshBowyerWatsonMeshGenerator: Gmsh 风格实现
"""

import numpy as np
from typing import List, Tuple, Optional
from math import sqrt
from scipy.spatial import KDTree
import heapq

from utils.message import verbose
from delaunay.types import MTri3, build_adjacency
from delaunay.predicates import orient2d, incircle, circumcenter_precise, point_in_triangle
from delaunay.cavity import recur_find_cavity, insert_vertex
from delaunay.boundary import (
    recover_edge_by_swaps,
    recover_edge_by_splitting,
    find_isolated_boundary_points
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
        self.protected_edges = set(frozenset(e) for e in (boundary_edges or []))
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
    
    def _point_in_circumcircle(self, point: np.ndarray, tri: MTri3) -> bool:
        """检查点是否在外接圆内。"""
        if tri.circumradius is None:
            self._compute_circumcircle(tri)
        
        v0, v1, v2 = tri.vertices
        return incircle(
            self.points[v0], self.points[v1], 
            self.points[v2], point
        ) > 0
    
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
        
        return MTri3(p1, p2, p3, idx=self._next_tri_id())
    
    def _triangulate(self) -> List[MTri3]:
        """初始三角剖分。"""
        super_tri = self._create_super_triangle()
        triangles = [super_tri]
        real_count = len(self.points) - 3
        
        verbose(f"  超级三角形：{super_tri.vertices}, 半径：{super_tri.circumradius:.2f}")
        
        for i in range(real_count):
            point = self.points[i]
            
            # 查找包含点的三角形
            containing_tri = None
            for tri in triangles:
                if not tri.deleted and point_in_triangle(point, tri.vertices, self.points):
                    containing_tri = tri
                    break
            
            if containing_tri is None:
                for tri in triangles:
                    if not tri.deleted and self._point_in_circumcircle(point, tri):
                        containing_tri = tri
                        break
            
            if containing_tri is None:
                continue
            
            # Cavity 搜索
            cavity_tris, shell_edges = recur_find_cavity(
                containing_tri, point, i, self.points,
                self.protected_edges, self._point_in_circumcircle
            )
            
            # 插入点
            new_tris, success = insert_vertex(
                shell_edges, cavity_tris, i, self.points, triangles,
                self._next_tri_id, self._compute_circumcircle,
                validate_star=False, validate_edges=False
            )
        
        # 删除超级三角形
        super_verts = set(super_tri.vertices)
        self.triangles = [
            tri for tri in triangles 
            if not any(v in super_verts for v in tri.vertices)
        ]
        self.points = self.points[:-3]
        
        # 构建邻接关系
        build_adjacency(self.triangles)
        
        return self.triangles
    
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
                verbose(f"  [进度] 迭代 {iteration} | 节点：{point_count} | 三角形：{tri_count}")
            
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
            in_hole = any(point_in_polygon(new_point, hole) for hole in self.holes)
            if in_hole:
                continue
            
            # 插入新点
            new_idx = len(self.points)
            self.points = np.vstack([self.points, new_point])
            
            # 找到包含新点的三角形
            containing_tri = worst_tri
            
            # Cavity 搜索
            cavity_tris, shell_edges = recur_find_cavity(
                containing_tri, new_point, new_idx, self.points,
                self.protected_edges, self._point_in_circumcircle
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
        
        verbose(f"  [完成] 插点完成 | 节点：{len(self.points)} | 三角形：{len(self.triangles)}")
    
    def _constrained_delaunay_triangulation(self):
        """Constrained Delaunay Triangulation。"""
        verbose("开始 Constrained Delaunay 边界恢复.
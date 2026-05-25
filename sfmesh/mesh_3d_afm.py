"""
三维曲面阵面推进法（3D AFM）模块

基于阵面推进法在三维曲面上生成三角形网格，包含：
- SurfaceMeshGenerator: 类接口，支持相交检测、曲率自适应、line_mesh 边界
"""
import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from OCC.Core.TopoDS import TopoDS_Face

from .surface_front import (
    SurfaceFront,
    NodeElement3D,
    SurfaceTriangle,
    create_initial_fronts_from_surface
)
from .surface_geometry import SurfaceGeometry
from .sizing_field import SurfaceSizingField
from .mesh_quality import SurfaceMeshQuality, check_edge_triangle_intersection
from .geom_utils import _are_coplanar_triangles_overlapping

from utils.message import info, debug
from utils.timer import TimeSpan
from data_structure.rtree_space import (
    build_space_index_3d_with_RTree,
    get_candidate_elements_id_3d,
    add_elems_to_space_index_3d_with_RTree
)


class SurfaceMeshGenerator:
    """
    曲面网格生成器
    
    使用阵面推进法在三维曲面上生成三角形网格
    """
    
    def __init__(
        self,
        surface: TopoDS_Face,
        global_spacing: float = 1.0,
        min_spacing: float = 0.01,
        max_spacing: float = 100.0,
        curvature_adaptation: bool = True,
        quality_threshold: float = 0.3,
        max_iterations: int = 100000,
        debug_level: int = 0,
        line_mesh: dict = None,
    ):
        """
        初始化曲面网格生成器

        Args:
            surface: OCC 曲面对象
            global_spacing: 全局网格尺寸
            min_spacing: 最小网格尺寸
            max_spacing: 最大网格尺寸
            curvature_adaptation: 是否启用曲率自适应
            quality_threshold: 质量阈值
            max_iterations: 最大迭代次数
            debug_level: 调试级别
            line_mesh: 预离散线网格（discretize_shape_edges 的输出），用于跨面共享边界
        """
        self.surface = surface
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.debug_level = debug_level
        self._line_mesh = line_mesh
        
        self.geometry = SurfaceGeometry()
        self.sizing_field = SurfaceSizingField(
            global_spacing=global_spacing,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            curvature_adaptation=curvature_adaptation,
            geometry_handler=self.geometry
        )
        
        self.front_list: List[SurfaceFront] = []
        self.node_list: List[NodeElement3D] = []
        self.triangle_list: List[SurfaceTriangle] = []
        
        self.node_coords: List[Tuple[float, float, float]] = []
        self.node_dict: Dict[int, NodeElement3D] = {}
        self.node_hash_set: Set[int] = set()
        self.triangle_set: Set[frozenset] = set()

        self.space_index_node = None
        self.space_index_front = None
        self.space_index_triangle = None
        self._triangle_dict: Dict[int, SurfaceTriangle] = {}

        self.num_nodes = 0
        self.num_triangles = 0
        
        self.search_radius_factor = 3.0
        self.quality_discount = 0.8
        
        self._initialize()
    
    def _initialize(self):
        """初始化网格生成器（3D AFM）"""
        info("初始化曲面网格生成器...")

        if self._line_mesh is not None:
            from .surface_front import create_fronts_from_line_mesh
            self.front_list = create_fronts_from_line_mesh(
                self.surface, self.geometry, self._line_mesh
            )
        else:
            self.front_list = create_initial_fronts_from_surface(
                self.surface,
                self.geometry,
                self.sizing_field
            )

        heapq.heapify(self.front_list)

        for front in self.front_list:
            for node in front.node_elems:
                if node.hash not in self.node_hash_set:
                    self.node_hash_set.add(node.hash)
                    self.node_list.append(node)
                    self.node_coords.append(node.coords)
                    self.node_dict[node.idx] = node
                    self.num_nodes = max(self.num_nodes, node.idx + 1)

        self._build_space_index()

        info(f"初始阵面数量: {len(self.front_list)}")
        info(f"初始节点数量: {len(self.node_list)}")
    
    def _build_space_index(self):
        """构建空间索引"""
        if self.node_list:
            self.node_dict, self.space_index_node = build_space_index_3d_with_RTree(
                self.node_list
            )
        
        if self.front_list:
            _, self.space_index_front = build_space_index_3d_with_RTree(
                list(self.front_list)
            )
        
        if self.triangle_list:
            self._triangle_dict, self.space_index_triangle = build_space_index_3d_with_RTree(
                self.triangle_list
            )
    
    def generate(self) -> List[SurfaceTriangle]:
        """
        使用 3D 阵面推进法（AFM）生成曲面网格

        Returns:
            生成的三角形列表
        """
        timer = TimeSpan("开始曲面网格生成...")

        iteration = 0
        while self.front_list and iteration < self.max_iterations:
            iteration += 1
            
            base_front = heapq.heappop(self.front_list)
            
            spacing = self.sizing_field.compute_front_spacing(base_front, self.surface)
            
            ideal_point, ideal_uv = self._compute_ideal_point(base_front, spacing)

            candidates = self._search_candidates(ideal_point, spacing)
            
            selected_node = self._select_best_node(base_front, ideal_point, candidates)
            
            if selected_node is None:
                base_front.al *= 1.2
                if base_front.al < 20:
                    heapq.heappush(self.front_list, base_front)
                continue
            
            if self._check_intersection(base_front, selected_node):
                base_front.al *= 1.2
                if base_front.al < 20:
                    heapq.heappush(self.front_list, base_front)
                continue
            
            self._update_mesh(base_front, selected_node)
            
            if iteration % 100 == 0:
                info(f"迭代 {iteration}: 阵面数={len(self.front_list)}, "
                     f"节点数={len(self.node_list)}, 三角形数={len(self.triangle_list)}")
        
        timer.show_to_console("曲面网格生成完成")
        
        self._print_statistics()
        
        return self.triangle_list
    
    def _compute_ideal_point(
        self,
        front: SurfaceFront,
        spacing: float
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float]]:
        """
        计算理想点：从阵面中点沿切平面垂直方向前进，投影到曲面

        Args:
            front: 当前阵面
            spacing: 网格尺寸

        Returns:
            (理想点坐标, 参数坐标)
        """
        distance = self.sizing_field.compute_ideal_point_distance(front, self.surface)
        return self.geometry.compute_ideal_point_on_surface(
            front.center, front.tangent_normal, distance, self.surface,
        )
    
    def _search_candidates(
        self,
        ideal_point: Tuple[float, float, float],
        spacing: float
    ) -> List[NodeElement3D]:
        """
        在理想点周围搜索候选节点

        以理想点为中心、0.6 * spacing 为半径的球内所有已有节点。

        Args:
            ideal_point: 理想点坐标
            spacing: 当地网格尺寸

        Returns:
            候选节点列表
        """
        if self.space_index_node is None:
            return []

        search_radius = 0.6 * spacing
        px, py, pz = ideal_point
        query_bbox = (
            px - search_radius, py - search_radius, pz - search_radius,
            px + search_radius, py + search_radius, pz + search_radius,
        )
        candidate_ids = list(self.space_index_node.intersection(query_bbox))

        r_sq = search_radius * search_radius
        candidates = []
        for nid in candidate_ids:
            if nid not in self.node_dict:
                continue
            node = self.node_dict[nid]
            dx = node.coords[0] - px
            dy = node.coords[1] - py
            dz = node.coords[2] - pz
            if dx * dx + dy * dy + dz * dz <= r_sq:
                candidates.append(node)

        return candidates
    
    def _select_best_node(
        self,
        front: SurfaceFront,
        ideal_point: Tuple[float, float, float],
        candidates: List[NodeElement3D]
    ) -> Optional[NodeElement3D]:
        """
        选择最佳节点

        Args:
            front: 当前阵面
            ideal_point: 理想点
            candidates: 候选节点列表

        Returns:
            最佳节点，如果没有合适的返回None
        """
        p0 = np.array(front.node_elems[0].coords)
        p1 = np.array(front.node_elems[1].coords)
        front_center = np.array(front.center)
        tangent = np.array(front.tangent_normal)

        front_len = np.linalg.norm(p1 - p0)
        min_height = front_len * 0.05  # 退化三角形高度阈值
        min_edge_len = self.sizing_field.global_spacing * 0.5  # 最小边长阈值

        # 理想点在阵面的哪一侧（正=推进方向）
        ideal_side = np.dot(np.array(ideal_point) - front_center, tangent)

        scored_candidates = []

        front_hash_key = frozenset([front.node_elems[0].hash, front.node_elems[1].hash])

        for node in candidates:
            if node.idx == front.node_elems[0].idx or node.idx == front.node_elems[1].idx:
                continue

            p2 = np.array(node.coords)

            # 拒绝已存在的三角形（防止重叠）
            tri_key = frozenset([front.node_elems[0].hash, front.node_elems[1].hash, node.hash])
            if tri_key in self.triangle_set:
                continue

            # 拒绝在阵面背面的候选节点（与理想点反向）
            candidate_side = np.dot(p2 - front_center, tangent)
            if ideal_side > 1e-12 and candidate_side < -1e-12:
                continue
            if ideal_side < -1e-12 and candidate_side > 1e-12:
                continue

            # 拒绝距离阵面边过近的候选节点（退化三角形）
            edge_vec = p1 - p0
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq > 1e-24:
                t = np.dot(p2 - p0, edge_vec) / edge_len_sq
                closest = p0 + np.clip(t, 0, 1) * edge_vec
                height = np.linalg.norm(p2 - closest)
                if height < min_height:
                    continue

            # 拒绝边长过短的候选节点（防止级联细分）
            d02 = np.linalg.norm(p2 - p0)
            d12 = np.linalg.norm(p2 - p1)
            if d02 < min_edge_len or d12 < min_edge_len:
                continue

            quality = self._compute_triangle_quality(p0, p1, p2)

            if quality > 0.1:
                scored_candidates.append((quality, node))

        ideal_node = self._create_ideal_node(ideal_point)
        ideal_quality = self._compute_triangle_quality(p0, p1, np.array(ideal_point))
        ideal_quality *= self.quality_discount

        if ideal_quality > 0:
            scored_candidates.append((ideal_quality, ideal_node))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        for quality, node in scored_candidates:
            return node

        return None
    
    def _compute_triangle_quality(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray
    ) -> float:
        """
        计算三角形质量
        
        Args:
            p0, p1, p2: 三角形顶点
        
        Returns:
            质量值
        """
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p0 - p2)
        
        if a < 1e-12 or b < 1e-12 or c < 1e-12:
            return 0.0
        
        s = (a + b + c) / 2.0
        
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            return 0.0
        
        area = np.sqrt(area_sq)
        
        sum_sq = a * a + b * b + c * c
        
        quality = 4.0 * np.sqrt(3) * area / sum_sq
        
        return min(1.0, max(0.0, quality))
    
    def _create_ideal_node(
        self,
        point: Tuple[float, float, float]
    ) -> NodeElement3D:
        """
        创建理想点节点
        
        Args:
            point: 点坐标
        
        Returns:
            节点对象
        """
        uv = self.geometry.project_point_to_surface(point, self.surface)
        normal = self.geometry.get_surface_normal(uv[0], uv[1], self.surface)
        
        node = NodeElement3D(
            coords=point,
            idx=self.num_nodes,
            surface=self.surface,
            uv_params=uv,
            normal=normal
        )
        
        return node
    
    def _check_intersection(
        self,
        front: SurfaceFront,
        node: NodeElement3D
    ) -> bool:
        """
        检查相交（RTree 加速 + 共面重叠检测）

        Args:
            front: 当前阵面
            node: 候选节点

        Returns:
            是否相交
        """
        p0 = np.array(front.node_elems[0].coords)
        p1 = np.array(front.node_elems[1].coords)
        p2 = np.array(node.coords)

        new_edges = [(p0, p2), (p2, p1)]

        shared_node_ids = {front.node_elems[0].idx, front.node_elems[1].idx, node.idx}

        if self.space_index_triangle is None or len(self.triangle_list) == 0:
            return False

        # RTree 查询候选三角形包围盒
        all_pts = np.array([p0, p1, p2])
        padding = 1e-6
        query_bbox = (
            all_pts[:, 0].min() - padding, all_pts[:, 1].min() - padding, all_pts[:, 2].min() - padding,
            all_pts[:, 0].max() + padding, all_pts[:, 1].max() + padding, all_pts[:, 2].max() + padding,
        )

        candidate_ids = list(self.space_index_triangle.intersection(query_bbox))

        surface_normal = np.array(front.normal)

        for tri_id in candidate_ids:
            if tri_id not in self._triangle_dict:
                continue
            existing_tri = self._triangle_dict[tri_id]

            # 共享 ≥2 个节点：合法相邻三角形，跳过
            existing_node_ids = set(existing_tri.node_ids)
            shared_count = len(existing_node_ids & shared_node_ids)
            if shared_count >= 2:
                continue

            # 边-三角形相交检测
            for edge_start, edge_end in new_edges:
                if check_edge_triangle_intersection(edge_start, edge_end, existing_tri):
                    return True

            # 共面重叠检测（共享 1 个顶点时可能发生重叠）
            existing_pts = [n.coords for n in existing_tri.nodes]
            new_pts = [tuple(p0), tuple(p1), tuple(p2)]
            if _are_coplanar_triangles_overlapping(new_pts, existing_pts, surface_normal):
                return True

        return False
    
    def _update_mesh(
        self,
        front: SurfaceFront,
        node: NodeElement3D
    ):
        """
        更新网格数据

        Args:
            front: 当前阵面
            node: 选中的节点
        """
        new_node_added = False
        if node.hash not in self.node_hash_set:
            self.node_hash_set.add(node.hash)
            self.node_list.append(node)
            self.node_coords.append(node.coords)
            self.node_dict[node.idx] = node
            self.num_nodes += 1
            new_node_added = True

        triangle = SurfaceTriangle(
            front.node_elems[0],
            front.node_elems[1],
            node,
            surface=self.surface,
            idx=self.num_triangles
        )
        self.triangle_list.append(triangle)
        self.triangle_set.add(frozenset([
            front.node_elems[0].hash, front.node_elems[1].hash, node.hash
        ]))
        self.num_triangles += 1

        # 更新空间索引
        if new_node_added:
            if self.space_index_node is not None:
                self.space_index_node, self.node_dict = add_elems_to_space_index_3d_with_RTree(
                    [node], self.space_index_node, self.node_dict
                )

        if self.space_index_triangle is None:
            self._triangle_dict, self.space_index_triangle = build_space_index_3d_with_RTree([triangle])
        else:
            self.space_index_triangle, self._triangle_dict = add_elems_to_space_index_3d_with_RTree(
                [triangle], self.space_index_triangle, self._triangle_dict,
            )

        new_front1 = SurfaceFront(
            front.node_elems[0],
            node,
            surface=self.surface,
            idx=len(self.front_list) + 1,
            bc_type="interior"
        )

        new_front2 = SurfaceFront(
            node,
            front.node_elems[1],
            surface=self.surface,
            idx=len(self.front_list) + 2,
            bc_type="interior"
        )

        min_front_len = self.sizing_field.global_spacing * 0.1
        if new_front1.length > min_front_len:
            heapq.heappush(self.front_list, new_front1)
        if new_front2.length > min_front_len:
            heapq.heappush(self.front_list, new_front2)
    
    def _print_statistics(self):
        """打印统计信息"""
        if self.triangle_list:
            SurfaceMeshQuality.evaluate_mesh(self.triangle_list, verbose=True)
    
    def export_to_vtk(self, filename: str):
        """
        导出为 Legacy ASCII VTK 格式

        Args:
            filename: 输出文件名（.vtk）
        """
        self._export_to_vtk_simple(filename)
    
    def _export_to_vtk_simple(self, filename: str):
        """
        简单VTK导出（不依赖VTK库）
        
        Args:
            filename: 输出文件名
        """
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Surface Mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            f.write(f"POINTS {len(self.node_list)} float\n")
            for node in self.node_list:
                f.write(f"{node.coords[0]} {node.coords[1]} {node.coords[2]}\n")
            
            f.write(f"\nCELLS {len(self.triangle_list)} {4 * len(self.triangle_list)}\n")
            for tri in self.triangle_list:
                f.write(f"3 {tri.nodes[0].idx} {tri.nodes[1].idx} {tri.nodes[2].idx}\n")
            
            f.write(f"\nCELL_TYPES {len(self.triangle_list)}\n")
            for _ in self.triangle_list:
                f.write("5\n")
        
        info(f"网格已导出到: {filename}")

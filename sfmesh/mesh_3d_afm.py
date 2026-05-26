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
from .mesh_quality import SurfaceMeshQuality
from .geom_utils import (
    point_in_triangle_3d, 
    segment_segment_distance_3d,
    check_triangle_intersection,
    _edge_triangle_intersection,
    _segments_intersect_3d,
    check_edge_triangle_intersection
)

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
        self.edge_count: Dict[frozenset, int] = {}

        self.num_nodes = 0
        self.num_triangles = 0

        self.search_radius_factor = 3.0
        self.quality_discount = 0.8

        # 曲面参数域边界（用于 UV 越界检测）
        self._surface_bounds = self._get_surface_bounds()

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

    def _get_surface_bounds(self):
        """获取曲面参数域边界"""
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        adaptor = BRepAdaptor_Surface(self.surface)
        return (
            adaptor.FirstUParameter(),
            adaptor.LastUParameter(),
            adaptor.FirstVParameter(),
            adaptor.LastVParameter(),
        )

    def _uv_out_of_bounds(self, uv, margin=0.1):
        """检查 UV 是否超出面的参数域"""
        u_min, u_max, v_min, v_max = self._surface_bounds
        u_range = u_max - u_min
        v_range = v_max - v_min
        u_margin = max(u_range * margin, 1e-6)
        v_margin = max(v_range * margin, 1e-6)
        u, v = uv
        return (u < u_min - u_margin or u > u_max + u_margin or
                v < v_min - v_margin or v > v_max + v_margin)
    
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

            if not self._update_mesh(base_front, selected_node):
                # UV 越界，跳过
                base_front.al *= 1.2
                if base_front.al < 20:
                    heapq.heappush(self.front_list, base_front)
                continue
            
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

        search_radius = 1.5 * spacing
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
        min_height = front_len * 0.01  # 退化三角形高度阈值（更宽松）
        min_edge_len = self.sizing_field.global_spacing * 0.3  # 最小边长阈值（降低）

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

            # 法向一致性检查：新三角形法向应与曲面法向夹角 < 90度
            new_tri_normal = np.cross(p1 - p0, p2 - p0)
            normal_norm = np.linalg.norm(new_tri_normal)
            if normal_norm > 1e-12:
                new_tri_normal = new_tri_normal / normal_norm
                # 获取新三角形重心处的曲面法向
                tri_center = (p0 + p1 + p2) / 3.0
                try:
                    tri_uv = self.geometry.project_point_to_surface(tuple(tri_center), self.surface)
                    surf_normal = self.geometry.get_surface_normal(tri_uv[0], tri_uv[1], self.surface)
                    surf_normal = np.array(surf_normal)
                    surf_norm = np.linalg.norm(surf_normal)
                    if surf_norm > 1e-12:
                        surf_normal = surf_normal / surf_norm
                        if np.dot(new_tri_normal, surf_normal) < 0.1:  # 允许小角度偏差
                            continue  # 法向不一致，拒绝
                except Exception:
                    pass  # 投影失败，跳过检查

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

        # 创建理想节点（仅当 UV 不越界时）
        ideal_node = self._create_ideal_node(ideal_point)
        
        if ideal_node is not None:
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
            节点对象，如果UV越界则返回None
        """
        uv = self.geometry.project_point_to_surface(point, self.surface)
        
        # 检查UV是否在参数域内
        u_min, u_max, v_min, v_max = self._get_surface_bounds()
        margin = 0.05  # 允许的小边界裕度
        if (uv[0] < u_min - margin or uv[0] > u_max + margin or
            uv[1] < v_min - margin or uv[1] > v_max + margin):
            return None
        
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
        检查新三角形是否与已有网格相交

        3D 混合策略：
        1. 边饱和检查（manifold 约束）
        2. 退化三角形检测
        3. RTree 3D 候选查询
        4. 按共享节点数分级检测：
           - shared_count >= 2: 共享边，合法邻接，跳过
           - shared_count == 1: 共享一个节点，检查非共享边是否交叉
           - shared_count == 0: 无共享节点，检查所有边对 + 点包含

        Args:
            front: 当前阵面
            node: 候选节点

        Returns:
            是否相交
        """
        n0 = front.node_elems[0]
        n1 = front.node_elems[1]
        n2 = node

        # 边饱和检查：每条边最多 2 个三角形
        for eh in [frozenset([n0.hash, n1.hash]), frozenset([n0.hash, n2.hash]), frozenset([n1.hash, n2.hash])]:
            if self.edge_count.get(eh, 0) >= 2:
                return True

        shared_node_ids = {n0.idx, n1.idx, n2.idx}

        if self.space_index_triangle is None or len(self.triangle_list) == 0:
            return False

        p0 = np.array(n0.coords)
        p1 = np.array(n1.coords)
        p2 = np.array(n2.coords)

        # 退化三角形检测
        new_tri_normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(new_tri_normal) < 1e-12:
            return True

        # RTree 查询候选三角形
        all_pts = np.array([p0, p1, p2])
        padding = self.sizing_field.global_spacing * 0.5
        query_bbox = (
            all_pts[:, 0].min() - padding,
            all_pts[:, 1].min() - padding,
            all_pts[:, 2].min() - padding,
            all_pts[:, 0].max() + padding,
            all_pts[:, 1].max() + padding,
            all_pts[:, 2].max() + padding,
        )
        candidate_ids = list(self.space_index_triangle.intersection(query_bbox))

        # 创建临时三角形用于相交检测
        new_tri = SurfaceTriangle(n0, n1, n2)

        for tri_id in candidate_ids:
            if tri_id not in self._triangle_dict:
                continue
            existing_tri = self._triangle_dict[tri_id]

            existing_node_ids = set(existing_tri.node_ids)
            shared_count = len(existing_node_ids & shared_node_ids)

            if shared_count >= 2:
                continue  # 共享边，合法邻接
            
            if shared_count == 1:
                # 共享一个顶点：需要检查所有边对是否相交
                # 获取共享顶点
                shared_idx = (existing_node_ids & shared_node_ids).pop()
                
                # 新三角形的所有边
                new_nodes = [n0, n1, n2]
                new_edges = []
                for a, b in [(0, 1), (1, 2), (2, 0)]:
                    new_edges.append((new_nodes[a], new_nodes[b]))
                
                # 现有三角形的所有边
                ex_nodes = existing_tri.nodes
                ex_edges = []
                for a, b in [(0, 1), (1, 2), (2, 0)]:
                    ex_edges.append((ex_nodes[a], ex_nodes[b]))
                
                # 检查所有边对（排除共享顶点的边对）
                for new_node_a, new_node_b in new_edges:
                    for ex_node_a, ex_node_b in ex_edges:
                        # 跳过共享顶点的边对
                        new_edge_ids = {new_node_a.idx, new_node_b.idx}
                        ex_edge_ids = {ex_node_a.idx, ex_node_b.idx}
                        if len(new_edge_ids & ex_edge_ids) > 0:
                            continue  # 这两条边共享顶点，跳过
                        
                        # 检查两条边是否相交
                        a1 = np.array(new_node_a.coords)
                        a2 = np.array(new_node_b.coords)
                        b1 = np.array(ex_node_a.coords)
                        b2 = np.array(ex_node_b.coords)
                        
                        if _segments_intersect_3d(a1, a2, b1, b2):
                            return True
                
                # 检查所有边是否与对方三角形内部相交（包括经过共享顶点的边）
                ex_pts = [np.array(ex_nodes[k].coords) for k in range(3)]
                new_pts = [p0, p1, p2]
                
                # 新三角形的每条边 vs 现有三角形
                for new_node_a, new_node_b in new_edges:
                    a1 = np.array(new_node_a.coords)
                    a2 = np.array(new_node_b.coords)
                    if _edge_triangle_intersection(a1, a2, ex_pts[0], ex_pts[1], ex_pts[2]):
                        return True
                
                # 现有三角形的每条边 vs 新三角形
                for ex_node_a, ex_node_b in ex_edges:
                    a1 = np.array(ex_node_a.coords)
                    a2 = np.array(ex_node_b.coords)
                    if _edge_triangle_intersection(a1, a2, new_pts[0], new_pts[1], new_pts[2]):
                        return True
                
                continue  # 共享顶点且不相交，合法邻接

            # shared_count == 0: 无共享节点，检查是否相交
            if check_triangle_intersection(new_tri, existing_tri):
                return True

        return False
    
    def _update_mesh(
        self,
        front: SurfaceFront,
        node: NodeElement3D
    ) -> bool:
        """
        更新网格数据

        Args:
            front: 当前阵面
            node: 选中的节点

        Returns:
            是否成功（False 表示节点 UV 严重越界被拒绝）
        """
        # UV 越界检查：严重越界则拒绝（防止环绕），不做 clamp（clamp 会破坏已验证的几何）
        if node.uv_params:
            if self._uv_out_of_bounds(node.uv_params, margin=0.2):
                return False

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

        # 更新边计数
        n0h = front.node_elems[0].hash
        n1h = front.node_elems[1].hash
        n2h = node.hash
        for eh in [frozenset([n0h, n1h]), frozenset([n0h, n2h]), frozenset([n1h, n2h])]:
            self.edge_count[eh] = self.edge_count.get(eh, 0) + 1

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

        return True

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

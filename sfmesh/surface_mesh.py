"""
曲面网格生成主模块

基于阵面推进法(Advancing Front Method)的三维曲面三角形网格生成
"""
import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE

from .surface_front import (
    SurfaceFront,
    NodeElement3D,
    SurfaceTriangle,
    create_initial_fronts_from_surface
)
from .surface_geometry import SurfaceGeometry
from .sizing_field import SurfaceSizingField, AdaptiveSizingField
from .mesh_quality import SurfaceMeshQuality, check_edge_triangle_intersection

from utils.message import info, debug, warning, error
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
        debug_level: int = 0
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
        """
        self.surface = surface
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.debug_level = debug_level
        
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

        self.num_nodes = 0
        self.num_triangles = 0
        
        self.search_radius_factor = 3.0
        self.quality_discount = 0.8
        
        self._initialize()
    
    def _initialize(self):
        """初始化网格生成器"""
        from .occ_utils import _is_closed_surface, _is_planar_face, _is_cylinder_face

        self._is_closed = _is_closed_surface(self.surface)
        self._is_planar = _is_planar_face(self.surface)
        self._is_cylinder = _is_cylinder_face(self.surface)

        if self._is_closed:
            info("初始化曲面网格生成器... (闭合曲面，使用参数化方法)")
            self.front_list = []
            return

        if self._is_cylinder:
            info("初始化曲面网格生成器... (圆柱面，使用2D展开流水线)")
            self.front_list = []
            return

        if self._is_planar:
            from .occ_utils import _is_disk_face
            if _is_disk_face(self.surface):
                info("初始化曲面网格生成器... (圆盘面，使用2D流水线)")
                self.front_list = []
                return

        info("初始化曲面网格生成器...")

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
            _, self.space_index_triangle = build_space_index_3d_with_RTree(
                self.triangle_list
            )
    
    def generate(self) -> List[SurfaceTriangle]:
        """
        执行网格生成

        - 闭合曲面（球面、椭球面等）：使用参数化网格方法
        - 平面：使用参数化网格方法
        - 圆柱面：使用 2D 展开流水线
        - 其他曲面：使用阵面推进法（AFM）

        Returns:
            生成的三角形列表
        """
        timer = TimeSpan("开始曲面网格生成...")

        if self._is_closed:
            from .primitives import _mesh_face_closed_surface
            spacing = self.sizing_field.global_spacing
            triangles, nodes = _mesh_face_closed_surface(self.surface, spacing, 0)
            self.triangle_list = triangles
            self.node_list = nodes
            self.num_triangles = len(triangles)
            self.num_nodes = len(nodes)
            timer.show_to_console("曲面网格生成完成（闭合曲面参数化）")
            self._print_statistics()
            return self.triangle_list

        if self._is_planar:
            from .occ_utils import _is_disk_face, _extract_disk_params
            if _is_disk_face(self.surface):
                from .pipeline_2d import _mesh_disk_2d
                params = _extract_disk_params(self.surface)
                spacing = self.sizing_field.global_spacing
                cx, cy = params['center'][0], params['center'][1]
                z = params['center'][2]
                nx, ny, nz = params['normal']
                normal_z = 1.0 if nz >= 0 else -1.0
                triangles, nodes = _mesh_disk_2d(
                    center_xy=(cx, cy), radius=params['radius'], z=z,
                    spacing=spacing, face_name="disk", normal_z=normal_z,
                )
                timer_msg = "曲面网格生成完成（圆盘2D流水线）"
            else:
                from .primitives import _mesh_face_parametric
                spacing = self.sizing_field.global_spacing
                triangles, nodes = _mesh_face_parametric(self.surface, spacing, 0)
                timer_msg = "曲面网格生成完成（平面参数化）"
            self.triangle_list = triangles
            self.node_list = nodes
            self.num_triangles = len(triangles)
            self.num_nodes = len(nodes)
            timer.show_to_console(timer_msg)
            self._print_statistics()
            return self.triangle_list

        if self._is_cylinder:
            from .pipeline_2d import _mesh_cylinder_unified
            from .occ_utils import _extract_cylinder_params
            params = _extract_cylinder_params(self.surface)
            spacing = self.sizing_field.global_spacing
            triangles, nodes, *_ = _mesh_cylinder_unified(
                base_center=params['base_center'],
                radius=params['radius'],
                height=params['height'],
                spacing=spacing,
            )
            self.triangle_list = triangles
            self.node_list = nodes
            self.num_triangles = len(triangles)
            self.num_nodes = len(nodes)
            timer.show_to_console("曲面网格生成完成（圆柱统一2D流水线）")
            self._print_statistics()
            return self.triangle_list

        iteration = 0
        while self.front_list and iteration < self.max_iterations:
            iteration += 1
            
            base_front = heapq.heappop(self.front_list)
            
            spacing = self.sizing_field.compute_front_spacing(base_front, self.surface)
            
            ideal_point, ideal_uv = self._compute_ideal_point(base_front, spacing)
            
            candidates = self._search_candidates(base_front, spacing)
            
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
        计算理想点
        
        Args:
            front: 当前阵面
            spacing: 网格尺寸
        
        Returns:
            (理想点坐标, 参数坐标)
        """
        distance = self.sizing_field.compute_ideal_point_distance(front, self.surface)
        
        ideal_point, ideal_uv = self.geometry.compute_ideal_point_on_surface(
            front.center,
            front.normal,
            front.tangent_normal,
            distance,
            self.surface
        )
        
        return ideal_point, ideal_uv
    
    def _search_candidates(
        self,
        front: SurfaceFront,
        spacing: float
    ) -> List[NodeElement3D]:
        """
        搜索候选节点
        
        Args:
            front: 当前阵面
            spacing: 网格尺寸
        
        Returns:
            候选节点列表
        """
        search_radius = front.al * spacing
        
        if self.space_index_node is None:
            return []
        
        candidate_ids = get_candidate_elements_id_3d(
            front, self.space_index_node, search_radius
        )
        
        candidates = [self.node_dict[nid] for nid in candidate_ids if nid in self.node_dict]
        
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
        检查相交

        Args:
            front: 当前阵面
            node: 候选节点

        Returns:
            是否相交
        """
        p0 = np.array(front.node_elems[0].coords)
        p1 = np.array(front.node_elems[1].coords)
        p2 = np.array(node.coords)

        new_edges = [
            (p0, p2),
            (p2, p1)
        ]

        # 只检查与最近三角形的相交，排除共享顶点的情况
        for edge_start, edge_end in new_edges:
            for triangle in self.triangle_list[-100:]:
                # 排除共享顶点的三角形
                tri_node_ids = set(triangle.node_ids)
                front_node_ids = {front.node_elems[0].idx, front.node_elems[1].idx}
                if node.idx in tri_node_ids:
                    continue
                if tri_node_ids & front_node_ids:
                    continue

                if check_edge_triangle_intersection(edge_start, edge_end, triangle):
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

        if self.space_index_triangle is not None:
            self.space_index_triangle, _ = add_elems_to_space_index_3d_with_RTree(
                [triangle], self.space_index_triangle, {}
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


def generate_surface_mesh_from_file(
    filename: str,
    global_spacing: float = 1.0,
    output_vtk: str = None
) -> List[SurfaceTriangle]:
    """
    从几何文件生成曲面网格

    Args:
        filename: 几何文件路径 (IGES/STEP)
        global_spacing: 全局网格尺寸
        output_vtk: 输出VTK文件路径（可选）

    Returns:
        生成的三角形列表
    """
    from fileIO.geometry_io import import_geometry_file

    shape = import_geometry_file(filename)
    all_triangles = generate_surface_mesh_from_shape(shape, global_spacing)

    if output_vtk:
        _export_combined_mesh(all_triangles, output_vtk)

    return all_triangles


def generate_surface_mesh_from_shape(
    shape: TopoDS_Shape,
    global_spacing: float = 1.0,
) -> List[SurfaceTriangle]:
    """
    从 OCC 形状生成曲面网格

    自动检测形状类型（圆柱体等），使用统一网格生成策略确保共享边界节点。

    Args:
        shape: OCC 形状对象
        global_spacing: 全局网格尺寸

    Returns:
        生成的三角形列表
    """
    from .occ_utils import _extract_faces, _is_cylinder_face, _extract_cylinder_params

    faces = _extract_faces(shape)

    # 检测是否为圆柱体（所有面都是圆柱面或圆盘面）
    cylinder_faces = [f for f in faces if _is_cylinder_face(f)]
    if len(cylinder_faces) == 1 and len(faces) == 3:
        # 标准圆柱体：1个侧面 + 2个端面
        params = _extract_cylinder_params(cylinder_faces[0])
        from .pipeline_2d import _mesh_cylinder_unified
        info("检测到圆柱体，使用统一网格生成...")
        triangles, nodes, *_ = _mesh_cylinder_unified(
            base_center=params['base_center'],
            radius=params['radius'],
            height=params['height'],
            spacing=global_spacing,
        )
        info(f"圆柱体网格: {len(triangles)} 三角形, {len(nodes)} 节点")
        return triangles

    # 通用形状：逐面生成
    all_triangles = []
    for i, face in enumerate(faces):
        info(f"\n处理曲面 {i+1}/{len(faces)}")
        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=global_spacing
        )
        triangles = generator.generate()
        all_triangles.extend(triangles)

    return all_triangles


def _export_combined_mesh(triangles: List[SurfaceTriangle], filename: str):
    """
    导出合并的网格
    
    Args:
        triangles: 三角形列表
        filename: 输出文件名
    """
    node_set = {}
    node_list = []
    
    for tri in triangles:
        for node in tri.nodes:
            if node.hash not in node_set:
                node_set[node.hash] = len(node_list)
                node_list.append(node)
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Surface Mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {len(node_list)} float\n")
        for node in node_list:
            f.write(f"{node.coords[0]} {node.coords[1]} {node.coords[2]}\n")
        
        f.write(f"\nCELLS {len(triangles)} {4 * len(triangles)}\n")
        for tri in triangles:
            idx0 = node_set[tri.nodes[0].hash]
            idx1 = node_set[tri.nodes[1].hash]
            idx2 = node_set[tri.nodes[2].hash]
            f.write(f"3 {idx0} {idx1} {idx2}\n")
        
        f.write(f"\nCELL_TYPES {len(triangles)}\n")
        for _ in triangles:
            f.write("5\n")
    
    info(f"网格已导出到: {filename}")

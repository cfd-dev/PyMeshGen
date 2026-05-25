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
from .geom_utils import _are_coplanar_triangles_overlapping
from .occ_utils import _is_point_in_face, _get_face_bbox

from utils.message import info, debug, warning, error
from utils.timer import TimeSpan
from data_structure.rtree_space import (
    build_space_index_3d_with_RTree,
    get_candidate_elements_id_3d,
    add_elems_to_space_index_3d_with_RTree
)


# ---------------------------------------------------------------------------
# 阵面推进法（Advancing Front Method）独立函数
# ---------------------------------------------------------------------------

def _compute_triangle_quality(p0, p1, p2) -> float:
    """计算三角形形状质量因子 (0~1, 1=等边三角形)"""
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
    return min(1.0, max(0.0, 4.0 * np.sqrt(3) * area / sum_sq))


def _check_intersection_afm(
    p0: np.ndarray,
    p1: np.ndarray,
    candidate_node: NodeElement3D,
    triangle_list: List[SurfaceTriangle],
    space_index_triangle,
    triangle_dict: dict,
    shared_node_ids: set,
    surface_normal: np.ndarray,
) -> bool:
    """RTree 加速的相交检测"""
    p2 = np.array(candidate_node.coords)
    new_edges = [(p0, p2), (p2, p1)]

    if space_index_triangle is None or len(triangle_list) == 0:
        return False

    # 计算候选三角形包围盒，查询 RTree
    all_pts = np.array([p0, p1, p2])
    padding = 1e-6
    query_bbox = (
        all_pts[:, 0].min() - padding, all_pts[:, 1].min() - padding, all_pts[:, 2].min() - padding,
        all_pts[:, 0].max() + padding, all_pts[:, 1].max() + padding, all_pts[:, 2].max() + padding,
    )

    candidate_ids = list(space_index_triangle.intersection(query_bbox))

    for tri_id in candidate_ids:
        if tri_id not in triangle_dict:
            continue
        existing_tri = triangle_dict[tri_id]

        existing_node_ids = set(existing_tri.node_ids)
        shared_count = len(existing_node_ids & shared_node_ids)

        # 共享2个节点（共享边）：合法相邻三角形，跳过
        if shared_count >= 2:
            continue

        # 检查每条新边是否与已有三角形相交
        for edge_start, edge_end in new_edges:
            if check_edge_triangle_intersection(edge_start, edge_end, existing_tri):
                return True

        # 共面重叠检测（共享1个顶点时可能发生重叠）
        existing_pts = [n.coords for n in existing_tri.nodes]
        new_pts = [tuple(p0), tuple(p1), tuple(p2)]
        if _are_coplanar_triangles_overlapping(new_pts, existing_pts, surface_normal):
            return True

    return False


def _select_best_node_afm(
    front: SurfaceFront,
    ideal_point: Tuple[float, float, float],
    candidates: List[NodeElement3D],
    triangle_set: set,
    triangle_list: List[SurfaceTriangle],
    space_index_triangle,
    triangle_dict: dict,
    surface_normal: np.ndarray,
    quality_discount: float = 0.8,
    boundary_node_indices: set = None,
) -> Optional[NodeElement3D]:
    """从候选节点中选择最佳节点（含相交检测）"""
    p0 = np.array(front.node_elems[0].coords)
    p1 = np.array(front.node_elems[1].coords)
    front_len = np.linalg.norm(p1 - p0)
    min_height = front_len * 0.05

    ideal_side = np.dot(np.array(ideal_point) - np.array(front.center), np.array(front.tangent_normal))
    ideal_pt = np.array(ideal_point)

    scored_candidates = []

    for node in candidates:
        if node.idx == front.node_elems[0].idx or node.idx == front.node_elems[1].idx:
            continue

        p2 = np.array(node.coords)

        # 拒绝重复三角形
        tri_key = frozenset([front.node_elems[0].hash, front.node_elems[1].hash, node.hash])
        if tri_key in triangle_set:
            continue

        # 侧向检查：候选节点必须在理想点同侧
        candidate_side = np.dot(p2 - np.array(front.center), np.array(front.tangent_normal))
        if ideal_side > 1e-12 and candidate_side < -1e-12:
            continue
        if ideal_side < -1e-12 and candidate_side > 1e-12:
            continue

        # 退化三角形检查
        edge_vec = p1 - p0
        edge_len_sq = np.dot(edge_vec, edge_vec)
        if edge_len_sq > 1e-24:
            t = np.dot(p2 - p0, edge_vec) / edge_len_sq
            closest = p0 + np.clip(t, 0, 1) * edge_vec
            height = np.linalg.norm(p2 - closest)
            if height < min_height:
                continue

        quality = _compute_triangle_quality(p0, p1, p2)
        if quality > 0.1:
            # 距离惩罚：离理想点越远，得分越低
            dist = np.linalg.norm(p2 - ideal_pt)
            dist_factor = 1.0 / (1.0 + dist / (front_len + 1e-12))
            scored_candidates.append((quality * dist_factor, node))

    if boundary_node_indices is None:
        boundary_node_indices = set()

    # 边界节点优先（促进阵面去重），同组内按质量排序
    scored_candidates.sort(
        key=lambda x: (1 if x[1].idx in boundary_node_indices else 0, x[0]),
        reverse=True,
    )

    shared_ids = {front.node_elems[0].idx, front.node_elems[1].idx}

    # 优先使用已有候选节点（需要相交检测）
    for quality, node in scored_candidates:
        if _check_intersection_afm(
            p0, p1, node, triangle_list,
            space_index_triangle, triangle_dict,
            shared_ids | {node.idx}, surface_normal,
        ):
            continue
        return node

    # 无合适已有节点时，创建理想点（新节点）
    ideal_quality = _compute_triangle_quality(p0, p1, np.array(ideal_point))
    if ideal_quality > 0.01:
        ideal_uv = None
        try:
            geom = SurfaceGeometry()
            ideal_uv = geom.project_point_to_surface(ideal_point, front.surface)
            # 检查理想点是否在面的边界内
            if not _is_point_in_face(ideal_uv[0], ideal_uv[1], front.surface):
                return None
            ideal_normal = geom.get_surface_normal(ideal_uv[0], ideal_uv[1], front.surface)
        except Exception:
            return None
        ideal_node = NodeElement3D(
            coords=ideal_point, idx=-1,
            surface=front.surface, uv_params=ideal_uv, normal=ideal_normal,
        )
        if not _check_intersection_afm(
            p0, p1, ideal_node, triangle_list,
            space_index_triangle, triangle_dict,
            shared_ids, surface_normal,
        ):
            return ideal_node

    return None


def _mesh_face_afm(
    face: TopoDS_Face,
    spacing: float,
    node_id_offset: int,
    max_iterations: int = 100000,
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    对单个面使用阵面推进法生成网格

    从面的边界出发，逐层向内部推进，每次生成一个三角形，
    直到整个面被填满。节点投影到几何曲面上确保几何保真度。

    Args:
        face: OCC TopoDS_Face
        spacing: 目标网格尺寸
        node_id_offset: 起始节点索引
        max_iterations: 安全迭代上限

    Returns:
        (triangles, nodes) 与 _mesh_face_parametric 接口一致
    """
    geometry = SurfaceGeometry()
    sizing_field = SurfaceSizingField(
        global_spacing=spacing,
        curvature_adaptation=False,
        geometry_handler=geometry,
    )

    # Step 1: 提取边界阵面
    initial_fronts = create_initial_fronts_from_surface(face, geometry, sizing_field)
    if not initial_fronts:
        return [], []

    # 计算面法向量和初始搜索半径
    face_normal = np.array(initial_fronts[0].normal)
    xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
    face_size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    initial_al = min(5.0, max(3.0, face_size / spacing * 0.5))

    # Step 2: 初始化数据结构
    front_list = list(initial_fronts)
    heapq.heapify(front_list)

    front_hash_set = set()
    for f in initial_fronts:
        f.al = initial_al
        front_hash_set.add(f.hash)

    node_list = []
    node_hash_set = set()
    node_dict = {}
    num_nodes = node_id_offset
    boundary_node_indices = set()

    for front in initial_fronts:
        for node in front.node_elems:
            if node.hash not in node_hash_set:
                node_hash_set.add(node.hash)
                node.idx = num_nodes
                node_list.append(node)
                node_dict[node.idx] = node
                boundary_node_indices.add(num_nodes)
                num_nodes += 1
            else:
                # 回填已有节点索引
                for existing in node_list:
                    if existing.hash == node.hash:
                        node.idx = existing.idx
                        break

    triangle_list = []
    triangle_set = set()

    _, space_index_node = build_space_index_3d_with_RTree(node_list)
    space_index_triangle = None
    triangle_dict = {}

    quality_discount = 0.8

    # Step 3: 主循环
    iteration = 0
    while front_list and iteration < max_iterations:
        iteration += 1
        base_front = heapq.heappop(front_list)

        # 跳过已去重的陈旧阵面
        if base_front.hash not in front_hash_set:
            continue

        # 计算间距和理想点
        spacing_local = sizing_field.compute_front_spacing(base_front, face)
        distance = sizing_field.compute_ideal_point_distance(base_front, face)
        try:
            tangent_normal = base_front.tangent_normal
            ideal_point, ideal_uv = geometry.compute_ideal_point_on_surface(
                base_front.center, tangent_normal, distance, face,
            )
            # 若理想点在面外，翻转切向再试
            if not _is_point_in_face(ideal_uv[0], ideal_uv[1], face):
                flipped = tuple(-x for x in tangent_normal)
                ideal_point, ideal_uv = geometry.compute_ideal_point_on_surface(
                    base_front.center, flipped, distance, face,
                )
        except Exception:
            base_front.al *= 1.2
            if base_front.al < 20.0:
                front_hash_set.add(base_front.hash)
                heapq.heappush(front_list, base_front)
            continue

        # 搜索候选节点
        search_radius = base_front.al * spacing_local
        candidate_ids = get_candidate_elements_id_3d(
            base_front, space_index_node, search_radius,
        )
        candidates = [node_dict[nid] for nid in candidate_ids if nid in node_dict]

        # 选择最佳节点
        selected_node = _select_best_node_afm(
            base_front, ideal_point, candidates,
            triangle_set, triangle_list,
            space_index_triangle, triangle_dict,
            face_normal, quality_discount,
            boundary_node_indices,
        )

        if selected_node is None:
            base_front.al *= 1.2
            if base_front.al < 20.0:
                front_hash_set.add(base_front.hash)
                heapq.heappush(front_list, base_front)
            continue

        # 新节点需要分配索引并加入空间索引
        if selected_node.hash not in node_hash_set:
            node_hash_set.add(selected_node.hash)
            selected_node.idx = num_nodes
            node_list.append(selected_node)
            node_dict[selected_node.idx] = selected_node
            num_nodes += 1
            space_index_node, node_dict = add_elems_to_space_index_3d_with_RTree(
                [selected_node], space_index_node, node_dict,
            )

        # 创建三角形
        triangle = SurfaceTriangle(
            base_front.node_elems[0], base_front.node_elems[1],
            selected_node, surface=face, idx=len(triangle_list),
        )
        triangle_list.append(triangle)
        triangle_set.add(frozenset([
            base_front.node_elems[0].hash, base_front.node_elems[1].hash,
            selected_node.hash,
        ]))

        # 更新三角形 RTree
        if space_index_triangle is None:
            _, space_index_triangle = build_space_index_3d_with_RTree([triangle])
            triangle_dict = {id(triangle): triangle}
        else:
            space_index_triangle, triangle_dict = add_elems_to_space_index_3d_with_RTree(
                [triangle], space_index_triangle, triangle_dict,
            )

        # 移除已消耗阵面
        front_hash_set.discard(base_front.hash)

        # 创建子阵面（含去重）
        new_front1 = SurfaceFront(
            base_front.node_elems[0], selected_node,
            surface=face, bc_type="interior",
        )
        new_front2 = SurfaceFront(
            selected_node, base_front.node_elems[1],
            surface=face, bc_type="interior",
        )

        for new_front in [new_front1, new_front2]:
            new_front.al = initial_al
            if new_front.hash in front_hash_set:
                # 该边已被另一三角形占用，移除旧阵面
                front_hash_set.discard(new_front.hash)
            else:
                front_hash_set.add(new_front.hash)
                heapq.heappush(front_list, new_front)

    return triangle_list, node_list


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

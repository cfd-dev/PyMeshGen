"""
基础几何体曲面网格生成模块

支持从 OCC 基础几何体（长方体、圆柱体）直接生成曲面网格，
无需从文件导入。使用阵面推进法（Advancing Front Method）生成高质量网格。
"""
import sys
import heapq
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fileIO.occ_loader import ensure_occ_loaded
ensure_occ_loaded()

from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Ax2, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCC.Core.TopoDS import TopoDS_Face, topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepClass import BRepClass_FaceClassifier

from .surface_mesh import _export_combined_mesh
from .surface_front import SurfaceFront, SurfaceTriangle, NodeElement3D, create_initial_fronts_from_surface
from .surface_geometry import SurfaceGeometry
from .sizing_field import SurfaceSizingField
from .mesh_quality import check_edge_triangle_intersection
from data_structure.rtree_space import (
    build_space_index_3d_with_RTree,
    get_candidate_elements_id_3d,
    add_elems_to_space_index_3d_with_RTree,
)
from data_structure.front2d import Front
from data_structure.basic_elements import NodeElementALM

from utils.message import info, warning

from .occ_utils import (
    _extract_faces, _get_face_bbox, _get_face_bbox_center,
    _classify_cube_faces, _classify_cylinder_faces,
    _is_point_in_face, _is_planar_face, _is_closed_surface,
)
from .geom_utils import (
    _project_to_2d, _segments_intersect_2d,
    _point_in_triangle_2d, _are_coplanar_triangles_overlapping,
)
from .pipeline_2d import (
    _mesh_face_2d_pipeline, _mesh_disk_2d, _mesh_lateral_cylinder_2d,
)


class PrimitiveMeshResult:
    """
    基础几何体网格生成结果

    Attributes:
        triangles: 所有三角形列表
        nodes: 所有节点列表（去重）
        face_map: 面索引到三角形列表的映射
        num_faces: 面的总数
        face_types: 每个面的类型描述
    """

    def __init__(self):
        self.triangles: List[SurfaceTriangle] = []
        self.nodes: List[NodeElement3D] = []
        self.face_map: Dict[int, List[SurfaceTriangle]] = {}
        self.face_types: Dict[int, str] = {}
        self.num_faces: int = 0


def _mesh_face_parametric(
    face: TopoDS_Face,
    spacing: float,
    node_id_offset: int,
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    对单个面进行参数化网格剖分，节点投影到几何曲面上

    利用 OCC 曲面评估器在参数空间 (u, v) 生成均匀网格，
    将参数坐标映射到三维空间，确保节点严格落在几何曲面上。
    通过 BRepClass_FaceClassifier 确保节点在面的边界内。
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

    geometry = SurfaceGeometry()
    adaptor = BRepAdaptor_Surface(face)
    u_min = adaptor.FirstUParameter()
    u_max = adaptor.LastUParameter()
    v_min = adaptor.FirstVParameter()
    v_max = adaptor.LastVParameter()

    # 利用包围盒估算物理尺寸，确定网格数
    xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    phys_len = max(dx, dy, dz)

    nu = max(6, int(round(phys_len / spacing)) + 1)
    nv = max(6, int(round(phys_len / spacing)) + 1)

    # 第一遍：生成候选节点，检查是否在面内
    node_dict = {}  # (ju, jv) -> node
    nid = node_id_offset

    for ju in range(nu):
        for jv in range(nv):
            u = u_min + ju * (u_max - u_min) / (nu - 1) if nu > 1 else u_min
            v = v_min + jv * (v_max - v_min) / (nv - 1) if nv > 1 else v_min

            if not _is_point_in_face(u, v, face):
                continue

            pnt = adaptor.Value(u, v)
            coords = (pnt.X(), pnt.Y(), pnt.Z())

            uv = geometry.project_point_to_surface(coords, face)
            proj_coords = geometry.evaluate_point(uv[0], uv[1], face)
            normal = geometry.get_surface_normal(uv[0], uv[1], face)

            node = NodeElement3D(
                coords=proj_coords, idx=nid,
                surface=face, uv_params=uv, normal=normal,
            )
            node_dict[(ju, jv)] = node
            nid += 1

    # 第二遍：生成三角形，连接相邻的面内节点
    triangles = []
    tri_idx = 0

    for ju in range(nu - 1):
        for jv in range(nv - 1):
            n00 = node_dict.get((ju, jv))
            n10 = node_dict.get((ju + 1, jv))
            n01 = node_dict.get((ju, jv + 1))
            n11 = node_dict.get((ju + 1, jv + 1))

            if n00 and n10 and n01:
                triangles.append(SurfaceTriangle(n00, n10, n01, idx=tri_idx))
                tri_idx += 1
            if n10 and n11 and n01:
                triangles.append(SurfaceTriangle(n10, n11, n01, idx=tri_idx))
                tri_idx += 1

    nodes = list(node_dict.values())
    return triangles, nodes


def _mesh_face_closed_surface(
    face: TopoDS_Face,
    spacing: float,
    node_id_offset: int,
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    对闭合曲面（球面、椭球面等）进行参数化网格剖分

    闭合曲面的参数域在极点方向有奇异性（球面 U=0/U=π 处所有 V 映射到同一点），
    直接在参数空间生成网格会产生退化三角形。

    本函数检测退化行（极点），跳过它们生成内部结构化网格，
    然后在两极用三角形扇填充。
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

    geometry = SurfaceGeometry()
    adaptor = BRepAdaptor_Surface(face)
    u_min = adaptor.FirstUParameter()
    u_max = adaptor.LastUParameter()
    v_min = adaptor.FirstVParameter()
    v_max = adaptor.LastVParameter()

    # 利用包围盒估算物理尺寸
    xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    phys_len = max(dx, dy, dz)
    n_div = max(6, int(round(phys_len / spacing)) + 1)

    # 检测哪个方向的边界是退化行（极点）
    # 采样边界行的多个点，如果全部映射到同一个 3D 点，则为退化行
    def _is_degenerate_row(u_val, n_samples=8):
        pts = []
        for i in range(n_samples):
            v = v_min + i * (v_max - v_min) / n_samples
            p = adaptor.Value(u_val, v)
            pts.append((p.X(), p.Y(), p.Z()))
        ref = pts[0]
        return all(
            ((p[0]-ref[0])**2 + (p[1]-ref[1])**2 + (p[2]-ref[2])**2) < 1e-10
            for p in pts
        )

    def _is_degenerate_col(v_val, n_samples=8):
        pts = []
        for i in range(n_samples):
            u = u_min + i * (u_max - u_min) / n_samples
            p = adaptor.Value(u, v_val)
            pts.append((p.X(), p.Y(), p.Z()))
        ref = pts[0]
        return all(
            ((p[0]-ref[0])**2 + (p[1]-ref[1])**2 + (p[2]-ref[2])**2) < 1e-10
            for p in pts
        )

    u_min_degen = _is_degenerate_row(u_min)
    u_max_degen = _is_degenerate_row(u_max)
    v_min_degen = _is_degenerate_col(v_min)
    v_max_degen = _is_degenerate_col(v_max)

    nodes = []
    node_dict = {}  # (ju, jv) -> node
    nid = node_id_offset
    triangles = []
    tri_idx = 0

    # U 方向有退化行（极点在 U 边界）→ 沿 U 方向跳过退化行
    if u_min_degen or u_max_degen:
        nu_interior = max(2, n_div - 1)
        nv = max(6, n_div)

        # 极点 1（U = u_min）
        if u_min_degen:
            pole1_coords = geometry.evaluate_point(u_min, (v_min + v_max) / 2, face)
            pole1_normal = geometry.get_surface_normal(u_min, (v_min + v_max) / 2, face)
            pole1_node = NodeElement3D(
                coords=pole1_coords, idx=nid,
                surface=face, uv_params=(u_min, (v_min + v_max) / 2),
                normal=pole1_normal,
            )
            nodes.append(pole1_node)
            nid += 1

        # 内部行（直接使用参数坐标，避免投影到极点）
        ju_start = 1 if u_min_degen else 0
        ju_end = nu_interior - 1 if u_max_degen else nu_interior
        for ju in range(ju_start, ju_end + 1):
            u = u_min + ju * (u_max - u_min) / nu_interior
            for jv in range(nv):
                v = v_min + jv * (v_max - v_min) / nv
                pnt = adaptor.Value(u, v)
                coords = (pnt.X(), pnt.Y(), pnt.Z())
                normal = geometry.get_surface_normal(u, v, face)
                node = NodeElement3D(
                    coords=coords, idx=nid,
                    surface=face, uv_params=(u, v), normal=normal,
                )
                node_dict[(ju, jv)] = node
                nodes.append(node)
                nid += 1

        # 极点 2（U = u_max）
        if u_max_degen:
            pole2_coords = geometry.evaluate_point(u_max, (v_min + v_max) / 2, face)
            pole2_normal = geometry.get_surface_normal(u_max, (v_min + v_max) / 2, face)
            pole2_node = NodeElement3D(
                coords=pole2_coords, idx=nid,
                surface=face, uv_params=(u_max, (v_min + v_max) / 2),
                normal=pole2_normal,
            )
            nodes.append(pole2_node)
            nid += 1

        # 极点 1 三角形扇
        if u_min_degen:
            for jv in range(nv):
                n1 = node_dict.get((ju_start, jv))
                n2 = node_dict.get((ju_start, (jv + 1) % nv))
                if n1 and n2:
                    triangles.append(SurfaceTriangle(pole1_node, n1, n2, idx=tri_idx))
                    tri_idx += 1

        # 中间四边形条带
        interior_keys = sorted(set(k[0] for k in node_dict))
        for idx_i in range(len(interior_keys) - 1):
            ju = interior_keys[idx_i]
            ju_next = interior_keys[idx_i + 1]
            for jv in range(nv):
                jv_next = (jv + 1) % nv
                tl = node_dict.get((ju, jv))
                tr = node_dict.get((ju, jv_next))
                bl = node_dict.get((ju_next, jv))
                br = node_dict.get((ju_next, jv_next))
                if tl and tr and bl:
                    triangles.append(SurfaceTriangle(tl, bl, tr, idx=tri_idx))
                    tri_idx += 1
                if tr and bl and br:
                    triangles.append(SurfaceTriangle(tr, bl, br, idx=tri_idx))
                    tri_idx += 1

        # 极点 2 三角形扇
        if u_max_degen:
            last_ju = interior_keys[-1]
            for jv in range(nv):
                n1 = node_dict.get((last_ju, jv))
                n2 = node_dict.get((last_ju, (jv + 1) % nv))
                if n1 and n2:
                    triangles.append(SurfaceTriangle(n1, pole2_node, n2, idx=tri_idx))
                    tri_idx += 1

        return triangles, nodes

    # V 方向有退化行（极点在 V 边界）→ 沿 V 方向跳过退化行
    elif v_min_degen or v_max_degen:
        nu = max(6, n_div)
        nv_interior = max(2, n_div - 1)

        if v_min_degen:
            pole1_coords = geometry.evaluate_point((u_min + u_max) / 2, v_min, face)
            pole1_normal = geometry.get_surface_normal((u_min + u_max) / 2, v_min, face)
            pole1_node = NodeElement3D(
                coords=pole1_coords, idx=nid,
                surface=face, uv_params=((u_min + u_max) / 2, v_min),
                normal=pole1_normal,
            )
            nodes.append(pole1_node)
            nid += 1

        jv_start = 1 if v_min_degen else 0
        jv_end = nv_interior - 1 if v_max_degen else nv_interior
        for jv in range(jv_start, jv_end + 1):
            v = v_min + jv * (v_max - v_min) / nv_interior
            for ju in range(nu):
                u = u_min + ju * (u_max - u_min) / nu
                pnt = adaptor.Value(u, v)
                coords = (pnt.X(), pnt.Y(), pnt.Z())
                normal = geometry.get_surface_normal(u, v, face)
                node = NodeElement3D(
                    coords=coords, idx=nid,
                    surface=face, uv_params=(u, v), normal=normal,
                )
                node_dict[(ju, jv)] = node
                nodes.append(node)
                nid += 1

        if v_max_degen:
            pole2_coords = geometry.evaluate_point((u_min + u_max) / 2, v_max, face)
            pole2_normal = geometry.get_surface_normal((u_min + u_max) / 2, v_max, face)
            pole2_node = NodeElement3D(
                coords=pole2_coords, idx=nid,
                surface=face, uv_params=((u_min + u_max) / 2, v_max),
                normal=pole2_normal,
            )
            nodes.append(pole2_node)
            nid += 1

        if v_min_degen:
            for ju in range(nu):
                n1 = node_dict.get((ju, jv_start))
                n2 = node_dict.get(((ju + 1) % nu, jv_start))
                if n1 and n2:
                    triangles.append(SurfaceTriangle(pole1_node, n1, n2, idx=tri_idx))
                    tri_idx += 1

        interior_keys = sorted(set(k[1] for k in node_dict))
        for idx_j in range(len(interior_keys) - 1):
            jv = interior_keys[idx_j]
            jv_next = interior_keys[idx_j + 1]
            for ju in range(nu):
                ju_next = (ju + 1) % nu
                tl = node_dict.get((ju, jv))
                tr = node_dict.get((ju_next, jv))
                bl = node_dict.get((ju, jv_next))
                br = node_dict.get((ju_next, jv_next))
                if tl and tr and bl:
                    triangles.append(SurfaceTriangle(tl, bl, tr, idx=tri_idx))
                    tri_idx += 1
                if tr and bl and br:
                    triangles.append(SurfaceTriangle(tr, bl, br, idx=tri_idx))
                    tri_idx += 1

        if v_max_degen:
            last_jv = interior_keys[-1]
            for ju in range(nu):
                n1 = node_dict.get((ju, last_jv))
                n2 = node_dict.get(((ju + 1) % nu, last_jv))
                if n1 and n2:
                    triangles.append(SurfaceTriangle(n1, pole2_node, n2, idx=tri_idx))
                    tri_idx += 1

        return triangles, nodes

    else:
        # 无退化行（如环面）— 普通参数化
        return _mesh_face_parametric(face, spacing, node_id_offset)




# ---------------------------------------------------------------------------
# 阵面推进法（Advancing Front Method）实现
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
                base_front.center, base_front.normal,
                tangent_normal, distance, face,
            )
            # 若理想点在面外，翻转切向再试
            if not _is_point_in_face(ideal_uv[0], ideal_uv[1], face):
                flipped = tuple(-x for x in tangent_normal)
                ideal_point, ideal_uv = geometry.compute_ideal_point_on_surface(
                    base_front.center, base_front.normal,
                    flipped, distance, face,
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


def _mesh_faces(
    faces: List[TopoDS_Face],
    face_types: Dict[int, str],
    spacing: float = 1.0,
) -> PrimitiveMeshResult:
    """
    对一组面逐个生成面网格

    - 平面使用参数化网格方法（高效、质量稳定）
    - 闭合曲面（球面、椭球面等）使用带极点处理的参数化方法
    - 其他曲面使用阵面推进法（AFM，从几何边界出发生成高质量网格）
    """
    result = PrimitiveMeshResult()
    result.num_faces = len(faces)
    result.face_types = face_types

    node_hash_set = set()
    node_id_offset = 0

    for i, face in enumerate(faces):
        ftype = face_types.get(i, "unknown")

        if _is_planar_face(face):
            info(f"生成面 {i + 1}/{len(faces)} ({ftype}) 网格 [参数化]...")
            triangles, face_nodes = _mesh_face_parametric(face, spacing, node_id_offset)
        elif _is_closed_surface(face):
            info(f"生成面 {i + 1}/{len(faces)} ({ftype}) 网格 [闭合曲面参数化]...")
            triangles, face_nodes = _mesh_face_closed_surface(face, spacing, node_id_offset)
        else:
            info(f"生成面 {i + 1}/{len(faces)} ({ftype}) 网格 [AFM]...")
            try:
                triangles, face_nodes = _mesh_face_afm(face, spacing, node_id_offset)
            except Exception as e:
                warning(f"AFM 失败 (面 {i}, {ftype})，回退到参数化方法: {e}")
                triangles, face_nodes = _mesh_face_parametric(face, spacing, node_id_offset)

        # 去重并收集节点
        unique_face_nodes = []
        for node in face_nodes:
            if node.hash not in node_hash_set:
                node_hash_set.add(node.hash)
                unique_face_nodes.append(node)

        result.face_map[i] = triangles
        result.triangles.extend(triangles)
        result.nodes.extend(unique_face_nodes)
        node_id_offset += len(face_nodes)

    return result


def generate_cube_mesh(
    corner1: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    corner2: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spacing: float = 0.5,
    output_vtk: str = None,
) -> PrimitiveMeshResult:
    """
    生成长方体的曲面网格

    逐面使用 2D 阵面推进流水线生成网格，共边节点通过坐标去重保持一致。

    Args:
        corner1: 长方体第一个角点坐标 (x, y, z)
        corner2: 长方体对角点坐标 (x, y, z)
        spacing: 全局网格尺寸
        output_vtk: 输出VTK文件路径（可选）

    Returns:
        PrimitiveMeshResult: 包含网格数据和面映射的结果

    Raises:
        ValueError: 如果角点坐标无效
    """
    for i in range(3):
        if abs(corner2[i] - corner1[i]) < 1e-12:
            raise ValueError(
                f"长方体第 {i} 方向长度为零: corner1[{i}]={corner1[i]}, corner2[{i}]={corner2[i]}"
            )

    x0, y0, z0 = min(corner1[0], corner2[0]), min(corner1[1], corner2[1]), min(corner1[2], corner2[2])
    x1, y1, z1 = max(corner1[0], corner2[0]), max(corner1[1], corner2[1]), max(corner1[2], corner2[2])

    # 8 个顶点
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # 底面
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # 顶面
    ]

    # 6 个面（投影到 2D 后为 CCW 排列，使 Front 左手法向量指向内部）
    face_defs = [
        ("bottom", [v[0], v[1], v[2], v[3]]),  # CCW in XY
        ("top",    [v[4], v[5], v[6], v[7]]),  # CCW in XY
        ("front",  [v[0], v[1], v[5], v[4]]),  # CCW in XZ
        ("back",   [v[3], v[2], v[6], v[7]]),  # CCW in XZ
        ("right",  [v[1], v[2], v[6], v[5]]),  # CCW in YZ
        ("left",   [v[0], v[3], v[7], v[4]]),  # CCW in YZ
    ]

    # 逐面生成网格
    face_results = []
    for face_name, corners in face_defs:
        info(f"生成面 {face_name} 网格...")
        tris, nodes = _mesh_face_2d_pipeline(corners, spacing, face_name)
        face_results.append((face_name, tris, nodes))

    # 合并结果，共边节点按坐标去重
    node_hash_to_global_idx = {}
    all_nodes = []
    all_triangles = []
    result = PrimitiveMeshResult()
    result.num_faces = 6

    global_idx = 0
    for face_idx, (face_name, tris, nodes) in enumerate(face_results):
        result.face_types[face_idx] = face_name

        # 建立当前面节点的 旧对象 → 全局索引 映射
        node_to_global = {}
        for node in nodes:
            h = node.hash
            if h not in node_hash_to_global_idx:
                node_hash_to_global_idx[h] = global_idx
                node.idx = global_idx
                all_nodes.append(node)
                global_idx += 1
            node_to_global[id(node)] = node_hash_to_global_idx[h]

        # 创建使用全局索引的三角形
        face_tris = []
        for tri in tris:
            new_tri = SurfaceTriangle(
                all_nodes[node_to_global[id(tri.nodes[0])]],
                all_nodes[node_to_global[id(tri.nodes[1])]],
                all_nodes[node_to_global[id(tri.nodes[2])]],
                surface=None, idx=len(all_triangles),
            )
            all_triangles.append(new_tri)
            face_tris.append(new_tri)

        result.face_map[face_idx] = face_tris

    result.triangles = all_triangles
    result.nodes = all_nodes

    if output_vtk:
        _export_combined_mesh(all_triangles, output_vtk)

    return result


def generate_cylinder_mesh(
    base_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    height: float = 1.0,
    spacing: float = 0.5,
    output_vtk: str = None,
) -> PrimitiveMeshResult:
    """
    生成圆柱体的曲面网格

    逐面使用 2D 阵面推进流水线生成网格：
    - 底面/顶面：圆形域在 XY 平面离散化后用 AFM 填充
    - 侧面：展开为 (s, z) 矩形后用 AFM 填充，再映射回 3D 圆柱坐标
    共边节点通过坐标去重保持一致。

    Args:
        base_center: 底面圆心坐标 (x, y, z)
        radius: 圆柱半径
        height: 圆柱高度（沿Z轴正方向）
        spacing: 全局网格尺寸
        output_vtk: 输出VTK文件路径（可选）

    Returns:
        PrimitiveMeshResult: 包含网格数据和面映射的结果

    Raises:
        ValueError: 如果半径或高度不是正数
    """
    if radius <= 0:
        raise ValueError(f"圆柱半径必须为正数: {radius}")
    if height <= 0:
        raise ValueError(f"圆柱高度必须为正数: {height}")

    cx, cy, z0 = base_center
    z1 = z0 + height

    # 三个面的定义
    face_mesh_funcs = [
        ("bottom", lambda: _mesh_disk_2d(
            center_xy=(cx, cy), radius=radius, z=z0,
            spacing=spacing, face_name="bottom", normal_z=-1.0,
        )),
        ("top", lambda: _mesh_disk_2d(
            center_xy=(cx, cy), radius=radius, z=z1,
            spacing=spacing, face_name="top", normal_z=1.0,
        )),
        ("lateral", lambda: _mesh_lateral_cylinder_2d(
            base_center=base_center, radius=radius, height=height,
            spacing=spacing, face_name="lateral",
        )),
    ]

    # 逐面生成网格
    face_results = []
    for face_name, mesh_func in face_mesh_funcs:
        info(f"生成面 {face_name} 网格...")
        tris, nodes = mesh_func()
        face_results.append((face_name, tris, nodes))

    # 合并结果，共边节点按坐标去重
    node_hash_to_global_idx = {}
    all_nodes = []
    all_triangles = []
    result = PrimitiveMeshResult()
    result.num_faces = 3

    global_idx = 0
    for face_idx, (face_name, tris, nodes) in enumerate(face_results):
        result.face_types[face_idx] = face_name

        node_to_global = {}
        for node in nodes:
            h = node.hash
            if h not in node_hash_to_global_idx:
                node_hash_to_global_idx[h] = global_idx
                node.idx = global_idx
                all_nodes.append(node)
                global_idx += 1
            node_to_global[id(node)] = node_hash_to_global_idx[h]

        face_tris = []
        for tri in tris:
            new_tri = SurfaceTriangle(
                all_nodes[node_to_global[id(tri.nodes[0])]],
                all_nodes[node_to_global[id(tri.nodes[1])]],
                all_nodes[node_to_global[id(tri.nodes[2])]],
                surface=None, idx=len(all_triangles),
            )
            all_triangles.append(new_tri)
            face_tris.append(new_tri)

        result.face_map[face_idx] = face_tris

    result.triangles = all_triangles
    result.nodes = all_nodes

    if output_vtk:
        _export_combined_mesh(all_triangles, output_vtk)

    return result


def generate_rectangle_mesh(
    corner1: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    corner2: Tuple[float, float, float] = (1.0, 0.0, 1.0),
    spacing: float = 0.1,
    output_vtk: str = None,
) -> PrimitiveMeshResult:
    """
    在指定平面内生成矩形域的三角形网格

    使用已有 2D 阵面推进流水线：离散化边界 → Front/NodeElementALM →
    QuadtreeSizing → Adfront2 → 边交换 + Laplacian 光滑。

    Args:
        corner1: 矩形第一个角点坐标
        corner2: 矩形对角点坐标
        spacing: 网格尺寸
        output_vtk: 输出VTK文件路径（可选）

    Returns:
        PrimitiveMeshResult

    Raises:
        ValueError: 如果角点坐标无效
    """
    p1 = np.array(corner1, dtype=float)
    p2 = np.array(corner2, dtype=float)
    diff = p2 - p1

    nonzero = [i for i in range(3) if abs(diff[i]) > 1e-12]
    if len(nonzero) < 2:
        raise ValueError(f"矩形需要两个方向有非零长度，当前差值: {diff}")

    corners_3d = [
        corner1,
        (corner2[0], corner1[1], corner1[2]),
        corner2,
        (corner1[0], corner2[1], corner2[2]),
    ]

    triangles, nodes_3d = _mesh_face_2d_pipeline(corners_3d, spacing, "rectangle")

    result = PrimitiveMeshResult()
    result.num_faces = 1
    result.face_types = {0: "rectangle"}
    result.face_map = {0: triangles}
    result.triangles = triangles
    result.nodes = nodes_3d

    if output_vtk:
        _export_combined_mesh(triangles, output_vtk)

    return result


def _compute_triangle_quality_standalone(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    p3: Tuple[float, float, float],
) -> float:
    """计算三角形质量（形状因子），范围 [0, 1]，等边三角形为 1.0"""
    a = np.linalg.norm(np.array(p2) - np.array(p1))
    b = np.linalg.norm(np.array(p3) - np.array(p2))
    c = np.linalg.norm(np.array(p1) - np.array(p3))
    s = (a + b + c) / 2.0
    if s < 1e-12:
        return 0.0
    area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
    sum_sq = a * a + b * b + c * c
    if sum_sq < 1e-12:
        return 0.0
    return min(1.0, max(0.0, 4.0 * np.sqrt(3) * area / sum_sq))


def generate_sphere_mesh(
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    spacing: float = 0.5,
    output_vtk: str = None,
) -> PrimitiveMeshResult:
    """
    生成球面的三角形网格

    使用球坐标参数化生成结构化网格：
    - 南北极点用三角形扇
    - 中间纬度带用四边形条带对角剖分

    Args:
        center: 球心坐标
        radius: 球半径
        spacing: 网格尺寸
        output_vtk: 输出VTK文件路径（可选）

    Returns:
        PrimitiveMeshResult

    Raises:
        ValueError: 如果半径不是正数
    """
    if radius <= 0:
        raise ValueError(f"球半径必须为正数: {radius}")

    cx, cy, cz = center
    n_theta = max(6, int(2 * np.pi * radius / spacing))
    n_phi = max(4, int(np.pi * radius / spacing))

    # 生成节点
    nodes = []
    node_idx = 0

    # 北极
    nx, ny, nz = cx, cy, cz + radius
    nodes.append(NodeElement3D(
        coords=(nx, ny, nz), idx=node_idx,
        normal=(0.0, 0.0, 1.0),
    ))
    node_idx += 1

    # 中间纬度带
    for j in range(1, n_phi):
        phi = j * np.pi / n_phi
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for i in range(n_theta):
            theta = i * 2 * np.pi / n_theta
            x = cx + radius * sin_phi * np.cos(theta)
            y = cy + radius * sin_phi * np.sin(theta)
            z = cz + radius * cos_phi
            nx_dir = sin_phi * np.cos(theta)
            ny_dir = sin_phi * np.sin(theta)
            nz_dir = cos_phi
            nodes.append(NodeElement3D(
                coords=(x, y, z), idx=node_idx,
                normal=(nx_dir, ny_dir, nz_dir),
            ))
            node_idx += 1

    # 南极
    sx, sy, sz = cx, cy, cz - radius
    nodes.append(NodeElement3D(
        coords=(sx, sy, sz), idx=node_idx,
        normal=(0.0, 0.0, -1.0),
    ))
    node_idx += 1

    triangles = []
    tri_idx = 0

    # 北极三角形扇
    north = nodes[0]
    for i in range(n_theta):
        n1 = nodes[1 + i]
        n2 = nodes[1 + (i + 1) % n_theta]
        tri = SurfaceTriangle(north, n1, n2, idx=tri_idx)
        triangles.append(tri)
        tri_idx += 1

    # 中间四边形条带
    for j in range(n_phi - 2):
        base_curr = 1 + j * n_theta
        base_next = 1 + (j + 1) * n_theta
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            tl = nodes[base_curr + i]
            tr = nodes[base_curr + i_next]
            bl = nodes[base_next + i]
            br = nodes[base_next + i_next]

            tri1 = SurfaceTriangle(tl, bl, tr, idx=tri_idx)
            triangles.append(tri1)
            tri_idx += 1
            tri2 = SurfaceTriangle(tr, bl, br, idx=tri_idx)
            triangles.append(tri2)
            tri_idx += 1

    # 南极三角形扇
    south = nodes[-1]
    base_last = 1 + (n_phi - 2) * n_theta
    for i in range(n_theta):
        n1 = nodes[base_last + i]
        n2 = nodes[base_last + (i + 1) % n_theta]
        tri = SurfaceTriangle(n1, south, n2, idx=tri_idx)
        triangles.append(tri)
        tri_idx += 1

    result = PrimitiveMeshResult()
    result.num_faces = 1
    result.face_types = {0: "sphere"}
    result.face_map = {0: triangles}
    result.triangles = triangles
    result.nodes = nodes

    if output_vtk:
        _export_combined_mesh(triangles, output_vtk)

    return result


def generate_ellipsoid_mesh(
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    semi_axes: Tuple[float, float, float] = (1.0, 0.75, 0.5),
    spacing: float = 0.2,
    output_vtk: str = None,
) -> PrimitiveMeshResult:
    """
    生成椭球面的三角形网格

    使用球坐标参数化，将单位球坐标缩放到椭球半轴。

    Args:
        center: 椭球中心坐标
        semi_axes: 三个半轴长度 (a, b, c) 对应 (x, y, z)
        spacing: 网格尺寸（基于最大半轴估算）
        output_vtk: 输出VTK文件路径（可选）

    Returns:
        PrimitiveMeshResult

    Raises:
        ValueError: 如果半轴不是正数
    """
    a, b, c = semi_axes
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"椭球半轴必须为正数: ({a}, {b}, {c})")

    cx, cy, cz = center
    r_max = max(a, b, c)
    n_theta = max(6, int(2 * np.pi * r_max / spacing))
    n_phi = max(4, int(np.pi * r_max / spacing))

    nodes = []
    node_idx = 0

    # 北极
    nodes.append(NodeElement3D(
        coords=(cx, cy, cz + c), idx=node_idx,
        normal=(0.0, 0.0, 1.0),
    ))
    node_idx += 1

    # 中间纬度带
    for j in range(1, n_phi):
        phi = j * np.pi / n_phi
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for i in range(n_theta):
            theta = i * 2 * np.pi / n_theta
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            x = cx + a * sin_phi * cos_theta
            y = cy + b * sin_phi * sin_theta
            z = cz + c * cos_phi

            # 椭球面法向量: (x/a^2, y/b^2, z/c^2) 归一化
            nx_dir = sin_phi * cos_theta / a
            ny_dir = sin_phi * sin_theta / b
            nz_dir = cos_phi / c
            norm = np.sqrt(nx_dir**2 + ny_dir**2 + nz_dir**2)
            if norm > 1e-12:
                nx_dir /= norm
                ny_dir /= norm
                nz_dir /= norm

            nodes.append(NodeElement3D(
                coords=(x, y, z), idx=node_idx,
                normal=(nx_dir, ny_dir, nz_dir),
            ))
            node_idx += 1

    # 南极
    nodes.append(NodeElement3D(
        coords=(cx, cy, cz - c), idx=node_idx,
        normal=(0.0, 0.0, -1.0),
    ))
    node_idx += 1

    triangles = []
    tri_idx = 0

    # 北极扇
    north = nodes[0]
    for i in range(n_theta):
        n1 = nodes[1 + i]
        n2 = nodes[1 + (i + 1) % n_theta]
        triangles.append(SurfaceTriangle(north, n1, n2, idx=tri_idx))
        tri_idx += 1

    # 中间条带
    for j in range(n_phi - 2):
        base_curr = 1 + j * n_theta
        base_next = 1 + (j + 1) * n_theta
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            tl = nodes[base_curr + i]
            tr = nodes[base_curr + i_next]
            bl = nodes[base_next + i]
            br = nodes[base_next + i_next]

            triangles.append(SurfaceTriangle(tl, bl, tr, idx=tri_idx))
            tri_idx += 1
            triangles.append(SurfaceTriangle(tr, bl, br, idx=tri_idx))
            tri_idx += 1

    # 南极扇
    south = nodes[-1]
    base_last = 1 + (n_phi - 2) * n_theta
    for i in range(n_theta):
        n1 = nodes[base_last + i]
        n2 = nodes[base_last + (i + 1) % n_theta]
        triangles.append(SurfaceTriangle(n1, south, n2, idx=tri_idx))
        tri_idx += 1

    result = PrimitiveMeshResult()
    result.num_faces = 1
    result.face_types = {0: "ellipsoid"}
    result.face_map = {0: triangles}
    result.triangles = triangles
    result.nodes = nodes

    if output_vtk:
        _export_combined_mesh(triangles, output_vtk)

    return result

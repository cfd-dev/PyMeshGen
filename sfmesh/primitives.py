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


def _extract_faces(shape) -> List[TopoDS_Face]:
    """从 OCC 形状中提取所有面"""
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while explorer.More():
        face = topods.Face(explorer.Current())
        faces.append(face)
        explorer.Next()
    return faces


def _get_face_bbox(face: TopoDS_Face):
    """获取面的包围盒，返回 (xmin, ymin, zmin, xmax, ymax, zmax)"""
    bbox = Bnd_Box()
    brepbndlib.Add(face, bbox)
    return bbox.Get()


def _get_face_bbox_center(face: TopoDS_Face) -> Tuple[float, float, float]:
    """获取面的包围盒中心"""
    xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
    return ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)


def _classify_cube_faces(
    faces: List[TopoDS_Face],
    corner1: Tuple[float, float, float],
    corner2: Tuple[float, float, float]
) -> Dict[int, str]:
    """根据包围盒中心坐标判断长方体各面类型"""
    x_min, x_max = min(corner1[0], corner2[0]), max(corner1[0], corner2[0])
    y_min, y_max = min(corner1[1], corner2[1]), max(corner1[1], corner2[1])
    z_min, z_max = min(corner1[2], corner2[2]), max(corner1[2], corner2[2])
    tol = 1e-6
    face_types = {}
    for i, face in enumerate(faces):
        cx, cy, cz = _get_face_bbox_center(face)
        if abs(cz - z_min) < tol:
            face_types[i] = "bottom"
        elif abs(cz - z_max) < tol:
            face_types[i] = "top"
        elif abs(cy - y_min) < tol:
            face_types[i] = "front"
        elif abs(cy - y_max) < tol:
            face_types[i] = "back"
        elif abs(cx - x_min) < tol:
            face_types[i] = "left"
        elif abs(cx - x_max) < tol:
            face_types[i] = "right"
        else:
            face_types[i] = f"face_{i}"
    return face_types


def _classify_cylinder_faces(
    faces: List[TopoDS_Face],
    base_z: float,
    top_z: float
) -> Dict[int, str]:
    """根据曲面类型和Z坐标判断圆柱体各面类型"""
    tol = 1e-6
    face_types = {}
    for i, face in enumerate(faces):
        surface = BRep_Tool.Surface(face)
        is_planar = surface.DynamicType().Name() == "Geom_Plane"
        if is_planar:
            _, _, cz = _get_face_bbox_center(face)
            if abs(cz - base_z) < tol:
                face_types[i] = "bottom"
            elif abs(cz - top_z) < tol:
                face_types[i] = "top"
            else:
                face_types[i] = f"disk_{i}"
        else:
            face_types[i] = "lateral"
    return face_types


def _is_point_in_face(u: float, v: float, face: TopoDS_Face) -> bool:
    """检查参数坐标 (u, v) 是否在面的边界内"""
    classifier = BRepClass_FaceClassifier()
    classifier.Perform(face, gp_Pnt2d(u, v), 1e-6)
    state = classifier.State()
    return state == TopAbs_IN or state == TopAbs_ON


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

    nu = max(2, int(round(phys_len / spacing)) + 1)
    nv = max(2, int(round(phys_len / spacing)) + 1)

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


def _project_to_2d(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """将3D点投影到2D平面（沿法向最大分量方向消除一个坐标轴）"""
    abs_n = np.abs(normal)
    if abs_n[0] >= abs_n[1] and abs_n[0] >= abs_n[2]:
        return points[:, [1, 2]]
    elif abs_n[1] >= abs_n[0] and abs_n[1] >= abs_n[2]:
        return points[:, [0, 2]]
    else:
        return points[:, [0, 1]]


def _segments_intersect_2d(a1, a2, b1, b2) -> bool:
    """判断两个2D线段是否相交（含共线重叠）"""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 快速排除端点重合
    for pa in (a1, a2):
        for pb in (b1, b2):
            if abs(pa[0] - pb[0]) < 1e-12 and abs(pa[1] - pb[1]) < 1e-12:
                return False

    # AABB 快速排斥
    if max(a1[0], a2[0]) < min(b1[0], b2[0]) - 1e-12:
        return False
    if max(b1[0], b2[0]) < min(a1[0], a2[0]) - 1e-12:
        return False
    if max(a1[1], a2[1]) < min(b1[1], b2[1]) - 1e-12:
        return False
    if max(b1[1], b2[1]) < min(a1[1], a2[1]) - 1e-12:
        return False

    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)

    if ((d1 > 1e-12 and d2 < -1e-12) or (d1 < -1e-12 and d2 > 1e-12)) and \
       ((d3 > 1e-12 and d4 < -1e-12) or (d3 < -1e-12 and d4 > 1e-12)):
        return True

    return False


def _point_in_triangle_2d(p, t0, t1, t2) -> bool:
    """判断2D点是否在三角形内部（含边界）"""
    d1 = (p[0] - t1[0]) * (t0[1] - t1[1]) - (p[1] - t1[1]) * (t0[0] - t1[0])
    d2 = (p[0] - t2[0]) * (t1[1] - t2[1]) - (p[1] - t2[1]) * (t1[0] - t2[0])
    d3 = (p[0] - t0[0]) * (t2[1] - t0[1]) - (p[1] - t0[1]) * (t2[0] - t0[0])
    has_neg = (d1 < -1e-12) or (d2 < -1e-12) or (d3 < -1e-12)
    has_pos = (d1 > 1e-12) or (d2 > 1e-12) or (d3 > 1e-12)
    return not (has_neg and has_pos)


def _are_coplanar_triangles_overlapping(
    tri_a: List[Tuple[float, float, float]],
    tri_b: List[Tuple[float, float, float]],
    surface_normal: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """检查两个共面三角形是否重叠（投影到2D后检测）"""
    pts_a = np.array(tri_a)
    pts_b = np.array(tri_b)
    pts_2d_a = _project_to_2d(pts_a, surface_normal)
    pts_2d_b = _project_to_2d(pts_b, surface_normal)

    edges_a = [(pts_2d_a[0], pts_2d_a[1]), (pts_2d_a[1], pts_2d_a[2]), (pts_2d_a[2], pts_2d_a[0])]
    edges_b = [(pts_2d_b[0], pts_2d_b[1]), (pts_2d_b[1], pts_2d_b[2]), (pts_2d_b[2], pts_2d_b[0])]

    for ea in edges_a:
        for eb in edges_b:
            if _segments_intersect_2d(ea[0], ea[1], eb[0], eb[1]):
                return True

    if _point_in_triangle_2d(pts_2d_a[0], pts_2d_b[0], pts_2d_b[1], pts_2d_b[2]):
        return True
    if _point_in_triangle_2d(pts_2d_b[0], pts_2d_a[0], pts_2d_a[1], pts_2d_a[2]):
        return True

    return False


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


def _is_planar_face(face: TopoDS_Face) -> bool:
    """判断面是否为平面"""
    surface = BRep_Tool.Surface(face)
    return surface.DynamicType().Name() == "Geom_Plane"


def _mesh_faces(
    faces: List[TopoDS_Face],
    face_types: Dict[int, str],
    spacing: float = 1.0,
) -> PrimitiveMeshResult:
    """
    对一组面逐个生成面网格

    平面使用参数化网格方法（高效、质量稳定），
    曲面使用阵面推进法（AFM，从几何边界出发生成高质量网格）。
    """
    result = PrimitiveMeshResult()
    result.num_faces = len(faces)
    result.face_types = face_types

    node_hash_set = set()
    node_id_offset = 0

    for i, face in enumerate(faces):
        ftype = face_types.get(i, "unknown")
        info(f"生成面 {i + 1}/{len(faces)} ({ftype}) 网格...")

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

    axis = gp_Ax2(gp_Pnt(*base_center), gp_Dir(0, 0, 1))
    shape = BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()

    faces = _extract_faces(shape)
    base_z = base_center[2]
    top_z = base_center[2] + height
    face_types = _classify_cylinder_faces(faces, base_z, top_z)

    result = _mesh_faces(faces, face_types, spacing=spacing)

    if output_vtk:
        _export_combined_mesh(result.triangles, output_vtk)

    return result


def _discretize_edge_2d(
    start_2d: Tuple[float, float],
    end_2d: Tuple[float, float],
    spacing: float,
) -> List[Tuple[float, float]]:
    """将 2D 线段均匀离散化为点列表"""
    s = np.array(start_2d)
    e = np.array(end_2d)
    length = np.linalg.norm(e - s)
    if length < 1e-14:
        return [start_2d]
    n = max(2, round(length / spacing) + 1)
    points = []
    for i in range(n + 1):
        t = i / n
        pt = s + t * (e - s)
        points.append((float(pt[0]), float(pt[1])))
    return points


def _mesh_face_2d_pipeline(
    corners_3d: List[Tuple[float, float, float]],
    spacing: float,
    face_name: str = "face",
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    对单个平面四边形面使用 2D 阵面推进流水线生成网格

    将 3D 平面四边形变换到 2D XY 空间，使用 Front → QuadtreeSizing →
    Adfront2 流水线生成网格，再经边交换和 Laplacian 光滑优化，
    最后变换回 3D 坐标。

    Args:
        corners_3d: 四个角点的 3D 坐标（按顺序排列，构成四边形）
        spacing: 网格尺寸
        face_name: 面名称（用于日志）

    Returns:
        (triangles, nodes_3d)
    """
    # 检测平面：确定常量轴和活跃轴
    c0, c1, c2, c3 = [np.array(p) for p in corners_3d]
    e0 = c1 - c0
    e1 = c3 - c0
    normal = np.cross(e0, e1)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-14:
        return [], []
    normal = normal / norm_len

    max_axis = int(np.argmax(np.abs(normal)))
    ax0, ax1 = [i for i in range(3) if i != max_axis]
    const_val = float(c0[max_axis])

    # 变换到 2D
    c2d = []
    for p in corners_3d:
        coord = [p[0], p[1], p[2]]
        c2d.append((coord[ax0], coord[ax1]))

    # 离散化四条边
    edge_points_2d = []
    for i in range(4):
        pts = _discretize_edge_2d(c2d[i], c2d[(i + 1) % 4], spacing)
        edge_points_2d.append(pts)

    # 创建 Front 对象
    all_fronts = []
    bc_type = "BCWall"
    for pts in edge_points_2d:
        for i in range(len(pts) - 1):
            node1 = NodeElementALM(
                coords=(pts[i][0], pts[i][1], 0.0),
                idx=-1, bc_type=bc_type, part_name=face_name,
            )
            node2 = NodeElementALM(
                coords=(pts[i + 1][0], pts[i + 1][1], 0.0),
                idx=-1, bc_type=bc_type, part_name=face_name,
            )
            front = Front(node1, node2, idx=-1, bc_type=bc_type, part_name=face_name)
            all_fronts.append(front)

    # QuadtreeSizing（扩展边界 + 越界保护）
    from meshsize.meshsize import QuadtreeSizing

    class _DummyVisual:
        ax = None

    _face_sz = max(
        max(c2d[i][0] for i in range(4)) - min(c2d[i][0] for i in range(4)),
        max(c2d[i][1] for i in range(4)) - min(c2d[i][1] for i in range(4)),
    )
    _extra_pad = max(_face_sz * 0.5, 5.0 * spacing) / max(_face_sz, 1e-12)

    class _PaddedSizingField(QuadtreeSizing):
        """扩展边界并在越界时返回 global_spacing"""
        def compute_global_parameters(self):
            super().compute_global_parameters()
            x0, y0, x1, y1 = self.bg_bounds
            dx, dy = x1 - x0, y1 - y0
            self.bg_bounds = (
                x0 - dx * _extra_pad, y0 - dy * _extra_pad,
                x1 + dx * _extra_pad, y1 + dy * _extra_pad,
            )

        def spacing_at(self, point):
            try:
                return super().spacing_at(point)
            except ValueError:
                return self.global_spacing

    sizing_system = _PaddedSizingField(
        initial_front=all_fronts,
        max_size=spacing * 10,
        resolution=0.1,
        decay=1.2,
        visual_obj=_DummyVisual(),
    )

    # Adfront2（带迭代上限保护）
    import heapq as _heapq
    from adfront2.adfront2 import Adfront2

    class _ParamObj:
        debug_level = 0
        mesh_type = 1

    front_heap = list(all_fronts)
    _heapq.heapify(front_heap)

    adfront = Adfront2(
        boundary_front=front_heap,
        sizing_system=sizing_system,
        node_coords=None,
        param_obj=_ParamObj(),
        visual_obj=_DummyVisual(),
    )

    # 硬性迭代上限，防止推进发散
    max_steps = max(50000, int((_face_sz / spacing) ** 2 * 15))
    orig_generate = adfront.generate_elements

    def _bounded_generate():
        step = 0
        while adfront.front_list and step < max_steps:
            step += 1
            adfront.base_front = _heapq.heappop(adfront.front_list)
            sp = sizing_system.spacing_at(adfront.base_front.center)
            adfront.add_new_point(sp)
            adfront.search_candidates(adfront.base_front.al * sp)
            adfront.select_point()
            adfront.update_data()
        adfront.construct_unstr_grid()
        return adfront.unstr_grid

    unstr_grid = _bounded_generate()

    # 网格优化：边交换 + Laplacian 光滑
    from optimize.optimize import edge_swap_delaunay, laplacian_smooth
    edge_swap_delaunay(unstr_grid)
    laplacian_smooth(unstr_grid, num_iter=3)

    # 变换回 3D 坐标
    grid_nodes = unstr_grid.node_coords
    grid_cells = unstr_grid.cell_container

    nodes_3d = []
    for idx, coords_2d in enumerate(grid_nodes):
        x2d, y2d = float(coords_2d[0]), float(coords_2d[1])
        coord_3d = [0.0, 0.0, 0.0]
        coord_3d[ax0] = x2d
        coord_3d[ax1] = y2d
        coord_3d[max_axis] = const_val
        node = NodeElement3D(
            coords=tuple(coord_3d), idx=idx,
            surface=None, uv_params=(0.0, 0.0),
            normal=tuple(float(x) for x in normal),
        )
        nodes_3d.append(node)

    triangles = []
    for cell in grid_cells:
        nids = cell.node_ids
        if len(nids) >= 3:
            tri = SurfaceTriangle(
                nodes_3d[nids[0]], nodes_3d[nids[1]], nodes_3d[nids[2]],
                surface=None, idx=len(triangles),
            )
            triangles.append(tri)

    return triangles, nodes_3d


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

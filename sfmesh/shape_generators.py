"""基础几何体形状生成器

基于 2D 间接法流水线生成特定几何体的曲面网格：
- 长方体（逐面 2D 流水线）
- 圆柱体（统一 2D 流水线）
- 矩形（单面 2D 流水线）
- 椭球（2D AFM + 参数化映射）
"""
import math
import numpy as np
from typing import List, Dict, Tuple

from .mesh_3d_afm import _export_combined_mesh
from .surface_front import SurfaceTriangle, NodeElement3D
from .mesh_2d_afm import (
    _mesh_face_2d_pipeline, _mesh_cylinder_unified,
    _create_fronts_from_2d_edges, _run_afm_2d_pipeline, _unstr_grid_to_3d,
    MetricAwareSizing,
)
from utils.message import info


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


def generate_cube_mesh(
    corner1: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    corner2: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spacing: float = 0.5,
    output_vtk: str = None,
):
    """
    生成长方体的曲面网格

    逐面使用 2D 阵面推进流水线生成网格，共边节点通过坐标去重保持一致。
    """
    for i in range(3):
        if abs(corner2[i] - corner1[i]) < 1e-12:
            raise ValueError(
                f"长方体第 {i} 方向长度为零: corner1[{i}]={corner1[i]}, corner2[{i}]={corner2[i]}"
            )

    x0, y0, z0 = min(corner1[0], corner2[0]), min(corner1[1], corner2[1]), min(corner1[2], corner2[2])
    x1, y1, z1 = max(corner1[0], corner2[0]), max(corner1[1], corner2[1]), max(corner1[2], corner2[2])

    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]

    face_defs = [
        ("bottom", [v[0], v[1], v[2], v[3]]),
        ("top",    [v[4], v[5], v[6], v[7]]),
        ("front",  [v[0], v[1], v[5], v[4]]),
        ("back",   [v[3], v[2], v[6], v[7]]),
        ("right",  [v[1], v[2], v[6], v[5]]),
        ("left",   [v[0], v[3], v[7], v[4]]),
    ]

    face_results = []
    for face_name, corners in face_defs:
        info(f"生成面 {face_name} 网格...")
        tris, nodes = _mesh_face_2d_pipeline(corners, spacing, face_name)
        face_results.append((face_name, tris, nodes))

    node_hash_to_global_idx = {}
    all_nodes = []
    all_triangles = []
    result = PrimitiveMeshResult()
    result.num_faces = 6

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


def generate_cylinder_mesh(
    base_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    height: float = 1.0,
    spacing: float = 0.5,
    output_vtk: str = None,
):
    """
    生成圆柱体的曲面网格

    使用统一的 2D 阵面推进流水线，确保端面和柱面共享边界节点。
    """
    if radius <= 0:
        raise ValueError(f"圆柱半径必须为正数: {radius}")
    if height <= 0:
        raise ValueError(f"圆柱高度必须为正数: {height}")

    all_triangles, all_nodes, face_tri_indices = _mesh_cylinder_unified(
        base_center=base_center, radius=radius, height=height, spacing=spacing,
    )

    face_tris_map = {0: [], 1: [], 2: []}
    name_to_idx = {"bottom": 0, "top": 1, "lateral": 2}
    for face_name, (start, end) in face_tri_indices.items():
        face_tris_map[name_to_idx[face_name]] = all_triangles[start:end]

    result = PrimitiveMeshResult()
    result.triangles = all_triangles
    result.nodes = all_nodes
    result.num_faces = 3
    result.face_types = {0: "bottom", 1: "top", 2: "lateral"}
    result.face_map = face_tris_map

    if output_vtk:
        _export_combined_mesh(all_triangles, output_vtk)

    return result


def generate_rectangle_mesh(
    corner1: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    corner2: Tuple[float, float, float] = (1.0, 0.0, 1.0),
    spacing: float = 0.1,
    output_vtk: str = None,
):
    """
    在指定平面内生成矩形域的三角形网格

    使用 2D 阵面推进流水线：离散化边界 → Front → QuadtreeSizing →
    Adfront2 → 边交换 + Laplacian 光滑。
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


def generate_ellipsoid_mesh_2d_afm(
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    semi_axes: Tuple[float, float, float] = (1.0, 0.75, 0.5),
    spacing: float = 0.2,
    output_vtk: str = None,
):
    """
    使用 2D 阵面推进流水线生成椭球面网格

    将椭球面按赤道拆分为南北两个半球面，在参数空间 (u, v) 中使用
    2D 阵面推进流水线生成网格，再映射到 3D 椭球面坐标，最后合并并
    去重赤道共享边界节点。
    """
    a, b, c = semi_axes
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"椭球半轴必须为正数: ({a}, {b}, {c})")

    cx, cy, cz = center

    # --- 1/4 网格 (u∈[0,π], v∈[0,v_pole])，不到极点 ---
    u_min, u_max = 0.0, math.pi
    v_lo, v_hi = 0.0, math.pi / 2.0

    L_equator = math.pi * max(a, b)
    n_equator = max(6, round(L_equator / spacing))
    du = (u_max - u_min) / n_equator
    dv = spacing / max(a, b, c)

    # 极点帽：一个经向步长（AFM 最小可行帽大小）
    r_hex = dv
    v_pole = v_hi - r_hex
    if v_pole < v_lo + 2 * dv:
        v_pole = v_lo + 2 * dv
        r_hex = v_hi - v_pole

    bottom = [(u_min + i * du, v_lo) for i in range(n_equator)]
    bottom.append((u_max, v_lo))

    nv_side = max(4, round((v_pole - v_lo) / dv) + 1)
    right = [(u_max, v_lo + i * (v_pole - v_lo) / nv_side) for i in range(nv_side + 1)]

    # 顶边：当极点帽较小时细化，确保段宽不超过帽高度
    top_du = min(du, r_hex * 1.5)
    n_top = max(n_equator, round((u_max - u_min) / top_du))
    top_du_actual = (u_max - u_min) / n_top
    top = [(u_max - i * top_du_actual, v_pole) for i in range(n_top)]
    top.append((u_min, v_pole))

    left = [(u_min, v_pole - i * (v_pole - v_lo) / nv_side) for i in range(nv_side + 1)]

    edge_pts = [bottom, right, top, left]
    all_fronts = _create_fronts_from_2d_edges(edge_pts, face_name="ellipsoid_quarter")

    # 度量张量：椭球面第一基本形式 E, F, G
    def _ellipsoid_metric(u, v):
        cosv = math.cos(v)
        sinv = math.sin(v)
        cosu = math.cos(u)
        sinu = math.sin(u)
        # r_u = (-a*cosv*sinu, b*cosv*cosu, 0)
        # r_v = (-a*sinv*cosu, -b*sinv*sinu, c*cosv)
        E = (a * cosv * sinu) ** 2 + (b * cosv * cosu) ** 2
        F = (a ** 2 - b ** 2) * sinv * cosv * sinu * cosu
        G = (a * sinv * cosu) ** 2 + (b * sinv * sinu) ** 2 + (c * cosv) ** 2
        return E, F, G

    metric_sizing = MetricAwareSizing(spacing, _ellipsoid_metric)

    face_sz = max(u_max - u_min, v_pole - v_lo)
    unstr_grid = _run_afm_2d_pipeline(
        all_fronts, spacing * 0.5, face_sz * 2,
        sizing_system=metric_sizing,
    )

    def _map_to_3d(u, v):
        cos_v = math.cos(v)
        sin_v = math.sin(v)
        return (
            a * cos_v * math.cos(u) + cx,
            b * cos_v * math.sin(u) + cy,
            c * sin_v + cz,
        )

    def _normal_func(u, v):
        cos_v = math.cos(v)
        sin_v = math.sin(v)
        nx = cos_v * math.cos(u) / a
        ny = cos_v * math.sin(u) / b
        nz = sin_v / c
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        return (nx / nn, ny / nn, nz / nn) if nn > 1e-14 else (0.0, 0.0, 1.0)

    tris_q, nodes_q = _unstr_grid_to_3d(unstr_grid, _map_to_3d, normal_func=_normal_func)

    all_nodes = []
    quarter_idx_map = {}
    for i, node in enumerate(nodes_q):
        quarter_idx_map[node.idx] = i
        all_nodes.append(NodeElement3D(coords=node.coords, idx=i, normal=node.normal))

    all_triangles = []
    for tri in tris_q:
        ids = [quarter_idx_map[n.idx] for n in tri.nodes]
        all_triangles.append(SurfaceTriangle(all_nodes[ids[0]], all_nodes[ids[1]], all_nodes[ids[2]]))

    # 极点帽
    tol_v = dv * 0.5
    boundary_nodes = []
    for i, node in enumerate(all_nodes):
        u_est = math.atan2(node.coords[1] - cy, node.coords[0] - cx) % (2 * math.pi)
        v_est = math.asin(max(-1, min(1, (node.coords[2] - cz) / c)))
        if abs(v_est - v_pole) < tol_v and u_est <= u_max + 0.1:
            boundary_nodes.append((u_est, i))
    boundary_nodes.sort(key=lambda x: x[0])

    pole_coords = _map_to_3d(0.0, v_hi)
    pole_normal = _normal_func(0.0, v_hi)
    pole_idx = len(all_nodes)
    all_nodes.append(NodeElement3D(coords=pole_coords, idx=pole_idx, normal=pole_normal))

    for j in range(len(boundary_nodes) - 1):
        _, bi0 = boundary_nodes[j]
        _, bi1 = boundary_nodes[j + 1]
        all_triangles.append(SurfaceTriangle(
            all_nodes[bi0], all_nodes[bi1], all_nodes[pole_idx],
        ))

    def _mirror_mesh(src_tri_indices, src_node_indices, mirror_axis):
        nonlocal all_nodes, all_triangles
        n_prev = len(all_nodes)

        mirror_map = {}
        for src_i in src_node_indices:
            node = all_nodes[src_i]
            x, y, z = node.coords
            nx, ny, nz = node.normal
            if mirror_axis == 'y':
                mc, mn = (x, -y, z), (nx, -ny, nz)
            else:
                mc, mn = (x, y, -z), (nx, ny, -nz)
            new_i = len(all_nodes)
            all_nodes.append(NodeElement3D(coords=mc, idx=new_i, normal=mn))
            mirror_map[src_i] = new_i

        tol = min(a, b, c) * 1e-4
        redirect = {}
        for src_i, new_i in mirror_map.items():
            mc = all_nodes[new_i].coords
            for j in range(n_prev):
                ec = all_nodes[j].coords
                if (abs(mc[0] - ec[0]) < tol and
                    abs(mc[1] - ec[1]) < tol and
                    abs(mc[2] - ec[2]) < tol):
                    redirect[new_i] = j
                    break

        for tri in src_tri_indices:
            ids = [mirror_map[n.idx] for n in tri.nodes]
            ids = [redirect.get(i, i) for i in ids]
            if ids[0] != ids[1] and ids[1] != ids[2] and ids[0] != ids[2]:
                all_triangles.append(SurfaceTriangle(
                    all_nodes[ids[0]], all_nodes[ids[2]], all_nodes[ids[1]],
                ))

    src_node_indices_1 = set(range(len(all_nodes)))
    src_tri_snapshot_1 = list(all_triangles)
    _mirror_mesh(src_tri_snapshot_1, src_node_indices_1, 'y')

    src_node_indices_2 = set(range(len(all_nodes)))
    src_tri_snapshot_2 = list(all_triangles)
    _mirror_mesh(src_tri_snapshot_2, src_node_indices_2, 'z')

    # 极点去重 + 退化三角形移除
    tol_merge = min(a, b, c) * 1e-6
    coord_key = lambda p: (round(p[0] / tol_merge), round(p[1] / tol_merge), round(p[2] / tol_merge))
    merge_map = {}
    node_redirect = list(range(len(all_nodes)))

    for i, node in enumerate(all_nodes):
        key = coord_key(node.coords)
        if key in merge_map:
            node_redirect[i] = merge_map[key]
        else:
            merge_map[key] = i
            node_redirect[i] = i

    new_idx_map = {}
    new_nodes = []
    new_global = 0
    for i in range(len(all_nodes)):
        rep = node_redirect[i]
        if rep not in new_idx_map:
            new_idx_map[rep] = new_global
            new_nodes.append(NodeElement3D(
                coords=all_nodes[rep].coords, idx=new_global,
                normal=all_nodes[rep].normal,
            ))
            new_global += 1

    final_triangles = []
    for tri in all_triangles:
        ids = [new_idx_map[node_redirect[n.idx]] for n in tri.nodes]
        if ids[0] != ids[1] and ids[1] != ids[2] and ids[0] != ids[2]:
            final_triangles.append(SurfaceTriangle(
                new_nodes[ids[0]], new_nodes[ids[1]], new_nodes[ids[2]],
            ))

    all_nodes = new_nodes
    all_triangles = final_triangles

    result = PrimitiveMeshResult()
    result.num_faces = 1
    result.face_types = {0: "ellipsoid"}
    result.face_map = {0: all_triangles}
    result.triangles = all_triangles
    result.nodes = all_nodes

    if output_vtk:
        _export_combined_mesh(all_triangles, output_vtk)

    return result

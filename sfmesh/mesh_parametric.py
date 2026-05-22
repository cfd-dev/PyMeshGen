"""
参数化间接法网格生成模块

在 2D 参数空间生成结构化网格，映射到 3D 曲面坐标。
适用于平面、闭合曲面（球面、椭球面）等可参数化的几何体。
"""
import math
import numpy as np
from typing import List, Tuple

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

from .surface_front import NodeElement3D, SurfaceTriangle
from .surface_geometry import SurfaceGeometry
from .mesh_3d_afm import _export_combined_mesh
from .occ_utils import _get_face_bbox, _is_point_in_face
from utils.message import info


def _mesh_face_parametric(
    face,
    spacing: float,
    node_id_offset: int,
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    对单个面进行参数化网格剖分，节点投影到几何曲面上

    利用 OCC 曲面评估器在参数空间 (u, v) 生成均匀网格，
    将参数坐标映射到三维空间，确保节点严格落在几何曲面上。
    通过 BRepClass_FaceClassifier 确保节点在面的边界内。
    """
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
    face,
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


def generate_sphere_mesh(
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    spacing: float = 0.5,
    output_vtk: str = None,
):
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
    from .shape_generators import PrimitiveMeshResult

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
):
    """
    生成椭球面的三角形网格

    使用结构化网格 + 极点扇形填充方法：
    1. 在参数空间 (u∈[0,2π], v∈[-π/2,π/2]) 中创建结构化网格
    2. 跳过极点退化行（v=±π/2），创建单个极点节点
    3. 中间区域用四边形条带拆分为三角形
    4. 两极用三角形扇填充

    Args:
        center: 椭球中心坐标
        semi_axes: 三个半轴长度 (a, b, c) 对应 (x, y, z)
        spacing: 网格尺寸
        output_vtk: 输出VTK文件路径（可选）

    Returns:
        PrimitiveMeshResult

    Raises:
        ValueError: 如果半轴不是正数
    """
    from .shape_generators import PrimitiveMeshResult

    a, b, c = semi_axes
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"椭球半轴必须为正数: ({a}, {b}, {c})")

    cx, cy, cz = center

    # 椭球参数方程:
    #   x = a * cos(v) * cos(u) + cx
    #   y = b * cos(v) * sin(u) + cy
    #   z = c * sin(v) + cz
    # u ∈ [0, 2π], v ∈ [-π/2, π/2]

    def _eval_uv(u, v):
        cos_v = math.cos(v)
        sin_v = math.sin(v)
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        return (
            a * cos_v * cos_u + cx,
            b * cos_v * sin_u + cy,
            c * sin_v + cz,
        )

    def _normal_uv(u, v):
        cos_v = math.cos(v)
        sin_v = math.sin(v)
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        nx = cos_v * cos_u / a
        ny = cos_v * sin_u / b
        nz = sin_v / c
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nn < 1e-14:
            return (0.0, 0.0, 1.0 if sin_v >= 0 else -1.0)
        return (nx / nn, ny / nn, nz / nn)

    # 离散化
    L_equator = math.pi * max(a, b)  # 赤道半周长
    L_meridian = math.pi * max(a, c)  # 经线全长（取较大半轴估算）
    nu = max(8, round(L_equator / spacing))
    nv = max(6, round(L_meridian / spacing))

    u_vals = [i * 2.0 * math.pi / nu for i in range(nu + 1)]
    v_vals = [-math.pi / 2.0 + j * math.pi / nv for j in range(nv + 1)]

    nodes = []
    node_dict = {}  # (iu, iv) -> node index
    nid = 0

    # 南极 (v = -π/2, 退化行)
    south_pole = NodeElement3D(
        coords=_eval_uv(0, -math.pi / 2), idx=nid,
        normal=_normal_uv(0, -math.pi / 2),
    )
    nodes.append(south_pole)
    nid += 1

    # 内部行 (iv = 1 .. nv-1, 跳过两极)
    for iv in range(1, nv):
        v = v_vals[iv]
        for iu in range(nu + 1):
            u = u_vals[iu]
            coords = _eval_uv(u, v)
            normal = _normal_uv(u, v)
            node = NodeElement3D(coords=coords, idx=nid, normal=normal)
            node_dict[(iu, iv)] = nid
            nodes.append(node)
            nid += 1

    # 北极 (v = +π/2, 退化行)
    north_pole = NodeElement3D(
        coords=_eval_uv(0, math.pi / 2), idx=nid,
        normal=_normal_uv(0, math.pi / 2),
    )
    nodes.append(north_pole)
    nid += 1

    # 生成三角形
    triangles = []

    # 南极三角形扇: 连接南极到第一行 (iv=1)
    iv = 1
    for iu in range(nu):
        n1 = nodes[node_dict[(iu, iv)]]
        n2 = nodes[node_dict[(iu + 1, iv)]]
        triangles.append(SurfaceTriangle(south_pole, n1, n2))

    # 中间四边形条带
    for iv in range(1, nv - 1):
        for iu in range(nu):
            tl = nodes[node_dict[(iu, iv)]]
            tr = nodes[node_dict[(iu + 1, iv)]]
            bl = nodes[node_dict[(iu, iv + 1)]]
            br = nodes[node_dict[(iu + 1, iv + 1)]]
            triangles.append(SurfaceTriangle(tl, bl, tr))
            triangles.append(SurfaceTriangle(tr, bl, br))

    # 北极三角形扇: 连接北极到最后一行 (iv=nv-1)
    iv = nv - 1
    for iu in range(nu):
        n1 = nodes[node_dict[(iu, iv)]]
        n2 = nodes[node_dict[(iu + 1, iv)]]
        triangles.append(SurfaceTriangle(n1, north_pole, n2))

    result = PrimitiveMeshResult()
    result.num_faces = 1
    result.face_types = {0: "ellipsoid"}
    result.face_map = {0: triangles}
    result.triangles = triangles
    result.nodes = nodes

    if output_vtk:
        _export_combined_mesh(triangles, output_vtk)

    return result

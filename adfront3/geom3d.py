"""三维几何测试函数

纯函数，不依赖任何类状态。用于四面体网格生成中的相交、包含等判断。
"""
import numpy as np


def point_in_tet(p, tet_coords):
    """检查点是否在四面体内部（使用有符号体积法）

    对每个顶点 pi，计算 (-1)^i * det(pt-pi, p_{j}-pi, p_{k}-pi)，
    其中 j=(i+2)%4, k=(i+3)%4。所有值同号表示点在内部。

    Args:
        p: (3,) 查询点坐标
        tet_coords: 四个顶点坐标列表 [p0, p1, p2, p3]

    Returns:
        bool: 点是否严格在四面体内部
    """
    pts = [np.array(c) for c in tet_coords]
    pt = np.array(p)

    # 参考行列式（四面体有符号体积的6倍）
    ref = np.dot(pts[1] - pts[0], np.cross(pts[2] - pts[0], pts[3] - pts[0]))
    if abs(ref) < 1e-30:
        return False

    # 对每个顶点 i，取边 j=(i+2)%4 和 k=(i+3)%4
    # 乘以 (-1)^i 使所有子体积符号一致
    eps = 1e-10 * abs(ref)
    for i in range(4):
        j = (i + 2) % 4
        k = (i + 3) % 4
        v = float(np.dot(pt - pts[i], np.cross(pts[j] - pts[i], pts[k] - pts[i])))
        signed_v = ((-1) ** i) * v
        if signed_v < eps:
            return False

    return True


def tets_intersect(coords1, coords2):
    """检查两个四面体是否相交

    通过检查一个四面体的顶点是否在另一个四面体内部来判断。

    Args:
        coords1: 第一个四面体的四个顶点坐标
        coords2: 第二个四面体的四个顶点坐标

    Returns:
        bool: 两个四面体是否相交
    """
    for c in coords1:
        if point_in_tet(c, coords2):
            return True
    for c in coords2:
        if point_in_tet(c, coords1):
            return True
    return False


def bbox_overlap_3d(coords1, coords2):
    """检查两个点集的包围盒是否重叠

    Args:
        coords1: 第一组点坐标列表
        coords2: 第二组点坐标列表

    Returns:
        bool: 包围盒是否重叠
    """
    eps = 1e-10
    min1 = [min(c[i] for c in coords1) - eps for i in range(3)]
    max1 = [max(c[i] for c in coords1) + eps for i in range(3)]
    min2 = [min(c[i] for c in coords2) - eps for i in range(3)]
    max2 = [max(c[i] for c in coords2) + eps for i in range(3)]
    return all(min1[i] <= max2[i] and max1[i] >= min2[i] for i in range(3))


def edge_intersects_triangle(edge, tri_coords):
    """检查线段是否与三角形相交

    使用参数化方法：先求线段与三角形平面的交点，
    再用重心坐标判断交点是否在三角形内部。

    Args:
        edge: 线段的两个端点坐标 (p0, p1)
        tri_coords: 三角形的三个顶点坐标 [a, b, c]

    Returns:
        bool: 线段是否与三角形相交
    """
    p0 = np.array(edge[0])
    p1 = np.array(edge[1])
    a, b, c = [np.array(x) for x in tri_coords]

    edge_vec = p1 - p0
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-30:
        return False

    normal = np.cross(b - a, c - a)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-30:
        return False
    normal = normal / normal_len

    denom = np.dot(normal, edge_vec)
    if abs(denom) < 1e-12:
        return False

    t = np.dot(normal, a - p0) / denom
    if t < 1e-8 or t > 1 - 1e-8:
        return False

    hit = p0 + t * edge_vec

    v0 = c - a
    v1 = b - a
    v2 = hit - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-30)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return u >= -1e-8 and v >= -1e-8 and (u + v) <= 1 + 1e-8


def tet_intersects_triangle(tet_coords, tri_coords):
    """检查四面体是否与三角形相交（边-面交叉检测）

    检查四面体的6条边是否与三角形相交。

    Args:
        tet_coords: 四面体的四个顶点坐标
        tri_coords: 三角形的三个顶点坐标

    Returns:
        bool: 四面体是否与三角形相交
    """
    edges = [
        (tet_coords[0], tet_coords[1]),
        (tet_coords[0], tet_coords[2]),
        (tet_coords[0], tet_coords[3]),
        (tet_coords[1], tet_coords[2]),
        (tet_coords[1], tet_coords[3]),
        (tet_coords[2], tet_coords[3]),
    ]
    for edge in edges:
        if edge_intersects_triangle(edge, tri_coords):
            return True
    return False


def point_in_triangle(point, tri_coords):
    """检查点是否在三角形内部（使用重心坐标）

    Args:
        point: (3,) 查询点坐标
        tri_coords: 三角形的三个顶点坐标 [a, b, c]

    Returns:
        bool: 点是否在三角形内部（含边界容差）
    """
    p = np.array(point)
    a, b, c = [np.array(x) for x in tri_coords]

    # 计算三角形法向量
    normal = np.cross(b - a, c - a)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-30:
        return False
    normal = normal / normal_len

    # 检查点是否在三角形平面上
    dist_to_plane = abs(np.dot(p - a, normal))
    if dist_to_plane > 1e-8:
        return False

    # 使用重心坐标检查点是否在三角形内部
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-30)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return u >= -1e-8 and v >= -1e-8 and (u + v) <= 1 + 1e-8


# ── Netgen IntersectTetTriangle 算法移植 ─────────────────────────
# 参考: gmsh_AFT/contrib/Netgen/libsrc/gprim/geomtest3d.cpp


def _find_shared_vertex_ids(tet_ids, tri_ids):
    """通过节点 ID 查找共享顶点（O(1) per pair）"""
    tet_set = set(tet_ids)
    shared_tet = []
    shared_tri = []
    for i, tid in enumerate(tri_ids):
        for j, eid in enumerate(tet_ids):
            if tid == eid:
                shared_tet.append(j)
                shared_tri.append(i)
                break
    return shared_tet, shared_tri


def _find_shared_vertex_geom(tet_coords, tri_coords, eps2):
    """通过几何距离查找共享顶点"""
    shared_tet = []
    shared_tri = []
    for i in range(3):
        for j in range(4):
            d2 = sum((tri_coords[i][k] - tet_coords[j][k]) ** 2 for k in range(3))
            if d2 < eps2:
                shared_tet.append(j)
                shared_tri.append(i)
                break
    return shared_tet, shared_tri


def local_coordinates_2d(e1, e2, v):
    """用 Cramer 法则求解 v = lam1*e1 + lam2*e2"""
    m11 = np.dot(e1, e1)
    m12 = np.dot(e1, e2)
    m22 = np.dot(e2, e2)
    rs1 = np.dot(v, e1)
    rs2 = np.dot(v, e2)
    det = m11 * m22 - m12 * m12
    if abs(det) < 1e-30:
        return None, None
    lam1 = (rs1 * m22 - rs2 * m12) / det
    lam2 = (m11 * rs2 - m12 * rs1) / det
    return lam1, lam2


def intersect_triangle_line(tri_coords, line_coords):
    """检测线段是否与三角形相交（Netgen IntersectTriangleLine 移植）

    通过求解 3x3 线性系统：[-vl, vt1, vt2] * [t, u, v]^T = vrs

    Args:
        tri_coords: 三角形三个顶点坐标
        line_coords: 线段两个端点坐标

    Returns:
        bool: 线段是否与三角形相交
    """
    tri = [np.array(c) for c in tri_coords]
    line = [np.array(c) for c in line_coords]

    vl = line[1] - line[0]
    vt1 = tri[1] - tri[0]
    vt2 = tri[2] - tri[0]
    vrs = line[0] - tri[0]

    # 3x3 线性系统: [-vl, vt1, vt2] * [t, u, v]^T = vrs
    A = np.column_stack([-vl, vt1, vt2])
    vl_len = np.linalg.norm(vl)
    vt1_len = np.linalg.norm(vt1)
    vt2_len = np.linalg.norm(vt2)
    arel = vl_len * vt1_len * vt2_len

    det = np.linalg.det(A)
    if abs(det) <= 1e-10 * arel:
        return False

    lami = np.linalg.solve(A, vrs)
    t, u, v = lami[0], lami[1], lami[2]

    # 线段参数 t ∈ [0, 1]，重心坐标 u >= 0, v >= 0, u+v <= 1
    if t < -1e-6 or t > 1 + 1e-6:
        return False
    if u < -1e-6 or v < -1e-6 or u + v > 1 + 1e-6:
        return False
    return True


def intersect_triangle_triangle(tri1_coords, tri2_coords, tri1_ids=None, tri2_ids=None):
    """检测两个三角形是否相交（Netgen IntersectTriangleTriangle 移植）

    仅当无共享顶点时进行 6 条边-三角形交叉检测。
    共享 1/2/3 个顶点不算相交。

    Args:
        tri1_coords: 第一个三角形的三个顶点坐标
        tri2_coords: 第二个三角形的三个顶点坐标
        tri1_ids: 第一个三角形的节点 ID 列表（可选）
        tri2_ids: 第二个三角形的节点 ID 列表（可选）

    Returns:
        bool: 两个三角形是否相交
    """
    t1 = [np.array(c) for c in tri1_coords]
    t2 = [np.array(c) for c in tri2_coords]

    eps2 = np.dot(t1[1] - t1[0], t1[1] - t1[0]) * 1e-16

    # 统计共享顶点数
    if tri1_ids is not None and tri2_ids is not None:
        cnt = len(set(tri1_ids) & set(tri2_ids))
    else:
        cnt = 0
        for i in range(3):
            for j in range(3):
                if np.dot(t2[j] - t1[i], t2[j] - t1[i]) < eps2:
                    cnt += 1
                    break

    if cnt != 0:
        return False

    # Case 0: 无共享顶点 — 6 条边-三角形交叉检测
    for i in range(3):
        j = (i + 1) % 3
        if intersect_triangle_line(tri1_coords, (tri2_coords[i], tri2_coords[j])):
            return True
    for i in range(3):
        j = (i + 1) % 3
        if intersect_triangle_line(tri2_coords, (tri1_coords[i], tri1_coords[j])):
            return True

    return False


def intersect_tet_triangle(tet_coords, tri_coords, tet_ids=None, tri_ids=None):
    """检测四面体是否与三角形相交（Netgen IntersectTetTriangle 移植）

    处理 4 种共享顶点情况：
    - 0 shared: 半空间测试 + 边-面交叉检测
    - 1 shared: 平面交线分析
    - 2 shared: 公共边投影分析
    - 3 shared: 共面不算相交

    Args:
        tet_coords: 四面体的四个顶点坐标
        tri_coords: 三角形的三个顶点坐标
        tet_ids: 四面体节点 ID 列表（可选，用于精确共享顶点检测）
        tri_ids: 三角形节点 ID 列表（可选）

    Returns:
        bool: 四面体是否与三角形相交
    """
    tet = [np.array(c) for c in tet_coords]
    tri = [np.array(c) for c in tri_coords]

    # 共享顶点检测
    if tet_ids is not None and tri_ids is not None:
        shared_tet, shared_tri = _find_shared_vertex_ids(tet_ids, tri_ids)
    else:
        eps2 = np.dot(tri[1] - tri[0], tri[1] - tri[0]) * 1e-16
        shared_tet, shared_tri = _find_shared_vertex_geom(tet_coords, tri_coords, eps2)

    cnt = len(shared_tet)

    # ── Case 0: 无共享顶点 ──
    if cnt == 0:
        # 计算四面体有符号体积确定法线方向
        det_val = np.dot(tet[1] - tet[0],
                         np.cross(tet[2] - tet[0], tet[3] - tet[0]))
        if abs(det_val) < 1e-30:
            return False

        # 半空间快速排除
        faces = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
        for f_idx, (i0, i1, i2) in enumerate(faces):
            opp = [i for i in range(4) if i not in (i0, i1, i2)][0]
            n = np.cross(tet[i1] - tet[i0], tet[i2] - tet[i0])
            if np.dot(n, tet[opp] - tet[i0]) > 0:
                n = -n
            nl = np.linalg.norm(n)
            if nl < 1e-30:
                continue
            n = n / nl
            eps = 1e-8 * nl
            all_outside = True
            for j in range(3):
                if np.dot(tri[j] - tet[i0], n) < eps:
                    all_outside = False
                    break
            if all_outside:
                return False

        # 检查三角形顶点是否在四面体内
        for j in range(3):
            inside = True
            for f_idx, (i0, i1, i2) in enumerate(faces):
                opp = [i for i in range(4) if i not in (i0, i1, i2)][0]
                n = np.cross(tet[i1] - tet[i0], tet[i2] - tet[i0])
                if np.dot(n, tet[opp] - tet[i0]) > 0:
                    n = -n
                nl = np.linalg.norm(n)
                if nl < 1e-30:
                    continue
                if np.dot(tri[j] - tet[i0], n / nl) >= 0:
                    inside = False
                    break
            if inside:
                return True

        # 6 条四面体边 vs 三角形
        for i in range(4):
            for j in range(i + 1, 4):
                if intersect_triangle_line(tri_coords, (tet_coords[i], tet_coords[j])):
                    return True

        # 3 条三角形边 vs 4 个四面体面
        faces_coords = [
            (tet_coords[1], tet_coords[2], tet_coords[3]),
            (tet_coords[0], tet_coords[3], tet_coords[2]),
            (tet_coords[0], tet_coords[1], tet_coords[3]),
            (tet_coords[0], tet_coords[2], tet_coords[1]),
        ]
        for i in range(3):
            j = (i + 1) % 3
            edge = (tri_coords[i], tri_coords[j])
            for fc in faces_coords:
                if intersect_triangle_line(fc, edge):
                    return True

        return False

    # ── Case 3: 3 个共享顶点（共面）──
    if cnt == 3:
        return False

    # ── Case 1: 1 个共享顶点 ──
    if cnt == 1:
        sp_tri = shared_tri[0]
        sp_tet = shared_tet[0]
        tri_rem = [i for i in range(3) if i != sp_tri]
        tet_rem = [i for i in range(4) if i != sp_tet]

        # 参考 Netgen: eps = diam * 1e-8, 并行阈值 = eps^4
        diam = np.linalg.norm(tri[1] - tri[0])
        eps = diam * 1e-8
        eps4 = eps ** 4

        ntri = np.cross(tri[tri_rem[0]] - tri[sp_tri],
                        tri[tri_rem[1]] - tri[sp_tri])

        for face_sp_idx in range(3):
            t1 = tet_rem[face_sp_idx]
            t2 = tet_rem[(face_sp_idx + 1) % 3]
            ntet = np.cross(tet[t1] - tet[sp_tet],
                            tet[t2] - tet[sp_tet])

            crline = np.cross(ntri, ntet)
            crline_len = np.linalg.norm(crline)
            if crline_len < eps4:
                continue

            vtri1 = tri[tri_rem[0]] - tri[sp_tri]
            vtri2 = tri[tri_rem[1]] - tri[sp_tri]
            vtet1 = tet[t1] - tet[sp_tet]
            vtet2 = tet[t2] - tet[sp_tet]

            # 参考 Netgen: vtri1*crline + vtri2*crline < 0 时反转
            if np.dot(crline, vtri1) + np.dot(crline, vtri2) < 0:
                crline = -crline

            crline = crline / crline_len

            lam1, lam2 = local_coordinates_2d(vtri1, vtri2, crline)
            lam3, lam4 = local_coordinates_2d(vtet1, vtet2, crline)
            if lam1 is None or lam3 is None:
                continue

            if lam1 >= -1e-8 and lam2 >= -1e-8 and lam3 >= -1e-8 and lam4 >= -1e-8:
                return True

        return False

    # ── Case 2: 2 个共享顶点（公共边）──
    if cnt == 2:
        tri_rem = [i for i in range(3) if i not in shared_tri]
        tet_rem = [i for i in range(4) if i not in shared_tet]
        if not tri_rem or not tet_rem:
            return False

        n = tri[shared_tri[1]] - tri[shared_tri[0]]
        n_len = np.linalg.norm(n)
        if n_len < 1e-30:
            return False
        n = n / n_len

        vtri = tri[tri_rem[0]] - tri[shared_tri[0]]
        vtet1 = tet[tet_rem[0]] - tet[shared_tri[0]]
        vtet2 = tet[tet_rem[1]] - tet[shared_tri[0]]

        # 投影到垂直于共享边的平面（参考 Netgen 原始实现）
        vtet1_proj = vtet1 - np.dot(vtet1, n) * n
        vtet2_proj = vtet2 - np.dot(vtet2, n) * n
        vtri_proj = vtri - np.dot(vtri, n) * n

        lam1, lam2 = local_coordinates_2d(vtet1_proj, vtet2_proj, vtri_proj)
        if lam1 is None:
            # 退化情况：vtet1 或 vtet2 投影为零向量
            # 选择非退化的投影向量，检查 vtri_proj 是否在其正半空间
            vtet1_len2 = np.dot(vtet1_proj, vtet1_proj)
            vtet2_len2 = np.dot(vtet2_proj, vtet2_proj)
            vtri_len2 = np.dot(vtri_proj, vtri_proj)
            if vtri_len2 < 1e-30:
                return False
            if vtet1_len2 > 1e-30:
                return np.dot(vtri_proj, vtet1_proj) >= -1e-8
            if vtet2_len2 > 1e-30:
                return np.dot(vtri_proj, vtet2_proj) >= -1e-8
            return False

        if lam1 < -1e-8 or lam2 < -1e-8:
            return False
        return True

    return False


def intersect_tet_triangle2(tet_coords, tri_coords, tet_ids=None, tri_ids=None):
    """二次相交检测（Netgen IntersectTetTriangle2 移植）

    当共享顶点数为 1 或 2 时，通过三角形-三角形交叉检测判断相交。
    共享 0 或 3 个顶点时返回 False（由 IntersectTetTriangle 处理）。

    Args:
        tet_coords: 四面体的四个顶点坐标
        tri_coords: 三角形的三个顶点坐标
        tet_ids: 四面体节点 ID 列表（可选）
        tri_ids: 三角形节点 ID 列表（可选）

    Returns:
        bool: 是否检测到相交
    """
    tet = [np.array(c) for c in tet_coords]
    tri = [np.array(c) for c in tri_coords]

    # 共享顶点检测
    if tet_ids is not None and tri_ids is not None:
        shared_tet, shared_tri = _find_shared_vertex_ids(tet_ids, tri_ids)
    else:
        eps2 = np.dot(tri[1] - tri[0], tri[1] - tri[0]) * 1e-16
        shared_tet, shared_tri = _find_shared_vertex_geom(tet_coords, tri_coords, eps2)

    cnt = len(shared_tet)

    if cnt != 1 and cnt != 2:
        return False

    if cnt == 1:
        sp_tet = shared_tet[0]
        sp_tri = shared_tri[0]
        tet_rem = [i for i in range(4) if i != sp_tet]
        tri_rem = [i for i in range(3) if i != sp_tri]

        # 4 个 tet 面 vs triangle
        faces = [
            (sp_tet, tet_rem[0], tet_rem[1]),
            (sp_tet, tet_rem[1], tet_rem[2]),
            (sp_tet, tet_rem[2], tet_rem[0]),
            (tet_rem[0], tet_rem[1], tet_rem[2]),
        ]
        for i0, i1, i2 in faces:
            tri1_coords = [tet_coords[i0], tet_coords[i1], tet_coords[i2]]
            tri1_ids = [tet_ids[i0], tet_ids[i1], tet_ids[i2]] if tet_ids else None
            if intersect_triangle_triangle(tri1_coords, tri_coords, tri1_ids, tri_ids):
                return True
        return False

    # cnt == 2: 公共边
    tet_rem = [i for i in range(4) if i not in shared_tet]
    tri1_coords = [tet_coords[shared_tet[1]], tet_coords[tet_rem[0]], tet_coords[tet_rem[1]]]
    tri2_coords = [tet_coords[shared_tet[0]], tet_coords[tet_rem[0]], tet_coords[tet_rem[1]]]
    tri1_ids = [tet_ids[shared_tet[1]], tet_ids[tet_rem[0]], tet_ids[tet_rem[1]]] if tet_ids else None
    tri2_ids = [tet_ids[shared_tet[0]], tet_ids[tet_rem[0]], tet_ids[tet_rem[1]]] if tet_ids else None

    if intersect_triangle_triangle(tri1_coords, tri_coords, tri1_ids, tri_ids):
        return True
    if intersect_triangle_triangle(tri2_coords, tri_coords, tri2_ids, tri_ids):
        return True
    return False

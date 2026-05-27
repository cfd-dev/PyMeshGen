"""
2D/3D 计算几何工具库

提供纯数值计算的几何判断工具：投影、线段相交、点在三角形内、共面重叠检测、
三角形相交检测等。无 OCC 依赖，仅依赖 NumPy。
"""

from typing import List, Tuple, Optional, Union
import numpy as np

# ============================================================================
# 全局常量与配置
# ============================================================================
DEFAULT_TOL = 1e-10       # 通用几何容差
DEGENERATE_TOL = 1e-24    # 退化检测容差（面积/长度平方级别）
COPLANAR_TOL = 1e-8       # 共面检测专用容差


# ============================================================================
# 2D 几何工具
# ============================================================================

def project_to_2d(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    将3D点集投影到2D平面。

    通过丢弃法向量绝对值最大的分量来避免投影退化。

    Args:
        points: (N, 3) 三维点集
        normal: (3,) 投影平面法向量

    Returns:
        (N, 2) 二维投影点集
    """
    abs_n = np.abs(normal)
    if abs_n[0] >= abs_n[1] and abs_n[0] >= abs_n[2]:
        return points[:, [1, 2]]
    elif abs_n[1] >= abs_n[0] and abs_n[1] >= abs_n[2]:
        return points[:, [0, 2]]
    else:
        return points[:, [0, 1]]


def segments_intersect_2d(
    a1: np.ndarray, a2: np.ndarray,
    b1: np.ndarray, b2: np.ndarray,
    tol: float = DEFAULT_TOL
) -> bool:
    """
    判断两条2D线段是否严格相交（不含端点重合）。

    使用叉积符号测试 + AABB快速排斥。

    Args:
        a1, a2: 线段A端点 (2,)
        b1, b2: 线段B端点 (2,)
        tol: 浮点容差

    Returns:
        True 表示两线段在内部相交
    """
    def cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 排除端点重合
    for pa in (a1, a2):
        for pb in (b1, b2):
            if np.max(np.abs(pa - pb)) < tol:
                return False

    # AABB 快速排斥
    if max(a1[0], a2[0]) < min(b1[0], b2[0]) - tol:
        return False
    if max(b1[0], b2[0]) < min(a1[0], a2[0]) - tol:
        return False
    if max(a1[1], a2[1]) < min(b1[1], b2[1]) - tol:
        return False
    if max(b1[1], b2[1]) < min(a1[1], a2[1]) - tol:
        return False

    d1 = cross2d(b1, b2, a1)
    d2 = cross2d(b1, b2, a2)
    d3 = cross2d(a1, a2, b1)
    d4 = cross2d(a1, a2, b2)

    if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
       ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
        return True

    return False


def point_in_triangle_2d(
    p: np.ndarray,
    t0: np.ndarray, t1: np.ndarray, t2: np.ndarray,
    tol: float = DEFAULT_TOL
) -> bool:
    """
    判断2D点是否在三角形内部（含边界）。

    使用同侧叉积法。

    Args:
        p: 待测点 (2,)
        t0, t1, t2: 三角形顶点 (2,)
        tol: 浮点容差

    Returns:
        True 表示点在三角形内或边上
    """
    d1 = (p[0] - t1[0]) * (t0[1] - t1[1]) - (p[1] - t1[1]) * (t0[0] - t1[0])
    d2 = (p[0] - t2[0]) * (t1[1] - t2[1]) - (p[1] - t2[1]) * (t1[0] - t2[0])
    d3 = (p[0] - t0[0]) * (t2[1] - t0[1]) - (p[1] - t0[1]) * (t2[0] - t0[0])

    has_neg = (d1 < -tol) or (d2 < -tol) or (d3 < -tol)
    has_pos = (d1 > tol) or (d2 > tol) or (d3 > tol)
    return not (has_neg and has_pos)


# ============================================================================
# 3D 基础几何工具
# ============================================================================

def point_in_triangle_3d(
    p: np.ndarray,
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
    tol: float = DEFAULT_TOL
) -> bool:
    """
    判断3D点是否在三角形内部（基于重心坐标）。

    注意：此函数假设点已在三角形平面上或非常接近平面。
    如需同时检查共面距离，请使用 point_triangle_distance_3d。

    Args:
        p: 待测点 (3,)
        a, b, c: 三角形顶点 (3,)
        tol: 重心坐标容差

    Returns:
        True 表示点在三角形内
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < DEGENERATE_TOL:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= -tol) and (v >= -tol) and (u + v <= 1.0 + tol)


def segment_intersects_triangle(
    p: np.ndarray, q: np.ndarray,
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
    tol: float = DEFAULT_TOL
) -> bool:
    """
    Möller–Trumbore 算法：判断线段 pq 是否与三角形 abc 相交。

    排除端点共享和边上的退化情况。

    Args:
        p, q: 线段端点 (3,)
        a, b, c: 三角形顶点 (3,)
        tol: 浮点容差

    Returns:
        True 表示线段穿过三角形内部
    """
    edge1 = b - a
    edge2 = c - a
    dir_vec = q - p

    h = np.cross(dir_vec, edge2)
    det = np.dot(edge1, h)
    if -tol < det < tol:
        return False

    f = 1.0 / det
    s = p - a
    u = f * np.dot(s, h)
    if u < -tol or u > 1.0 + tol:
        return False

    q_vec = np.cross(s, edge1)
    v = f * np.dot(dir_vec, q_vec)
    if v < -tol or u + v > 1.0 + tol:
        return False

    t = f * np.dot(edge2, q_vec)
    return tol < t < 1.0 - tol


def segment_segment_distance_3d(
    p1: np.ndarray, p2: np.ndarray,
    q1: np.ndarray, q2: np.ndarray
) -> float:
    """
    计算两条3D线段之间的最短距离。

    基于参数化最近点求解，处理平行/退化线段。

    Args:
        p1, p2: 线段P端点 (3,)
        q1, q2: 线段Q端点 (3,)

    Returns:
        最短欧氏距离
    """
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1

    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    if a < DEGENERATE_TOL and e < DEGENERATE_TOL:
        return float(np.linalg.norm(r))

    if a < DEGENERATE_TOL:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = np.dot(d1, r)
        if e < DEGENERATE_TOL:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b_val = np.dot(d1, d2)
            denom = a * e - b_val * b_val
            if abs(denom) > DEGENERATE_TOL:
                s = float(np.clip((b_val * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0
            t = (b_val * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b_val - c) / a, 0.0, 1.0))

    closest_p = p1 + s * d1
    closest_q = q1 + t * d2
    return float(np.linalg.norm(closest_p - closest_q))


def point_triangle_distance_3d(
    p: np.ndarray,
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """
    计算3D点到三角形的最短距离（Voronoi区域法）。

    自动处理点在顶点、边、面投影等各种情况。

    Args:
        p: 查询点 (3,)
        a, b, c: 三角形顶点 (3,)

    Returns:
        最短欧氏距离
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(p - (a + v * ab)))

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(p - (a + w * ac)))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(p - (b + w * (c - b))))

    # 点在三角形面投影区域内
    n = np.cross(ab, ac)
    n_len = np.linalg.norm(n)
    if n_len < DEGENERATE_TOL:
        return 0.0
    return float(abs(np.dot(ap, n)) / n_len)


# ============================================================================
# 共面三角形重叠检测
# ============================================================================

def are_coplanar_triangles_overlapping(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    surface_normal: np.ndarray,
    tol: float = COPLANAR_TOL
) -> bool:
    """
    检查两个已知共面的三角形是否重叠。

    通过投影到2D后检测边相交和包含关系。

    Args:
        tri_a: (3, 3) 三角形A顶点
        tri_b: (3, 3) 三角形B顶点
        surface_normal: (3,) 公共法向量
        tol: 容差

    Returns:
        True 表示两三角形有重叠区域
    """
    pts_2d_a = project_to_2d(tri_a, surface_normal)
    pts_2d_b = project_to_2d(tri_b, surface_normal)

    edges_a = [
        (pts_2d_a[0], pts_2d_a[1]),
        (pts_2d_a[1], pts_2d_a[2]),
        (pts_2d_a[2], pts_2d_a[0]),
    ]
    edges_b = [
        (pts_2d_b[0], pts_2d_b[1]),
        (pts_2d_b[1], pts_2d_b[2]),
        (pts_2d_b[2], pts_2d_b[0]),
    ]

    for ea in edges_a:
        for eb in edges_b:
            if segments_intersect_2d(ea[0], ea[1], eb[0], eb[1], tol):
                return True

    # 无边相交时检查包含关系
    if point_in_triangle_2d(pts_2d_a[0], pts_2d_b[0], pts_2d_b[1], pts_2d_b[2], tol):
        return True
    if point_in_triangle_2d(pts_2d_b[0], pts_2d_a[0], pts_2d_a[1], pts_2d_a[2], tol):
        return True

    return False


# ============================================================================
# 3D 三角形/边相交检测（核心算法）
# ============================================================================

def _edge_intersects_triangle_core(
    edge_start: np.ndarray, edge_end: np.ndarray,
    t1: np.ndarray, t2: np.ndarray, t3: np.ndarray,
    tol: float = DEFAULT_TOL
) -> bool:
    """
    核心：检测3D线段是否与三角形相交（含共面穿透）。

    处理三种情况：
    1. 线段穿越三角形平面 → 求交点并验证
    2. 线段与三角形共面 → 退化为2D边相交检测
    3. 线段端点在三角形上 → 根据上下文判断

    Args:
        edge_start, edge_end: 线段端点 (3,)
        t1, t2, t3: 三角形顶点 (3,)
        tol: 容差

    Returns:
        True 表示存在有效相交
    """
    normal = np.cross(t2 - t1, t3 - t1)
    norm_len = np.linalg.norm(normal)
    if norm_len < DEGENERATE_TOL:
        return False
    normal = normal / norm_len

    d1 = np.dot(edge_start - t1, normal)
    d2 = np.dot(edge_end - t1, normal)

    # 情况1: 两端在同一侧且不接近平面 → 不相交
    if d1 * d2 > tol and abs(d1) > tol and abs(d2) > tol:
        return False

    # 情况2: 共面
    if abs(d1) < tol and abs(d2) < tol:
        tri_edges = [(t1, t2), (t2, t3), (t3, t1)]
        for te_s, te_e in tri_edges:
            # 共面时使用3D线段距离判断代替2D投影，避免额外投影开销
            dist = segment_segment_distance_3d(edge_start, edge_end, te_s, te_e)
            if dist < tol * 10.0:
                # 排除纯端点接触
                for ep in (edge_start, edge_end):
                    for tv in (te_s, te_e):
                        if np.linalg.norm(ep - tv) < tol * 10.0:
                            break
                    else:
                        continue
                    break
                else:
                    return True

        # 检查线段中点是否在三角形内（完全包含情况）
        mid = (edge_start + edge_end) / 2.0
        if point_in_triangle_3d(mid, t1, t2, t3, tol):
            start_inside = point_in_triangle_3d(edge_start, t1, t2, t3, tol)
            end_inside = point_in_triangle_3d(edge_end, t1, t2, t3, tol)
            if not start_inside or not end_inside:
                return True
        return False

    # 情况3: 穿越平面
    denom = d1 - d2
    if abs(denom) < DEGENERATE_TOL:
        return False

    t_param = d1 / (denom + np.copysign(DEGENERATE_TOL, denom))
    if t_param < tol or t_param > 1.0 - tol:
        return False

    intersection = edge_start + t_param * (edge_end - edge_start)

    if not point_in_triangle_3d(intersection, t1, t2, t3, tol):
        return False

    # 排除交点恰好是三角形顶点的情况（视为非有效穿透）
    for vertex in (t1, t2, t3):
        if np.linalg.norm(intersection - vertex) < tol * 100.0:
            return False

    return True


# ============================================================================
# 高层API：支持自定义网格对象
# ============================================================================

def _extract_triangle_coords(triangle) -> np.ndarray:
    """
    从网格三角形对象中提取顶点坐标。

    支持两种格式：
    - 自定义对象: triangle.nodes[i].coords
    - NumPy数组: (3, 3) 直接返回

    Args:
        triangle: 三角形对象或数组

    Returns:
        (3, 3) 顶点坐标数组
    """
    if isinstance(triangle, np.ndarray):
        return triangle
    return np.array([
        triangle.nodes[0].coords,
        triangle.nodes[1].coords,
        triangle.nodes[2].coords,
    ], dtype=np.float64)


def check_triangle_intersection(
    tri1, tri2,
    tolerance: float = DEFAULT_TOL
) -> bool:
    """
    检查两个三角形是否真正相交（非仅共面接触）。

    算法流程：
    1. AABB包围盒快速排除
    2. 分离平面测试（双方向）
    3. 边-三角形穿透检测
    4. 顶点包含检测（排除共享顶点）

    Args:
        tri1, tri2: 三角形对象（需有 nodes[i].coords）或 (3,3) ndarray
        tolerance: 几何容差

    Returns:
        True 表示两三角形存在有效相交
    """
    p = _extract_triangle_coords(tri1)
    q = _extract_triangle_coords(tri2)

    p1, p2, p3 = p[0], p[1], p[2]
    q1, q2, q3 = q[0], q[1], q[2]

    # Step 1: AABB 快速排斥
    p_min = np.minimum(np.minimum(p1, p2), p3)
    p_max = np.maximum(np.maximum(p1, p2), p3)
    q_min = np.minimum(np.minimum(q1, q2), q3)
    q_max = np.maximum(np.maximum(q1, q2), q3)

    if np.any(p_max < q_min - tolerance) or np.any(p_min > q_max + tolerance):
        return False

    # Step 2: 分离平面测试
    n1 = np.cross(p2 - p1, p3 - p1)
    n2 = np.cross(q2 - q1, q3 - q1)
    n1_len = np.linalg.norm(n1)
    n2_len = np.linalg.norm(n2)

    if n1_len < DEGENERATE_TOL or n2_len < DEGENERATE_TOL:
        return False

    n1 = n1 / n1_len
    n2 = n2 / n2_len

    def _all_same_side(pts: np.ndarray, plane_pt: np.ndarray, plane_n: np.ndarray) -> bool:
        dists = np.dot(pts - plane_pt, plane_n)
        return bool(np.all(dists > tolerance) or np.all(dists < -tolerance))

    if _all_same_side(q, p1, n1) or _all_same_side(p, q1, n2):
        return False

    # Step 3: 边-三角形穿透检测
    edges_p = [(p1, p2), (p2, p3), (p3, p1)]
    edges_q = [(q1, q2), (q2, q3), (q3, q1)]

    for es, ee in edges_p:
        if _edge_intersects_triangle_core(es, ee, q1, q2, q3, tolerance):
            return True
    for es, ee in edges_q:
        if _edge_intersects_triangle_core(es, ee, p1, p2, p3, tolerance):
            return True

    # Step 4: 顶点包含检测（排除共享顶点）
    shared_tol = tolerance * 100.0

    def _is_shared(pt: np.ndarray, vertices: np.ndarray) -> bool:
        return any(np.linalg.norm(pt - v) < shared_tol for v in vertices)

    for pt in p:
        if point_in_triangle_3d(pt, q1, q2, q3, tolerance) and not _is_shared(pt, q):
            return True
    for pt in q:
        if point_in_triangle_3d(pt, p1, p2, p3, tolerance) and not _is_shared(pt, p):
            return True

    return False


def check_edge_triangle_intersection(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    triangle,
    tolerance: float = DEFAULT_TOL
) -> bool:
    """
    检查3D线段是否与三角形相交。

    Args:
        edge_start, edge_end: 线段端点 (3,)
        triangle: 三角形对象（需有 nodes[i].coords）或 (3,3) ndarray
        tolerance: 几何容差

    Returns:
        True 表示线段穿过三角形内部
    """
    verts = _extract_triangle_coords(triangle)
    return _edge_intersects_triangle_core(
        edge_start, edge_end,
        verts[0], verts[1], verts[2],
        tolerance
    )
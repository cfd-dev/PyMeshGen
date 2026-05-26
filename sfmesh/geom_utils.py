"""
2D/3D 计算几何函数

提供纯数值计算的几何判断工具：投影、线段相交、点在三角形内、共面重叠检测等。
无 OCC 依赖。
"""
from typing import List, Tuple
import numpy as np


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


def segment_intersects_triangle(p, q, a, b, c, tol=1e-8) -> bool:
    """Möller–Trumbore 算法：判断线段 pq 是否与三角形 abc 相交（不含端点共享）"""
    edge1 = b - a
    edge2 = c - a
    dir_vec = q - p

    h = np.cross(dir_vec, edge2)
    a_det = np.dot(edge1, h)
    if -tol < a_det < tol:
        return False

    f = 1.0 / a_det
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


def point_in_triangle_3d(p, a, b, c, tol=0.05) -> bool:
    """判断点 p 是否在三角形 abc 的内部（基于重心坐标和距离）"""
    v0 = c - a
    v1 = b - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return False

    bv = (d11 * d20 - d01 * d21) / denom
    bw = (d00 * d21 - d01 * d20) / denom
    bu = 1.0 - bv - bw

    if bu < -tol or bv < -tol or bw < -tol:
        return False

    normal = np.cross(v1, v0)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return False
    normal /= norm_len
    dist = abs(np.dot(p - a, normal))

    edge_len = max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b))
    if dist > edge_len * 0.1:
        return False

    return True


def segment_segment_distance_3d(p1, p2, q1, q2) -> float:
    """计算两条 3D 线段之间的最短距离"""
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1

    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    if a < 1e-24 and e < 1e-24:
        return np.linalg.norm(r)

    if a < 1e-24:
        s = 0.0
        t = np.clip(f / e, 0, 1)
    else:
        c = np.dot(d1, r)
        if e < 1e-24:
            t = 0.0
            s = np.clip(-c / a, 0, 1)
        else:
            b_val = np.dot(d1, d2)
            denom = a * e - b_val * b_val
            if abs(denom) > 1e-24:
                s = np.clip((b_val * f - c * e) / denom, 0, 1)
            else:
                s = 0.0
            t = (b_val * s + f) / e
            if t < 0:
                t = 0
                s = np.clip(-c / a, 0, 1)
            elif t > 1:
                t = 1
                s = np.clip((b_val - c) / a, 0, 1)

    closest_p = p1 + s * d1
    closest_q = q1 + t * d2
    return np.linalg.norm(closest_p - closest_q)


def point_triangle_distance_3d(p, a, b, c) -> float:
    """计算 3D 点到三角形的最短距离"""
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return np.linalg.norm(p - a)

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return np.linalg.norm(p - b)

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return np.linalg.norm(p - (a + v * ab))

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return np.linalg.norm(p - c)

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return np.linalg.norm(p - (a + w * ac))

    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.linalg.norm(p - (b + w * (c - b)))

    # 点在三角形内部
    n = np.cross(ab, ac)
    n_len = np.linalg.norm(n)
    if n_len < 1e-24:
        return 0.0
    n /= n_len
    return abs(np.dot(ap, n))


def _point_in_triangle_3d(point, t1, t2, t3, tolerance=1e-10):
    """
    检查3D点是否在三角形内（使用重心坐标）
    """
    v0 = t3 - t1
    v1 = t2 - t1
    v2 = point - t1
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < tolerance:
        return False
    
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= -tolerance) and (v >= -tolerance) and (u + v <= 1 + tolerance)


def _segments_intersect_3d(a1, a2, b1, b2, tolerance=1e-10):
    """
    检查两条3D线段是否相交
    """
    d1 = a2 - a1
    d2 = b2 - b1
    r = a1 - b1
    
    cross_d1_d2 = np.cross(d1, d2)
    cross_r_d2 = np.cross(r, d2)
    cross_r_d1 = np.cross(r, d1)
    
    denom = np.dot(cross_d1_d2, cross_d1_d2)
    
    if denom < tolerance:
        return False
    
    t = np.dot(cross_r_d2, cross_d1_d2) / denom
    s = np.dot(cross_r_d1, cross_d1_d2) / denom
    
    if -tolerance <= t <= 1 + tolerance and -tolerance <= s <= 1 + tolerance:
        point_on_a = a1 + t * d1
        point_on_b = b1 + s * d2
        return np.linalg.norm(point_on_a - point_on_b) < tolerance * 10
    
    return False


def _edge_triangle_intersection(edge_start, edge_end, t1, t2, t3, tolerance=1e-10):
    """
    检查边是否与三角形相交（包含内部穿透）
    """
    normal = np.cross(t2 - t1, t3 - t1)
    norm = np.linalg.norm(normal)
    if norm < tolerance:
        return False
    normal = normal / norm
    
    d1 = np.dot(edge_start - t1, normal)
    d2 = np.dot(edge_end - t1, normal)
    
    if d1 * d2 > tolerance and abs(d1) > tolerance and abs(d2) > tolerance:
        return False
    
    if abs(d1) < tolerance and abs(d2) < tolerance:
        tri_edges = [(t1, t2), (t2, t3), (t3, t1)]
        for te_start, te_end in tri_edges:
            if _segments_intersect_3d(edge_start, edge_end, te_start, te_end, tolerance):
                return True
        
        mid_point = (edge_start + edge_end) / 2.0
        if _point_in_triangle_3d(mid_point, t1, t2, t3, tolerance):
            if not _point_in_triangle_3d(edge_start, t1, t2, t3, tolerance) or \
               not _point_in_triangle_3d(edge_end, t1, t2, t3, tolerance):
                return True
        
        return False
    
    t = d1 / (d1 - d2 + 1e-20)
    if t < tolerance or t > 1 - tolerance:
        return False
    
    intersection = edge_start + t * (edge_end - edge_start)
    
    if _point_in_triangle_3d(intersection, t1, t2, t3, tolerance):
        for vertex in [t1, t2, t3]:
            if np.linalg.norm(intersection - vertex) < tolerance * 100:
                return False
        return True
    
    return False


def check_triangle_intersection(
    tri1,
    tri2,
    tolerance: float = 1e-10
) -> bool:
    """
    检查两个三角形是否相交（真正的相交，不只是平面相交）
    
    算法步骤：
    1. 快速AABB包围盒排除
    2. 分离轴测试（SAT）
    3. 边-三角形相交检测
    4. 共面情况处理
    
    Args:
        tri1, tri2: 两个三角形（需要有 nodes 属性，每个 node 有 coords 属性）
        tolerance: 容差
    
    Returns:
        是否相交
    """
    p1 = np.array(tri1.nodes[0].coords)
    p2 = np.array(tri1.nodes[1].coords)
    p3 = np.array(tri1.nodes[2].coords)
    
    q1 = np.array(tri2.nodes[0].coords)
    q2 = np.array(tri2.nodes[1].coords)
    q3 = np.array(tri2.nodes[2].coords)
    
    p_min = np.minimum(np.minimum(p1, p2), p3)
    p_max = np.maximum(np.maximum(p1, p2), p3)
    q_min = np.minimum(np.minimum(q1, q2), q3)
    q_max = np.maximum(np.maximum(q1, q2), q3)
    
    if np.any(p_max < q_min - tolerance) or np.any(p_min > q_max + tolerance):
        return False
    
    n1 = np.cross(p2 - p1, p3 - p1)
    n2 = np.cross(q2 - q1, q3 - q1)
    
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < tolerance or n2_norm < tolerance:
        return False
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    def signed_distance(point, plane_point, plane_normal):
        return np.dot(point - plane_point, plane_normal)
    
    d1_q1 = signed_distance(q1, p1, n1)
    d1_q2 = signed_distance(q2, p1, n1)
    d1_q3 = signed_distance(q3, p1, n1)
    
    if (d1_q1 > tolerance and d1_q2 > tolerance and d1_q3 > tolerance):
        return False
    if (d1_q1 < -tolerance and d1_q2 < -tolerance and d1_q3 < -tolerance):
        return False
    
    d2_p1 = signed_distance(p1, q1, n2)
    d2_p2 = signed_distance(p2, q1, n2)
    d2_p3 = signed_distance(p3, q1, n2)
    
    if (d2_p1 > tolerance and d2_p2 > tolerance and d2_p3 > tolerance):
        return False
    if (d2_p1 < -tolerance and d2_p2 < -tolerance and d2_p3 < -tolerance):
        return False
    
    edges1 = [(p1, p2), (p2, p3), (p3, p1)]
    edges2 = [(q1, q2), (q2, q3), (q3, q1)]
    
    for edge_start, edge_end in edges1:
        if _edge_triangle_intersection(edge_start, edge_end, q1, q2, q3, tolerance):
            return True
    
    for edge_start, edge_end in edges2:
        if _edge_triangle_intersection(edge_start, edge_end, p1, p2, p3, tolerance):
            return True
    
    tri2_vertices = [q1, q2, q3]
    for pt in [p1, p2, p3]:
        if _point_in_triangle_3d(pt, q1, q2, q3, tolerance):
            is_shared_vertex = False
            for vertex in tri2_vertices:
                if np.linalg.norm(pt - vertex) < tolerance * 100:
                    is_shared_vertex = True
                    break
            if not is_shared_vertex:
                return True
    
    tri1_vertices = [p1, p2, p3]
    for pt in [q1, q2, q3]:
        if _point_in_triangle_3d(pt, p1, p2, p3, tolerance):
            is_shared_vertex = False
            for vertex in tri1_vertices:
                if np.linalg.norm(pt - vertex) < tolerance * 100:
                    is_shared_vertex = True
                    break
            if not is_shared_vertex:
                return True
    
    return False


def check_edge_triangle_intersection(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    triangle,
    tolerance: float = 1e-10
) -> bool:
    """
    检查边与三角形是否相交
    
    Args:
        edge_start, edge_end: 边的端点
        triangle: 三角形（需要有 nodes 属性，每个 node 有 coords 属性）
        tolerance: 容差
    
    Returns:
        是否相交
    """
    p1 = np.array(triangle.nodes[0].coords)
    p2 = np.array(triangle.nodes[1].coords)
    p3 = np.array(triangle.nodes[2].coords)
    
    normal = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(normal)
    if norm < tolerance:
        return False
    normal = normal / norm
    
    d1 = np.dot(edge_start - p1, normal)
    d2 = np.dot(edge_end - p1, normal)
    
    if abs(d1) < tolerance and abs(d2) < tolerance:
        return False
    
    if d1 * d2 > tolerance:
        return False
    
    t = d1 / (d1 - d2 + 1e-20)
    intersection = edge_start + t * (edge_end - edge_start)
    
    v0 = p3 - p1
    v1 = p2 - p1
    v2 = intersection - p1
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < tolerance:
        return False
    
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= -tolerance) and (v >= -tolerance) and (u + v <= 1 + tolerance)

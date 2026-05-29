"""
2D/3D 计算几何工具库 (v2.0 - 严谨性修复版)

提供纯数值计算的几何判断工具：投影、线段相交、点在三角形内、共面重叠检测、
三角形相交检测等。无 OCC 依赖，仅依赖 NumPy。

主要算法：
- 2D线段相交：叉积符号测试 + 共线重叠区间检测 + T型相交检测
- 3D边-三角形相交：平面穿越(Möller–Trumbore) + 共面退化(2D投影)
- 3D三角形相交：AABB排斥 → 分离平面 → 共享边排除 → 边穿透 → 顶点包含
- 点到三角形距离：Voronoi区域法（顶点/边/面分类）
- 线段间距离：参数化最近点求解（含平行退化回退）
"""

from typing import List, Tuple, Optional, Union
import numpy as np

# ============================================================================
# 全局常量与配置
# ============================================================================
DEFAULT_TOL = 1e-10       # 通用几何容差（坐标级别）
DEGENERATE_TOL = 1e-24    # 退化检测容差（面积/长度平方级别）
COPLANAR_TOL = 1e-8       # 共面检测专用容差（比 DEFAULT_TOL 宽松，减少误判）
# 浮点最小正规数，用于除零保护（替代硬编码 1e-300）
FLOAT_MIN = np.finfo(np.float64).tiny  # ≈ 2.2e-308


# ============================================================================
# 2D 几何工具
# ============================================================================

def project_to_2d(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    将3D点集投影到2D平面。
    通过丢弃法向量绝对值最大的分量来避免投影退化。

    Args:
        points: (N, 3) 或 (3,) 三维点集
        normal: (3,) 投影平面法向量

    Returns:
        (N, 2) 或 (2,) 二维投影点集
    """
    abs_n = np.abs(normal)
    if abs_n[0] >= abs_n[1] and abs_n[0] >= abs_n[2]:
        return points[..., [1, 2]]
    elif abs_n[1] >= abs_n[0] and abs_n[1] >= abs_n[2]:
        return points[..., [0, 2]]
    else:
        return points[..., [0, 1]]


def _on_segment_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """
    检查已知共线的点 p 是否在线段 ab 的 AABB 范围内（含容差）。

    前提：调用方已确认 p 与 ab 共线（叉积 ≈ 0）。
    仅做区间包含判断，不重复共线性检查。

    Args:
        p: 待检查点 (2,), 已知与 ab 共线
        a, b: 线段端点 (2,)
        tol: 容差

    Returns:
        True 表示 p 在线段 ab 的包围盒内
    """
    return (min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and
            min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol)


def segments_intersect_2d(
    a1: np.ndarray, a2: np.ndarray,
    b1: np.ndarray, b2: np.ndarray,
    tol: float = DEFAULT_TOL
) -> bool:
    """
    判断两条2D线段是否有实质交集。

    检测三种相交模式：
    1. 共线重叠：四叉积均≈0 且投影区间有实质重叠（排除纯端点接触）
    2. 严格跨立：叉积异号（标准跨立实验）
    3. T型相交：某端点落在另一线段上（一叉积≈0 且在AABB内）

    端点重合（非共线情况下）不视为相交。

    Args:
        a1, a2: 线段A端点 (2,)
        b1, b2: 线段B端点 (2,)
        tol: 浮点容差

    Returns:
        True 表示两线段有实质交集
    """
    def cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 叉积 d1,d2: a端点到直线b的有符号距离（2倍面积）
    # 叉积 d3,d4: b端点到直线a的有符号距离
    d1 = cross2d(b1, b2, a1)
    d2 = cross2d(b1, b2, a2)
    d3 = cross2d(a1, a2, b1)
    d4 = cross2d(a1, a2, b2)

    # ---- 情况1: 共线重叠 ----
    # 四叉积均≈0 → 共线，投影到主轴检查区间是否有实质重叠
    if abs(d1) < tol and abs(d2) < tol and abs(d3) < tol and abs(d4) < tol:
        # 选择跨度更大的轴投影，避免退化为点时的数值问题
        if abs(a2[0] - a1[0]) >= abs(a2[1] - a1[1]):
            a_min, a_max = min(a1[0], a2[0]), max(a1[0], a2[0])
            b_min, b_max = min(b1[0], b2[0]), max(b1[0], b2[0])
        else:
            a_min, a_max = min(a1[1], a2[1]), max(a1[1], a2[1])
            b_min, b_max = min(b1[1], b2[1]), max(b1[1], b2[1])
        overlap_start = max(a_min, b_min)
        overlap_end = min(a_max, b_max)
        # 重叠长度 > tol 才算实质重叠（排除纯端点接触）
        if overlap_start < overlap_end - tol:
            return True
        return False

    # ---- 非共线情况: 排除端点重合 ----
    # 端点重合在 advancing front 中是拓扑邻接，不视为几何相交
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

    # ---- 情况2 & 3: 非共线相交 ----
    # 用三值符号函数替代乘法判断，避免小浮点数乘积缩放导致的误判
    # sign=+1: 在正侧, sign=-1: 在负侧, sign=0: 在直线上（≈容差内）
    def sign(x):
        return 1 if x > tol else (-1 if x < -tol else 0)

    s1, s2, s3, s4 = sign(d1), sign(d2), sign(d3), sign(d4)

    # 标准跨立：a的两端在直线b两侧 且 b的两端在直线a两侧
    if s1 * s2 < 0 and s3 * s4 < 0:
        return True

    # T型相交：某端点恰好落在另一线段上（叉积≈0 且 在AABB内）
    # 例：a1在直线b上且在b1b2区间内 → 两线段在a1处相交
    if s1 == 0 and _on_segment_2d(a1, b1, b2, tol):
        return True
    if s2 == 0 and _on_segment_2d(a2, b1, b2, tol):
        return True
    if s3 == 0 and _on_segment_2d(b1, a1, a2, tol):
        return True
    if s4 == 0 and _on_segment_2d(b2, a1, a2, tol):
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
    tol: float = DEFAULT_TOL,
    check_coplanar: bool = True
) -> bool:
    """
    判断3D点是否在三角形内部（基于重心坐标）。

    算法：计算重心坐标 (u, v)，判断 u≥0, v≥0, u+v≤1。
    可选的共面性预检可防止悬空点被误判（例如投影误差导致的偏离）。

    Args:
        p: 待测点 (3,)
        a, b, c: 三角形顶点 (3,)
        tol: 重心坐标容差（含边界膨胀，边界上的点视为在内部）
        check_coplanar: 是否先检查点到三角形平面的距离。
            设为 False 可跳过预检（当交点由平面方程精确求得时）。

    Returns:
        True 表示点在三角形内（含边界容差）
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a

    # [FIX] 共面性预检
    if check_coplanar:
        n = np.cross(v0, v1)
        n_len = np.linalg.norm(n)
        if n_len > DEGENERATE_TOL:
            dist = abs(np.dot(v2, n)) / n_len
            if dist > tol:
                return False

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < DEGENERATE_TOL * (dot00 * dot11 + FLOAT_MIN):
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
            if abs(denom) > DEGENERATE_TOL * (a * e + FLOAT_MIN):
                s = float(np.clip((b_val * f - c * e) / denom, 0.0, 1.0))
                t = (b_val * s + f) / e
            else:
                # 近似平行（denom ≈ 0）：解析解不稳定，回退到枚举端点组合
                # 检查4种端点到线段的最近距离，取最小值
                best = float('inf')
                best_s, best_t = 0.0, 0.0
                for s_cand in (0.0, 1.0):
                    t_cand = float(np.clip((b_val * s_cand + f) / e, 0.0, 1.0))
                    dist = np.linalg.norm((p1 + s_cand * d1) - (q1 + t_cand * d2))
                    if dist < best:
                        best, best_s, best_t = dist, s_cand, t_cand
                for t_cand in (0.0, 1.0):
                    s_cand = float(np.clip((b_val * t_cand - c) / a, 0.0, 1.0))
                    dist = np.linalg.norm((p1 + s_cand * d1) - (q1 + t_cand * d2))
                    if dist < best:
                        best, best_s, best_t = dist, s_cand, t_cand
                s, t = best_s, best_t
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

    按 Voronoi 区域逐步判断最近元素：顶点 → 边 → 面投影。
    每个除法分母增加 FLOAT_MIN 保护，防止退化几何下的 ZeroDivisionError。

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

    # 边AB区域：最近点在边AB上（参数 v ∈ [0,1]）
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        # copysign(FLOAT_MIN) 防止 d1≈d3 时除零，同时保持符号正确
        v = d1 / (d1 - d3 + np.copysign(FLOAT_MIN, d1 - d3))
        return float(np.linalg.norm(p - (a + v * ab)))

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))

    # 边AC区域：最近点在边AC上（参数 w ∈ [0,1]）
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6 + np.copysign(FLOAT_MIN, d2 - d6))
        return float(np.linalg.norm(p - (a + w * ac)))

    # 边BC区域：最近点在边BC上（参数 w ∈ [0,1]）
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom_edge = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / (denom_edge + np.copysign(FLOAT_MIN, denom_edge))
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

    注意：此处的"重叠"包含仅边界/边重合的退化情况。
    如需严格内部交集面积 > 0 的判断，请使用多边形裁剪算法。

    Args:
        tri_a: (3, 3) 三角形A顶点
        tri_b: (3, 3) 三角形B顶点
        surface_normal: (3,) 公共法向量
        tol: 容差

    Returns:
        True 表示两三角形有重叠区域（含边界接触）
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

def _point_on_segment_2d(
    p: np.ndarray, s1: np.ndarray, s2: np.ndarray,
    tol: float = DEFAULT_TOL
) -> bool:
    """判断2D点是否在线段上（含端点）。"""
    cp = (s2[0] - s1[0]) * (p[1] - s1[1]) - (s2[1] - s1[1]) * (p[0] - s1[0])
    if abs(cp) > tol:
        return False
    dot = (p[0] - s1[0]) * (p[0] - s2[0]) + (p[1] - s1[1]) * (p[1] - s2[1])
    return dot < tol


def _edge_intersects_triangle_core(
    edge_start: np.ndarray, edge_end: np.ndarray,
    t1: np.ndarray, t2: np.ndarray, t3: np.ndarray,
    tol: float = DEFAULT_TOL
) -> bool:
    """
    核心：检测3D线段是否与三角形相交（含共面穿透）。

    [FIX] 修复了同侧判断中使用乘积导致的浮点量级缩放误判问题。
    [FIX] 共面分支中避免频繁 np.array() 拼接，改为逐点投影。

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

    # d1, d2: 边端点到三角形平面的有符号距离
    d1 = np.dot(edge_start - t1, normal)
    d2 = np.dot(edge_end - t1, normal)

    # ---- 情况1: 两端在同一侧 → 不相交 ----
    # 原代码用 d1*d2 > tol 判断，当 d1,d2 均为小正数时乘积缩放会导致误判。
    # 改为显式比较，避免浮点量级问题。
    if (d1 > tol and d2 > tol) or (d1 < -tol and d2 < -tol):
        return False

    # ---- 情况2: 共面 → 投影到2D做标准相交检测 ----
    # 共面时3D相交退化为2D问题。投影到法向量最大分量对应的坐标平面，
    # 用2D线段相交（含共线重叠）+ 点在三角形内 + 顶点在边上三重检测。
    if abs(d1) < tol and abs(d2) < tol:
        proj_normal = np.cross(t2 - t1, t3 - t1)
        # 逐点投影避免紧密循环中 np.array() 拼接开销
        es_2d = project_to_2d(edge_start, proj_normal)
        ee_2d = project_to_2d(edge_end, proj_normal)
        tri_2d = np.array([
            project_to_2d(t1, proj_normal),
            project_to_2d(t2, proj_normal),
            project_to_2d(t3, proj_normal),
        ])
        te_2d = [(tri_2d[0], tri_2d[1]), (tri_2d[1], tri_2d[2]), (tri_2d[2], tri_2d[0])]

        # 检测1: 边与三角形边相交（含共线重叠和T型相交）
        for ts, te in te_2d:
            if segments_intersect_2d(es_2d, ee_2d, ts, te, tol):
                return True

        # 检测2: 边端点在三角形内（排除共享端点，避免拓扑邻接误报）
        shared_tol = tol * 10.0
        for ep, ep_3d in [(es_2d, edge_start), (ee_2d, edge_end)]:
            if point_in_triangle_2d(ep, tri_2d[0], tri_2d[1], tri_2d[2], tol):
                if all(np.linalg.norm(ep_3d - v) > shared_tol for v in (t1, t2, t3)):
                    return True

        # 检测3: 三角形顶点在边上（边穿过三角形顶点的退化情况）
        for tv, tv_3d in zip(tri_2d, (t1, t2, t3)):
            if all(np.linalg.norm(tv_3d - ep) > shared_tol for ep in (edge_start, edge_end)):
                if _point_on_segment_2d(tv, es_2d, ee_2d, tol):
                    return True

        return False

    # ---- 情况3: 穿越平面 ----
    # 边从平面一侧穿到另一侧，求交点并验证是否在三角形内
    denom = d1 - d2
    if abs(denom) < DEGENERATE_TOL:
        return False

    # t_param: 边上交点的参数 (0=起点, 1=终点)
    # copysign(FLOAT_MIN) 保证分母不为零且符号正确
    t_param = d1 / (denom + np.copysign(FLOAT_MIN, denom))
    # 排除端点附近的交点（tol < t < 1-tol），避免边-顶点接触误报
    if t_param < tol or t_param > 1.0 - tol:
        return False

    intersection = edge_start + t_param * (edge_end - edge_start)

    # 关闭共面预检：交点由平面方程精确求得，必然共面，无需重复验证
    if not point_in_triangle_3d(intersection, t1, t2, t3, tol, check_coplanar=False):
        return False

    # 排除交点恰好是三角形顶点的情况（视为边-顶点接触，非有效穿透）
    for vertex in (t1, t2, t3):
        if np.linalg.norm(intersection - vertex) < tol * 10.0:
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
    检查两个三角形是否真正相交（非仅拓扑邻接）。

    算法流程：
    1. AABB包围盒快速排斥
    2. 分离平面测试（双方向）
    3. 共享边排除（≥2个共享顶点 → 拓扑邻接，非几何穿透）
    4. 边-三角形穿透检测（含共面退化处理）
    5. 顶点包含检测（排除共享顶点）

    Args:
        tri1, tri2: 三角形对象（需有 nodes[i].coords）或 (3,3) ndarray
        tolerance: 几何容差

    Returns:
        True 表示两三角形存在有效几何穿透
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

    # Step 2.5: 共享边排除（含蝴蝶形检测）
    # 共享一条边（≥2个共享顶点）通常是拓扑邻接，但如果非共享边交叉
    # （蝴蝶形/bowtie），则仍属于几何相交。
    shared_tol = tolerance * 10.0
    shared_p = [i for i in range(3) if any(np.linalg.norm(p[i] - q[j]) < shared_tol for j in range(3))]
    shared_q = [j for j in range(3) if any(np.linalg.norm(q[j] - p[i]) < shared_tol for i in range(3))]
    if len(shared_p) >= 2 and len(shared_q) >= 2:
        # 找到非共享顶点，检查非共享边是否交叉
        non_shared_p = [i for i in range(3) if i not in shared_p]
        non_shared_q = [j for j in range(3) if j not in shared_q]
        if non_shared_p and non_shared_q:
            cp = p[non_shared_p[0]]  # tri1 的非共享顶点
            cq = q[non_shared_q[0]]  # tri2 的非共享顶点
            # 检查共享边的两个端点与非共享顶点组成的边是否交叉
            s1, s2 = p[shared_p[0]], p[shared_p[1]]
            if segment_segment_distance_3d(s1, cp, s2, cq) < tolerance:
                return True
            if segment_segment_distance_3d(s2, cp, s1, cq) < tolerance:
                return True
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
    shared_tol_v = tolerance * 100.0

    def _is_shared(pt: np.ndarray, vertices: np.ndarray) -> bool:
        return any(np.linalg.norm(pt - v) < shared_tol_v for v in vertices)

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


# ============================================================================
# 三角形质量与退化检测
# ============================================================================

def triangle_quality_from_coords(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> float:
    """
    从顶点坐标计算三角形形状质量因子。

    公式: 4√3 · Area / (a² + b² + c²)
    范围: [0, 1]，1 = 等边三角形

    Args:
        p0, p1, p2: 三角形顶点坐标 (3,)

    Returns:
        质量值 (0~1)
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


def check_triangle_degenerate(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    min_height: float, min_edge_len: float
) -> bool:
    """
    检查三角形是否退化：顶点到对边距离过小 或 边长过短。

    Args:
        p0, p1, p2: 三角形顶点坐标 (3,)
        min_height: 最小高度阈值（p2 到边 p0p1 的距离）
        min_edge_len: 最小边长阈值（边 p0p2 和 p1p2）

    Returns:
        True 表示三角形退化（应拒绝）
    """
    edge_vec = p1 - p0
    edge_len_sq = np.dot(edge_vec, edge_vec)
    if edge_len_sq > 1e-24:
        t = np.dot(p2 - p0, edge_vec) / edge_len_sq
        closest = p0 + np.clip(t, 0, 1) * edge_vec
        if np.linalg.norm(p2 - closest) < min_height:
            return True

    d02 = np.linalg.norm(p2 - p0)
    d12 = np.linalg.norm(p2 - p1)
    if d02 < min_edge_len or d12 < min_edge_len:
        return True

    return False
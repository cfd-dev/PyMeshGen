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

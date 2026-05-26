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
    
def _check_intersection(self, front, node) -> bool:
    """
    3D 相交检测：边交叉 + 节点包含

    使用边-边交叉检测（投影到2D），避免共享顶点在曲面上产生假阳性。
    不使用法向翻转检测（front 方向不保证与曲面法向一致）。
    """
    p0 = np.array(front.node_elems[0].coords)
    p1 = np.array(front.node_elems[1].coords)
    p2 = np.array(node.coords)

    shared_node_ids = {front.node_elems[0].idx, front.node_elems[1].idx, node.idx}

    if not self.space_index_triangle or not self.triangle_list:
        return False

    # 1. 退化三角形检测
    new_tri_normal = np.cross(p1 - p0, p2 - p0)
    norm_len = np.linalg.norm(new_tri_normal)
    if norm_len < 1e-12:
        return True
    new_tri_normal /= norm_len

    # 2. RTree 查询候选三角形，检测边交叉与节点包含
    all_pts = np.array([p0, p1, p2])
    padding = self.sizing_field.global_spacing * 0.1
    query_bbox = (
        all_pts[:, 0].min() - padding,
        all_pts[:, 1].min() - padding,
        all_pts[:, 2].min() - padding,
        all_pts[:, 0].max() + padding,
        all_pts[:, 1].max() + padding,
        all_pts[:, 2].max() + padding,
    )
    candidate_ids = list(self.space_index_triangle.intersection(query_bbox))

    # 新三角形投影到2D
    new_pts_2d = _project_to_2d(all_pts, new_tri_normal)
    new_edges_2d = [
        (new_pts_2d[0], new_pts_2d[2]),
        (new_pts_2d[2], new_pts_2d[1]),
    ]

    for tri_id in candidate_ids:
        existing_tri = self._triangle_dict.get(tri_id)
        if not existing_tri:
            continue

        existing_node_ids = set(existing_tri.node_ids)
        shared_count = len(existing_node_ids & shared_node_ids)

        if shared_count >= 2:
            continue  # 共享边，合法邻接

        q0, q1, q2 = [np.array(n.coords) for n in existing_tri.nodes]

        if shared_count == 0:
            # 边-边交叉检测（投影到2D）
            exist_pts_2d = _project_to_2d(np.array([q0, q1, q2]), new_tri_normal)
            exist_edges_2d = [
                (exist_pts_2d[0], exist_pts_2d[1]),
                (exist_pts_2d[1], exist_pts_2d[2]),
                (exist_pts_2d[2], exist_pts_2d[0]),
            ]
            for ne in new_edges_2d:
                for ee in exist_edges_2d:
                    if _segments_intersect_2d(ne[0], ne[1], ee[0], ee[1]):
                        return True

            # 节点包含检测：新节点是否掉入现有三角形内部
            if _point_in_triangle_2d(new_pts_2d[2], exist_pts_2d[0], exist_pts_2d[1], exist_pts_2d[2]):
                if np.linalg.norm(p2 - q0) > 1e-6:
                    return True

    return False

def _segment_intersects_triangle(self, p, q, a, b, c, tol=1e-4) -> bool:
    """Möller–Trumbore 算法：判断线段 pq 是否与三角形 abc 相交"""
    edge1 = b - a
    edge2 = c - a
    dir_vec = q - p

    h = np.cross(dir_vec, edge2)
    a_det = np.dot(edge1, h)
    if -tol < a_det < tol:
        return False  # 平行

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
    # 确保交点在线段 pq 的内部 (排除端点共享的情况)
    if tol < t < 1.0 - tol:
        return True

    return False

def _point_projects_inside_triangle(self, p, a, b, c) -> bool:
    """判断点 p 是否在三角形 abc 的内部 (基于重心坐标和距离)"""
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

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    # 如果投影在内部，且距离三角形平面极近，则认为穿透
    if (u >= -0.05) and (v >= -0.05) and (w >= -0.05):
        normal = np.cross(v1, v0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            return False
        normal /= norm_len
        dist = abs(np.dot(p - a, normal))
        if dist < self.sizing_field.global_spacing * 0.15:
            return True
    return False
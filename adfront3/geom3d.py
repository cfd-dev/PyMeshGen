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

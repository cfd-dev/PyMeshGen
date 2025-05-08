from math import sqrt, isnan, isinf
import numpy as np
import math


def point_to_segment_distance(point, segment_start, segment_end):
    """
    计算点到线段的最短距离（使用numpy）
    :param point: 点的坐标 (x, y)
    :param segment_start: 线段起点 (x, y)
    :param segment_end: 线段终点 (x, y)
    :return: 最短距离
    """
    point = np.array(point, dtype=np.float64)
    s = np.array(segment_start, dtype=np.float64)
    e = np.array(segment_end, dtype=np.float64)

    # 线段方向向量
    v = e - s
    w = point - s

    # 计算投影参数t
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(point - s)

    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(point - e)

    # 投影点在中间
    t = c1 / c2
    projection = s + t * v
    return np.linalg.norm(point - projection)


def segments_closest_distance(A, B, C, D):
    """
    计算两条线段的最小距离（使用numpy）
    :param A: 线段AB的起点 (x, y)
    :param B: 线段AB的终点 (x, y)
    :param C: 线段CD的起点 (x, y)
    :param D: 线段CD的终点 (x, y)
    :return: 最小距离
    """
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    D = np.array(D, dtype=np.float64)

    # 计算端点到对面线段的距离
    d1 = point_to_segment_distance(A, C, D)
    d2 = point_to_segment_distance(B, C, D)
    d3 = point_to_segment_distance(C, A, B)
    d4 = point_to_segment_distance(D, A, B)
    candidates = [d1, d2, d3, d4]

    # 计算线段间的内部距离
    AB = B - A
    CD = D - C
    AC = C - A

    # 向量点积
    AB_dot_AB = np.dot(AB, AB)
    AB_dot_CD = np.dot(AB, CD)
    CD_dot_CD = np.dot(CD, CD)
    AC_dot_AB = np.dot(AC, AB)
    AC_dot_CD = np.dot(AC, CD)

    # 计算行列式D
    D_det = AB_dot_AB * CD_dot_CD - AB_dot_CD**2

    # 处理平行或接近平行的情况
    if D_det < 1e-10:  # 更严格的平行判断阈值
        return min(candidates)

    # 解参数s和t
    s_num = AC_dot_AB * CD_dot_CD - AC_dot_CD * AB_dot_CD
    t_num = AC_dot_AB * AB_dot_CD - AC_dot_CD * AB_dot_AB
    s = s_num / D_det
    t = t_num / D_det

    if 0 <= s <= 1 and 0 <= t <= 1:
        P = A + s * AB
        Q = C + t * CD
        distance = np.linalg.norm(P - Q)
        candidates.append(distance)

    return min(candidates)


def min_distance_between_segments(A, B, C, D):
    """
    主函数：计算两条线段的最小距离
    :param A: 线段AB的起点 (x, y)
    :param B: 线段AB的终点 (x, y)
    :param C: 线段CD的起点 (x, y)
    :param D: 线段CD的终点 (x, y)
    :return: 最小距离
    """
    return segments_closest_distance(A, B, C, D)


def normal_vector2d(front):
    """计算二维平面阵面的单位法向量"""
    node1, node2 = [front.node_elems[i].coords for i in range(2)]
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]

    # 计算向量模长
    magnitude = sqrt(dx**2 + dy**2)

    # 处理零向量情况
    if magnitude < 1e-12:
        return (0.0, 0.0)

    # 单位化法向量
    return (-dy / magnitude, dx / magnitude)


def calculate_angle(p1, p2, p3):
    """计算空间中三个点的夹角（角度制），自动适配2D/3D坐标"""
    # 自动补充z坐标（二维点z=0）
    coord1 = [*p1, 0] if len(p1) == 2 else p1
    coord2 = [*p2, 0] if len(p2) == 2 else p2
    coord3 = [*p3, 0] if len(p3) == 2 else p3

    v1 = np.array([coord1[i] - coord2[i] for i in range(3)])
    v2 = np.array([coord3[i] - coord2[i] for i in range(3)])

    # 计算向量模长
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # 处理零向量情况
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    # 计算夹角的余弦值
    cos_theta = np.dot(v1, v2) / (magnitude_v1 * magnitude_v2)

    # 处理余弦值超出范围的情况
    cos_theta = max(-1.0, min(1.0, cos_theta))

    # 计算夹角（弧度制）
    theta = np.arccos(cos_theta)

    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    if cross_z < 0:
        theta = 2 * np.pi - theta

    return np.degrees(theta)


def calculate_distance(p1, p2):
    """计算二维/三维点间距"""
    return sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calculate_distance2(p1, p2):
    """计算二维/三维点间距"""
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


def triangle_area(p1, p2, p3):
    """计算三角形面积（支持2D/3D点）"""
    # 向量叉积法计算面积
    v1 = (
        [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
        if len(p1) > 2
        else [p2[0] - p1[0], p2[1] - p1[1], 0]
    )
    v2 = (
        [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]]
        if len(p1) > 2
        else [p3[0] - p1[0], p3[1] - p1[1], 0]
    )
    cross = (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )
    return 0.5 * sqrt(sum(x**2 for x in cross))


def is_left2d(p1, p2, p3):
    """判断点p3是否在p1-p2向量的左侧"""
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p1[0], p3[1] - p1[1])
    # 向量叉积
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_product > 0


def is_convex(a, b, c, d, node_coords):
    """改进的凸性检查，支持节点索引或坐标"""

    # 确保顶点按顺序排列（如a→b→c→d→a）
    def get_coord(x):
        return node_coords[x] if isinstance(x, int) else x

    points = [get_coord(a), get_coord(b), get_coord(c), get_coord(d)]

    cross_products = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        p3 = points[(i + 2) % 4]

        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        cross = np.cross(v1, v2)
        cross_products.append(cross)

    # 检查所有叉积符号是否一致（全为正或全为负）
    positive = all(cp > 0 for cp in cross_products)
    negative = all(cp < 0 for cp in cross_products)

    return positive or negative


def fast_distance_check(p0, p1, q0, q1, safe_distance_sq):
    """快速距离检查"""
    # 线段端点距离检查
    for p in [p0, p1]:
        for q in [q0, q1]:
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            if dx * dx + dy * dy < safe_distance_sq:
                return True

    # 线段间距离检查
    return min_distance_between_segments(p0, p1, q0, q1) < sqrt(safe_distance_sq)


# 计算三角形最小角
def calculate_min_angle(cell, node_coords):
    from basic_elements import Triangle

    if isinstance(cell, Triangle):
        cell = cell.node_ids

    if len(cell) != 3:
        return 0.0
    p1, p2, p3 = cell
    angles = []
    for i in range(3):
        prev = cell[(i - 1) % 3]
        next_p = cell[(i + 1) % 3]
        v1 = np.array(node_coords[prev]) - np.array(node_coords[cell[i]])
        v2 = np.array(node_coords[next_p]) - np.array(node_coords[cell[i]])
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angles.append(np.rad2deg(angle))
    return min(angles) if angles else 0.0


# 检查三角形是否有效（非退化）
def is_valid_triangle(cell, node_coords):
    from basic_elements import Triangle

    if isinstance(cell, Triangle):
        cell = cell.node_ids

    if len(cell) != 3:
        return False
    p1, p2, p3 = cell

    # 面积计算
    v1 = np.array(node_coords[p2]) - np.array(node_coords[p1])
    v2 = np.array(node_coords[p3]) - np.array(node_coords[p1])
    area = 0.5 * np.abs(np.cross(v1, v2))  # 使用向量叉积计算面积
    return area > 1e-10


# 检查两点是否重合 TODO: epsilon取值对实际问题的影响
def points_equal(p1, p2, epsilon=1e-6):
    return abs(p1[0] - p2[0]) < epsilon and abs(p1[1] - p2[1]) < epsilon


def segments_intersect(a1, a2, b1, b2):
    """判断两个二维线段是否相交（包含共线部分重叠但不包含端点重合的情况）

    参数：
        a1, a2: 线段A的端点坐标 (x, y)
        b1, b2: 线段B的端点坐标 (x, y)

    返回值：
        bool: 是否相交（True表示相交）
    """
    # 阶段1：快速排除端点重合的情况
    if any(points_equal(a, b) for a in (a1, a2) for b in (b1, b2)):
        return False

    # 阶段2：轴对齐包围盒（AABB）快速排斥实验
    # 计算线段A的包围盒
    a_min_x = min(a1[0], a2[0])
    a_max_x = max(a1[0], a2[0])
    a_min_y = min(a1[1], a2[1])
    a_max_y = max(a1[1], a2[1])

    # 计算线段B的包围盒
    b_min_x = min(b1[0], b2[0])
    b_max_x = max(b1[0], b2[0])
    b_min_y = min(b1[1], b2[1])
    b_max_y = max(b1[1], b2[1])

    # 包围盒不相交则直接返回False
    if (
        (a_max_x < b_min_x)
        or (b_max_x < a_min_x)
        or (a_max_y < b_min_y)
        or (b_max_y < a_min_y)
    ):
        return False

    # 阶段3：跨立实验（使用向量叉积判断线段相对位置）
    def cross(o, a, b):
        """计算向量oa和ob的叉积（z分量）"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 计算四个关键叉积值
    c1 = cross(a1, a2, b1)  # b1相对a1a2的位置
    c2 = cross(a1, a2, b2)  # b2相对a1a2的位置
    c3 = cross(b1, b2, a1)  # a1相对b1b2的位置
    c4 = cross(b1, b2, a2)  # a2相对b1b2的位置

    # 标准相交情况：线段AB相互跨立
    if (c1 * c2 < 0) and (c3 * c4 < 0):
        return True

    # 阶段4：处理共线情况（所有叉积为0时）
    # if c1 == 0 and c2 == 0 and c3 == 0 and c4 == 0:
    #     # 检查坐标轴投影是否有重叠（允许部分重叠但不完全重合）
    #     x_overlap = max(a_min_x, b_min_x) <= min(a_max_x, b_max_x)
    #     y_overlap = max(a_min_y, b_min_y) <= min(a_max_y, b_max_y)
    #     return x_overlap and y_overlap
    epsilon = 1e-8
    if all(abs(val) < epsilon for val in [c1, c2, c3, c4]):
        # 检查坐标轴投影是否有重叠（允许部分重叠但不完全重合）
        # 改用更精确的浮点数比较
        x_overlap = (a_max_x >= b_min_x - epsilon) and (a_min_x <= b_max_x + epsilon)
        y_overlap = (a_max_y >= b_min_y - epsilon) and (a_min_y <= b_max_y + epsilon)
        return x_overlap and y_overlap

    return False


def is_point_inside_or_on(p, a, b, c):
    # 检查点p是否在三角形内部或边上，但排除共享顶点的情况
    if any(points_equal(p, vtx) for vtx in [a, b, c]):
        return False  # 共享顶点，视为不相交

    def cross_sign(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = cross_sign(a, b, p)
    d2 = cross_sign(b, c, p)
    d3 = cross_sign(c, a, p)

    has_neg = (d1 < -1e-8) or (d2 < -1e-8) or (d3 < -1e-8)
    has_pos = (d1 > 1e-8) or (d2 > 1e-8) or (d3 > 1e-8)

    return not (has_neg and has_pos)


def triangle_intersect_triangle(p1, p2, p3, p4, p5, p6):
    """判断两个三角形是否相交，仅共享一条边不算相交、仅共享一个顶点不算相交"""
    tri1_edges = [(p1, p2), (p2, p3), (p3, p1)]
    tri2_edges = [(p4, p5), (p5, p6), (p6, p4)]

    # 检查边是否相交（排除共边）
    for e1 in tri1_edges:
        a1, a2 = e1
        for e2 in tri2_edges:
            b1, b2 = e2
            if segments_intersect(a1, a2, b1, b2):
                # 检查是否为共边（顶点完全相同）
                if (points_equal(a1, b1) and points_equal(a2, b2)) or (
                    points_equal(a1, b2) and points_equal(a2, b1)
                ):
                    continue  # 共边，视为不相交
                else:
                    return True  # 非共边的相交

    # 检查顶点是否在另一个三角形内部或边上（严格内部或边上，但排除共享顶点）
    for p in [p1, p2, p3]:
        if any(points_equal(p, tri_p) for tri_p in [p4, p5, p6]):
            continue
        if is_point_inside_or_on(p, p4, p5, p6):
            return True

    for p in [p4, p5, p6]:
        if any(points_equal(p, self_p) for self_p in [p1, p2, p3]):
            continue
        if is_point_inside_or_on(p, p1, p2, p3):
            return True

    # 检查当前三角形是否完全在另一个三角形内部（非共享顶点）
    all_in_self = True
    for p in [p1, p2, p3]:
        if not is_point_inside_or_on(p, p4, p5, p6):
            all_in_self = False
            break
    if all_in_self:
        return True

    # 检查另一个三角形是否完全在当前三角形内部
    all_in_other = True
    for p in [p4, p5, p6]:
        if not is_point_inside_or_on(p, p1, p2, p3):
            all_in_other = False
            break
    if all_in_other:
        return True

    return False


def quadrilateral_area(p1, p2, p3, p4):
    """使用鞋带公式计算任意简单四边形面积（顶点需按顺序排列）"""
    points = np.array([p1, p2, p3, p4])
    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_valid_quadrilateral(p1, p2, p3, p4):
    """检查四边形是否非退化（面积>阈值）"""
    area = quadrilateral_area(p1, p2, p3, p4)
    return area > 1e-10


def is_point_inside_quad(p, quad_points):
    """
    判断点p是否在四边形内部（不包括边界）
    :param p: 待判断的点 (x, y)
    :param quad_points: 四边形顶点列表 [p1, p2, p3, p4]，按顺时针或逆时针顺序排列
    :return: True如果在内部，False否则
    """
    x, y = p
    n = len(quad_points)
    inside = False

    for i in range(n):
        a, b = quad_points[i], quad_points[(i + 1) % n]
        # 检查点是否在顶点上
        if abs(a[0] - x) < 1e-8 and abs(a[1] - y) < 1e-8:
            return False

        # 检查点是否在边上
        if (min(a[0], b[0]) <= x <= max(a[0], b[0])) and (
            min(a[1], b[1]) <= y <= max(a[1], b[1])
        ):
            # 计算点到边的距离
            cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
            if abs(cross) < 1e-8:
                return False

        # 射线法核心逻辑
        if (a[1] > y) != (b[1] > y):
            intersect_x = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < intersect_x:
                inside = not inside

    return inside


def quad_intersects_triangle(
    quad_p1, quad_p2, quad_p3, quad_p4, tri_p1, tri_p2, tri_p3
):
    # 新增三点重合检查
    quad_points = [quad_p1, quad_p2, quad_p3, quad_p4]
    tri_points = [tri_p1, tri_p2, tri_p3]

    # 统计四边形顶点在三角形顶点中的重复数量
    overlap_count = sum(
        any(points_equal(qp, tp) for tp in tri_points) for qp in quad_points
    )
    if overlap_count >= 3:
        return True

    quad_edges = [
        (quad_p1, quad_p2),
        (quad_p2, quad_p3),
        (quad_p3, quad_p4),
        (quad_p4, quad_p1),
    ]
    tri_edges = [(tri_p1, tri_p2), (tri_p2, tri_p3), (tri_p3, tri_p1)]

    # 检查边的相交
    for edge_quad in quad_edges:
        a1, a2 = edge_quad
        for edge_tri in tri_edges:
            b1, b2 = edge_tri
            if segments_intersect(a1, a2, b1, b2):
                return True  # 边相交，直接返回True

    # 检查四边形顶点是否在三角形内部或边上
    for p in [quad_p1, quad_p2, quad_p3, quad_p4]:
        if is_point_inside_or_on(p, tri_p1, tri_p2, tri_p3):
            return True

    # 检查三角形顶点是否在四边形内部或边上
    for p in [tri_p1, tri_p2, tri_p3]:
        if is_point_inside_quad(p, [quad_p1, quad_p2, quad_p3, quad_p4]):
            return True

    # 检查四边形是否完全在三角形内部（冗余，但保留以防万一）
    all_inside_tri = True
    for p in [quad_p1, quad_p2, quad_p3, quad_p4]:
        if not is_point_inside_or_on(p, tri_p1, tri_p2, tri_p3):
            all_inside_tri = False
            break
    if all_inside_tri:
        return True

    # 检查三角形是否完全在四边形内部
    all_inside_quad = True
    for p in [tri_p1, tri_p2, tri_p3]:
        if not is_point_inside_quad(p, [quad_p1, quad_p2, quad_p3, quad_p4]):
            all_inside_quad = False
            break
    if all_inside_quad:
        return True

    return False


def is_same_edge(e1, e2):
    # 判断两条边是否是同一条边（共边）
    a1, a2 = e1
    b1, b2 = e2
    return (points_equal(a1, b1) and points_equal(a2, b2)) or (
        points_equal(a1, b2) and points_equal(a2, b1)
    )


def is_shared_vertex(p, other_quad):
    # 判断顶点p是否是另一个四边形的顶点
    return any(points_equal(p, v) for v in other_quad)


def quad_inside_another(quad_inside, quad_outside):
    # 判断quad_inside是否完全在quad_outside内部（排除共享顶点）
    for p in quad_inside:
        if any(points_equal(p, v) for v in quad_outside):
            continue
        if not is_point_inside_quad(p, quad_outside):
            return False
    return True


def quad_intersects_quad(q1, q2):
    # q1和q2是四边形顶点列表，顺序为[p1,p2,p3,p4]
    q1_edges = [(q1[0], q1[1]), (q1[1], q1[2]), (q1[2], q1[3]), (q1[3], q1[0])]
    q2_edges = [(q2[0], q2[1]), (q2[1], q2[2]), (q2[2], q2[3]), (q2[3], q2[0])]

    # 检查边相交
    for e1 in q1_edges:
        a1, a2 = e1
        for e2 in q2_edges:
            b1, b2 = e2
            if is_same_edge(e1, e2):
                continue
            if segments_intersect(a1, a2, b1, b2):
                return True

    # 检查q1的边与q2的对角线是否相交
    q2_diags = [(q2[0], q2[2]), (q2[1], q2[3])]
    for e1 in q1_edges:
        a1, a2 = e1
        for diag in q2_diags:
            d1, d2 = diag
            if segments_intersect(a1, a2, d1, d2):
                return True

    # 检查q2的边与q1的对角线是否相交
    q1_diags = [(q1[0], q1[2]), (q1[1], q1[3])]
    for e2 in q2_edges:
        b1, b2 = e2
        for diag in q1_diags:
            d1, d2 = diag
            if segments_intersect(d1, d2, b1, b2):
                return True

    # 检查顶点是否在对方内部或边上（排除共享顶点）
    for p in q1:
        if not is_shared_vertex(p, q2) and is_point_inside_quad(p, q2):
            return True
    for p in q2:
        if not is_shared_vertex(p, q1) and is_point_inside_quad(p, q1):
            return True

    # 检查完全包含
    if quad_inside_another(q1, q2) or quad_inside_another(q2, q1):
        return True

    return False


# def is_point_on_segment(p, a, b):
#     """判断点p是否在线段ab上（包括端点，排除共线但不在线段的情况）"""
#     # 叉积判断共线
#     cross = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
#     if abs(cross) > 1e-8:
#         return False

#     # 坐标范围检查
#     min_x = min(a[0], b[0]) - 1e-8
#     max_x = max(a[0], b[0]) + 1e-8
#     min_y = min(a[1], b[1]) - 1e-8
#     max_y = max(a[1], b[1]) + 1e-8
#     return (min_x <= p[0] <= max_x) and (min_y <= p[1] <= max_y)

# def point_in_polygon(p, polygon):
#     x, y = p
#     n = len(polygon)
#     if n == 0:
#         return False

#     # 转换为元组避免列表与元组比较问题
#     p_tuple = (x, y)
#     polygon = [tuple(point) for point in polygon]

#     # 检查是否是顶点
#     if p_tuple in polygon:
#         return False

#     # 检查是否在边上（非顶点）
#     for i in range(n):
#         a = polygon[i]
#         b = polygon[(i+1) % n]
#         if is_point_on_segment(p, a, b):
#             return False

#     # 射线法判断内部（奇数次交叉为内部）
#     inside = False
#     for i in range(n):
#         a, b = polygon[i], polygon[(i + 1) % n]
#         xa, ya = a
#         xb, yb = b

#         # 边AB跨越射线的y坐标
#         if (ya <= y < yb) != (yb <= y < ya):
#             # 计算交点x坐标
#             x_inter = (y - ya) * (xb - xa) / (yb - ya) + xa
#             if x_inter >= x:
#                 inside = not inside

#     return inside


def is_point_on_segment(p, a, b):
    cross = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    if abs(cross) > 1e-8:
        return False
    min_x = min(a[0], b[0]) - 1e-8
    max_x = max(a[0], b[0]) + 1e-8
    min_y = min(a[1], b[1]) - 1e-8
    max_y = max(a[1], b[1]) + 1e-8
    return (min_x <= p[0] <= max_x) and (min_y <= p[1] <= max_y)


def point_in_polygon(p, polygon):
    x, y = p
    n = len(polygon)
    if n == 0:
        return False
    p_tuple = (x, y)
    polygon = [tuple(point) for point in polygon]

    if p_tuple in polygon:
        return False

    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if is_point_on_segment(p, a, b):
            return False

    inside = False
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        xa, ya = a
        xb, yb = b

        if xa == xb:  # 处理垂直线
            if (
                (ya < y and yb >= y)
                or (yb < y and ya >= y)
                or (ya > y and yb <= y)
                or (yb > y and ya <= y)
            ):
                if xa >= x:
                    inside = not inside
            continue

        # 非垂直线，判断是否跨越y
        if (
            (ya < y and yb >= y)
            or (yb < y and ya >= y)
            or (ya > y and yb <= y)
            or (yb > y and ya <= y)
        ):
            # 计算交点x坐标
            try:
                x_inter = (y - ya) * (xb - xa) / (yb - ya) + xa
            except ZeroDivisionError:
                # 避免因浮点误差导致的除零，但非垂直线应不会出现
                continue
            if x_inter >= x:
                inside = not inside

    return inside


def centroid(polygon_points):
    # 计算多边形的形心
    x = np.mean(polygon_points[:, 0])
    y = np.mean(polygon_points[:, 1])
    return np.array([x, y])


def unit_direction_vector(node1, node2):
    """计算单位方向向量（增强版）"""
    dim = len(node1)
    if dim not in (2, 3) or len(node2) != dim:
        raise ValueError("Nodes must be 2D or 3D with same dimension")

    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]
    dz = (node2[2] - node1[2]) if dim == 3 else 0.0

    length = (dx**2 + dy**2 + dz**2) ** 0.5
    if length == 0:
        return (0.0,) * dim
    return (dx / length, dy / length, dz / length)[:dim]

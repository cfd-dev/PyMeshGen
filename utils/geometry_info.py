from math import sqrt, isnan, isinf
import numpy as np
import matplotlib.pyplot as plt


def normal_vector2d(front):
    """计算二维平面阵面的单位法向量"""
    node1, node2 = front.nodes_coords
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]

    # 计算向量模长
    magnitude = sqrt(dx**2 + dy**2)

    # 处理零向量情况
    if magnitude < 1e-12:
        return (0.0, 0.0)

    # 单位化法向量
    return (-dy / magnitude, dx / magnitude)


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


def triangle_quality(p1, p2, p3):
    """计算三角形网格质量（两种方法）"""
    a = calculate_distance(p1, p2)
    b = calculate_distance(p2, p3)
    c = calculate_distance(p3, p1)
    area = triangle_area(p1, p2, p3)

    # 方法1：内外接圆半径比（注释保留）
    # denominator = a + b + c
    # if denominator == 0:
    #     return 0.0
    # r = 2.0 * area / denominator  # 内切圆半径
    # R = (a * b * c) / (4.0 * area) if area != 0 else 0  # 外接圆半径
    # quality_method = 3.0 * r / R if R != 0 else 0
    # upper_limit = 1.5

    # 方法2：面积与边长平方比（默认使用该方法）
    denominator = a**2 + b**2 + c**2
    quality_method = 4.0 * sqrt(3.0) * area / denominator if denominator != 0 else 0
    upper_limit = 1.0
    lower_limit = 0.0
    # 新增异常检测
    if (
        isnan(quality_method)
        or isinf(quality_method)
        or quality_method < lower_limit
        or quality_method > upper_limit
    ):
        raise ValueError(
            f"三角形质量计算异常：quality={quality_method}，节点坐标：{p1}, {p2}, {p3}"
        )

    return quality_method


def is_left(p1, p2, p3):
    """判断点p3是否在p1-p2向量的左侧"""
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p1[0], p3[1] - p1[1])
    # 向量叉积
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_product > 0


# 辅助函数：判断四边形是否为凸
def is_convex(a, b, c, d, node_coords):
    ab = np.array(node_coords[b]) - np.array(node_coords[a])
    ac = np.array(node_coords[c]) - np.array(node_coords[a])
    ad = np.array(node_coords[d]) - np.array(node_coords[a])
    cross_c = np.cross(ab, ac)
    cross_d = np.cross(ab, ad)
    return cross_c * cross_d < 0  # 符号不同则在两侧


# 辅助函数：计算三角形最小角
def calculate_min_angle(cell, node_coords):
    if isinstance(cell, Triangle):
        cell = cell.node_idx

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


# 辅助函数：检查三角形是否有效（非退化）
def is_valid_triangle(cell, node_coords):
    if isinstance(cell, Triangle):
        cell = cell.node_idx

    if len(cell) != 3:
        return False
    p1, p2, p3 = cell

    # 面积计算
    v1 = np.array(node_coords[p2]) - np.array(node_coords[p1])
    v2 = np.array(node_coords[p3]) - np.array(node_coords[p1])
    area = 0.5 * np.abs(np.cross(v1, v2))  # 使用向量叉积计算面积
    return area > 1e-10


class NodeElement:
    def __init__(self, node_coords, idx):
        # 四舍五入到小数点后6位以消除浮点误差
        self.node_coords = [round(coord, 6) for coord in node_coords]
        self.idx = idx
        # 使用处理后的坐标生成哈希
        self.hash = hash(tuple(self.node_coords))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if isinstance(other, NodeElement):
            return self.node_coords == other.node_coords
        return False


class LineSegment:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.length = calculate_distance(p1, p2)
        self.bbox = [
            min(p1[0], p2[0]),
            min(p1[1], p2[1]),
            max(p1[0], p2[0]),
            max(p1[1], p2[1]),
        ]

    def is_intersect(self, line):
        """判断两条线段是否相交"""
        # 快速排斥实验
        if (
            self.bbox[0] > line.bbox[2]
            or self.bbox[2] < line.bbox[0]
            or self.bbox[1] > line.bbox[3]
            or self.bbox[3] < line.bbox[1]
        ):
            return False

        # 跨立实验
        if is_left(self.p1, self.p2, line.p1) != is_left(
            self.p1, self.p2, line.p2
        ) and is_left(line.p1, line.p2, self.p1) != is_left(line.p1, line.p2, self.p2):
            # 新增端点重合检查
            if (
                self.p1 == line.p1
                or self.p1 == line.p2
                or self.p2 == line.p1
                or self.p2 == line.p2
            ):
                return False
            return True

        return False


def points_equal(p1, p2, epsilon=1e-6):
    return abs(p1[0] - p2[0]) < epsilon and abs(p1[1] - p2[1]) < epsilon


def segments_intersect(a1, a2, b1, b2):
    # 检查两个线段是否在非端点处相交（包括部分重叠但非共线的情况）
    if any(points_equal(a, b) for a in (a1, a2) for b in (b1, b2)):
        return False

    a_min_x = min(a1[0], a2[0])
    a_max_x = max(a1[0], a2[0])
    a_min_y = min(a1[1], a2[1])
    a_max_y = max(a1[1], a2[1])

    b_min_x = min(b1[0], b2[0])
    b_max_x = max(b1[0], b2[0])
    b_min_y = min(b1[1], b2[1])
    b_max_y = max(b1[1], b2[1])

    if (
        (a_max_x < b_min_x)
        or (b_max_x < a_min_x)
        or (a_max_y < b_min_y)
        or (b_max_y < a_min_y)
    ):
        return False

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    c1 = cross(a1, a2, b1)
    c2 = cross(a1, a2, b2)
    c3 = cross(b1, b2, a1)
    c4 = cross(b1, b2, a2)

    if (c1 * c2 < 0) and (c3 * c4 < 0):
        return True

    # 处理共线但部分重叠的情况（非完全重合）
    if c1 == 0 and c2 == 0 and c3 == 0 and c4 == 0:
        x_overlap = max(a_min_x, b_min_x) <= min(a_max_x, b_max_x)
        y_overlap = max(a_min_y, b_min_y) <= min(a_max_y, b_max_y)
        return x_overlap and y_overlap

    return False


def is_point_inside(p, a, b, c):
    # 判断点p是否严格在三角形内部（不包括边）
    def cross_sign(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = cross_sign(a, b, p)
    d2 = cross_sign(b, c, p)
    d3 = cross_sign(c, a, p)

    has_neg = (d1 < -1e-8) or (d2 < -1e-8) or (d3 < -1e-8)
    has_pos = (d1 > 1e-8) or (d2 > 1e-8) or (d3 > 1e-8)

    return not (has_neg or has_pos) and not (d1 == d2 == d3 == 0)


class Triangle:
    def __init__(self, p1, p2, p3, node_idx=None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.node_idx = node_idx
        self.area = None
        self.quality = None
        self.bbox = [
            min(p1[0], p2[0], p3[0]),
            min(p1[1], p2[1], p3[1]),
            max(p1[0], p2[0], p3[0]),
            max(p1[1], p2[1], p3[1]),
        ]

    def __hash__(self):
        return hash((self.p1, self.p2, self.p3))

    def __eq__(self, other):
        if isinstance(other, Triangle):
            return self.p1 == other.p1 and self.p2 == other.p2 and self.p3 == other.p3
        return False

    def get_edges(self):
        return [(self.p1, self.p2), (self.p2, self.p3), (self.p3, self.p1)]

    def init_metrics(self):
        self.area = triangle_area(self.p1, self.p2, self.p3)
        self.quality = triangle_quality(self.p1, self.p2, self.p3)

    def is_intersect(self, triangle):
        self_edges = [(self.p1, self.p2), (self.p2, self.p3), (self.p3, self.p1)]
        tri_edges = [
            (triangle.p1, triangle.p2),
            (triangle.p2, triangle.p3),
            (triangle.p3, triangle.p1),
        ]

        # 检查边是否相交（排除共边）
        for e1 in self_edges:
            a1, a2 = e1
            for e2 in tri_edges:
                b1, b2 = e2
                if segments_intersect(a1, a2, b1, b2):
                    # 检查是否为共边（顶点完全相同）
                    if (points_equal(a1, b1) and points_equal(a2, b2)) or (
                        points_equal(a1, b2) and points_equal(a2, b1)
                    ):
                        continue  # 共边，视为不相交
                    else:
                        return True  # 非共边的相交

        # 检查顶点是否在另一个三角形内部（严格内部）
        for p in [self.p1, self.p2, self.p3]:
            if any(
                points_equal(p, tri_p)
                for tri_p in [triangle.p1, triangle.p2, triangle.p3]
            ):
                continue
            if is_point_inside(p, triangle.p1, triangle.p2, triangle.p3):
                return True

        for p in [triangle.p1, triangle.p2, triangle.p3]:
            if any(points_equal(p, self_p) for self_p in [self.p1, self.p2, self.p3]):
                continue
            if is_point_inside(p, self.p1, self.p2, self.p3):
                return True

        return False

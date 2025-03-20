from math import sqrt, isnan, isinf


def normal_vector2d(front):
    """计算二维平面阵面的法向量"""
    node1, node2 = front.nodes_coords
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]
    return dy, -dx


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

    # 方法1：内外接圆半径比（默认使用该方法）
    denominator = a + b + c
    if denominator == 0:
        return 0.0
    r = 2.0 * area / denominator  # 内切圆半径
    R = (a * b * c) / (4.0 * area) if area != 0 else 0  # 外接圆半径
    quality_method1 = 3.0 * r / R if R != 0 else 0

    # 方法2：面积与边长平方比（注释保留）
    # denominator = a**2 + b**2 + c**2
    # quality_method2 = 4.0 * sqrt(3.0) * area / denominator if denominator !=0 else 0

    # 新增异常检测
    if (
        isnan(quality_method1)
        or isinf(quality_method1)
        or quality_method1 < 0
        or quality_method1 > 1
    ):
        raise ValueError(
            f"三角形质量计算异常：quality={quality_method1}，节点坐标：{p1}, {p2}, {p3}"
        )

    return quality_method1


def is_left(p1, p2, p3):
    """判断点p3是否在p1-p2向量的左侧"""
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p1[0], p3[1] - p1[1])
    # 向量叉积
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_product > 0


class NodeElement:
    def __init__(self, node):
        self.node = node


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
            return True

        return False

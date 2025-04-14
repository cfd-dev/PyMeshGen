import numpy as np
from math import sqrt, isnan, isinf
from geom_toolkit import (
    calculate_distance,
    triangle_area,
    calculate_angle,
    is_valid_quadrilateral,
    is_convex,
)


def triangle_shape_quality(p1, p2, p3):
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
    # quality = 3.0 * r / R if R != 0 else 0
    # lower_limit = 0.0
    # upper_limit = 1.5

    # 方法2：面积与边长平方比（默认使用该方法）
    denominator = a**2 + b**2 + c**2
    quality = 4.0 * sqrt(3.0) * area / denominator if denominator != 0 else 0
    upper_limit = 1.0
    lower_limit = 0.0
    # 新增异常检测
    if (
        isnan(quality)
        or isinf(quality)
        or quality < lower_limit
        or quality > upper_limit
    ):
        raise ValueError(
            f"三角形质量计算异常：quality={quality}，节点坐标：{p1}, {p2}, {p3}"
        )

    return quality


def triangle_skewness(p1, p2, p3):
    """计算三角形的偏斜度"""
    # 计算三角形的三个内角
    angles = [
        calculate_angle(p3, p2, p1),
        calculate_angle(p1, p3, p2),
        calculate_angle(p2, p1, p3),
    ]

    # 检查内角和是否为180度
    if abs(np.sum(angles) - 180) > 1e-3:
        skewness = 0.0
        return skewness

    # 计算最大和最小角度
    max_angle = max(angles)
    min_angle = min(angles)

    skew1 = (max_angle - 60) / 120
    skew2 = (60 - min_angle) / 60
    skewness = 1.0 - max([skew1, skew2])

    return skewness


def quadrilateral_skewness(p1, p2, p3, p4):
    """计算四边形的偏斜度"""
    # 四边形的四个内角
    angles = [
        calculate_angle(p3, p2, p1),
        calculate_angle(p4, p3, p2),
        calculate_angle(p1, p4, p3),
        calculate_angle(p2, p1, p4),
    ]

    # 检查内角和是否为360度
    if abs(np.sum(angles) - 360) > 1e-3:
        skewness = 0.0
        return skewness

    # 计算最大和最小角度
    max_angle = max(angles)
    min_angle = min(angles)

    skew1 = (max_angle - 90) / 90
    skew2 = (90 - min_angle) / 90
    skewness = 1.0 - max([skew1, skew2])

    return skewness


def quadrilateral_aspect_ratio(p1, p2, p3, p4):
    """计算四边形的长宽比"""
    # 计算所有边长
    edges = [
        calculate_distance(p1, p2),
        calculate_distance(p2, p3),
        calculate_distance(p3, p4),
        calculate_distance(p4, p1),
    ]

    max_edge = max(edges)
    min_edge = min(edges)
    return max_edge / min_edge if min_edge > 1e-12 else 0.0


def quadrilateral_quality2(p1, p2, p3, p4):
    quality = quadrilateral_shape_quality(p1, p2, p3, p4)
    skewness = quadrilateral_skewness(p1, p2, p3, p4)
    aspect_ratio = quadrilateral_aspect_ratio(p1, p2, p3, p4)

    # 只要有一个质量为0，则返回0
    if quality * skewness * aspect_ratio < 1e-10:
        return 0.0

    as_quality = 0.0 if aspect_ratio == 0 else (1.0 / aspect_ratio)
    # return (quality + skewness + as_quality) / 3.0
    return (quality + 4.0 * skewness + as_quality) / 6.0


def quadrilateral_shape_quality(p1, p2, p3, p4):
    """计算四边形质量（基于子三角形质量的最小组合）"""
    from basic_elements import LineSegment

    # 检查四边形有效性
    if not is_valid_quadrilateral(p1, p2, p3, p4):
        return 0.0

    # 凸性检查（使用现有is_convex函数适配）
    if not is_convex(0, 1, 2, 3, [p1, p2, p3, p4]):
        return 0.0

    # 计算对角线交点
    seg1 = LineSegment(p1, p3)
    seg2 = LineSegment(p2, p4)
    if not seg1.is_intersect(seg2):
        return 0.0

    # 获取交点坐标（新增交点计算逻辑）
    def line_intersection(line1, line2):
        # 实现线段交点计算
        x1, y1 = line1.p1
        x2, y2 = line1.p2
        x3, y3 = line2.p1
        x4, y4 = line2.p2

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return None
        u = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        x = x1 + u * (x2 - x1)
        y = y1 + u * (y2 - y1)
        return (x, y)

    po = line_intersection(seg1, seg2)

    try:
        # 计算四个子三角形质量
        q1 = triangle_shape_quality(p1, p2, po)
        q2 = triangle_shape_quality(p2, p3, po)
        q3 = triangle_shape_quality(p3, p4, po)
        q4 = triangle_shape_quality(p4, p1, po)
    except ValueError:
        return 0.0

    # 组合质量指标
    tmp = sorted([q1, q2, q3, q4])
    if tmp[-1] == 0:
        return 0.0
    quality = (tmp[0] * tmp[1]) / (tmp[2] * tmp[3])

    # 异常检测（保持与triangle_quality一致）
    if isnan(quality) or isinf(quality):
        raise ValueError(f"四边形质量异常：{quality}，顶点：{p1}, {p2}, {p3}, {p4}")

    return quality

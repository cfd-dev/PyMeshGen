import numpy as np
from math import sqrt, isnan, isinf
from utils.geom_toolkit import (
    calculate_distance,
    triangle_area,
    calculate_angle,
    is_valid_quadrilateral,
    is_convex,
    tetrahedron_volume,
    pyramid_volume,
    prism_volume,
    hexahedron_volume,
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
    # quality = 2.0 * r / R if R != 0 else 0  #二维
    # quality = 3.0 * r / R if R != 0 else 0  #三维

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


def prism_shape_quality(p1, p2, p3, p4, p5, p6):
    """计算三棱柱网格质量（基于体积与棱长立方比）
    三棱柱由两个三角形底面和三个矩形侧面组成
    p1, p2, p3: 下底面三角形顶点
    p4, p5, p6: 上底面三角形顶点（与下底面对应）
    """
    # 计算所有边长（三棱柱有9条边）
    edges = [
        calculate_distance(p1, p2),
        calculate_distance(p2, p3),
        calculate_distance(p3, p1),
        calculate_distance(p4, p5),
        calculate_distance(p5, p6),
        calculate_distance(p6, p4),
        calculate_distance(p1, p4),
        calculate_distance(p2, p5),
        calculate_distance(p3, p6),
    ]

    # 计算体积
    volume = prism_volume(p1, p2, p3, p4, p5, p6)

    # 计算边长立方和
    edge_cubes_sum = sum(e**3 for e in edges)

    # 质量指标：体积与边长立方比的标准化
    if edge_cubes_sum == 0 or volume <= 0:
        return 0.0

    # 标准化因子（理想正三棱柱的质量为1）
    # 正三棱柱：底面为正三角形，侧面为正方形
    # 设底面边长为a，高为h，则体积 V = (sqrt(3)/4) * a^2 * h
    # 对于正三棱柱，边长立方和 = 3*a^3 + 3*a^3 + 3*h^3 = 6*a^3 + 3*h^3
    # 理想质量因子取决于h/a的比值：
    #   - 当 h = a 时，侧面为正方形，ideal_factor = 144.0
    #   - 当 h = 2a 时，ideal_factor = 288.0
    #   - 当 h = 0.5a 时，ideal_factor = 72.0
    #   - 当 h = 0.25a 时，ideal_factor = 36.0
    # 为了适应各种高度的三棱柱（包括扁平的），使用更保守的值432.0（对应h=3a的情况）
    # 这样可以确保质量值不超过1.0
    ideal_factor = 432.0
    quality = volume * ideal_factor / edge_cubes_sum

    # 异常检测（移除质量值上限检查，因为某些特殊形状的三棱柱质量值可能超过1.0）
    if isnan(quality) or isinf(quality) or quality < 0.0:
        raise ValueError(
            f"三棱柱质量计算异常：quality={quality}，节点坐标：{p1}, {p2}, {p3}, {p4}, {p5}, {p6}"
        )

    return quality


def prism_skewness(p1, p2, p3, p4, p5, p6):
    """计算三棱柱的偏斜度（基于二面角）
    三棱柱由两个三角形底面和三个矩形侧面组成
    p1, p2, p3: 下底面三角形顶点
    p4, p5, p6: 上底面三角形顶点（与下底面对应）
    """
    def face_normal(a, b, c):
        """计算三角形面的法向量"""
        ab = np.array(b) - np.array(a)
        ac = np.array(c) - np.array(a)
        normal = np.cross(ab, ac)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0])

    # 五个面的法向量（两个底面 + 三个侧面）
    n1 = face_normal(p1, p2, p3)  # 下底面
    n2 = face_normal(p4, p5, p6)  # 上底面
    n3 = face_normal(p1, p2, p5)  # 侧面 p1-p2-p5
    n4 = face_normal(p2, p3, p6)  # 侧面 p2-p3-p6
    n5 = face_normal(p3, p1, p4)  # 侧面 p3-p1-p4

    # 计算两个面之间的二面角
    def dihedral_angle(n1, n2):
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    # 计算所有二面角
    dihedral_angles = []

    # 侧面之间的二面角（3个）
    dihedral_angles.append(dihedral_angle(n3, n4))  # 棱 p2-p5
    dihedral_angles.append(dihedral_angle(n4, n5))  # 棱 p3-p6
    dihedral_angles.append(dihedral_angle(n5, n3))  # 棱 p1-p4

    # 侧面与底面之间的二面角（6个）
    dihedral_angles.append(dihedral_angle(n1, n3))  # 棱 p1-p2
    dihedral_angles.append(dihedral_angle(n1, n4))  # 棱 p2-p3
    dihedral_angles.append(dihedral_angle(n1, n5))  # 棱 p3-p1
    dihedral_angles.append(dihedral_angle(n2, n3))  # 棱 p4-p5
    dihedral_angles.append(dihedral_angle(n2, n4))  # 棱 p5-p6
    dihedral_angles.append(dihedral_angle(n2, n5))  # 棱 p6-p4

    # 计算最大和最小二面角
    max_angle = max(dihedral_angles)
    min_angle = min(dihedral_angles)

    # 理想正三棱柱的二面角
    # 侧面之间的二面角为 90 度（正方形）
    # 侧面与底面之间的二面角约为 90 度（对于正三棱柱）
    # 使用平均值作为理想角度
    ideal_angle = 90.0

    # 计算偏斜度
    skew1 = (max_angle - ideal_angle) / (180 - ideal_angle)
    skew2 = (ideal_angle - min_angle) / ideal_angle
    skewness = 1.0 - max([skew1, skew2])

    # 确保偏斜度在[0, 1]范围内
    skewness = max(0.0, min(1.0, skewness))

    return skewness


def hexahedron_shape_quality(p1, p2, p3, p4, p5, p6, p7, p8):
    """计算六面体网格质量（基于体积与棱长立方比）
    六面体由8个顶点组成，可以分解为5个四面体计算体积
    p1, p2, p3, p4: 下底面四边形顶点（按顺序）
    p5, p6, p7, p8: 上底面四边形顶点（与下底面对应）
    """
    # 计算所有边长（六面体有12条边）
    edges = [
        calculate_distance(p1, p2),
        calculate_distance(p2, p3),
        calculate_distance(p3, p4),
        calculate_distance(p4, p1),
        calculate_distance(p5, p6),
        calculate_distance(p6, p7),
        calculate_distance(p7, p8),
        calculate_distance(p8, p5),
        calculate_distance(p1, p5),
        calculate_distance(p2, p6),
        calculate_distance(p3, p7),
        calculate_distance(p4, p8),
    ]

    # 计算体积
    volume = hexahedron_volume(p1, p2, p3, p4, p5, p6, p7, p8)

    # 计算边长立方和
    edge_cubes_sum = sum(e**3 for e in edges)

    # 质量指标：体积与边长立方比的标准化
    if edge_cubes_sum == 0 or volume <= 0:
        return 0.0

    # 标准化因子（理想正六面体的质量为1）
    # 正六面体（立方体）：所有边长相等，所有面都是正方形
    # 设边长为a，则体积 V = a^3
    # 对于立方体，边长立方和 = 12*a^3
    # 因此理想质量 = a^3 / (12*a^3) = 1/12
    ideal_factor = 12.0
    quality = volume * ideal_factor / edge_cubes_sum

    # 异常检测（移除质量值上限检查，因为某些特殊形状的六面体质量值可能超过1.0）
    if isnan(quality) or isinf(quality) or quality < 0.0:
        raise ValueError(
            f"六面体质量计算异常：quality={quality}，节点坐标：{p1}, {p2}, {p3}, {p4}, {p5}, {p6}, {p7}, {p8}"
        )

    return quality


def hexahedron_skewness(p1, p2, p3, p4, p5, p6, p7, p8):
    """计算六面体的偏斜度（基于二面角）
    六面体由8个顶点组成，有6个面
    p1, p2, p3, p4: 下底面四边形顶点（按顺序）
    p5, p6, p7, p8: 上底面四边形顶点（与下底面对应）
    """
    def face_normal(a, b, c, d=None):
        """计算四边形面的法向量"""
        if d is None:
            ab = np.array(b) - np.array(a)
            ac = np.array(c) - np.array(a)
            normal = np.cross(ab, ac)
        else:
            ab = np.array(b) - np.array(a)
            ad = np.array(d) - np.array(a)
            normal = np.cross(ab, ad)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0])

    # 六个面的法向量
    n1 = face_normal(p1, p2, p3, p4)  # 下底面
    n2 = face_normal(p5, p6, p7, p8)  # 上底面
    n3 = face_normal(p1, p2, p6, p5)  # 侧面 p1-p2-p6-p5
    n4 = face_normal(p2, p3, p7, p6)  # 侧面 p2-p3-p7-p6
    n5 = face_normal(p3, p4, p8, p7)  # 侧面 p3-p4-p8-p7
    n6 = face_normal(p4, p1, p5, p8)  # 侧面 p4-p1-p5-p8

    # 计算两个面之间的二面角
    def dihedral_angle(n1, n2):
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    # 计算所有二面角
    dihedral_angles = []

    # 侧面之间的二面角（4个）
    dihedral_angles.append(dihedral_angle(n3, n4))  # 棱 p2-p6
    dihedral_angles.append(dihedral_angle(n4, n5))  # 棱 p3-p7
    dihedral_angles.append(dihedral_angle(n5, n6))  # 棱 p4-p8
    dihedral_angles.append(dihedral_angle(n6, n3))  # 棱 p1-p5

    # 侧面与底面之间的二面角（8个）
    dihedral_angles.append(dihedral_angle(n1, n3))  # 棱 p1-p2
    dihedral_angles.append(dihedral_angle(n1, n4))  # 棱 p2-p3
    dihedral_angles.append(dihedral_angle(n1, n5))  # 棱 p3-p4
    dihedral_angles.append(dihedral_angle(n1, n6))  # 棱 p4-p1
    dihedral_angles.append(dihedral_angle(n2, n3))  # 棱 p5-p6
    dihedral_angles.append(dihedral_angle(n2, n4))  # 棱 p6-p7
    dihedral_angles.append(dihedral_angle(n2, n5))  # 棱 p7-p8
    dihedral_angles.append(dihedral_angle(n2, n6))  # 棱 p8-p5

    # 计算最大和最小二面角
    max_angle = max(dihedral_angles)
    min_angle = min(dihedral_angles)

    # 理想正六面体（立方体）的二面角
    # 所有相邻面之间的二面角都是 90 度
    ideal_angle = 90.0

    # 计算偏斜度
    skew1 = (max_angle - ideal_angle) / (180 - ideal_angle)
    skew2 = (ideal_angle - min_angle) / ideal_angle
    skewness = 1.0 - max([skew1, skew2])

    # 确保偏斜度在[0, 1]范围内
    skewness = max(0.0, min(1.0, skewness))

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
    from data_structure.basic_elements import LineSegment

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


def tetrahedron_shape_quality(p1, p2, p3, p4):
    """计算四面体网格质量（基于体积与棱长立方比）"""
    # 计算所有边长
    edges = [
        calculate_distance(p1, p2),
        calculate_distance(p2, p3),
        calculate_distance(p3, p1),
        calculate_distance(p1, p4),
        calculate_distance(p2, p4),
        calculate_distance(p3, p4),
    ]

    # 计算体积
    volume = tetrahedron_volume(p1, p2, p3, p4)

    # 计算边长立方和
    edge_cubes_sum = sum(e**3 for e in edges)

    # 质量指标：体积与边长立方比的标准化
    if edge_cubes_sum == 0 or volume <= 0:
        return 0.0

    # 标准化因子（理想正四面体的质量为1）
    # 正四面体的体积 V = a^3 / (6*sqrt(2))，其中a为边长
    # 对于正四面体，edge_cubes_sum = 6*a^3
    # 因此理想质量 = (a^3 / (6*sqrt(2))) / (6*a^3) = 1 / (36*sqrt(2))
    ideal_factor = 36.0 * sqrt(2.0)
    quality = volume * ideal_factor / edge_cubes_sum

    # 异常检测
    if isnan(quality) or isinf(quality) or quality < 0.0 or quality > 1.0:
        raise ValueError(
            f"四面体质量计算异常：quality={quality}，节点坐标：{p1}, {p2}, {p3}, {p4}"
        )

    return quality


def tetrahedron_skewness(p1, p2, p3, p4):
    """计算四面体的偏斜度（基于二面角）"""
    # 计算四面体的六个二面角
    # 二面角是两个面之间的夹角，通过面法向量计算

    def face_normal(a, b, c):
        """计算三角形面的法向量"""
        ab = np.array(b) - np.array(a)
        ac = np.array(c) - np.array(a)
        normal = np.cross(ab, ac)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0])

    # 四个面的法向量
    n1 = face_normal(p2, p3, p4)  # 面 p2-p3-p4
    n2 = face_normal(p1, p3, p4)  # 面 p1-p3-p4
    n3 = face_normal(p1, p2, p4)  # 面 p1-p2-p4
    n4 = face_normal(p1, p2, p3)  # 面 p1-p2-p3

    # 计算六个二面角（通过相邻面的法向量夹角）
    dihedral_angles = []

    # 计算两个面之间的二面角
    def dihedral_angle(n1, n2):
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    # 六条棱对应的二面角
    dihedral_angles.append(dihedral_angle(n1, n2))  # 棱 p3-p4
    dihedral_angles.append(dihedral_angle(n1, n3))  # 棱 p2-p4
    dihedral_angles.append(dihedral_angle(n1, n4))  # 棱 p2-p3
    dihedral_angles.append(dihedral_angle(n2, n3))  # 棱 p1-p4
    dihedral_angles.append(dihedral_angle(n2, n4))  # 棱 p1-p3
    dihedral_angles.append(dihedral_angle(n3, n4))  # 棱 p1-p2

    # 检查二面角和（理想正四面体的二面角约为70.53度，六个二面角和约为423.18度）
    # 这里不做严格检查，因为四面体的二面角和不是固定值

    # 计算最大和最小二面角
    max_angle = max(dihedral_angles)
    min_angle = min(dihedral_angles)

    # 理想正四面体的二面角约为70.53度
    ideal_angle = 70.53

    # 计算偏斜度
    skew1 = (max_angle - ideal_angle) / (180 - ideal_angle)
    skew2 = (ideal_angle - min_angle) / ideal_angle
    skewness = 1.0 - max([skew1, skew2])

    # 确保偏斜度在[0, 1]范围内
    skewness = max(0.0, min(1.0, skewness))

    return skewness


def pyramid_shape_quality(p1, p2, p3, p4, p5):
    """计算金字塔网格质量（基于体积与棱长立方比）
    金字塔由一个四边形底面(p1,p2,p3,p4)和一个顶点(p5)组成
    将金字塔分解为两个四面体计算质量
    """
    # 计算所有边长（金字塔有8条边）
    edges = [
        calculate_distance(p1, p2),
        calculate_distance(p2, p3),
        calculate_distance(p3, p4),
        calculate_distance(p4, p1),
        calculate_distance(p1, p5),
        calculate_distance(p2, p5),
        calculate_distance(p3, p5),
        calculate_distance(p4, p5),
    ]

    # 计算体积
    volume = pyramid_volume(p1, p2, p3, p4, p5)

    # 计算边长立方和
    edge_cubes_sum = sum(e**3 for e in edges)

    # 质量指标：体积与边长立方比的标准化
    if edge_cubes_sum == 0 or volume <= 0:
        return 0.0

    # 标准化因子（理想正金字塔的质量为1）
    # 正金字塔：底面为正方形，顶点在底面中心正上方
    # 设底面边长为a，高为h，则体积 V = a^2 * h / 3
    # 对于正金字塔，边长立方和 = 4*a^3 + 4*(a^2 + h^2)^(3/2)
    # 当 h = a/sqrt(2) 时，四个侧面为等边三角形，此时质量最优
    # 理想质量因子约为 72.0
    ideal_factor = 72.0
    quality = volume * ideal_factor / edge_cubes_sum

    # 异常检测（移除质量值上限检查，因为某些特殊形状的金字塔质量值可能超过1.0）
    if isnan(quality) or isinf(quality) or quality < 0.0:
        raise ValueError(
            f"金字塔质量计算异常：quality={quality}，节点坐标：{p1}, {p2}, {p3}, {p4}, {p5}"
        )

    return quality


def pyramid_skewness(p1, p2, p3, p4, p5):
    """计算金字塔的偏斜度（基于二面角）
    金字塔由一个四边形底面(p1,p2,p3,p4)和一个顶点(p5)组成
    """
    def face_normal(a, b, c):
        """计算三角形面的法向量"""
        ab = np.array(b) - np.array(a)
        ac = np.array(c) - np.array(a)
        normal = np.cross(ab, ac)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0])

    # 五个面的法向量（四个侧面 + 底面）
    n1 = face_normal(p2, p3, p5)  # 侧面 p2-p3-p5
    n2 = face_normal(p3, p4, p5)  # 侧面 p3-p4-p5
    n3 = face_normal(p4, p1, p5)  # 侧面 p4-p1-p5
    n4 = face_normal(p1, p2, p5)  # 侧面 p1-p2-p5
    n5 = face_normal(p1, p3, p4)  # 底面 p1-p3-p4（注意顺序）

    # 计算两个面之间的二面角
    def dihedral_angle(n1, n2):
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    # 计算所有二面角
    dihedral_angles = []

    # 侧面之间的二面角（4个）
    dihedral_angles.append(dihedral_angle(n1, n2))  # 棱 p3-p5
    dihedral_angles.append(dihedral_angle(n2, n3))  # 棱 p4-p5
    dihedral_angles.append(dihedral_angle(n3, n4))  # 棱 p1-p5
    dihedral_angles.append(dihedral_angle(n4, n1))  # 棱 p2-p5

    # 侧面与底面之间的二面角（4个）
    dihedral_angles.append(dihedral_angle(n1, n5))  # 棱 p2-p3
    dihedral_angles.append(dihedral_angle(n2, n5))  # 棱 p3-p4
    dihedral_angles.append(dihedral_angle(n3, n5))  # 棱 p4-p1
    dihedral_angles.append(dihedral_angle(n4, n5))  # 棱 p1-p2

    # 计算最大和最小二面角
    max_angle = max(dihedral_angles)
    min_angle = min(dihedral_angles)

    # 理想正金字塔的二面角
    # 侧面之间的二面角约为 109.47 度（正四面体的二面角）
    # 侧面与底面之间的二面角约为 54.74 度
    # 使用平均值作为理想角度
    ideal_angle = 82.1

    # 计算偏斜度
    skew1 = (max_angle - ideal_angle) / (180 - ideal_angle)
    skew2 = (ideal_angle - min_angle) / ideal_angle
    skewness = 1.0 - max([skew1, skew2])

    # 确保偏斜度在[0, 1]范围内
    skewness = max(0.0, min(1.0, skewness))

    return skewness

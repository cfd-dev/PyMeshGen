"""
Bowyer-Watson 网格生成器 - 几何谓词

实现鲁棒的几何谓词：
- orient2d: 2D 方向测试
- incircle: 2D 圆内测试
- circumcenter: 外接圆圆心计算
"""

import numpy as np
from decimal import Decimal, getcontext

getcontext().prec = 50  # 高精度用于外接圆计算


def orient2d(a, b, p) -> float:
    """2D 方向测试。

    判断点 p 相对于有向直线 (a→b) 的位置。
    
    Args:
        a, b: 直线端点 [x, y]
        p: 测试点 [x, y]
    
    Returns:
        > 0: p 在直线左侧
        = 0: p 在直线上  
        < 0: p 在直线右侧
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def incircle(a, b, c, p) -> float:
    """Incircle 测试（Shewchuk）。

    判断点 p 是否在三角形 (a,b,c) 的外接圆内。
    
    Returns:
        > 0: p 在圆内
        = 0: p 在圆上
        < 0: p 在圆外
    """
    # 相对于 p 的坐标
    adx, ady = a[0] - p[0], a[1] - p[1]
    bdx, bdy = b[0] - p[0], b[1] - p[1]
    cdx, cdy = c[0] - p[0], c[1] - p[1]

    # 提升坐标
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy

    # 3x3 行列式
    det = (adx * (bdy * clift - cdy * blift) -
           ady * (bdx * clift - cdx * blift) +
           alift * (bdx * cdy - cdx * bdy))

    # 乘以 orient2d 符号
    orient = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    
    return det * orient


def circumcenter_precise(a, b, c):
    """高精度计算外接圆圆心。

    使用 Decimal 避免浮点误差。
    
    Returns:
        (cx, cy): 外接圆心坐标
    """
    ax, ay = Decimal(a[0]), Decimal(a[1])
    bx, by = Decimal(b[0]), Decimal(b[1])
    cx, cy = Decimal(c[0]), Decimal(c[1])

    d = Decimal('2') * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    if abs(d) < Decimal('1e-30'):
        # 退化三角形，返回重心
        return (a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0

    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy

    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d

    return float(ux), float(uy)


def point_in_triangle(p, tri_vertices, points) -> bool:
    """检查点是否在三角形内部。

    使用重心坐标法。
    
    Args:
        p: 测试点坐标 [x, y]
        tri_vertices: 三角形顶点索引 (v0, v1, v2)
        points: 所有点的坐标数组
    
    Returns:
        True 如果点在三角形内（包括边界）
    """
    v0, v1, v2 = tri_vertices
    a, b, c = points[v0], points[v1], points[v2]
    
    # 计算重心坐标
    denom = ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
    if abs(denom) < 1e-12:
        return False
    
    u = (((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom)
    v = (((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom)
    w = 1.0 - u - v
    
    return u >= -1e-10 and v >= -1e-10 and w >= -1e-10


def segments_intersect(p1, p2, p3, p4, strict=True) -> bool:
    """检查两条线段是否相交。

    Args:
        p1, p2: 第一条线段端点
        p3, p4: 第二条线段端点
        strict: True=严格相交（不包括端点接触）
    
    Returns:
        True 如果相交
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    eps = 1e-12

    if strict:
        return ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
               ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps))
    else:
        return ((d1 >= -eps and d2 <= eps) or (d1 <= eps and d2 >= -eps)) and \
               ((d3 >= -eps and d4 <= eps) or (d3 <= eps and d4 >= -eps))

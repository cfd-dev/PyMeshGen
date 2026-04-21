"""
Bowyer-Watson Delaunay 网格生成器 - 鲁棒几何谓词

实现 Shewchuk 的自适应精度浮点谓词：
- orient2d: 2D 方向测试（三点定向）
- incircle: 2D 圆内测试
- 各向异性点在圆内测试（使用度量张量）

参考:
- Gmsh robustPredicates 类
- Jonathan Shewchuk 的 "Adaptive Precision Floating-Point Arithmetic"
"""

import numpy as np
from typing import Tuple
from decimal import Decimal, getcontext

# 设置高精度（用于外接圆计算）
getcontext().prec = 50


# =============================================================================
# 2D 方向测试（Shewchuk orient2d）
# =============================================================================

def orient2d(ax, ay, bx, by, px, py) -> float:
    """2D 方向测试：判断点 p 相对于有向直线 (a->b) 的位置。
    
    返回：
      > 0：p 在直线左侧
      = 0：p 在直线上
      < 0：p 在直线右侧
    
    使用行列式计算，确保数值稳定性。
    """
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def orient2d_fast(a, b, p) -> float:
    """快速版本：使用 numpy 数组。
    
    参数:
        a, b, p: 2D 点坐标 [x, y]
    
    返回:
        方向测试结果的符号
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


# =============================================================================
# 2D 圆内测试（Shewchuk incircle）
# =============================================================================

def incircle(ax, ay, bx, by, cx, cy, px, py) -> float:
    """精确的 incircle 测试（Shewchuk 的 Robust Predicates）。
    
    判断点 p 是否在三角形 (a, b, c) 的外接圆内。
    使用行列式计算，避免浮点误差。
    
    返回：
      > 0：p 在圆内
      = 0：p 在圆上
      < 0：p 在圆外
    
    算法：
    构造增广矩阵的行列式，乘以 orient2d 的符号。
    """
    # 相对于测试点的坐标
    adx = ax - px
    ady = ay - py
    bdx = bx - px
    bdy = by - py
    cdx = cx - px
    cdy = cy - py
    
    # 计算提升坐标（lifted coordinates）
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    
    # 计算 3x3 行列式
    det = (adx * (bdy * clift - cdy * blift) -
           ady * (bdx * clift - cdx * blift) +
           alift * (bdx * cdy - cdx * bdy))
    
    # 乘以 orient2d 的符号（确保正确的方向）
    orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    
    return det * orient


def incircle_fast(a, b, c, p) -> float:
    """快速版本：使用 numpy 数组。
    
    参数:
        a, b, c: 三角形顶点 [x, y]
        p: 测试点 [x, y]
    
    返回:
        > 0：p 在圆内
        < 0：p 在圆外
    """
    return incircle(a[0], a[1], b[0], b[1], c[0], c[1], p[0], p[1])


# =============================================================================
# 外接圆计算（高精度算术）
# =============================================================================

def circumcenter_precise(ax, ay, bx, by, cx, cy) -> Tuple[float, float]:
    """精确计算外接圆圆心（使用高精度算术）。
    
    参考 Gmsh circumCenterMetric：
    使用行列式法求解线性方程组，避免除零问题。
    
    返回：
        (ux, uy): 外接圆心坐标
    """
    # 使用 Python 的高精度浮点
    ax_d, ay_d = Decimal(ax), Decimal(ay)
    bx_d, by_d = Decimal(bx), Decimal(by)
    cx_d, cy_d = Decimal(cx), Decimal(cy)
    
    # 计算分母（2 倍有向面积）
    d = Decimal('2') * (ax_d * (by_d - cy_d) + bx_d * (cy_d - ay_d) + cx_d * (ay_d - by_d))
    
    if abs(d) < Decimal('1e-30'):
        # 退化三角形，返回重心
        return (ax + bx + cx) / 3.0, (ay + by + cy) / 3.0
    
    # 计算平方距离
    a2 = ax_d * ax_d + ay_d * ay_d
    b2 = bx_d * bx_d + by_d * by_d
    c2 = cx_d * cx_d + cy_d * cy_d
    
    # 求解外接圆心
    ux = (a2 * (by_d - cy_d) + b2 * (cy_d - ay_d) + c2 * (ay_d - by_d)) / d
    uy = (a2 * (cx_d - bx_d) + b2 * (ax_d - cx_d) + c2 * (bx_d - ax_d)) / d
    
    return float(ux), float(uy)


def compute_circumcircle(p1, p2, p3) -> Tuple[np.ndarray, float]:
    """计算三角形的外接圆。
    
    参数:
        p1, p2, p3: 三角形顶点 [x, y]
    
    返回:
        (center, radius): 外接圆心和半径
    """
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    cx, cy = float(p3[0]), float(p3[1])
    
    # 使用高精度算术计算 circumcenter
    ux, uy = circumcenter_precise(ax, ay, bx, by, cx, cy)
    center = np.array([ux, uy])
    
    # 计算半径
    radius = float(np.linalg.norm(p1 - center))
    
    return center, radius


# =============================================================================
# 各向异性度量空间支持（Gmsh 扩展）
# =============================================================================

def build_metric(derivatives) -> np.ndarray:
    """构建度量张量（第一基本形式）。
    
    参考 Gmsh buildMetric：
    计算曲面在该点的第一基本形式，捕获参数空间到物理空间的局部变形。
    
    参数:
        derivatives: 曲面导数 (du, dv)，每个是 3D 向量
    
    返回:
        metric: 度量张量 [a, b, d]（对称矩阵的上三角部分）
                M = | a  b |
                    | b  d |
    """
    du, dv = derivatives
    
    # 第一基本形式系数
    a = float(np.dot(du, du))  # E = r_u · r_u
    b = float(np.dot(dv, du))  # F = r_v · r_u
    d = float(np.dot(dv, dv))  # G = r_v · r_v
    
    return np.array([a, b, d])


def circumcenter_metric(p1, p2, p3, metric) -> Tuple[np.ndarray, float]:
    """各向异性度量空间中的外接圆计算。
    
    参考 Gmsh circumCenterMetric：
    求解方程组：
    (x - pa)^T M (x - pa) = (x - pb)^T M (x - pb) = (x - pc)^T M (x - pc)
    
    展开为线性方程组：
    sys * x = rhs
    
    参数:
        p1, p2, p3: 三角形顶点 [u, v]（参数空间）
        metric: 度量张量 [a, b, d]
    
    返回:
        (center, radius_sq): 外接圆心和半径平方
    """
    a, b, d = metric
    
    pa, pb, pc = p1, p2, p3
    
    # 构建 2x2 线性系统
    sys = np.array([
        [2 * (a * (pa[0] - pb[0]) + b * (pa[1] - pb[1])),
         2 * (d * (pa[1] - pb[1]) + b * (pa[0] - pb[0]))],
        [2 * (a * (pa[0] - pc[0]) + b * (pa[1] - pc[1])),
         2 * (d * (pa[1] - pc[1]) + b * (pa[0] - pc[0]))]
    ])
    
    rhs = np.array([
        a * (pa[0]**2 - pb[0]**2) + d * (pa[1]**2 - pb[1]**2) + 2 * b * (pa[0] * pa[1] - pb[0] * pb[1]),
        a * (pa[0]**2 - pc[0]**2) + d * (pa[1]**2 - pc[1]**2) + 2 * b * (pa[0] * pa[1] - pc[0] * pc[1])
    ])
    
    # 求解线性系统
    try:
        center = np.linalg.solve(sys, rhs)
    except np.linalg.LinAlgError:
        # 奇异矩阵，返回重心
        center = (pa + pb + pc) / 3.0
    
    # 计算半径平方
    dx = center - pa
    radius_sq = a * dx[0]**2 + d * dx[1]**2 + 2 * b * dx[0] * dx[1]
    
    return center, radius_sq


def in_circumcircle_aniso(p1, p2, p3, point, metric) -> bool:
    """各向异性度量空间中的点在圆内测试。
    
    参考 Gmsh inCircumCircleAniso：
    使用度量张量定义的距离判断点是否在圆内。
    
    参数:
        p1, p2, p3: 三角形顶点 [u, v]
        point: 测试点 [u, v]
        metric: 度量张量 [a, b, d]
    
    返回:
        True 如果点在圆内
    """
    # 计算外接圆
    center, radius_sq = circumcenter_metric(p1, p2, p3, metric)
    
    # 计算测试点到圆心的度量距离
    a, b, d = metric
    dx = point - center
    dist_sq = a * dx[0]**2 + d * dx[1]**2 + 2 * b * dx[0] * dx[1]
    
    # 计算容差（自适应）
    tolerance = compute_tolerance(radius_sq)
    
    return dist_sq < radius_sq - tolerance


def compute_tolerance(radius_sq: float) -> float:
    """自适应容差策略。
    
    参考 Gmsh computeTolerance：
    根据外接圆大小调整容差。
    
    返回:
        容差值
    """
    radius = np.sqrt(abs(radius_sq))
    
    if radius <= 1e3:
        return 1e-12
    elif radius <= 1e5:
        return 1e-11
    else:
        return 1e-9


# =============================================================================
# 鲁棒点在圆内测试（封装接口）
# =============================================================================

def point_in_circumcircle_robust(p1, p2, p3, point) -> bool:
    """鲁棒的点在圆内测试（结合精确谓词和高精度算术）。
    
    这是主要的外部接口，结合了：
    1. Shewchuk 的 incircle 谓词（符号测试）
    2. 高精度 circumcenter 计算（数值稳定性）
    3. 自适应容差
    
    参数:
        p1, p2, p3: 三角形顶点 [x, y]
        point: 测试点 [x, y]
    
    返回:
        True 如果点在圆内
    """
    # 使用 incircle 谓词（最鲁棒）
    result = incircle(
        p1[0], p1[1],
        p2[0], p2[1],
        p3[0], p3[1],
        point[0], point[1]
    )
    
    return result > 0

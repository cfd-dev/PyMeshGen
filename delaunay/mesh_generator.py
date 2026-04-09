"""
Delaunay 网格生成器
使用 Bowyer-Watson 算法生成高质量的三角形网格
"""

import numpy as np
from typing import List, Tuple, Optional


class Triangle:
    """三角形类"""

    def __init__(self, p1: int, p2: int, p3: int):
        self.vertices = sorted([p1, p2, p3])
        self.circumcenter = None
        self.circumradius = None

    def __eq__(self, other):
        return self.vertices == other.vertices

    def __hash__(self):
        return hash(tuple(self.vertices))

    def __contains__(self, point_idx: int) -> bool:
        return point_idx in self.vertices

    def get_edges(self) -> List[Tuple[int, int]]:
        """返回三角形的三条边"""
        return [
            (self.vertices[0], self.vertices[1]),
            (self.vertices[1], self.vertices[2]),
            (self.vertices[0], self.vertices[2]),
        ]


class BowyerWatson:
    """Bowyer-Watson Delaunay 三角剖分算法"""

    def __init__(self, points: np.ndarray):
        self.points = points
        self.triangles = []
        self.super_triangle = None

    def compute_circumcircle(self, tri: Triangle) -> Tuple[np.ndarray, float]:
        """计算三角形的外接圆"""
        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

        if abs(d) < 1e-10:
            center = np.array([(ax + bx + cx) / 3, (ay + by + cy) / 3])
            radius = np.linalg.norm(p1 - center)
            return center, radius

        ux = (
            (ax**2 + ay**2) * (by - cy)
            + (bx**2 + by**2) * (cy - ay)
            + (cx**2 + cy**2) * (ay - by)
        ) / d
        uy = (
            (ax**2 + ay**2) * (cx - bx)
            + (bx**2 + by**2) * (ax - cx)
            + (cx**2 + cy**2) * (bx - ax)
        ) / d

        center = np.array([ux, uy])
        radius = np.linalg.norm(p1 - center)

        return center, radius

    def point_in_circumcircle(self, point: np.ndarray, tri: Triangle) -> bool:
        """检查点是否在三角形的外接圆内"""
        if tri.circumcenter is None:
            tri.circumcenter, tri.circumradius = self.compute_circumcircle(tri)

        distance = np.linalg.norm(point - tri.circumcenter)
        return distance < tri.circumradius

    def create_super_triangle(self) -> Triangle:
        """创建包含所有点的超级三角形"""
        min_x = np.min(self.points[:, 0])
        max_x = np.max(self.points[:, 0])
        min_y = np.min(self.points[:, 1])
        max_y = np.max(self.points[:, 1])

        dx = max_x - min_x
        dy = max_y - min_y
        delta = max(dx, dy) * 10

        p1 = len(self.points)
        p2 = len(self.points) + 1
        p3 = len(self.points) + 2

        super_points = np.array(
            [
                [min_x - delta, min_y - delta],
                [min_x + 3 * delta, min_y - delta],
                [min_x + delta, max_y + 3 * delta],
            ]
        )

        self.points = np.vstack([self.points, super_points])

        return Triangle(p1, p2, p3)

    def triangulate(self) -> List[Triangle]:
        """执行 Bowyer-Watson 算法"""
        self.super_triangle = self.create_super_triangle()
        self.triangles = [self.super_triangle]

        for i in range(len(self.points) - 3):
            point = self.points[i]

            bad_triangles = []
            for tri in self.triangles:
                if self.point_in_circumcircle(point, tri):
                    bad_triangles.append(tri)

            polygon = []
            for tri in bad_triangles:
                for edge in tri.get_edges():
                    is_shared = False
                    for other_tri in bad_triangles:
                        if tri != other_tri and edge in other_tri.get_edges():
                            is_shared = True
                            break
                    if not is_shared:
                        polygon.append(edge)

            for tri in bad_triangles:
                self.triangles.remove(tri)

            for edge in polygon:
                new_tri = Triangle(edge[0], edge[1], i)
                new_tri.circumcenter, new_tri.circumradius = self.compute_circumcircle(
                    new_tri
                )
                self.triangles.append(new_tri)

        final_triangles = []
        for tri in self.triangles:
            if not (
                self.super_triangle.vertices[0] in tri
                or self.super_triangle.vertices[1] in tri
                or self.super_triangle.vertices[2] in tri
            ):
                final_triangles.append(tri)

        self.points = self.points[:-3]
        self.triangles = final_triangles

        return final_triangles

    def get_simplices(self) -> np.ndarray:
        """获取三角形顶点索引数组"""
        return np.array([tri.vertices for tri in self.triangles])


def compute_triangle_quality(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三角形质量
    使用纵横比（aspect ratio）作为质量度量
    值越接近1，三角形质量越好
    """
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)

    s = (a + b + c) / 2
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))

    if area < 1e-10:
        return 0.0

    longest_edge = max(a, b, c)
    inscribed_radius = area / s

    if inscribed_radius < 1e-10:
        return 0.0

    quality = 2 * inscribed_radius / longest_edge

    return min(quality, 1.0)


def evaluate_mesh_quality(points: np.ndarray, triangles: List[Triangle]) -> dict:
    """评估网格质量"""
    qualities = []
    min_angles = []
    max_angles = []

    for tri in triangles:
        p1 = points[tri.vertices[0]]
        p2 = points[tri.vertices[1]]
        p3 = points[tri.vertices[2]]

        quality = compute_triangle_quality(p1, p2, p3)
        qualities.append(quality)

        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p3 - p2

        angle1 = np.arccos(
            np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10),
                -1,
                1,
            )
        )
        angle2 = np.arccos(
            np.clip(
                np.dot(-v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3) + 1e-10),
                -1,
                1,
            )
        )
        angle3 = np.pi - angle1 - angle2

        angles = [angle1, angle2, angle3]
        min_angles.append(np.min(angles) * 180 / np.pi)
        max_angles.append(np.max(angles) * 180 / np.pi)

    return {
        "mean_quality": np.mean(qualities),
        "min_quality": np.min(qualities),
        "max_quality": np.max(qualities),
        "std_quality": np.std(qualities),
        "mean_min_angle": np.mean(min_angles),
        "mean_max_angle": np.mean(max_angles),
        "num_triangles": len(triangles),
    }


def laplacian_smoothing(
    points: np.ndarray,
    triangles: List[Triangle],
    boundary_mask: np.ndarray,
    iterations: int = 5,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Laplacian 平滑优化网格
    通过迭代调整顶点位置来改善网格质量
    
    参数:
        points: 点坐标数组
        triangles: 三角形列表
        boundary_mask: 边界点掩码，True 表示边界点（不移动）
        iterations: 迭代次数
        alpha: 平滑系数
    """
    smoothed_points = points.copy()

    for _ in range(iterations):
        new_points = smoothed_points.copy()

        neighbor_dict = {}
        for tri in triangles:
            for i in range(3):
                v = tri.vertices[i]
                if v not in neighbor_dict:
                    neighbor_dict[v] = set()
                neighbor_dict[v].add(tri.vertices[(i + 1) % 3])
                neighbor_dict[v].add(tri.vertices[(i + 2) % 3])

        for v, neighbors in neighbor_dict.items():
            # 边界点不移动
            if boundary_mask[v]:
                continue
                
            if len(neighbors) > 0:
                neighbor_center = np.mean(
                    [smoothed_points[n] for n in neighbors], axis=0
                )
                new_points[v] = smoothed_points[v] + alpha * (
                    neighbor_center - smoothed_points[v]
                )

        smoothed_points = new_points

    return smoothed_points


def generate_boundary_points(boundary_points: int = 10) -> np.ndarray:
    """
    生成正方形边界点（逆时针顺序，无重复）
    
    参数:
        boundary_points: 每条边上的点数（包括端点）
    
    返回:
        points: 边界点坐标数组，形状为 (4*(boundary_points-1), 2)
    """
    points = []

    # 底边：从左到右 (y=0, x从0到1)，不包括右端点
    for i in range(boundary_points - 1):
        t = i / (boundary_points - 1)
        points.append([t, 0.0])
    
    # 右边：从下到上 (x=1, y从0到1)，不包括上端点
    for i in range(boundary_points - 1):
        t = i / (boundary_points - 1)
        points.append([1.0, t])
    
    # 顶边：从右到左 (y=1, x从1到0)，不包括左端点
    for i in range(boundary_points - 1):
        t = 1.0 - i / (boundary_points - 1)
        points.append([t, 1.0])
    
    # 左边：从上到下 (x=0, y从1到0)，不包括下端点
    for i in range(boundary_points - 1):
        t = 1.0 - i / (boundary_points - 1)
        points.append([0.0, t])

    return np.array(points)


def insert_points_iteratively(
    bw: BowyerWatson,
    target_triangle_count: Optional[int] = None,
    boundary_count: int = 0,
    max_edge_length: Optional[float] = None,
    size_field: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> None:
    """
    迭代插入内部点，使用外接圆圆心策略
    选择质量最差的三角形的外接圆圆心作为新点位置
    
    参数:
        bw: BowyerWatson 实例
        target_triangle_count: 目标三角形数量（可选）
        boundary_count: 边界点数量
        max_edge_length: 全局最大边长（可选）
        size_field: 尺寸场函数 f(x, y) -> max_edge_length，输入点坐标，返回允许的最大边长
        seed: 随机种子
    
    终止条件（满足任一即停止）:
        1. 达到目标三角形数量
        2. 所有三角形的边长都小于 max_edge_length 或 size_field 指定的尺寸
    """
    if seed is not None:
        np.random.seed(seed)

    # 根据 Euler 公式计算目标节点数（如果指定了目标三角形数）
    target_total_points = None
    if target_triangle_count is not None:
        target_total_points = (target_triangle_count + 2 + boundary_count) // 2

    while True:
        # 安全检查：防止无限循环
        if len(bw.points) > 5000:  # 最大节点数限制
            print(f"  警告：达到最大节点数限制 (5000)，停止插点")
            break
        
        # 检查终止条件 1: 达到目标三角形数量
        if target_total_points is not None and len(bw.points) >= target_total_points:
            break
            
        if len(bw.triangles) == 0:
            break

        # 寻找需要细分的三角形
        worst_quality = float('inf')
        worst_triangle = None
        needs_refinement = False
        
        for tri in bw.triangles:
            if tri.circumcenter is None:
                tri.circumcenter, tri.circumradius = bw.compute_circumcircle(tri)

            p1 = bw.points[tri.vertices[0]]
            p2 = bw.points[tri.vertices[1]]
            p3 = bw.points[tri.vertices[2]]

            quality = compute_triangle_quality(p1, p2, p3)
            
            # 计算三角形的最大边长
            edge_lengths = [
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p3 - p2),
                np.linalg.norm(p1 - p3)
            ]
            max_edge = max(edge_lengths)
            
            # 确定该三角形的目标尺寸
            if size_field is not None:
                # 使用尺寸场：取三角形中心的尺寸
                center = (p1 + p2 + p3) / 3
                # 支持两种调用方式：size_field(center) 或 size_field(x, y)
                try:
                    target_size = size_field(center)
                except TypeError:
                    target_size = size_field(center[0], center[1])
            elif max_edge_length is not None:
                target_size = max_edge_length
            else:
                target_size = None
            
            # 检查是否需要细分（边长过大或质量差）
            should_split = False
            if target_size is not None and max_edge > target_size:
                should_split = True
            elif target_size is None and quality < 0.5:
                # 如果没有尺寸限制，只细分质量差的三角形
                should_split = True
            
            if should_split:
                needs_refinement = True
                # 选择质量最差的三角形进行细分
                if quality < worst_quality:
                    worst_quality = quality
                    worst_triangle = tri

        # 终止条件 2: 所有三角形都满足尺寸要求
        if not needs_refinement:
            break
            
        if worst_triangle is None:
            break

        new_point = worst_triangle.circumcenter.copy()

        # 确保新点在正方形区域内
        margin = 0.001
        if margin < new_point[0] < 1 - margin and margin < new_point[1] < 1 - margin:

            # 检查与已有点的最小距离
            min_dist = float("inf")
            for existing_point in bw.points:
                dist = np.linalg.norm(new_point - existing_point)
                min_dist = min(min_dist, dist)

            # 如果距离足够远，则添加新点
            if min_dist > 0.01:
                bw.points = np.vstack([bw.points, new_point])
                bw.triangles = []
                bw.super_triangle = None
                bw.triangulate()
        else:
            # 如果点在边界附近，尝试在三角形内部随机添加点
            p1 = bw.points[worst_triangle.vertices[0]]
            p2 = bw.points[worst_triangle.vertices[1]]
            p3 = bw.points[worst_triangle.vertices[2]]
            
            # 在三角形内随机生成点
            r1, r2 = np.random.rand(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            new_point = p1 + r1 * (p2 - p1) + r2 * (p3 - p1)
            
            margin = 0.001
            if margin < new_point[0] < 1 - margin and margin < new_point[1] < 1 - margin:
                min_dist = float("inf")
                for existing_point in bw.points:
                    dist = np.linalg.norm(new_point - existing_point)
                    min_dist = min(min_dist, dist)

                if min_dist > 0.01:
                    bw.points = np.vstack([bw.points, new_point])
                    bw.triangles = []
                    bw.super_triangle = None
                    bw.triangulate()
            else:
                break


def generate_uniform_points(
    n_points: int, boundary_points: int = 10, seed: Optional[int] = None
) -> np.ndarray:
    """
    生成更均匀分布的点集
    使用泊松盘采样的简化版本
    """
    if seed is not None:
        np.random.seed(seed)

    points = []

    for i in range(boundary_points):
        t = i / (boundary_points - 1)
        points.append([t, 0.0])
        points.append([t, 1.0])
        points.append([0.0, t])
        points.append([1.0, t])

    internal_points = []
    grid_size = int(np.sqrt(n_points))
    spacing = 1.0 / (grid_size + 1)

    for i in range(grid_size):
        for j in range(grid_size):
            x = spacing * (i + 1) + np.random.uniform(-spacing * 0.3, spacing * 0.3)
            y = spacing * (j + 1) + np.random.uniform(-spacing * 0.3, spacing * 0.3)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            internal_points.append([x, y])

    remaining = n_points - len(internal_points)
    if remaining > 0:
        extra_points = np.random.rand(remaining, 2) * 0.9 + 0.05
        internal_points.extend(extra_points.tolist())

    points.extend(internal_points)

    return np.array(points)


def create_delaunay_mesh(
    target_triangle_count: Optional[int] = None,
    boundary_points: int = 15,
    max_edge_length: Optional[float] = None,
    size_field: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    smoothing_iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    使用 Bowyer-Watson 算法创建高质量的 Delaunay 网格

    参数:
        target_triangle_count: 目标三角形数量（可选）
        boundary_points: 每条边界上的点数
        max_edge_length: 全局最大边长（可选）
        size_field: 尺寸场函数 f(x, y) -> max_edge_length（可选）
        seed: 随机种子
        smoothing_iterations: 平滑迭代次数

    返回:
        points: 点坐标数组
        simplices: 三角形索引数组
        boundary_mask: 边界点掩码（True 表示边界点）
        quality_info: 网格质量信息
    """
    points = generate_boundary_points(boundary_points)
    boundary_count = len(points)

    bw = BowyerWatson(points.copy())
    triangles = bw.triangulate()

    insert_points_iteratively(
        bw,
        target_triangle_count=target_triangle_count,
        boundary_count=boundary_count,
        max_edge_length=max_edge_length,
        size_field=size_field,
        seed=seed
    )

    points = bw.points
    triangles = bw.triangles
    
    # 更新边界掩码（添加内部点后，边界点仍然是前 boundary_count 个）
    boundary_mask = np.zeros(len(points), dtype=bool)
    boundary_mask[:boundary_count] = True

    if smoothing_iterations > 0:
        points = laplacian_smoothing(
            points, triangles, boundary_mask, iterations=smoothing_iterations
        )
        # 光滑后需要重新三角剖分，但边界点保持不变
        bw = BowyerWatson(points.copy())
        triangles = bw.triangulate()
        # 更新 boundary_mask（点顺序不变）
        boundary_mask = np.zeros(len(points), dtype=bool)
        boundary_mask[:boundary_count] = True

    simplices = bw.get_simplices()

    quality_info = evaluate_mesh_quality(points, triangles)

    return points, simplices, boundary_mask, quality_info


def extract_edges_from_triangles(triangles: List[Triangle]) -> np.ndarray:
    """从三角形列表中提取边"""
    edge_set = set()
    for tri in triangles:
        for edge in tri.get_edges():
            edge_set.add((min(edge), max(edge)))

    return np.array(list(edge_set)).T

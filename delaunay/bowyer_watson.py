"""
Bowyer-Watson Delaunay 网格生成器

基于 Bowyer-Watson 算法的二维三角形网格生成，支持：
1. 以离散边界网格作为输入
2. 使用 QuadtreeSizing 尺寸场控制网格尺寸
3. 自动生成高质量三角形网格

优化特性：
- 增量式点插入（避免全量重剖分）
- 空间索引加速最近邻搜索
- 缓存三角形质量和尺寸计算
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from math import sqrt
from scipy.spatial import KDTree

from utils.message import info, debug, warning, verbose
from utils.timer import TimeSpan


class Triangle:
    """三角形单元类"""

    __slots__ = ['vertices', 'circumcenter', 'circumradius', 'idx']

    def __init__(self, p1: int, p2: int, p3: int, idx: int = -1):
        # 顶点索引排序以便于去重和比较
        self.vertices = tuple(sorted([p1, p2, p3]))
        self.circumcenter = None  # 外接圆圆心
        self.circumradius = None  # 外接圆半径
        self.idx = idx  # 三角形索引

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return False
        return self.vertices == other.vertices

    def __hash__(self):
        return hash(self.vertices)

    def __contains__(self, point_idx: int) -> bool:
        return point_idx in self.vertices

    def get_edges(self) -> List[Tuple[int, int]]:
        """返回三角形的三条边（排序后的元组）"""
        v0, v1, v2 = self.vertices
        return [
            (v0, v1),
            (v1, v2),
            (v0, v2),
        ]

    def __repr__(self):
        return f"Triangle({self.vertices})"


class BowyerWatsonMeshGenerator:
    """
    Bowyer-Watson Delaunay 网格生成器
    
    特性：
    - 支持从离散边界网格输入
    - 集成 QuadtreeSizing 尺寸场控制
    - 支持尺寸场和全局尺寸控制
    - Laplacian 平滑优化
    """

    def __init__(
        self,
        boundary_points: np.ndarray,
        boundary_edges: Optional[List[Tuple[int, int]]] = None,
        sizing_system=None,
        max_edge_length: Optional[float] = None,
        smoothing_iterations: int = 3,
        seed: Optional[int] = None,
    ):
        """
        初始化 Bowyer-Watson 网格生成器

        参数:
            boundary_points: 边界点坐标数组，形状为 (N, 2)
            boundary_edges: 边界边列表，每个元素为 (idx1, idx2)
            sizing_system: QuadtreeSizing 尺寸场对象（可选）
            max_edge_length: 全局最大边长（可选，优先使用 sizing_system）
            smoothing_iterations: Laplacian 平滑迭代次数
            seed: 随机种子
        """
        self.original_points = boundary_points.copy()
        self.boundary_edges = boundary_edges or []
        self.sizing_system = sizing_system
        self.max_edge_length = max_edge_length
        self.smoothing_iterations = smoothing_iterations
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # 工作状态变量
        self.points = None  # 所有点（边界+内部）
        self.triangles = []  # 三角形列表
        self.boundary_mask = None  # 边界点掩码
        self.boundary_count = 0  # 边界点数量
        self._kdtree = None  # KD-tree 用于加速最近邻搜索

    def _compute_circumcircle(self, tri: Triangle) -> Tuple[np.ndarray, float]:
        """
        计算三角形的外接圆

        返回:
            (center, radius): 外接圆圆心和半径
        """
        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        ax, ay = float(p1[0]), float(p1[1])
        bx, by = float(p2[0]), float(p2[1])
        cx, cy = float(p3[0]), float(p3[1])

        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

        if abs(d) < 1e-12:
            # 三点共线或非常接近，返回重心
            center = np.array([(ax + bx + cx) / 3.0, (ay + by + cy) / 3.0])
            radius = float(np.linalg.norm(p1 - center))
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
        radius = float(np.linalg.norm(p1 - center))

        return center, radius

    def _point_in_circumcircle(self, point: np.ndarray, tri: Triangle) -> bool:
        """
        检查点是否在三角形的外接圆内（含边界）
        """
        if tri.circumcenter is None:
            tri.circumcenter, tri.circumradius = self._compute_circumcircle(tri)

        distance = float(np.linalg.norm(point - tri.circumcenter))
        # 使用小的容差值避免浮点误差
        return distance < tri.circumradius * (1.0 + 1e-10)

    def _create_super_triangle(self) -> Triangle:
        """
        创建包含所有点的超级三角形
        """
        min_x = np.min(self.points[:, 0])
        max_x = np.max(self.points[:, 0])
        min_y = np.min(self.points[:, 1])
        max_y = np.max(self.points[:, 1])

        dx = max_x - min_x
        dy = max_y - min_y
        delta = max(dx, dy) * 10.0

        # 超级三角形的三个顶点索引
        p1 = len(self.points)
        p2 = len(self.points) + 1
        p3 = len(self.points) + 2

        # 添加超级三角形顶点
        super_verts = np.array([
            [min_x - delta, min_y - delta],
            [max_x + delta, min_y - delta],
            [(min_x + max_x) / 2.0, max_y + 3.0 * delta],
        ])
        self.points = np.vstack([self.points, super_verts])

        return Triangle(p1, p2, p3)

    def _triangulate(self) -> List[Triangle]:
        """
        执行 Bowyer-Watson 三角剖分算法

        算法流程：
        1. 创建超级三角形包含所有点
        2. 逐点插入，删除外接圆包含该点的三角形
        3. 重新连接形成新的三角形
        4. 删除包含超级三角形顶点的三角形

        返回:
            三角形列表
        """
        # 创建超级三角形
        super_tri = self._create_super_triangle()
        triangles = [super_tri]

        # 获取真实点数量（不包括超级三角形顶点）
        real_point_count = len(self.points) - 3

        # 逐点插入
        for i in range(real_point_count):
            point = self.points[i]

            # 找到所有外接圆包含当前点的三角形
            bad_triangles = []
            for tri in triangles:
                if self._point_in_circumcircle(point, tri):
                    bad_triangles.append(tri)

            # 找到坏三角形的边界边（不共享的边）
            boundary_edges_dict = {}
            for tri in bad_triangles:
                for edge in tri.get_edges():
                    # 使用排序后的边作为键
                    edge_key = tuple(sorted(edge))
                    if edge_key in boundary_edges_dict:
                        boundary_edges_dict[edge_key] += 1
                    else:
                        boundary_edges_dict[edge_key] = 1

            # 只出现一次的边是边界边
            polygon_edges = [
                edge for edge, count in boundary_edges_dict.items() if count == 1
            ]

            # 删除坏三角形
            for tri in bad_triangles:
                triangles.remove(tri)

            # 创建新三角形连接边界边和新点
            for edge in polygon_edges:
                new_tri = Triangle(edge[0], edge[1], i)
                new_tri.circumcenter, new_tri.circumradius = self._compute_circumcircle(new_tri)
                triangles.append(new_tri)

        # 移除包含超级三角形顶点的三角形
        super_verts = {super_tri.vertices[0], super_tri.vertices[1], super_tri.vertices[2]}
        final_triangles = [
            tri for tri in triangles
            if not any(v in super_verts for v in tri.vertices)
        ]

        # 移除超级三角形顶点
        self.points = self.points[:-3]
        self.triangles = final_triangles

        return final_triangles

    def _compute_triangle_quality(self, tri: Triangle) -> float:
        """
        计算三角形质量（基于纵横比）

        质量定义：quality = 2 * r_inscribed / r_circumscribed
        值越接近 1，三角形质量越好（等边三角形质量为 1）
        """
        p1 = self.points[tri.vertices[0]]
        p2 = self.points[tri.vertices[1]]
        p3 = self.points[tri.vertices[2]]

        a = float(np.linalg.norm(p2 - p1))
        b = float(np.linalg.norm(p3 - p2))
        c = float(np.linalg.norm(p1 - p3))

        # 半周长
        s = (a + b + c) / 2.0

        # 面积（海伦公式）
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq < 0:
            return 0.0
        area = sqrt(max(area_sq, 0.0))

        if area < 1e-12:
            return 0.0

        # 内切圆半径
        inscribed_radius = area / s

        # 外接圆半径
        if tri.circumradius is None:
            tri.circumcenter, tri.circumradius = self._compute_circumcircle(tri)
        circumscribed_radius = tri.circumradius

        if circumscribed_radius < 1e-12:
            return 0.0

        # 质量 = 2 * r_inscribed / r_circumscribed
        quality = 2.0 * inscribed_radius / circumscribed_radius
        return min(quality, 1.0)

    def _get_target_size_for_triangle(self, tri: Triangle) -> Optional[float]:
        """
        获取三角形的目标尺寸

        优先级：
        1. sizing_system（尺寸场）
        2. max_edge_length（全局尺寸）
        3. None（不限制）
        """
        if self.sizing_system is not None:
            # 使用尺寸场：取三角形中心处的尺寸
            center = np.mean([
                self.points[tri.vertices[0]],
                self.points[tri.vertices[1]],
                self.points[tri.vertices[2]],
            ], axis=0)
            try:
                return self.sizing_system.spacing_at(center)
            except Exception as e:
                debug(f"获取尺寸场失败: {e}，使用全局尺寸")
                return self.max_edge_length
        else:
            return self.max_edge_length

    def _insert_point_incremental(self, point_idx: int, triangles: List[Triangle]) -> List[Triangle]:
        """
        增量式插入单个点到现有三角剖分中
        
        这是优化的关键：只更新受影响的三角形，而不是重新剖分所有点
        
        参数:
            point_idx: 新插入点的索引
            triangles: 当前三角形列表
            
        返回:
            更新后的三角形列表
        """
        point = self.points[point_idx]
        
        # 找到所有外接圆包含新点的三角形（bad triangles）
        bad_triangles = []
        for tri in triangles:
            if self._point_in_circumcircle(point, tri):
                bad_triangles.append(tri)
        
        if not bad_triangles:
            return triangles
        
        # 找到 bad triangles 的边界边（不共享的边）
        boundary_edges_dict = {}
        for tri in bad_triangles:
            for edge in tri.get_edges():
                edge_key = tuple(sorted(edge))
                if edge_key in boundary_edges_dict:
                    boundary_edges_dict[edge_key] += 1
                else:
                    boundary_edges_dict[edge_key] = 1
        
        # 只出现一次的边是边界边
        polygon_edges = [
            edge for edge, count in boundary_edges_dict.items() if count == 1
        ]
        
        # 删除 bad triangles
        bad_set = set(id(tri) for tri in bad_triangles)
        triangles = [tri for tri in triangles if id(tri) not in bad_set]
        
        # 创建新三角形连接边界边和新点
        for edge in polygon_edges:
            new_tri = Triangle(edge[0], edge[1], point_idx)
            new_tri.circumcenter, new_tri.circumradius = self._compute_circumcircle(new_tri)
            triangles.append(new_tri)
        
        return triangles

    def _insert_points_iteratively(self, target_triangle_count: Optional[int] = None):
        """
        迭代插入内部点，使用外接圆圆心策略

        优化策略：
        - 使用增量式插入（避免全量重剖分）
        - 使用 KD-tree 加速最近邻搜索
        - 批量处理候选三角形

        终止条件（满足任一即停止）：
        1. 达到目标三角形数量
        2. 所有三角形都满足尺寸和质量要求
        3. 达到最大节点数限制
        """
        boundary_count = self.boundary_count
        initial_point_count = len(self.points)
        inserted_points = 0  # 已插入的内部点计数器

        # 根据 Euler 公式计算目标节点数（如果指定了目标三角形数）
        target_total_points = None
        if target_triangle_count is not None:
            target_total_points = (target_triangle_count + 2 + boundary_count) // 2

        max_iterations = 0
        while True:
            max_iterations += 1
            
            # 定期输出进度（每 10 次迭代输出一次）
            if max_iterations % 10 == 0 or max_iterations == 1:
                current_triangles = len(self.triangles)
                current_points = len(self.points)
                current_inserted = current_points - initial_point_count
                verbose(f"  [进度] 迭代 {max_iterations} | "
                       f"节点: {current_points} (边界: {boundary_count}, 内部: {current_inserted}) | "
                       f"三角形: {current_triangles}")
            
            # 安全检查：防止无限循环
            if len(self.points) > 100000:
                warning(f"达到最大节点数限制 (100000)，停止插点")
                break

            if max_iterations > 50000:
                warning(f"达到最大迭代次数限制 (50000)，停止插点")
                break

            # 检查终止条件 1: 达到目标三角形数量
            if target_total_points is not None and len(self.points) >= target_total_points:
                verbose(f"  [进度] 达到目标节点数: {len(self.points)}")
                break

            if len(self.triangles) == 0:
                break

            # 寻找需要细分的三角形
            worst_quality = float('inf')
            worst_triangle = None
            needs_refinement = False

            for tri in self.triangles:
                # 计算三角形质量
                quality = self._compute_triangle_quality(tri)

                # 计算最大边长
                edge_lengths = [
                    float(np.linalg.norm(self.points[tri.vertices[1]] - self.points[tri.vertices[0]])),
                    float(np.linalg.norm(self.points[tri.vertices[2]] - self.points[tri.vertices[1]])),
                    float(np.linalg.norm(self.points[tri.vertices[0]] - self.points[tri.vertices[2]])),
                ]
                max_edge = max(edge_lengths)

                # 获取目标尺寸
                target_size = self._get_target_size_for_triangle(tri)

                # 检查是否需要细分
                should_split = False
                if target_size is not None and max_edge > target_size * 1.1:
                    should_split = True
                elif target_size is None and quality < 0.5:
                    should_split = True

                if should_split:
                    needs_refinement = True
                    if quality < worst_quality:
                        worst_quality = quality
                        worst_triangle = tri

            # 终止条件 2: 所有三角形都满足要求
            if not needs_refinement:
                verbose(f"  [进度] 所有三角形满足尺寸和质量要求")
                break

            if worst_triangle is None:
                break

            # 新点位置：外接圆圆心
            if worst_triangle.circumcenter is None:
                worst_triangle.circumcenter, worst_triangle.circumradius = \
                    self._compute_circumcircle(worst_triangle)

            new_point = worst_triangle.circumcenter.copy()

            # 计算边界范围
            x_min = np.min(self.original_points[:, 0])
            x_max = np.max(self.original_points[:, 0])
            y_min = np.min(self.original_points[:, 1])
            y_max = np.max(self.original_points[:, 1])

            margin = 0.001 * max(x_max - x_min, y_max - y_min)

            # 检查新点是否在边界范围内
            if not (x_min - margin < new_point[0] < x_max + margin and
                    y_min - margin < new_point[1] < y_max + margin):
                # 如果外接圆中心在边界外，尝试在三角形内部随机生成点
                p1 = self.points[worst_triangle.vertices[0]]
                p2 = self.points[worst_triangle.vertices[1]]
                p3 = self.points[worst_triangle.vertices[2]]

                r1, r2 = np.random.rand(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                new_point = p1 + r1 * (p2 - p1) + r2 * (p3 - p1)

            # 使用 KD-tree 加速最近邻搜索
            if len(self.points) > 0:
                # 构建 KD-tree（只在需要时）
                if max_iterations == 1 or max_iterations % 50 == 0:
                    self._kdtree = KDTree(self.points)
                
                min_dist, _ = self._kdtree.query(new_point)
            else:
                min_dist = float('inf')

            # 最小距离阈值
            avg_edge = np.mean([
                float(np.linalg.norm(self.original_points[1] - self.original_points[0]))
                if len(self.original_points) > 1 else 0.1
            ])
            min_dist_threshold = 0.01 * avg_edge if avg_edge > 0 else 0.01

            # 如果距离足够远，则添加新点
            if min_dist > min_dist_threshold:
                # 添加新点
                new_point_idx = len(self.points)
                self.points = np.vstack([self.points, new_point])
                inserted_points += 1
                
                # 增量式插入（关键优化：避免全量重剖分）
                self.triangles = self._insert_point_incremental(new_point_idx, self.triangles)
            else:
                verbose(f"  [进度] 无法插入新点（距离过近），停止插点")
                break
        
        # 最终进度输出
        final_triangles = len(self.triangles)
        final_points = len(self.points)
        final_inserted = final_points - initial_point_count
        verbose(f"  [完成] 插点完成 | "
               f"节点: {final_points} (边界: {boundary_count}, 内部: {final_inserted}) | "
               f"三角形: {final_triangles}")

    def _laplacian_smoothing(self, iterations: int = 3, alpha: float = 0.5):
        """
        Laplacian 平滑优化网格

        通过迭代调整内部点位置来改善网格质量
        边界点保持不动

        参数:
            iterations: 迭代次数
            alpha: 平滑系数 (0-1)
        """
        if iterations <= 0:
            return

        smoothed_points = self.points.copy()

        for iteration in range(iterations):
            new_points = smoothed_points.copy()

            # 构建邻接表
            neighbor_dict = {}
            for tri in self.triangles:
                for i in range(3):
                    v = tri.vertices[i]
                    if v not in neighbor_dict:
                        neighbor_dict[v] = set()
                    neighbor_dict[v].add(tri.vertices[(i + 1) % 3])
                    neighbor_dict[v].add(tri.vertices[(i + 2) % 3])

            # 更新内部点位置（边界点不动）
            for v, neighbors in neighbor_dict.items():
                # 边界点不移动
                if self.boundary_mask[v]:
                    continue

                if len(neighbors) > 0:
                    neighbor_center = np.mean(
                        [smoothed_points[n] for n in neighbors], axis=0
                    )
                    new_points[v] = smoothed_points[v] + alpha * (
                        neighbor_center - smoothed_points[v]
                    )

            smoothed_points = new_points

        self.points = smoothed_points

    def generate_mesh(
        self,
        target_triangle_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成三角形网格

        参数:
            target_triangle_count: 目标三角形数量（可选）

        返回:
            (points, simplices, boundary_mask):
                - points: 点坐标数组，形状为 (N, 2)
                - simplices: 三角形索引数组，形状为 (M, 3)
                - boundary_mask: 边界点掩码，形状为 (N,)，True 表示边界点
        """
        timer = TimeSpan("开始 Bowyer-Watson 网格生成...")

        # 初始化
        self.boundary_count = len(self.original_points)
        self.points = self.original_points.copy()
        self.boundary_mask = np.zeros(len(self.points), dtype=bool)
        self.boundary_mask[:self.boundary_count] = True

        verbose(f"边界点数量: {self.boundary_count}")

        # 第一次三角剖分
        verbose(f"阶段 1/3: 初始三角剖分...")
        self._triangulate()
        verbose(f"  初始三角形数量: {len(self.triangles)}")

        # 迭代插入内部点
        verbose(f"阶段 2/3: 迭代插入内部点...")
        self._insert_points_iteratively(target_triangle_count)

        # Laplacian 平滑
        if self.smoothing_iterations > 0:
            verbose(f"阶段 3/3: Laplacian 平滑 ({self.smoothing_iterations} 次迭代)...")
            # 更新 boundary_mask 以匹配当前点数
            current_point_count = len(self.points)
            self.boundary_mask = np.zeros(current_point_count, dtype=bool)
            self.boundary_mask[:self.boundary_count] = True

            self._laplacian_smoothing(self.smoothing_iterations)

            # 平滑后需要重新三角剖分
            verbose(f"  平滑后重新三角剖分...")
            self.triangles = []
            self._triangulate()
        else:
            verbose(f"阶段 3/3: 跳过平滑（未启用）")

        # 准备输出
        points = self.points.copy()
        simplices = np.array([tri.vertices for tri in self.triangles])
        # 确保 boundary_mask 是最新的
        boundary_mask = np.zeros(len(points), dtype=bool)
        boundary_mask[:self.boundary_count] = True

        # 统计信息
        num_points = len(points)
        num_triangles = len(simplices)
        num_boundary = np.sum(boundary_mask)
        num_internal = num_points - num_boundary

        verbose(f"网格生成完成:")
        verbose(f"  - 总节点数: {num_points}")
        verbose(f"  - 边界节点: {num_boundary}")
        verbose(f"  - 内部节点: {num_internal}")
        verbose(f"  - 三角形数: {num_triangles}")

        timer.show_to_console("Bowyer-Watson 网格生成完成")

        return points, simplices, boundary_mask


def create_bowyer_watson_mesh(
    boundary_front,
    sizing_system,
    target_triangle_count: Optional[int] = None,
    max_edge_length: Optional[float] = None,
    smoothing_iterations: int = 3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bowyer-Watson 网格生成接口函数

    参数:
        boundary_front: 边界阵面列表（Front 对象），参考 Adfront2 的初始输入
        sizing_system: QuadtreeSizing 尺寸场对象
        target_triangle_count: 目标三角形数量（可选）
        max_edge_length: 全局最大边长（可选，优先使用 sizing_system）
        smoothing_iterations: Laplacian 平滑迭代次数
        seed: 随机种子

    返回:
        (points, simplices, boundary_mask):
            - points: 点坐标数组，形状为 (N, 2)
            - simplices: 三角形索引数组，形状为 (M, 3)
            - boundary_mask: 边界点掩码，形状为 (N,)，True 表示边界点
    """
    timer = TimeSpan("开始 Bowyer-Watson 网格生成流程...")

    # 1. 从边界阵面提取边界点和边界边
    boundary_points = []
    boundary_edges = []
    node_index_map = {}  # Front 节点 hash -> 索引映射
    current_idx = 0

    for front in boundary_front:
        for node_elem in front.node_elems:
            node_hash = node_elem.hash
            if node_hash not in node_index_map:
                node_index_map[node_hash] = current_idx
                boundary_points.append(node_elem.coords)
                current_idx += 1

        # 提取边界边
        if len(front.node_elems) >= 2:
            idx1 = node_index_map[front.node_elems[0].hash]
            idx2 = node_index_map[front.node_elems[1].hash]
            boundary_edges.append((idx1, idx2))

    boundary_points = np.array(boundary_points)
    verbose(f"边界点数: {len(boundary_points)}")
    verbose(f"边界边数: {len(boundary_edges)}")

    # 2. 创建网格生成器
    generator = BowyerWatsonMeshGenerator(
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        sizing_system=sizing_system,
        max_edge_length=max_edge_length,
        smoothing_iterations=smoothing_iterations,
        seed=seed,
    )

    # 3. 生成网格
    points, simplices, boundary_mask = generator.generate_mesh(
        target_triangle_count=target_triangle_count
    )

    timer.show_to_console("Bowyer-Watson 网格生成流程完成")

    return points, simplices, boundary_mask

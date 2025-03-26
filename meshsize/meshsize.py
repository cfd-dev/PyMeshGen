from math import log
import math
import matplotlib.pyplot as plt


class QuadTreeNode:
    """四叉树节点类，用于存储网格划分信息"""

    def __init__(self, bounds):
        self.level = 0  # 节点层级（0表示根节点）
        self.subflag = 0  # 细分标志（0:未细分，1:已细分，-1:需强制细分）
        self.parent = None  # 父节点指针
        self.children = None  # 子节点列表（最多4个子节点）
        self.bounds = bounds  # 节点边界坐标 (x_min, y_min, x_max, y_max)
        self.spacing = [0.0] * 4  # 四个角点的网格尺寸（NW, NE, SW, SE）
        self.isvalid = True  # 节点是否有效（用于遍历过滤）

    @staticmethod
    def _draw_node(node, ax, marker):
        """Recursive node drawing method"""
        if node.children:
            for child in node.children:
                QuadTreeNode._draw_node(child, ax, marker)
        else:
            QuadTreeNode._draw_square(node.bounds, ax, marker)

    @staticmethod
    def _draw_square(bounds, ax, marker):
        """Draw single node's boundaries"""
        x_min, y_min, x_max, y_max = bounds
        ax.plot(
            [x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            marker,
            linewidth=0.5,
        )

    def is_empty(self):
        """检查节点是否为空（无效区域）"""
        return self.bounds[0] >= self.bounds[2] or self.bounds[1] >= self.bounds[3]


def draw_quadtree(quadtree, ax, marker="g-"):
    """Draw entire quadtree structure"""
    for node in quadtree:
        QuadTreeNode._draw_node(node, ax, marker)
    ax.axis("equal")
    plt.show()


def enumerate_quadtree(quadtree, func, data):
    """
    遍历叉树的所有节点

    参数:
    quadtree (list): 叉树节点列表
    func (function): 对每个有效节点应用的函数，接受节点和data作为参数。
    data: 传递给func的额外数据。
    """
    for node in quadtree:
        if not node.isvalid:
            continue
        # 对当前有效节点应用函数
        func(node, data)

        # 如果节点被分割，递归处理所有子节点
        if node.subflag > 0:
            for i in range(4):
                child = node.children[i]
                # 递归调用时，将子节点作为单元素列表传入
                enumerate_quadtree([child], func, data)


def CountNode(node, data):
    data["nCells"] += 1
    data["maxLevel"] = max(data["maxLevel"], node.level)


NEIGHBORS_MAP2D = [
    [-1, 1, -1, 2],  # 子节点0 (NW)
    [0, -1, -1, 3],  # 子节点1 (NE)
    [-1, 3, 0, -1],  # 子节点2 (SW)
    [2, -1, 1, -1],  # 子节点3 (SE)
]


class QuadtreeSizing:
    """四叉树网格生成器，用于自适应网格尺寸计算"""

    def __init__(
        self,
        initial_front=None,  # 初始阵面列表（包含几何边界信息）
        max_size=1.0,  # 最大允许网格尺寸
        resolution=0.1,  # 网格细分分辨率阈值（尺寸差百分比）
        decay=1.2,  # 尺寸场衰减系数（>1时尺寸随距离增大）
        visual_obj=None,  # matplotlib可视化对象
    ):

        self.max_size = max_size  # 最大网格尺寸
        self.resolution = resolution  # 分辨率
        self.decay = decay  # 网格尺寸场的decay
        self.initial_front = initial_front  # 阵面列表

        self.quad_tree = None  # 四叉树
        self.bg_bounds = []  # 背景网格边界
        self.global_spacing = None  # 全局网格尺寸
        self.depth = None  # 叉树深度
        self.bg_divisions = [1, 1]  # 背景网格维度

        self.ax = visual_obj.ax

        self.generate_bg_mesh()

    def draw_bgmesh(self):
        draw_quadtree(self.quad_tree, self.ax)

    def generate_bg_mesh(self):
        # 初始化计算域的BoundingBox及全局尺寸及叉树深度
        self.compute_global_parameters()

        # 初始叉树网格
        self.initial_quadtree()

        # draw_quadtree(self.quad_tree, self.ax)

        # 细分网格
        self.refine_quadtree()

        # draw_quadtree(self.quad_tree, self.ax)

        # 加密区扩散，避免level相差>=2
        self.level_refinement()

        # draw_quadtree(self.quad_tree, self.ax)

        # 根据decay参数计算网格尺度场的decay
        self.compute_spacing_decay()

        # 网格尺度场过渡
        self.spacing_transition()

        self.grid_summary()

        return

    def compute_global_parameters(self):

        print(f"计算域的BoundingBox...")
        all_points = []
        min_edge_len = float("inf")
        max_edge_len = 0
        for front in self.initial_front:
            for node_elem in front.node_elems:
                all_points.append(node_elem.coords)
            min_edge_len = min(min_edge_len, front.length)
            max_edge_len = max(max_edge_len, front.length)

        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]

        # 计算初始边界范围（扩展1%防止边界情况）
        pad = 0.02
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_range = x_max - x_min
        y_range = y_max - y_min

        self.bg_bounds = (
            x_min - x_range * pad,
            y_min - y_range * pad,
            x_max + x_range * pad,
            y_max + y_range * pad,
        )

        print(f"计算最大尺度及树深度...")
        # 计算全局尺寸及叉树深度
        if self.max_size > min_edge_len and self.max_size < max_edge_len:
            self.global_spacing = self.max_size
        else:
            self.global_spacing = max_edge_len

        self._calculate_max_depth(min_edge_len)

    def _calculate_max_depth(self, min_edge_len):
        """计算四叉树最大深度：
        基于公式：depth = ceil(log2(global_spacing / min_edge_length)) + 安全裕度
        保证能解析最小几何特征
        """
        # 计算理论深度
        ratio = self.global_spacing / min_edge_len
        theoretical_depth = math.log(ratio) / math.log(2)  # 换底公式

        # 添加安全裕度（建议1-2层）
        safety_margin = 1

        # self.depth = round(1.4427 * log(ratio) + 0.5)
        self.depth = math.ceil(theoretical_depth) + safety_margin

    def initial_quadtree(self):
        """初始化四叉树网格"""
        print(f"初始quadtree生成...")

        # 计算背景网格分割数
        dist = [
            self.bg_bounds[2] - self.bg_bounds[0],  # x方向长度
            self.bg_bounds[3] - self.bg_bounds[1],  # y方向长度
        ]

        # 计算每个方向的分割数
        self.bg_divisions = [
            max(1, int(dist[0] / self.global_spacing) + 1),
            max(1, int(dist[1] / self.global_spacing) + 1),
        ]

        # 计算实际间距（调整后）
        actual_spacing = [
            dist[0] / self.bg_divisions[0],
            dist[1] / self.bg_divisions[1],
        ]

        # 初始化四叉树节点数组
        self.quad_tree = []

        # 生成初始网格节点
        for j in range(self.bg_divisions[1]):
            for i in range(self.bg_divisions[0]):
                # 计算节点边界
                node_bounds = (
                    self.bg_bounds[0] + i * actual_spacing[0],
                    self.bg_bounds[1] + j * actual_spacing[1],
                    self.bg_bounds[0] + (i + 1) * actual_spacing[0],
                    self.bg_bounds[1] + (j + 1) * actual_spacing[1],
                )

                # 创建节点对象
                node = QuadTreeNode(node_bounds)
                node.level = 0
                node.subflag = 0
                node.spacing = [self.global_spacing] * 4
                node.children = []

                self.quad_tree.append(node)

    @staticmethod
    def divide_bounds(parent_bounds):
        """将父节点边界四等分，返回四个子边界列表（二维）"""
        x_min, y_min, x_max, y_max = parent_bounds
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        return [
            (x_min, y_min, x_center, y_center),  # 西北象限 (NW)
            (x_center, y_min, x_max, y_center),  # 东北象限 (NE)
            (x_min, y_center, x_center, y_max),  # 西南象限 (SW)
            (x_center, y_center, x_max, y_max),  # 东南象限 (SE)
        ]

    def _locate_quadtree(self, point, node):
        """定位点在四叉树中的叶节点"""
        while node.children:
            x_center = (node.bounds[0] + node.bounds[2]) / 2
            y_center = (node.bounds[1] + node.bounds[3]) / 2

            # 判断点所在象限
            quadrant = 0
            if point[0] > x_center:
                quadrant += 1
            if point[1] > y_center:
                quadrant += 2
            node = node.children[quadrant]
        return node

    def refine_quadtree(self):
        """基于表面网格尺寸的细化：
        1. 遍历所有表面网格单元
        2. 定位到对应的四叉树节点
        3. 根据尺寸差判断是否需要细分
        4. 递归细分直到满足条件
        """

        def _should_refine(node, target_size):
            """细分条件判断"""
            if node.level >= self.depth:
                return False

            current_spacing = current.spacing[0]
            if current_spacing <= target_size:
                return False
            diff = (current_spacing - target_size) / current_spacing
            return diff > self.resolution

        print(f"细化初始quadtree...")

        # 遍历所有表面节点
        for front in self.initial_front:
            face_center = front.center
            target_size = front.length

            # 从背景网格根节点开始定位
            current = None
            for root in self.quad_tree:  # 遍历所有背景网格根节点
                if (
                    root.bounds[0] <= face_center[0] <= root.bounds[2]
                    and root.bounds[1] <= face_center[1] <= root.bounds[3]
                ):
                    current = root
                    break

            while _should_refine(current, target_size):
                # 执行细分
                if not current.children:
                    current.subflag = 1
                    current.children = [
                        QuadTreeNode(bounds)
                        for bounds in self.divide_bounds(current.bounds)
                    ]
                    for i, child in enumerate(current.children):
                        child.level = current.level + 1
                        child.parent = current
                        child.spacing = [s / 2 for s in current.spacing]

                # 重新定位到子节点
                current = self._locate_quadtree(face_center, current)

    def level_refinement(self):
        """层级平衡细化：
        保证相邻节点层级差不超过1，避免尺寸突变
        使用广度优先搜索处理需要强制细分的节点
        """

        def mark_neighbors(node, neighbors):
            """递归标记需要细分的节点"""
            changed = 0
            if node.subflag > 0 and node.children:
                cs = [None] * 4
                # node._draw_node(node, self.ax, "r--")
                # 处理当前节点的子节点
                for n in range(4):
                    for j in range(4):  # 四个方向
                        i = NEIGHBORS_MAP2D[n][j]
                        if i < 0:
                            if neighbors[j] and neighbors[j].subflag <= 0:
                                cs[j] = neighbors[j]
                                # cs[j]._draw_node(cs[j], self.ax, "b--")
                            else:
                                opposite = j ^ 1  # 取相反方向
                                i = NEIGHBORS_MAP2D[n][opposite]
                                cs[j] = (
                                    neighbors[j].children[i] if neighbors[j] else None
                                )
                                # cs[j]._draw_node(cs[j], self.ax, "b--")
                        else:
                            cs[j] = node.children[i]
                            # cs[j]._draw_node(cs[j], self.ax, "b--")

                    changed += mark_neighbors(node.children[n], cs)

            elif node.level >= 2:  # 需要平衡的层级差
                # node._draw_node(node, self.ax, "r--")
                for j in range(4):  # 四个方向
                    neighbor = neighbors[j]
                    if (
                        neighbor
                        and neighbor.level < node.level - 1
                        and neighbor.subflag == 0
                    ):
                        neighbor.subflag = -1
                        changed += 1
            return changed

        print(f"平衡quadtree，保证层级差<2...")
        # 主循环
        changed = 1
        while changed:
            changed = 0
            ndiv = 0

            # 遍历背景网格
            for j in range(self.bg_divisions[1]):
                for i in range(self.bg_divisions[0]):
                    current = self.quad_tree[ndiv]
                    neighbors = [None] * 4  # 西、东、南、北

                    # current._draw_node(current, self.ax, "r-")
                    # 获取相邻节点
                    if i > 0:
                        neighbors[0] = self.quad_tree[ndiv - 1]  # 西
                    if i < self.bg_divisions[0] - 1:
                        neighbors[1] = self.quad_tree[ndiv + 1]  # 东
                    if j > 0:
                        neighbors[2] = self.quad_tree[ndiv - self.bg_divisions[0]]  # 南
                    if j < self.bg_divisions[1] - 1:
                        neighbors[3] = self.quad_tree[ndiv + self.bg_divisions[0]]  # 北

                    # for ineighbor in neighbors:
                    #     if ineighbor:
                    #         ineighbor._draw_node(ineighbor, self.ax, "b-")

                    changed += mark_neighbors(current, neighbors)
                    ndiv += 1

            # 执行实际细分
            from collections import deque

            queue = deque(self.quad_tree)  # 初始队列包含所有根节点
            while queue:
                node = queue.popleft()
                if node.subflag < 0:
                    self._force_refine(node)
                # 将子节点加入队列以继续处理
                if node.children:
                    queue.extend(node.children)

    def _force_refine(self, node):
        """执行实际的节点细分"""
        if not node.children and node.level < self.depth:
            node.subflag = 1
            node.children = [
                QuadTreeNode(bounds) for bounds in self.divide_bounds(node.bounds)
            ]
            for i, child in enumerate(node.children):
                child.level = node.level + 1
                child.parent = node
                child.spacing = [s / 2 for s in node.spacing]

    def compute_spacing_decay(self):
        """计算考虑表面节点影响的网格尺寸"""

        def source_spacing(base_size, dist, decay):
            """基于指数衰减的尺寸计算"""
            pwr = 0.5 * dist * (decay - 1.0) / base_size
            pwr = min(pwr, 50.0)  # 限制最大指数
            return base_size * math.exp(pwr)

        print(f"计算尺寸场decay...")

        if self.decay < 1.0:
            return

        # 遍历四叉树所有节点
        from collections import deque

        queue = deque(self.quad_tree)  # 初始队列包含所有根节点
        while queue:
            node = queue.popleft()

            min_sp = float("inf")
            x_center = (node.bounds[0] + node.bounds[2]) / 2
            y_center = (node.bounds[1] + node.bounds[3]) / 2
            # 遍历所有表面节点
            for front in self.initial_front:
                face_center = front.center
                target_size = front.length

                # 计算子节点中心到表面节点的距离
                dx = x_center - face_center[0]
                dy = y_center - face_center[1]
                dist = math.sqrt(dx**2 + dy**2)

                # 计算该表面节点影响的尺寸
                sp = source_spacing(target_size, dist, self.decay)
                min_sp = min(min_sp, sp)

            # 限制最小尺寸不超过全局尺寸
            node.spacing = [min(min_sp, self.global_spacing) for _ in range(4)]

            # 将子节点加入队列以继续处理
            if node.children:
                queue.extend(node.children)

    def spacing_transition(self):
        """基于相邻节点尺寸的平滑过渡"""

        def update_sizes(node, neighbors):
            """递归标记需要细分的节点"""
            changed = 0
            if node.subflag > 0 and node.children:
                cs = [None] * 4
                # 处理当前节点的子节点
                for n in range(4):
                    for j in range(4):  # 四个方向
                        i = NEIGHBORS_MAP2D[n][j]
                        if i < 0:
                            if neighbors[j] and neighbors[j].subflag <= 0:
                                cs[j] = neighbors[j]
                            else:
                                opposite = j ^ 1  # 取相反方向
                                i = NEIGHBORS_MAP2D[n][opposite]
                                cs[j] = (
                                    neighbors[j].children[i] if neighbors[j] else None
                                )
                        else:
                            cs[j] = node.children[i]

                    changed += update_sizes(node.children[n], cs)

            else:
                for n in range(4):
                    neighbor = neighbors[n]
                    if not neighbor or neighbor.subflag > 0:
                        continue

                    if neighbor.level == node.level:
                        for i in range(4):
                            j = NEIGHBORS_MAP2D[i][n ^ 1]
                            if j >= 0 and neighbor.spacing[j] > node.spacing[i]:
                                neighbor.spacing[j] = node.spacing[i]
                                changed += 1
            return changed

        print(f"尺寸场光滑transition...")
        # 主循环
        changed = 1
        while changed:
            changed = 0
            ndiv = 0

            # 遍历背景网格
            for j in range(self.bg_divisions[1]):
                for i in range(self.bg_divisions[0]):
                    current = self.quad_tree[ndiv]
                    neighbors = [None] * 4  # 西、东、南、北

                    # 获取相邻节点
                    if i > 0:
                        neighbors[0] = self.quad_tree[ndiv - 1]  # 西
                    if i < self.bg_divisions[0] - 1:
                        neighbors[1] = self.quad_tree[ndiv + 1]  # 东
                    if j > 0:
                        neighbors[2] = self.quad_tree[ndiv - self.bg_divisions[0]]  # 南
                    if j < self.bg_divisions[1] - 1:
                        neighbors[3] = self.quad_tree[ndiv + self.bg_divisions[0]]  # 北

                    changed += update_sizes(current, neighbors)
                    ndiv += 1

    def grid_summary(self):
        """输出网格统计信息"""
        print(f"输出背景网格信息...")
        data = {"nCells": 0, "maxLevel": 0}

        enumerate_quadtree(self.quad_tree, CountNode, data)

        print(f"网格统计:")
        print(f"- 总节点数: {data['nCells']}")
        print(f"- 最大深度: {data['maxLevel']}")

    def spacing_at(self, point):
        """计算指定点的网格尺寸（二维双线性插值）"""
        # 定位到包含该点的背景网格根节点
        root_node = None
        for node in self.quad_tree:
            x_min, y_min, x_max, y_max = node.bounds
            if (x_min <= point[0] <= x_max) and (y_min <= point[1] <= y_max):
                root_node = node
                break

        if not root_node:
            raise ValueError(f"Point {point} outside quadtree bounds")

        # 定位到叶节点
        leaf = self._locate_quadtree(point, root_node)

        # 计算局部坐标系参数 (归一化坐标)
        x_min, y_min, x_max, y_max = leaf.bounds
        rx = (point[0] - x_min) / (x_max - x_min) if x_max != x_min else 0.0
        ry = (point[1] - y_min) / (y_max - y_min) if y_max != y_min else 0.0

        # 二维双线性插值（四个角点权重）
        return (
            leaf.spacing[0] * (1 - rx) * (1 - ry)  # 西北角
            + leaf.spacing[1] * rx * (1 - ry)  # 东北角
            + leaf.spacing[2] * (1 - rx) * ry  # 西南角
            + leaf.spacing[3] * rx * ry
        )  # 东南角

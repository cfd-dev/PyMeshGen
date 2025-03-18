from math import log
import math
import matplotlib.pyplot as plt


class QuadTreeNode:
    def __init__(self, bounds):
        self.level = 0
        self.subflag = 0
        self.parent = None
        self.children = None
        self.bounds = bounds  # (x_min, y_min, x_max, y_max)
        self.spacing = [0.0] * 4  # spacing at 4 corners
        self.isvalid = True

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
        """Check if node has zero area"""
        return self.bounds[0] >= self.bounds[2] or self.bounds[1] >= self.bounds[3]


def draw_quadtree(quadtree, ax, marker="g-"):
    """Draw entire quadtree structure"""
    for node in quadtree:
        QuadTreeNode._draw_node(node, ax, marker)
    ax.axis("equal")
    plt.show()


def enumerate_quadtree(quadtree, func, data):
    """
    遍历八叉树的所有节点，无论是否被分割。

    参数:
    octree (list): 八叉树节点列表，每个节点为具有isvalid、subflag和child属性的对象。
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
                child = node.child[i]
                # 递归调用时，将子节点作为单元素列表传入
                enumerate_quadtree(child, func, data)


class QuadtreeSizing:
    def __init__(
        self,
        initial_front=None,
        max_size=1.0,
        resolution=0.1,
        decay=1.2,
        fig=None,
        ax=None,
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

        self.fig = fig
        self.ax = ax
        self.generate_bg_mesh()

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

        draw_quadtree(self.quad_tree, self.ax)

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
            all_points.extend(front.nodes_coords)
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
        """四叉树深度计算"""
        # 计算理论深度
        ratio = self.global_spacing / min_edge_len
        theoretical_depth = math.log(ratio) / math.log(2)  # 换底公式

        # 添加安全裕度（建议1-2层）
        safety_margin = 1

        # self.depth = round(1.4427 * log(ratio) + 0.5)
        self.depth = math.ceil(theoretical_depth) + safety_margin

    def initial_quadtree(self):
        """初始化四叉树网格"""
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

    def refine_quadtree(self):
        """实现基于表面网格尺寸的四叉树细分"""

        def _should_refine(node, target_size):
            """细分条件判断"""
            if node.level >= self.depth:
                return False

            current_spacing = current.spacing[0]
            if current_spacing <= target_size:
                return False
            diff = (current_spacing - target_size) / current_spacing
            return diff > self.resolution

        def _locate_quadtree(point, node):
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

        # 遍历所有表面节点
        for front in self.initial_front:
            face_center = front.front_center
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
                current = _locate_quadtree(face_center, current)

    def level_refinement(self):
        """平衡树层级差异,保证相邻节点之间层级差不大于1"""
        BG_MAP_SIDES = [
            [-1, 1, -1, 2],  # 子节点0 (NW)
            [0, -1, -1, 3],  # 子节点1 (NE)
            [-1, 3, 0, -1],  # 子节点2 (SW)
            [2, -1, 1, -1],  # 子节点3 (SE)
        ]

        def mark_neighbors(node, neighbors):
            """递归标记需要细分的节点"""
            changed = 0
            if node.subflag > 0 and node.children:
                cs = [None] * 4
                # node._draw_node(node, self.ax, "r--")
                # 处理当前节点的子节点
                for n in range(4):
                    for j in range(4):  # 四个方向
                        i = BG_MAP_SIDES[n][j]
                        if i < 0:
                            if neighbors[j] and neighbors[j].subflag <= 0:  # 西
                                cs[j] = neighbors[j]
                                # cs[j]._draw_node(cs[j], self.ax, "b--")
                            else:
                                opposite = j ^ 1  # 取相反方向
                                i = BG_MAP_SIDES[n][opposite]
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

    def traverse_tree(self, node):
        """递归枚举四叉树节点"""
        if not node.children:
            yield node
        else:
            for child in node.children:
                yield from traverse_tree(child)

    def compute_spacing_decay(self):
        """计算考虑表面节点影响的网格尺寸"""

        def source_spacing(base_size, dist, decay):
            """基于指数衰减的尺寸计算"""
            pwr = 0.5 * dist * (decay - 1.0) / base_size
            pwr = min(pwr, 50.0)  # 限制最大指数
            return base_size * math.exp(pwr)

        if self.decay < 1.0:
            return

        # 遍历四叉树所有节点
        for node in self.traverse_tree(self.quad_tree):
            min_sp = float("inf")
            x_center = (node.bounds[0] + node.bounds[2]) / 2
            y_center = (node.bounds[1] + node.bounds[3]) / 2
            # 遍历所有表面节点
            for front in self.initial_front:
                face_center = front.front_center
                target_size = front.length

                # 计算子节点中心到表面节点的距离
                dx = x_center - face_center[0]
                dy = y_center - face_center[1]
                dist = math.sqrt(dx**2 + dy**2)

                # 计算该表面节点影响的尺寸
                sp = source_spacing(target_size, dist, self.decay)
                min_sp = min(min_sp, sp)

            # 限制最小尺寸不超过全局尺寸
            node.sp = min(min_sp, self.global_spacing)

    def spacing_transition(self):
        """基于相邻节点尺寸的平滑过渡"""
        from collections import deque

        # 方向映射：西(0)、东(1)、南(2)、北(3)
        bg_map_sides = [
            [-1, 1, -1, 2],  # 子节点0 (NW)
            [0, -1, -1, 3],  # 子节点1 (NE)
            [-1, 3, 0, -1],  # 子节点2 (SW)
            [2, -1, 1, -1],  # 子节点3 (SE)
        ]

        def get_neighbor(node, direction):
            """通过父节点链获取指定方向的叶节点"""
            if not node.parent:
                return None

            sibling_idx = node.parent.children.index(node)
            neighbor_idx = bg_map_sides[sibling_idx][direction]

            if neighbor_idx != -1:
                return node.parent.children[neighbor_idx]
            else:
                parent_neighbor = get_neighbor(node.parent, direction)
                if parent_neighbor and parent_neighbor.children:
                    return parent_neighbor.children[sibling_idx]
            return None

        # 迭代3次确保收敛
        for _ in range(3):
            updates = []

            # 收集所有叶节点
            leaves = []
            queue = deque([self.quad_tree])
            while queue:
                node = queue.popleft()
                if node.children:
                    queue.extend(node.children)
                else:
                    leaves.append(node)

            # 计算相邻节点尺寸影响
            for node in leaves:
                avg_spacing = node.spacing.copy()
                count = 1

                # 检查四个方向
                for dir in range(4):
                    neighbor = get_neighbor(node, dir)
                    if neighbor and not neighbor.children:
                        # 获取对应方向的角点索引
                        corner_map = [
                            [0, 1],  # 西侧影响左边界（0,2角点）
                            [1, 3],  # 东侧影响右边界（1,3角点）
                            [0, 2],  # 南侧影响下边界（0,1角点）
                            [2, 3],  # 北侧影响上边界（2,3角点）
                        ][dir]

                        # 累加相邻节点对应角点尺寸
                        for idx in corner_map:
                            avg_spacing[idx] += neighbor.spacing[idx]
                            count += 1

                    # 计算平均值并更新
                    for i in range(4):
                        avg_spacing[i] /= count
                        # 平滑过渡系数 (0.2表示保留80%原尺寸，20%邻居影响)
                        node.spacing[i] = node.spacing[i] * 0.8 + avg_spacing[i] * 0.2

                    updates.append((node, avg_spacing))

            # 应用更新
            for node, new_spacing in updates:
                node.spacing = [min(ns, s) for ns, s in zip(new_spacing, node.spacing)]

    def grid_summary(self):
        """输出网格统计信息"""
        node_count = 0
        max_level = 0
        for node in self.traverse_tree(self.quad_tree):
            node_count += 1
            max_level = max(max_level, node.level)

        print(f"网格统计:")
        print(f"- 总节点数: {node_count}")
        print(f"- 最大深度: {max_level}")
        print(f"- 最小尺寸: {self.global_spacing / (2 ** max_level):.4f}")

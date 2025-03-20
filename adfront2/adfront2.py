import heapq
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "data_structure"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))
import geometry_info as geo_info
import front2d
import matplotlib.pyplot as plt


class Unstructed_Grid:
    def __init__(self, cell_nodes, node_coords, boundary_nodes):
        self.dim = None
        self.cell_nodes = cell_nodes
        self.node_coords = node_coords
        self.boundary_nodes = boundary_nodes
        self.num_cells = len(cell_nodes)
        self.num_nodes = len(node_coords)
        self.num_boundary_nodes = len(boundary_nodes)
        self.num_edges = 0
        self.num_faces = 0
        self.edges = []

        self.dim = len(node_coords[0])

    def calculate_edges(self):
        """计算网格的边"""
        edge_set = set()
        for cell in self.cell_nodes:
            for i in range(len(cell)):
                edge = tuple(sorted([cell[i], cell[(i + 1) % len(cell)]]))
                if edge not in edge_set:
                    edge_set.add(edge)

        self.edges = list(edge_set)
        self.num_edges = len(self.edges)

    def visualize_unstr_grid_2d(self):
        """可视化二维网格"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制所有节点
        xs = [n[0] for n in self.node_coords]
        ys = [n[1] for n in self.node_coords]
        ax.scatter(xs, ys, c="gray", s=10, alpha=0.3, label="Nodes")

        # 绘制边
        if self.dim == 2:
            self.calculate_edges()
        for edge in self.edges:
            x = [self.node_coords[i][0] for i in edge]
            y = [self.node_coords[i][1] for i in edge]
            ax.plot(x, y, c="red", alpha=0.5, lw=1.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Unstructured Grid Visualization")
        ax.axis("equal")

        plt.show(block=False)


class Adfront2:
    def __init__(self, boundary_front, sizing_system, ax=None):
        self.ax = ax

        # 阵面推进参数
        self.al = 3.0  # 在几倍范围内搜索
        self.discount = 0.85  # Pbest质量系数，discount越小，选择Pbest的概率越小
        self.mesh_type = 1  # 1-三角形，2-直角三角形，3-三角形/四边形混合
        self.quality_criteria = 0.5  # 单元质量标准，值越大，要求越高
        self.plot_front = False  # 是否实时绘图

        self.front_list = boundary_front  # 初始边界阵面列表
        self.sizing_system = sizing_system  # 尺寸场系统
        self.base_front = None  # 当前基准阵面
        self.pbest = None  #  当前Pbest
        self.pselected = None  # 当前选中的节点
        self.best_flag = False  # 标记是否采用pbest
        self.node_candidates = None  # 候选节点列表
        self.front_candidates = None  # 候选阵面列表
        self.search_radius = None  # 搜索半径

        self.num_cells = 0  # 单元计数器
        self.num_nodes = 0  # 节点计数器

        self.cell_nodes = []  # 单元节点列表
        self.node_coords = []  # 节点坐标列表
        self.node_ids = {}  # 节点ID映射  {id: (x,y)}
        self.node_id_map = {}  # 节点ID映射 {(x,y): id}
        self.boundary_nodes = {}  # 边界节点映射 {id: bc_type}
        self.initialize()

    def initialize(self):
        """初始化节点编号系统"""
        node_id_map = {}  # 存储节点坐标到ID的映射 {(x,y): id}
        self.node_coords = []  # 按ID顺序存储所有节点坐标
        self.node_ids = {}  # 反向映射 {id: (x,y)}
        current_id = 0

        # 存储边界节点的ID及其边界类型
        self.boundary_nodes = {}
        # 遍历所有阵面提取节点
        for front in self.front_list:
            front.node_ids = []
            for node in front.nodes_coords:
                # 将节点坐标转为可哈希的元组
                node_tuple = tuple(
                    round(coord, 6) for coord in node
                )  # 使用6位小数精度去重

                # 为新节点分配ID
                if node_tuple not in node_id_map:
                    node_id_map[node_tuple] = current_id
                    self.node_coords.append(node)  # 保留原始坐标
                    self.node_ids[current_id] = node_tuple
                    self.boundary_nodes[current_id] = front.bc_type
                    current_id += 1

                # 将节点ID添加到阵面上
                front.node_ids.append(node_id_map[node_tuple])

        # 将映射关系保存到实例中
        self.node_id_map = node_id_map

        # 更新节点计数器
        self.num_nodes = current_id

    def draw_front_list(self, ax=None):
        """绘制阵面列表"""
        for front in self.front_list:
            front.draw_front(ax)

    def debug_draw(self, ax):
        """绘制候选节点、候选阵面"""
        if ax and self.plot_front:
            # 绘制基准阵面
            self.base_front.draw_front("r-", self.ax)

            # 绘制Pbest
            # self.ax.plot(self.pbest[0], self.pbest[1], "r.", markersize=10)

            # 绘制虚线圆
            # from matplotlib.patches import Circle

            # self.ax.add_patch(
            #     Circle(
            #         (self.pbest[0], self.pbest[1]),
            #         self.search_radius,
            #         edgecolor="b",
            #         linestyle="--",
            #         fill=False,
            #     )
            # )

            # # 绘制候选节点
            # for node in self.node_candidates:
            #     ax.plot(node[0], node[1], "r.")
            # # 绘制候选阵面
            # for front in self.front_candidates:
            #     front.draw_front("y-", ax)

    def generate_elements(self):
        while self.front_list:
            self.base_front = heapq.heappop(self.front_list)

            spacing = self.sizing_system.spacing_at(self.base_front.front_center)

            self.add_new_point(spacing)

            self.search_candidates(self.al * spacing)

            self.debug_draw(self.ax)

            self.select_point()

            self.update_cells()

            self.show_progress()

        return Unstructed_Grid(self.cell_nodes, self.node_coords, self.boundary_nodes)

    def show_progress(self):
        if self.num_cells % 100 == 0 or len(self.front_list) == 0:
            print(f"当前阵面数量：{len(self.front_list)}")
            print(f"当前节点数量：{self.num_nodes}")
            print(f"当前单元数量：{self.num_cells} \n")

    def update_cells(self):
        # 更新节点
        if self.best_flag and self.pselected is not None:
            node_tuple = tuple(round(coord, 6) for coord in self.pselected)
            # 将pselected加入到node_id_map中去
            if node_tuple not in self.node_id_map:
                self.node_id_map[node_tuple] = self.num_nodes
                self.node_coords.append(self.pselected)  # 保留原始坐标
                self.node_ids[self.num_nodes] = node_tuple
                # 更新节点计数器
                self.num_nodes += 1

        # 更新阵面
        if self.pselected is not None:
            new_front1 = front2d.Front(
                [self.base_front.nodes_coords[0], self.pselected],
                "interior",
                "internal",
            )
            new_front2 = front2d.Front(
                [self.pselected, self.base_front.nodes_coords[1]],
                "interior",
                "internal",
            )

            # 判断front_list中是否存在new_front1，若不存在，
            # 则将其压入front_list，若已存在，则将其从front_list中删除
            # 判断front_list中是否存在相同front_center的阵面
            exists_new1 = any(
                front.front_center == new_front1.front_center
                for front in self.front_list
            )
            exists_new2 = any(
                front.front_center == new_front2.front_center
                for front in self.front_list
            )

            # 使用列表推导式过滤已存在的阵面
            if not exists_new1:
                heapq.heappush(self.front_list, new_front1)
                if self.ax and self.plot_front:
                    new_front1.draw_front("g-", self.ax)
            else:
                # 移除所有相同位置的旧阵面
                self.front_list = [
                    front
                    for front in self.front_list
                    if front.front_center != new_front1.front_center
                ]
                heapq.heapify(self.front_list)  # 重新堆化

            if not exists_new2:
                heapq.heappush(self.front_list, new_front2)
                if self.ax and self.plot_front:
                    new_front2.draw_front("g-", self.ax)
            else:
                self.front_list = [
                    front
                    for front in self.front_list
                    if front.front_center != new_front2.front_center
                ]
                heapq.heapify(self.front_list)

        # 更新单元
        # self.cell_nodes.append(
        #     geo_info.Triangle(
        #         self.base_front.nodes_coords[0],
        #         self.base_front.nodes_coords[1],
        #         self.pselected,
        #     )
        # )
        # 更新单元（使用节点ID）
        node1 = self.node_id_map[
            tuple(round(coord, 6) for coord in self.base_front.nodes_coords[0])
        ]
        node2 = self.node_id_map[
            tuple(round(coord, 6) for coord in self.base_front.nodes_coords[1])
        ]
        node3 = self.node_id_map[tuple(round(coord, 6) for coord in self.pselected)]

        # 检查单元是否已存在（考虑节点顺序不同的情况）
        new_cell = tuple(sorted((node1, node2, node3)))  # 排序节点顺序
        if new_cell not in {tuple(sorted(c)) for c in self.cell_nodes}:
            self.cell_nodes.append((node1, node2, node3))
            self.num_cells += 1
        else:
            print(f"发现重复单元：{new_cell}")

    def select_point(self):
        # 存储带质量的候选节点元组 (质量, 节点)
        scored_candidates = []

        # 遍历所有候选节点计算质量
        for node in self.node_candidates:
            quality = geo_info.triangle_quality(
                self.base_front.nodes_coords[0], self.base_front.nodes_coords[1], node
            )
            scored_candidates.append((quality, node))

        # 添加Pbest节点的质量（带折扣系数）
        pbest_quality = (
            geo_info.triangle_quality(
                self.base_front.nodes_coords[0],
                self.base_front.nodes_coords[1],
                self.pbest,
            )
            if self.pbest
            else 0
        )
        scored_candidates.append((pbest_quality * self.discount, self.pbest))

        # 去掉quality为0的节点
        scored_candidates = [(q, n) for q, n in scored_candidates if q > 0]

        # 按质量降序排序（质量高的在前）
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        for idx, (quality, node) in enumerate(scored_candidates):
            if not geo_info.is_left(
                self.base_front.nodes_coords[0], self.base_front.nodes_coords[1], node
            ):
                continue

            if self.is_cross(node):
                continue

            # 质量不足时，仅允许最后一个候选放宽标准
            # if quality < self.quality_criteria and idx != len(scored_candidates) - 1:
            #     continue

            self.pselected = node

            if self.pselected == self.pbest:
                self.best_flag = True

            return self.pselected

    def is_cross(self, node):
        for front in self.front_candidates:
            front_line = geo_info.LineSegment(
                front.nodes_coords[0], front.nodes_coords[1]
            )
            line1 = geo_info.LineSegment(node, self.base_front.nodes_coords[0])
            line2 = geo_info.LineSegment(node, self.base_front.nodes_coords[1])

            if front_line.is_intersect(line1) or front_line.is_intersect(line2):
                return True

        return False

    def search_candidates(self, radius):
        self.node_candidates = []
        self.front_candidates = []
        self.search_radius = radius
        radius2 = radius * radius

        point = self.pbest
        possible_fronts = []
        for front in self.front_list:
            if (
                point[0] > front.bbox[0] - radius
                and point[0] < front.bbox[1] + radius
                and point[1] > front.bbox[2] - radius
                and point[1] < front.bbox[3] + radius
            ):
                possible_fronts.append(front)

        seen_nodes = set()
        seen_fronts = set()
        for front in possible_fronts:
            front_tuple = tuple(front.front_center)
            for node in front.nodes_coords:
                node_tuple = tuple(node)
                if geo_info.calculate_distance2(point, node) <= radius2:
                    if node_tuple not in seen_nodes:
                        self.node_candidates.append(node)
                        seen_nodes.add(node_tuple)
                    if front_tuple not in seen_fronts:
                        self.front_candidates.append(front)
                        seen_fronts.add(front_tuple)

        return self.node_candidates, self.front_candidates

    def add_new_point(self, spacing):
        normal_vec = geo_info.normal_vector2d(self.base_front)

        # 分量式计算向量相加
        if self.mesh_type == 1:
            fc = self.base_front.front_center
            self.pbest = [
                fc[0] + normal_vec[0] * spacing,
                fc[1] + normal_vec[1] * spacing,
            ]
        elif self.mesh_type == 2:
            node_coord = self.base_front.nodes_coords[0]
            self.pbest = [
                node_coord[0] + normal_vec[0] * spacing,
                node_coord[1] + normal_vec[1] * spacing,
            ]
        elif self.mesh_type == 3:
            pass
        else:
            fc = self.base_front.front_center
            self.pbest = [
                fc[0] + normal_vec[0] * spacing,
                fc[1] + normal_vec[1] * spacing,
            ]

        return self.pbest

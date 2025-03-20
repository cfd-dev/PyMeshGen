import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
import geometry_info as geo_info


class Adfront2:
    def __init__(self, initial_front, sizing_system):
        # 阵面推进参数
        self.al = 3.0  # 在几倍范围内搜索
        self.discount = 0.85  # Pbest质量系数，discount越小，选择Pbest的概率越小
        self.mesh_type = 1  # 1-三角形，2-直角三角形，3-三角形/四边形混合
        self.quality_criteria = 0.5  # 单元质量标准，值越大，要求越高
        self.sort_front = True  # 是否对阵面排序
        self.plot_front = True  # 是否实时绘图

        self.front_list = initial_front
        self.sizing_system = sizing_system
        self.base_front = None
        self.pbest = None
        self.pselected = None
        self.best_flag = False
        self.node_candidates = None
        self.front_candidates = None

        self.num_cells = 0
        self.num_nodes = 0

        self.cell_nodes = []
        self.node_coords = []
        self.node_ids = {}
        self.node_id_map = {}

        self.initialize()

    def initialize(self):
        """初始化节点编号系统"""
        node_id_map = {}  # 存储节点坐标到ID的映射 {(x,y): id}
        self.node_coords = []  # 按ID顺序存储所有节点坐标
        self.node_ids = {}  # 反向映射 {id: (x,y)}
        current_id = 0

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
                    current_id += 1

                # 将节点ID添加到阵面上
                front.node_ids.append(node_id_map[node_tuple])

        # 将映射关系保存到实例中
        self.node_id_map = node_id_map

        # 更新节点计数器
        self.num_nodes = current_id

    def generate_elements(self):
        while self.front_list:
            self.base_front = heapq.heappop(self.front_list)

            spacing = self.sizing_system(self.base_front.front_center)

            self.add_new_point(spacing)

            self.search_candidates(self.al * spacing)

            self.select_point()

            self.update_cells()

    def update_cells(self):
        if self.best_flag and self.pselected is not None:
            node_tuple = tuple(round(coord, 6) for coord in self.pselected)
            self.cell_nodes.append(self.node_id_map.get(node_tuple))

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

        # 按质量降序排序（质量高的在前）
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        for quality, node in scored_candidates:
            if quality < self.quality_criteria:
                continue

            if not geo_info.is_left(
                self.base_front.nodes_coords[0], self.base_front.nodes_coords[1], node
            ):
                continue

            if self.is_cross(node):
                continue

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
            front_tuple = tuple(front.face_center)
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
        if self.mesh_type == 1:
            self.pbest = self.base_front.front_center + normal_vec * spacing
        elif self.mesh_type == 2:
            self.pbest = self.base_front.nodes_coords[0] + normal_vec * spacing
        elif self.mesh_type == 3:
            pass
        else:
            self.pbest = self.base_front.front_center + normal_vec * spacing

        return self.pbest

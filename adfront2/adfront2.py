import heapq
import matplotlib.pyplot as plt

from utils.geom_toolkit import (
    is_left2d,
    calculate_distance2,
)
from optimize.mesh_quality import triangle_shape_quality
from data_structure.front2d import Front
from data_structure.basic_elements import (
    NodeElement,
    Triangle,
    Quadrilateral,
    Unstructured_Grid,
    LineSegment,
)
from utils.timer import TimeSpan
from utils.message import info, debug, verbose, warning, error
from data_structure.rtree_space import (
    build_space_index_with_RTree,
    add_elems_to_space_index_with_RTree,
    get_candidate_elements_id,
    build_space_index_with_cartesian_grid,
    add_elems_to_space_index_with_cartesian_grid,
    get_candidate_elements,
)


class Adfront2:
    def __init__(
        self,
        boundary_front,
        sizing_system,
        node_coords=None,
        param_obj=None,
        visual_obj=None,
    ):
        self.ax = visual_obj.ax
        # 调试级别，0-不输出，1-输出基本信息，2-输出详细信息
        self.debug_level = param_obj.debug_level

        # 阵面推进参数
        self.al = 3.0  # 在几倍范围内搜索
        self.discount = 0.8  # Pbest质量系数，discount越小，选择Pbest的概率越小
        # 1-三角形，2-直角三角形，3-三角形/四边形混合
        self.mesh_type = param_obj.mesh_type
        self.progress_interval = 500  # 进度输出间隔

        self.front_list = boundary_front  # 初始边界阵面列表，堆
        self.sizing_system = sizing_system  # 尺寸场系统对象
        self.param_obj = param_obj  # 参数对象，包含部件参数
        self.base_front = None  # 当前基准阵面
        self.pbest = None  #  当前Pbest，NodeElement对象
        self.pselected = None  # 当前选中的节点，NodeElement对象
        self.best_flag = False  # 标记是否采用pbest

        self.search_radius = None  # 搜索半径
        self.node_candidates = None  # 候选节点列表
        self.front_candidates = None  # 候选阵面列表
        self.cell_candidates = None  # 候选单元列表

        # 搜索方法，"bbox"或"rtree"或"cartesian"，默认为"bbox"
        self.search_method = "bbox"
        self.space_grid_size = 1.0  # 背景网格大小，默认为1.0
        self.space_index_node = None  # 节点空间索引
        self.node_dict = {}  # 节点id字典，用于快速查找
        self.space_index_front = None  # 阵面空间索引
        self.front_dict = {}  # 阵面id字典，用于快速查找
        self.space_index_cell = None  # 单元空间索引
        self.cell_dict = {}  # 单元id字典，用于快速查找

        self.num_cells = 0  # 单元计数器
        self.num_nodes = 0  # 节点计数器

        self.cell_container = None  # 单元对象（Triangle）对象列表
        self.node_coords = node_coords  # 节点坐标列表
        self.front_node_list = None  # 阵面节点列表

        self.boundary_nodes = None  # 边界节点对象列表
        self.unstr_grid = None  # Unstructured_Grid网格对象

        self.node_hash_list = None  # 节点hash列表，用于判断是否重复
        self.cell_hash_list = None  # 单元hash列表，用于判断是否重复

        self.initialize()

    def initialize(self):
        """初始化hash列表、坐标、边界点和优先级"""
        # 初始化搜索半径系数
        for front in self.front_list:
            front.al = self.al

        self.boundary_nodes = set()
        self.node_hash_list = set()
        self.cell_hash_list = set()
        self.cell_container = []
        self.front_node_list = []

        # 如果未传入已生成网格的节点坐标，则生成新的节点坐标
        flag_given_node = True
        if self.node_coords is None:
            flag_given_node = False  # 未给定节点坐标
            self.node_coords = []  # 准备初始化节点坐标

        for front in self.front_list:
            front.priority = True  # 初始化优先级为True
            front.al = self.al  # 初始化搜索半径系数
            for node_elem in front.node_elems:
                if node_elem.hash not in self.node_hash_list:
                    self.node_hash_list.add(node_elem.hash)
                    self.boundary_nodes.add(node_elem)  # 添加边界节点
                    if not flag_given_node:  # 如果未给定节点坐标，则添加节点坐标
                        self.node_coords.append(node_elem.coords)

        self.front_node_list = list(self.boundary_nodes)
        self.num_nodes = len(self.node_coords)

        self.build_space_index()

    def build_space_index(self):
        """构建空间索引"""
        if self.search_method == "cartesian":
            self.build_front_and_cell_cartesian_index()
        elif self.search_method == "rtree":
            self.build_front_and_cell_rtree()

    def build_front_and_cell_cartesian_index(self):
        """构建阵面和单元的笛卡尔网格索引"""
        self.space_grid_size = self.sizing_system.global_spacing
        self.space_index_node = build_space_index_with_cartesian_grid(
            self.front_node_list, self.space_grid_size
        )
        self.space_index_front = build_space_index_with_cartesian_grid(
            self.front_list, self.space_grid_size
        )
        self.space_index_cell = build_space_index_with_cartesian_grid(
            self.cell_container, self.space_grid_size
        )

    def draw_front_list(self, ax=None):
        """绘制阵面列表"""
        for front in self.front_list:
            front.draw_front("r-", self.ax)

    def debug_draw(self):
        """绘制候选节点、候选阵面"""
        if self.base_front.node_ids == [650, 986]:
            self.debug_save()
            kkk = 0

        if not __debug__ or self.ax is None or self.debug_level < 1:
            return

        if self.debug_level >= 1:
            # 绘制基准阵面
            self.base_front.draw_front("r-", self.ax)
            # 绘制Pbest，适应多个pbest的情况
            for point in self.pbest if isinstance(self.pbest, list) else [self.pbest]:
                self.ax.plot(point.coords[0], point.coords[1], "r.", markersize=10)

            # 绘制节点编号
            # for idx, (x, y) in enumerate(self.node_coords):
            #     self.ax.text(x, y, str(i), fontsize=8, ha="center", va="top")

        if self.debug_level >= 2:
            # 绘制虚线圆
            from matplotlib.patches import Circle

            self.ax.add_patch(
                Circle(
                    (self.pbest.coords[0], self.pbest.coords[1]),
                    self.search_radius,
                    edgecolor="b",
                    linestyle="--",
                    fill=False,
                )
            )

            # 绘制候选节点
            for node_elem in self.node_candidates:
                self.ax.plot(node_elem.coords[0], node_elem.coords[1], "r.")
            # 绘制候选阵面
            for front in self.front_candidates:
                front.draw_front("y-", self.ax)

    def generate_elements(self):
        timer = TimeSpan("开始推进生成三角形...")
        while self.front_list:
            self.base_front = heapq.heappop(self.front_list)

            spacing = self.sizing_system.spacing_at(self.base_front.center)

            self.add_new_point(spacing)

            self.search_candidates(self.base_front.al * spacing)

            self.debug_draw()

            self.select_point()

            self.update_data()

            self.show_progress()

        self.construct_unstr_grid()

        timer.show_to_console("完成三角形网格生成.")

        return self.unstr_grid

    def search_candidates(self, search_radius):
        """搜索候选节点和阵面"""
        self.search_radius = search_radius

        if self.search_method == "cartesian":
            self.search_candidates_with_cartesian_grid()
        elif self.search_method == "rtree":
            self.search_candidates_with_rtree()
        elif self.search_method == "bbox":
            self.search_candidates_with_bbox()
        else:
            warning("无效的搜索方法，采用默认bbox搜索方法.")
            self.search_candidates_with_bbox()

    def search_candidates_with_cartesian_grid(self):
        """使用笛卡尔网格搜索候选节点和阵面"""
        self.node_candidates = get_candidate_elements(
            self.base_front,
            self.space_index_node,
            self.space_grid_size,
            self.search_radius,
        )
        self.front_candidates = get_candidate_elements(
            self.base_front,
            self.space_index_front,
            self.space_grid_size,
            self.search_radius,
        )
        self.cell_candidates = get_candidate_elements(
            self.base_front,
            self.space_index_cell,
            self.space_grid_size,
            self.search_radius,
        )

    def build_front_and_cell_rtree(self):
        """构建RTree索引"""
        self.node_dict, self.space_index_node = build_space_index_with_RTree(
            self.front_node_list
        )

        self.front_dict, self.space_index_front = build_space_index_with_RTree(
            self.front_list
        )

        self.cell_dict, self.space_index_cell = build_space_index_with_RTree(
            self.cell_container
        )

    def search_candidates_with_rtree(self):
        """使用RTree索引搜索候选节点和阵面"""
        self.node_candidates = []
        self.front_candidates = []
        self.cell_candidates = []

        # 搜索节点
        node_ids = get_candidate_elements_id(
            self.base_front, self.space_index_node, self.search_radius
        )
        self.node_candidates = [self.node_dict[node_id] for node_id in node_ids]

        # 搜索阵面
        front_ids = get_candidate_elements_id(
            self.base_front, self.space_index_front, self.search_radius
        )
        self.front_candidates = [self.front_dict[front_id] for front_id in front_ids]

        # 搜索单元
        cell_ids = get_candidate_elements_id(
            self.base_front, self.space_index_cell, self.search_radius
        )
        self.cell_candidates = [self.cell_dict[cell_id] for cell_id in cell_ids]

    def construct_unstr_grid(self):
        self.unstr_grid = Unstructured_Grid(
            self.cell_container, self.node_coords, self.boundary_nodes
        )

    def show_progress(self):
        if self.num_cells % self.progress_interval == 0 or len(self.front_list) == 0:
            info(f"当前阵面数量：{len(self.front_list)}")
            info(f"当前节点数量：{self.num_nodes}")
            info(f"当前单元数量：{self.num_cells}\n")

            self.debug_save()

    def add_elems_to_space_index(self, elems, space_index, elem_dict):
        """将新阵面和节点添加到空间索引"""
        if self.search_method == "cartesian":
            add_elems_to_space_index_with_cartesian_grid(
                elems, space_index, self.space_grid_size
            )
        elif self.search_method == "rtree":
            add_elems_to_space_index_with_RTree(elems, space_index, elem_dict)

    def update_data(self):
        if self.pselected is None:
            return

        # Safety check: ensure base front has at least 2 nodes
        if len(self.base_front.node_elems) < 2:
            warning(f"阵面{self.base_front.node_ids}节点数量不足，无法更新数据！")
            return

        # 更新节点
        self.update_nodes()

        # 获取当前基准阵面的部件信息
        part_name = getattr(self.base_front, 'part_name', 'Fluid')  # 默认为'Fluid'

        # 更新阵面
        new_front1 = Front(
            self.base_front.node_elems[0],
            self.pselected,
            -1,
            "interior",
            part_name,  # 使用当前阵面的部件信息
        )
        new_front2 = Front(
            self.pselected,
            self.base_front.node_elems[1],
            -1,
            "interior",
            part_name,  # 使用当前阵面的部件信息
        )

        self.update_fronts([new_front1, new_front2])
        heapq.heapify(self.front_list)  # 重新堆化

        # 更新单元
        new_cell = Triangle(
            self.base_front.node_elems[0],
            self.base_front.node_elems[1],
            self.pselected,
            self.num_cells,
            node_ids=[self.base_front.node_elems[0].idx,
                     self.base_front.node_elems[1].idx,
                     self.pselected.idx]
        )

        # 为新单元添加部件信息
        # 如果当前阵面是边界阵面（part_name不是'Fluid'），则使用阵面的part_name
        # 否则，新生成的内部单元默认为'Fluid'
        if part_name != 'Fluid':
            new_cell.part_name = part_name
        else:
            new_cell.part_name = 'Fluid'

        self.update_cells(new_cell)

    def select_point(self):
        # Safety check: ensure base front has at least 2 nodes
        if len(self.base_front.node_elems) < 2:
            warning(f"阵面{self.base_front.node_ids}节点数量不足，无法进行选择！")
            self.base_front.al *= 1.2
            heapq.heappush(self.front_list, self.base_front)
            return None

        # 预计算基准点坐标
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords

        # 存储带质量的候选节点元组 (质量, 节点)
        scored_candidates = []
        # 遍历所有候选节点计算质量
        for node_elem in self.node_candidates:
            quality = triangle_shape_quality(p0, p1, node_elem.coords)
            if quality > 0:
                scored_candidates.append((quality, node_elem))

        # 添加Pbest节点的质量（带折扣系数）
        pbest_quality = (
            triangle_shape_quality(p0, p1, self.pbest.coords) * self.discount
            if self.pbest
            else 0
        )

        if pbest_quality > 0:
            scored_candidates.append((pbest_quality, self.pbest))

        # 按质量降序排序（质量高的在前）
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        self.pselected = None
        self.best_flag = False
        for quality, node_elem in scored_candidates:
            if not is_left2d(p0, p1, node_elem.coords):
                continue

            if self.is_cross(node_elem):
                continue

            self.pselected = node_elem
            break

        self.best_flag = self.pselected == self.pbest

        if self.pselected == None:
            warning(
                f"阵面{self.base_front.node_ids}候选点列表中没有合适的点，扩大搜索范围！"
            )
            self.base_front.al *= 1.2
            heapq.heappush(self.front_list, self.base_front)  # 重新将基准阵面加入堆中
            # self.debug_level = 1

        if self.base_front.al > 200:
            # 异常退出
            raise Exception("基准阵面搜索半径超过20，可能存在问题")

        return self.pselected

    def debug_save(self):
        if self.debug_level < 1:
            return

        self.construct_unstr_grid()
        self.unstr_grid.save_debug_file(f"cells{self.num_cells}")

    def is_cross(self, node_elem):
        # Safety check: ensure base front has at least 2 nodes
        if len(self.base_front.node_elems) < 2:
            return False  # Can't form a segment with less than 2 nodes

        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]

        line1 = LineSegment(node_elem, p0)
        line2 = LineSegment(node_elem, p1)

        for front in self.front_candidates:
            # Safety check: ensure front has at least 2 nodes
            if len(front.node_elems) < 2:
                continue  # Skip fronts with insufficient nodes
            front_line = LineSegment(front.node_elems[0], front.node_elems[1])
            if front_line.is_intersect(line1) or front_line.is_intersect(line2):
                return True

        cell_to_add = Triangle(node_elem, p0, p1)

        for check_cell in self.cell_candidates:
            if isinstance(check_cell, Quadrilateral):
                if check_cell.is_intersect_triangle(cell_to_add):
                    return True
            elif isinstance(check_cell, Triangle):
                if check_cell.is_intersect(cell_to_add):
                    return True

        return False

    def search_candidates_with_bbox(self):
        self.node_candidates = []
        self.front_candidates = []
        self.cell_candidates = []

        point = []
        # 统一处理单个/多个pbest点
        if isinstance(self.pbest, list) and len(self.pbest) > 1:
            p1, p2 = (p.coords for p in self.pbest[:2])
            point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        else:
            point = self.pbest.coords

        # 使用预计算的边界框信息进行快速筛选
        min_x = point[0] - self.search_radius
        max_x = point[0] + self.search_radius
        min_y = point[1] - self.search_radius
        max_y = point[1] + self.search_radius

        # 使用生成器表达式减少内存使用
        possible_fronts = (
            front
            for front in self.front_list
            if (
                front.bbox[0] <= max_x
                and front.bbox[2] >= min_x
                and front.bbox[1] <= max_y
                and front.bbox[3] >= min_y
            )
        )

        seen_nodes = set()
        seen_fronts = set()
        radius2 = self.search_radius * self.search_radius
        for front in possible_fronts:
            front_hash = front.hash
            for node_elem in front.node_elems:
                node_hash = node_elem.hash
                node_coord = node_elem.coords
                # 若节点在范围内，且没有处理过，则先将节点加入候选点列表，
                # 同时节点所在的阵面也要加入候选阵面列表（如果没有被处理过的话）
                if calculate_distance2(point, node_coord) <= radius2:
                    if node_hash not in seen_nodes:
                        self.node_candidates.append(node_elem)
                        seen_nodes.add(node_hash)
                    if front_hash not in seen_fronts:
                        self.front_candidates.append(front)
                        seen_fronts.add(front_hash)

        self.cell_candidates = [
            cell
            for cell in self.cell_container
            if (
                cell.bbox[0] <= max_x
                and cell.bbox[2] >= min_x
                and cell.bbox[1] <= max_y
                and cell.bbox[3] >= min_y
            )
        ]

    def add_new_point(self, spacing):
        normal_vec = self.base_front.normal

        # 分量式计算向量相加
        fc = self.base_front.center
        if self.mesh_type == 1:
            pbest = [
                fc[0] + normal_vec[0] * spacing,
                fc[1] + normal_vec[1] * spacing,
            ]
        elif self.mesh_type == 2:
            node_coord = self.base_front.node_elems[0].coords
            pbest = [
                node_coord[0] + normal_vec[0] * spacing,
                node_coord[1] + normal_vec[1] * spacing,
            ]
        elif self.mesh_type == 3:
            pass
        else:
            pbest = [
                fc[0] + normal_vec[0] * spacing,
                fc[1] + normal_vec[1] * spacing,
            ]

        # 创建节点时考虑部件信息
        part_name = getattr(self.base_front, 'part_name', 'Fluid')  # 默认为'Fluid'
        self.pbest = NodeElement(
            pbest,
            self.num_nodes,
            part_name=part_name,  # 传递部件信息
            bc_type="interior",
        )

        return self.pbest

    def update_nodes(self):
        """更新节点"""
        # 应对pselected可能为NodeElement，也可能为多个NodeElement的情况，统一转换为list
        # 注意不能直接将self.pselected转换为list，应该后面update_fronts时还需要使用self.pselected
        pselected = (
            [self.pselected]
            if isinstance(self.pselected, NodeElement)
            else self.pselected
        )

        for i, node in enumerate(pselected):
            node_hash = node.hash
            if node_hash not in self.node_hash_list:
                self.node_hash_list.add(node_hash)
                self.node_coords.append(node.coords)
                pselected[i].idx = self.num_nodes
                self.add_elems_to_space_index(
                    [node], self.space_index_node, self.node_dict
                )

                self.num_nodes += 1

    def update_fronts(self, new_fronts):
        """更新阵面"""
        # 判断front_list中是否存在new_front，若不存在，
        # 则将其压入front_list，若已存在，则将其从front_list中删除
        front_hashes = {f.hash for f in self.front_list}
        for chk_fro in new_fronts:
            if chk_fro.hash not in front_hashes:
                self.front_list.append(chk_fro)
                # heapq.heappush(self.front_list, chk_fro)
                self.add_elems_to_space_index(
                    [chk_fro], self.space_index_front, self.front_dict
                )

                if self.ax and self.debug_level >= 1:
                    chk_fro.draw_front("g-", self.ax)
            else:  # 移除相同位置的旧阵面
                self.front_list = [
                    tmp_fro
                    for tmp_fro in self.front_list
                    if tmp_fro.hash != chk_fro.hash
                ]

    def update_cells(self, new_cell):
        """更新单元"""
        if new_cell.hash not in self.cell_hash_list:
            self.cell_hash_list.add(new_cell.hash)
            self.cell_container.append(new_cell)

            self.add_elems_to_space_index(
                [new_cell], self.space_index_cell, self.cell_dict
            )

            self.num_cells += 1
        else:
            warning(f"发现重复单元：{new_cell.node_ids}")
            self.debug_level = 1

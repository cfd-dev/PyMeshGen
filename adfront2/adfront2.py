import heapq
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "data_structure"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))
import geometry_info as geo_info
import front2d
import matplotlib.pyplot as plt
from geometry_info import NodeElement, Unstructured_Grid


class Adfront2:
    def __init__(self, boundary_front, sizing_system, ax=None):
        self.ax = ax

        # 阵面推进参数
        self.al = 3.0  # 在几倍范围内搜索
        self.discount = 0.8  # Pbest质量系数，discount越小，选择Pbest的概率越小
        self.mesh_type = 1  # 1-三角形，2-直角三角形，3-三角形/四边形混合
        # self.quality_criteria = 0.5  # 单元质量标准，值越大，要求越高
        self.plot_front = False  # 是否实时绘图

        self.front_list = boundary_front  # 初始边界阵面列表，堆
        self.sizing_system = sizing_system  # 尺寸场系统对象
        self.base_front = None  # 当前基准阵面
        self.pbest = None  #  当前Pbest，NodeElement对象
        self.pselected = None  # 当前选中的节点，NodeElement对象
        self.best_flag = False  # 标记是否采用pbest
        self.node_candidates = None  # 候选节点列表
        self.front_candidates = None  # 候选阵面列表
        self.cell_candidates = None  # 候选单元列表
        self.search_radius = None  # 搜索半径
        self.debug_switch = False  # 调试模式开关

        self.num_cells = 0  # 单元计数器
        self.num_nodes = 0  # 节点计数器

        self.cell_container = None  # 单元对象（Triangle）对象列表
        self.node_coords = None  # 节点坐标列表

        self.boundary_nodes = None  # 边界节点对象列表
        self.unstr_grid = None  # Unstructured_Grid网格对象

        self.node_hash_list = None  # 节点hash列表，用于判断是否重复
        self.cell_hash_list = None  # 单元hash列表，用于判断是否重复

        self.initialize()

    def initialize(self):
        """初始化hash列表、坐标、边界点和优先级"""
        self.boundary_nodes = set()
        self.node_hash_list = set()
        self.cell_hash_list = set()
        self.cell_container = []
        self.node_coords = []

        # hash_idx_map = {}  # 节点hash值到节点索引的映射
        node_count = 0
        for front in self.front_list:
            # front.node_ids = []
            for node_elem in front.node_elems:
                if node_elem.hash not in self.node_hash_list:
                    # node_elem.idx = node_count
                    # hash_idx_map[node_elem.hash] = node_elem.idx
                    self.node_hash_list.add(node_elem.hash)
                    self.node_coords.append(node_elem.coords)
                    self.boundary_nodes.add(node_elem)
                    node_count += 1
                # else:
                # node_elem.idx = hash_idx_map[node_elem.hash]

                # front.node_ids.append(node_elem.idx)
                front.priority = True

        self.num_nodes = node_count

    def draw_front_list(self, ax=None):
        """绘制阵面列表"""
        for front in self.front_list:
            front.draw_front(ax)

    def debug_draw(self, ax):
        """绘制候选节点、候选阵面"""
        if not __debug__:
            return

        if ax is None or self.plot_front is False:
            return

        # if self.base_front.node_ids == (895, 738):
        #     self.unstr_grid.save_to_vtkfile("./out/debug_output_mesh.vtk")
        #     kkk = 0

        # 绘制基准阵面
        self.base_front.draw_front("r-", self.ax)

        # # 绘制节点编号
        # for idx, (x, y) in enumerate(self.node_coords):
        #     ax.text(x, y, str(i), fontsize=8, ha="center", va="center")

        # 绘制Pbest
        self.ax.plot(self.pbest.coords[0], self.pbest.coords[1], "r.", markersize=10)

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
            ax.plot(node_elem.coords[0], node_elem.coords[1], "r.")
        # 绘制候选阵面
        for front in self.front_candidates:
            front.draw_front("y-", ax)

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

        self.construct_unstructured_grid()

        return self.unstr_grid

    def construct_unstructured_grid(self):
        self.unstr_grid = Unstructured_Grid(
            self.cell_container, self.node_coords, self.boundary_nodes
        )

    def show_progress(self):
        if self.num_cells % 500 == 0 or len(self.front_list) == 0:
            print(f"当前阵面数量：{len(self.front_list)}")
            print(f"当前节点数量：{self.num_nodes}")
            print(f"当前单元数量：{self.num_cells} \n")

            if self.debug_switch:
                self.unstr_grid.save_to_vtkfile("./out/debug_output_mesh.vtk")

    def update_cells(self):
        # 更新节点
        if self.best_flag and self.pselected is not None:
            node_hash = self.pselected.hash
            if node_hash not in self.node_hash_list:
                self.node_hash_list.add(node_hash)
                self.node_coords.append(self.pselected.coords)
                self.num_nodes += 1

        # 更新阵面
        if self.pselected is not None:
            new_front1 = front2d.Front(
                self.base_front.node_elems[0],
                self.pselected,
                -1,
                "interior",
                "internal",
            )
            new_front2 = front2d.Front(
                self.pselected,
                self.base_front.node_elems[1],
                -1,
                "interior",
                "internal",
            )

            # 判断front_list中是否存在new_front，若不存在，
            # 则将其压入front_list，若已存在，则将其从front_list中删除
            exists_new1 = any(
                front.hash == new_front1.hash for front in self.front_list
            )
            exists_new2 = any(
                front.hash == new_front2.hash for front in self.front_list
            )

            if not exists_new1:
                heapq.heappush(self.front_list, new_front1)
                if self.ax and self.plot_front:
                    new_front1.draw_front("g-", self.ax)
            else:
                # 移除相同位置的旧阵面
                self.front_list = [
                    front for front in self.front_list if front.hash != new_front1.hash
                ]
                heapq.heapify(self.front_list)  # 重新堆化

            if not exists_new2:
                heapq.heappush(self.front_list, new_front2)
                if self.ax and self.plot_front:
                    new_front2.draw_front("g-", self.ax)
            else:
                self.front_list = [
                    front for front in self.front_list if front.hash != new_front2.hash
                ]
                heapq.heapify(self.front_list)

        # 更新单元
        new_cell = geo_info.Triangle(
            self.base_front.node_elems[0],
            self.base_front.node_elems[1],
            self.pselected,
            self.num_cells,
        )

        if new_cell.hash not in self.cell_hash_list:
            self.cell_hash_list.add(new_cell.hash)
            self.cell_container.append(new_cell)
            self.num_cells += 1
        else:
            print(f"发现重复单元：{new_cell}")
            self.debug_switch = True

    def select_point(self):
        # 存储带质量的候选节点元组 (质量, 节点)
        scored_candidates = []
        # 遍历所有候选节点计算质量
        for node_elem in self.node_candidates:
            quality = geo_info.triangle_quality(
                self.base_front.node_elems[0].coords,
                self.base_front.node_elems[1].coords,
                node_elem.coords,
            )
            scored_candidates.append((quality, node_elem))

        # 添加Pbest节点的质量（带折扣系数）
        pbest_quality = (
            geo_info.triangle_quality(
                self.base_front.node_elems[0].coords,
                self.base_front.node_elems[1].coords,
                self.pbest.coords,
            )
            if self.pbest
            else 0
        )
        scored_candidates.append((pbest_quality * self.discount, self.pbest))

        # 去掉quality为0的节点
        scored_candidates = [(q, n) for q, n in scored_candidates if q > 0]

        # 按质量降序排序（质量高的在前）
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        self.pselected = None
        self.best_flag = False
        for idx, (quality, node_elem) in enumerate(scored_candidates):
            if not geo_info.is_left2d(
                self.base_front.node_elems[0].coords,
                self.base_front.node_elems[1].coords,
                node_elem.coords,
            ):
                continue

            if self.is_cross(node_elem):
                continue

            self.pselected = node_elem
            break

        if self.pselected == self.pbest:
            self.best_flag = True

        if self.pselected == None:
            self.construct_unstructured_grid()
            self.unstr_grid.save_to_vtkfile("./out/output_mesh.vtk")
            raise (f"候选点列表中没有合适的点，可能需要扩大搜索范围，请检查！")

        return self.pselected

    def is_cross(self, node_elem):
        for front in self.front_candidates:
            front_line = geo_info.LineSegment(front.node_elems[0], front.node_elems[1])
            line1 = geo_info.LineSegment(node_elem, self.base_front.node_elems[0])
            line2 = geo_info.LineSegment(node_elem, self.base_front.node_elems[1])

            if front_line.is_intersect(line1) or front_line.is_intersect(line2):
                return True

        cell_to_add = geo_info.Triangle(
            node_elem,
            self.base_front.node_elems[0],
            self.base_front.node_elems[1],
        )

        for tri_cell in self.cell_candidates:
            if len(tri_cell.node_ids) != 3:
                raise (f"节点数量错误：{tri_cell.node_ids}")

            if tri_cell.is_intersect(cell_to_add):
                return True
        return False

    def search_candidates(self, radius):
        self.node_candidates = []
        self.front_candidates = []
        self.cell_candidates = []

        self.search_radius = radius
        radius2 = radius * radius

        point = self.pbest.coords
        possible_fronts = []
        for front in self.front_list:
            if (
                point[0] > front.bbox[0] - radius  # xmin
                and point[0] < front.bbox[2] + radius  # xmax
                and point[1] > front.bbox[1] - radius  # ymin
                and point[1] < front.bbox[3] + radius  # ymax
            ):
                possible_fronts.append(front)

        seen_nodes = set()
        seen_fronts = set()
        for front in possible_fronts:
            front_hash = front.hash
            for node_elem in front.node_elems:
                node_hash = node_elem.hash
                node_coord = node_elem.coords
                if geo_info.calculate_distance2(point, node_coord) <= radius2:
                    if node_hash not in seen_nodes:
                        self.node_candidates.append(node_elem)
                        seen_nodes.add(node_hash)
                    if front_hash not in seen_fronts:
                        self.front_candidates.append(front)
                        seen_fronts.add(front_hash)

        possible_cells = []
        for cell in self.cell_container:
            if (
                point[0] > cell.bbox[0] - radius  # xmin
                and point[0] < cell.bbox[2] + radius  # xmax
                and point[1] > cell.bbox[1] - radius  # ymin
                and point[1] < cell.bbox[3] + radius  # ymax
            ):
                possible_cells.append(cell)

        seen_cells = set()
        for cell in possible_cells:
            cell_hash = cell.hash
            if cell_hash not in seen_cells:
                self.cell_candidates.append(cell)
                seen_cells.add(cell_hash)

    def add_new_point(self, spacing):
        normal_vec = geo_info.normal_vector2d(self.base_front)

        # 分量式计算向量相加
        fc = self.base_front.front_center
        if self.mesh_type == 1:
            pbest = [
                fc[0] + normal_vec[0] * spacing,
                fc[1] + normal_vec[1] * spacing,
            ]
        elif self.mesh_type == 2:
            node_coord = self.base_front.nodes_coords[0]
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

        self.pbest = NodeElement(
            pbest,
            self.num_nodes,
            "interior",
        )

        return self.pbest

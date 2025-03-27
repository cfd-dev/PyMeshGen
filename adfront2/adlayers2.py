import numpy as np
from math import pi
import sys
from pathlib import Path
import heapq

sys.path.append(str(Path(__file__).parent.parent / "utils"))
sys.path.append(str(Path(__file__).parent.parent / "adfront2"))
from geometry_info import NodeElement, NodeElementALM, Quadrilateral, Unstructured_Grid
from front2d import Front


class Adlayers2:
    def __init__(self, boundary_front, sizing_system, param_obj=None, visual_obj=None):
        self.ax = visual_obj.ax
        self.debug_level = (
            param_obj.debug_level
        )  # 调试级别，0-不输出，1-输出基本信息，2-输出详细信息

        # 层推进全局参数
        self.max_layers = 100  # 最大推进层数
        self.full_layers = 0  # 完整推进层数
        self.multi_direction = False  # 是否多方向推进

        self.initial_front_list = boundary_front  # 初始阵面
        self.sizing_system = sizing_system  # 尺寸场系统对象
        self.part_params = param_obj.part_params  # 层推进部件参数

        self.current_part = None  # 当前推进部件

        self.normal_points = []  # 节点推进方向
        self.normal_fronts = []  # 阵面法向
        self.front_node_list = []  # 当前阵面节点列表
        self.relax_factor = 0.2  # 节点推进方向光滑松弛因子，0-不光滑，1-完全光滑
        self.smooth_iterions = 3  # laplacian光滑次数

        self.ilayer = 0  # 当前推进层数
        self.num_nodes = 0  # 节点数量
        self.num_cells = 0  # 单元数量
        self.cell_container = []  # 单元容器

        self.unstr_grid = None  # 非结构化网格
        self.node_coords = []  # 节点坐标
        self.boundary_nodes = []  # 边界节点
        self.all_boundary_fronts = []  # 所有边界阵面

        self.initialize_nodes()
        self.match_parts_with_fronts()

    def generate_elements(self):
        """生成边界层网格"""
        for part in self.part_params:
            if not part.PRISM_SWITCH:
                continue

            # 将部件参数设置为当前推进参数
            self.current_part = part
            self.max_layers = part.max_layers
            self.full_layers = part.full_layers
            self.multi_direction = part.multi_direction

            for self.ilayer in range(self.max_layers):

                print(f"第{self.ilayer + 1}层推进中...")

                self.prepare_geometry_info()

                self.visualize_point_normals()

                self.calculate_marching_distance()

                self.advancing_fronts()

                self.show_progress()

                self.ilayer += 1

        self.construct_unstr_grid()

        return self.unstr_grid, self.all_boundary_fronts

    def construct_unstr_grid(self):
        """构造非结构化网格"""
        self.unstr_grid = Unstructured_Grid(
            self.cell_container, self.node_coords, self.boundary_nodes
        )

        if self.debug_level == 1:
            self.unstr_grid.save_debug_file(f"layer{self.ilayer + 1}")

        # 汇总所有边界阵面
        self.all_boundary_fronts = []
        for part in self.part_params:
            self.all_boundary_fronts.extend(part.front_list)

        heapq.heapify(self.all_boundary_fronts)

    def show_progress(self):
        """显示推进进度"""
        print(f"第{self.ilayer + 1}层推进..., Done.")
        print(f"当前节点数量：{self.num_nodes}")
        print(f"当前单元数量：{self.num_cells} \n")

        if self.debug_level >= 2:
            self.construct_unstr_grid()
            self.unstr_grid.save_debug_file(f"layer{self.ilayer + 1}")

    def advancing_fronts(self):
        # 逐个部件进行推进
        new_interior_list = []  # 新增的边界层法向面，设置为内部面
        new_prism_cap_list = []  # 新增的边界层流向面

        # 逐个阵面进行推进
        for front in self.current_part.front_list:
            if front.bc_type == "interior":
                # 未推进的阵面仍然加入到新阵面列表中
                new_interior_list.append(front)
                continue

            new_cell_nodes = front.node_elems.copy()

            # 如果early_stop_flag为True，则跳过该阵面
            if new_cell_nodes[0].early_stop_flag or new_cell_nodes[1].early_stop_flag:
                # 未推进的阵面仍然加入到新阵面列表中
                new_prism_cap_list.append(front)
                continue

            # 逐个节点进行推进
            for node_elem in front.node_elems:
                if node_elem.early_stop_flag:
                    continue

                if node_elem.idx == 630:
                    print("stop")

                if node_elem.corresponding_node is None:
                    new_node = (
                        np.array(node_elem.coords)
                        + np.array(node_elem.marching_direction)
                        * node_elem.marching_distance
                    )

                    # TODO: 相交判断

                    # 更新节点坐标
                    self.node_coords.append(new_node.tolist())

                    # 创建新节点元素
                    new_node_elem = NodeElementALM(
                        coords=new_node.tolist(),
                        idx=self.num_nodes,
                        bc_type="interior",
                    )

                    node_elem.corresponding_node = new_node_elem
                    new_cell_nodes.append(new_node_elem)
                    self.num_nodes += 1
                else:
                    new_cell_nodes.append(node_elem.corresponding_node)

            # 创建新阵面
            alm_front = Front(
                new_cell_nodes[2],
                new_cell_nodes[3],
                -1,
                "prism-cap",
                self.current_part.name,
            )
            new_front1 = Front(
                new_cell_nodes[0],
                new_cell_nodes[2],
                -1,
                "interior",
                self.current_part.name,
            )
            new_front2 = Front(
                new_cell_nodes[3],
                new_cell_nodes[1],
                -1,
                "interior",
                self.current_part.name,
            )

            # 检查新阵面是否已经存在于堆中
            # exists_alm = any(front.hash == alm_front.hash for front in new_prism_cap_list)
            new_prism_cap_list.append(alm_front)
            if self.ax and self.debug_level >= 1:
                alm_front.draw_front("g-", self.ax)

            exists_new1 = any(
                front.hash == new_front1.hash for front in new_interior_list
            )
            exists_new2 = any(
                front.hash == new_front2.hash for front in new_interior_list
            )

            if not exists_new1:
                new_interior_list.append(new_front1)
                if self.ax and self.debug_level >= 1:
                    new_front1.draw_front("g-", self.ax)
            else:
                # 移除相同位置的旧阵面
                new_interior_list = [
                    front
                    for front in new_interior_list
                    if front.hash != new_front1.hash
                ]

            if not exists_new2:
                new_interior_list.append(new_front2)
                if self.ax and self.debug_level >= 1:
                    new_front2.draw_front("g-", self.ax)
            else:
                new_interior_list = [
                    front
                    for front in new_interior_list
                    if front.hash != new_front2.hash
                ]

            # 创建新单元
            new_cell = Quadrilateral(
                new_cell_nodes[0],
                new_cell_nodes[1],
                new_cell_nodes[3],
                new_cell_nodes[2],
                self.num_cells,
            )

            self.cell_container.append(new_cell)
            self.num_cells += 1

            # 早停条件判断
            cell_size = new_cell.get_element_size()
            cell_aspect_ratio = new_cell.get_aspect_ratio()
            isotropic_size = self.sizing_system.spacing_at(front.center)
            if (
                cell_size < 1.1 * isotropic_size
                and cell_aspect_ratio <= 1.1
                and self.ilayer >= self.full_layers - 1
            ):
                new_cell_nodes[2].early_stop_flag = True
                new_cell_nodes[3].early_stop_flag = True

        # 更新part阵面列表
        self.current_part.front_list = []
        for front in new_prism_cap_list:
            self.current_part.front_list.append(front)
        for front in new_interior_list:
            self.current_part.front_list.append(front)

    def calculate_marching_distance(self):
        """计算节点推进距离"""
        global_marching_distance = 0.0  # 全局推进距离

        if self.current_part.growth_method == "geometric":
            # 计算几何增长距离
            first_height = self.current_part.first_height
            growth_rate = self.current_part.growth_rate
            global_marching_distance = first_height * growth_rate**self.ilayer

        for front in self.current_part.front_list:
            # 计算节点推进距离
            for node in front.node_elems:
                node.marching_distance = global_marching_distance
                front1, front2 = node.node2front[:2]

                # 节点推进方向与阵面法向的夹角, 节点推进方向投影到面法向
                proj1 = np.dot(node.marching_direction, front1.normal)
                proj2 = np.dot(node.marching_direction, front2.normal)

                if (
                    proj1 * proj2 < 0
                    and front1.bc_type != "interior"
                    and front2.bc_type != "interior"
                ):
                    print(
                        f"node{node.idx}推进方向与相邻阵面法向夹角大于90°，可能出现质量差单元！"
                    )

                # 节点推进距离
                node.marching_distance = (
                    global_marching_distance
                    * node.local_step_factor
                    / np.mean([proj1, proj2])
                )  # min(abs(proj1), abs(proj2))

    def visualize_point_normals(self):
        """可视化节点推进方向"""
        if self.ax is None or self.debug_level < 1:
            return

        for node in self.front_node_list:
            # 绘制front_node_list
            self.ax.plot(
                [node.coords[0]],
                [node.coords[1]],
                "ro",
                markersize=5,
                label="front_node_list",
            )

            # 绘制推进方向
            self.ax.arrow(
                node.coords[0],
                node.coords[1],
                node.marching_direction[0],
                node.marching_direction[1],
                head_width=0.05,
                head_length=0.1,
                fc="k",
                ec="k",
            )

    def compute_point_normals(self):
        """计算节点推进方向"""
        for node_elem in self.front_node_list:
            if len(node_elem.node2front) < 2:
                continue

            # 对于凸角点，在此不计算，也不光滑
            if len(node_elem.marching_direction) > 1:
                continue

            front1, front2 = node_elem.node2front[:2]
            normal1 = np.array(front1.normal)
            normal2 = np.array(front2.normal)

            # 应对节点只有一侧有流向阵面，另一侧是法向阵面的情况
            if front1.bc_type == "interior":
                node_elem.marching_direction = tuple(normal2)
                continue
            elif front2.bc_type == "interior":
                node_elem.marching_direction = tuple(normal1)
                continue

            # 计算初始推进方向（法向量平均）
            avg_direction = (normal1 + normal2) / 2.0
            norm = np.linalg.norm(avg_direction)
            avg_direction /= norm

            new_direction = avg_direction if norm > 1e-6 else normal1

            # 加权光滑
            w1 = self.relax_factor
            wf1 = 0.5
            wf2 = 1 - wf1

            iterations = 0
            max_iterations = 50
            smooth = True
            while smooth and iterations < max_iterations:
                iterations += 1
                old_direction = new_direction.copy()

                new_direction = (1 - w1) * old_direction + w1 * (
                    wf1 * normal1 + +wf2 * normal2
                )
                new_direction /= np.linalg.norm(new_direction)

                # 计算法向与相邻面的夹角及其与平均夹角的偏差df
                dot_product1 = np.dot(new_direction, normal1)
                dot_product2 = np.dot(new_direction, normal2)
                angle1 = np.arccos(np.clip(dot_product1, -1.0, 1.0))  # 添加数值裁剪
                angle2 = np.arccos(np.clip(dot_product2, -1.0, 1.0))  # 添加数值裁剪

                avg_angle = (angle1 + angle2) / 2

                # 计算夹角偏差
                df1 = abs(angle1 - avg_angle)
                df2 = abs(angle2 - avg_angle)

                # 更新权重
                epsilon = 1e-10
                wf1_bar = wf1 * (1 - df1 / (avg_angle + epsilon))
                wf2_bar = wf2 * (1 - df2 / (avg_angle + epsilon))
                wf1 = wf1_bar + wf1 * (1 - (wf1_bar + wf2_bar))
                wf2 = wf2_bar + wf2 * (1 - (wf1_bar + wf2_bar))

                if np.linalg.norm(new_direction - old_direction) < 1e-3:
                    smooth = False

            node_elem.marching_direction = tuple(new_direction)

    def laplacian_smooth_normals(self):
        """拉普拉斯平滑节点推进方向"""
        for node_elem in self.front_node_list:
            num_neighbors = len(node_elem.node2node)
            if num_neighbors < 2:
                continue

            iteration = 0
            while iteration < self.smooth_iterions:
                summation = np.zeros_like(np.array(node_elem.marching_direction))
                for neighbor in node_elem.node2node:
                    summation += np.array(neighbor.marching_direction)

                new_direction = (1 - self.relax_factor) * np.array(
                    node_elem.marching_direction
                ) + self.relax_factor / num_neighbors * summation

                node_elem.marching_direction = tuple(
                    new_direction / np.linalg.norm(new_direction)
                )
                iteration += 1

    def prepare_geometry_info(self):
        """准备几何信息"""
        self.compute_front_geometry()

        self.compute_point_normals()

        self.laplacian_smooth_normals()

    def initialize_nodes(self):
        """初始化节点"""
        processed_nodes = set()
        for front in self.initial_front_list:
            for node_elem in front.node_elems:
                if node_elem.hash not in processed_nodes:
                    self.boundary_nodes.append(node_elem)
                    self.node_coords.append(node_elem.coords)
                    processed_nodes.add(node_elem.hash)

        self.num_nodes = len(self.node_coords)

    def compute_front_geometry(self):
        """计算阵面几何信息"""
        # node2front
        self.front_node_list = []
        processed_nodes = set()
        hash_idx_map = {}  # 节点hash值到节点索引的映射
        for front in self.current_part.front_list:
            for i, node_elem in enumerate(front.node_elems):
                if node_elem.hash not in processed_nodes:
                    if not isinstance(node_elem, NodeElementALM):
                        # 将所有节点均转换为NodeElementALM类型
                        front.node_elems[i] = NodeElementALM.from_existing_node(
                            node_elem
                        )

                    processed_nodes.add(node_elem.hash)
                    hash_idx_map[node_elem.hash] = front.node_elems[i]

                    # 为方便对节点进行遍历，收集所有节点
                    self.front_node_list.append(front.node_elems[i])
                else:
                    # 处理过的节点，直接取hash值对应的NodeElementALM对象
                    front.node_elems[i] = hash_idx_map[node_elem.hash]

                front.node_elems[i].node2front.append(front)

        # 计算node2node
        for front in self.current_part.front_list:
            nodes = front.node_elems
            num_nodes = len(nodes)
            for i, node in enumerate(nodes):
                # 环形处理相邻节点索引
                prev_index = (i - 1) % num_nodes
                next_index = (i + 1) % num_nodes
                prev_node = nodes[prev_index]
                next_node = nodes[next_index]

                # 获取当前节点已存在的哈希集合
                existing_hashes = {n.hash for n in node.node2node}

                # 添加前节点（排除自身且未添加过的节点）
                if (
                    prev_node.hash != node.hash
                    and prev_node.hash not in existing_hashes
                ):
                    node.node2node.append(prev_node)
                    existing_hashes.add(prev_node.hash)

                # 添加后节点（排除自身且未添加过的节点）
                if (
                    next_node.hash != node.hash
                    and next_node.hash not in existing_hashes
                ):
                    node.node2node.append(next_node)
                    existing_hashes.add(next_node.hash)

        # 计算阵面间夹角、凹凸标记、局部步长因子、多方向推进数量
        for node_elem in self.front_node_list:
            if len(node_elem.node2front) < 2:
                continue

            front1, front2 = node_elem.node2front[:2]
            normal1 = np.array(front1.normal)
            normal2 = np.array(front2.normal)
            # 计算夹角（0-180度）
            cos_theta = np.dot(normal1, normal2) / (
                np.linalg.norm(normal1) * np.linalg.norm(normal2)
            )
            node_elem.angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            # 判断凹凸性（通过叉积符号）
            # 判断连接顺序：front2在前，front1在后，则交换normal顺序，便于叉乘
            if front2.node_ids[1] == front1.node_ids[0]:
                temp = normal2
                normal2 = normal1
                normal1 = temp

            thetam = 0  # 局部步长因子计算中的夹角
            cross = np.cross(normal1, normal2)
            if cross < -1e-6:  # 凸角
                node_elem.convex_flag = True
                node_elem.concav_flag = False
                thetam = np.radians(node_elem.angle)
            elif cross > 1e-6:  # 凹角
                node_elem.convex_flag = False
                node_elem.concav_flag = True
                thetam = -np.radians(node_elem.angle)
            else:  # 共线
                node_elem.convex_flag = False
                node_elem.concav_flag = False

            # 计算多方向推进数量和局部步长因子
            if self.multi_direction and node_elem.convex_flag:
                node_elem.num_multi_direction = (
                    int(np.radians(node_elem.angle) / (1.1 * pi / 3)) + 1
                )
                delta = np.radians(node_elem.angle) / (
                    node_elem.num_multi_direction - 1
                )
                initial_vectors = normal1

                for i in range(node_elem.num_multi_direction):
                    angle = -i * delta
                    rotation_matrix = np.array(
                        [
                            [np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)],
                        ]
                    )
                    rotated_vector = np.dot(rotation_matrix, initial_vectors)
                    node_elem.marching_direction.append(tuple(rotated_vector))

                node_elem.local_step_factor = 1.0
            else:
                node_elem.num_multi_direction = 1
                node_elem.local_step_factor = 1 - np.sign(thetam) * abs(thetam) / pi

    def match_parts_with_fronts(self):
        """匹配部件和初始阵面"""
        for front in self.initial_front_list:
            for part in self.part_params:
                if part.name == front.part_name:
                    part.front_list.append(front)
                    break

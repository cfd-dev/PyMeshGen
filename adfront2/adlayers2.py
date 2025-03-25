import numpy as np
from math import pi
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
sys.path.append(str(Path(__file__).parent.parent / "adfront2"))
from geometry_info import NodeElement, NodeElementALM, Quadrilateral, Unstructured_Grid
from front2d import Front


class PartMeshParameters:
    """网格生成部件参数"""

    def __init__(
        self,
        name,
        max_size=1.0,
        PRISM_SWITCH=False,
        first_height=0.1,
        growth_rate=1.2,
        growth_method="geometric",
    ):
        self.name = name  # 部件名称
        self.max_size = max_size  # 最大网格尺寸
        self.PRISM_SWITCH = PRISM_SWITCH  # 是否生成边界层网格
        self.first_height = first_height  # 第一层网格高度
        self.growth_rate = growth_rate  # 网格高度增长比例
        self.growth_method = growth_method  # 网格高度增长方法
        self.front_heap = []  # 阵面堆


class Adlayers2:
    def __init__(self, part_params, initial_front, ax=None):
        self.ax = ax
        # 层推进全局参数
        self.max_layers = 100  # 最大推进层数
        self.full_layers = 0  # 完整推进层数
        self.multi_direction = False  # 是否多方向推进

        self.initial_front = initial_front  # 初始阵面堆
        self.part_params = part_params  # 层推进部件参数
        self.normal_points = []  # 节点推进方向
        self.normal_fronts = []  # 阵面法向
        self.front_node_list = []  # 当前阵面节点列表
        self.relax_factor = 0.5  # 节点坐标松弛因子
        self.smooth_iterions = 500  # laplacian光滑次数

        self.ilayer = 0  # 当前推进层数
        self.num_nodes = 0  # 节点数量
        self.num_cells = 0  # 单元数量
        self.cell_container = []  # 单元容器
        self.debug_switch = True  # 是否调试模式
        self.unstr_grid = None  # 非结构化网格
        self.node_coords = []  # 节点坐标
        self.boundary_nodes = []  # 边界节点

    def generate_elements(self):
        """生成边界层网格"""
        for self.ilayer in range(self.max_layers):

            print("第{}层推进...".format(self.ilayer + 1))

            self.prepare_geometry_info()

            self.visualize_point_normals()

            self.calculate_marching_distance()

            self.advancing_fronts()

            self.show_progress()

            self.ilayer += 1

        self.construct_unstr_grid()

        return unstr_grid

    def construct_unstr_grid(self):
        """构造非结构化网格"""
        self.unstr_grid = Unstructured_Grid(
            self.cell_container, self.node_coords, self.boundary_nodes
        )

    def show_progress(self):
        """显示推进进度"""
        print("第{}层推进...,Done.".format(self.ilayer + 1))
        print(f"当前节点数量：{self.num_nodes}")
        print(f"当前单元数量：{self.num_cells} \n")

        if self.debug_switch:
            self.construct_unstr_grid()
            self.unstr_grid.save_to_vtkfile("./out/debug_output_mesh.vtk")

    def advancing_fronts(self):
        # 逐个部件进行推进
        for part in self.part_params:
            if not part.PRISM_SWITCH:
                continue

            new_interior_list = []
            new_prism_cap_list = []

            # 逐个阵面进行推进
            for front in part.front_heap:
                if front.bc_type == "interior":
                    continue

                new_cell_nodes = front.node_elems.copy()
                # 逐个节点进行推进
                for node_elem in front.node_elems:
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
                    part.name,
                )
                new_front1 = Front(
                    new_cell_nodes[0], new_cell_nodes[2], -1, "interior", part.name
                )
                new_front2 = Front(
                    new_cell_nodes[3], new_cell_nodes[1], -1, "interior", part.name
                )

                # 检查新阵面是否已经存在于堆中
                # exists_alm = any(front.hash == alm_front.hash for front in part.front_heap)
                new_prism_cap_list.append(alm_front)
                if self.ax and self.debug_switch:
                    alm_front.draw_front("g-", self.ax)

                exists_new1 = any(
                    front.hash == new_front1.hash for front in new_interior_list
                )
                exists_new2 = any(
                    front.hash == new_front2.hash for front in new_interior_list
                )

                if not exists_new1:
                    new_interior_list.append(new_front1)
                    if self.ax and self.debug_switch:
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
                    if self.ax and self.debug_switch:
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

            part.front_heap = []
            for front in new_prism_cap_list:
                part.front_heap.append(front)
            for front in new_interior_list:
                part.front_heap.append(front)

    def calculate_marching_distance(self):
        """计算节点推进距离"""
        global_marching_distance = 0.0  # 全局推进距离

        for part in self.part_params:
            if not part.PRISM_SWITCH:
                continue

            if part.growth_method == "geometric":
                # 计算几何增长距离
                global_marching_distance = (
                    part.first_height * part.growth_rate**self.ilayer
                )

            for front in part.front_heap:
                # 计算节点推进距离
                for node in front.node_elems:
                    node.marching_distance = global_marching_distance

                    front1, front2 = node.node2front[:2]

                    # 节点推进方向与阵面法向的夹角, 节点推进方向投影到面法向
                    proj1 = np.dot(node.marching_direction, front1.normal)
                    proj2 = np.dot(node.marching_direction, front2.normal)

                    # 节点推进距离
                    node.marching_distance = (
                        global_marching_distance
                        * node.local_step_factor
                        / min(proj1, proj2)
                    )

    def visualize_point_normals(self):
        """可视化节点推进方向"""
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

            front1, front2 = node_elem.node2front[:2]
            normal1 = np.array(front1.normal)
            normal2 = np.array(front2.normal)

            # 计算推进方向（法向量平均）
            avg_direction = (normal1 + normal2) / 2.0
            norm = np.linalg.norm(avg_direction)
            if norm > 1e-6:
                node_elem.marching_direction = tuple(avg_direction / norm)
            else:
                node_elem.marching_direction = tuple(normal1)

            # 加权光滑
            w1 = self.relax_factor
            wf1 = (1 - self.relax_factor) / 2
            wf2 = 1 - w1 - wf1

            iterations = 0
            smooth = True
            while smooth:
                iterations += 1
                old_direction = np.array(node_elem.marching_direction)

                node_elem.marching_direction = tuple(
                    w1 * old_direction + wf1 * normal1 + +wf2 * normal2
                ) / np.linalg.norm(w1 * old_direction + wf1 * normal1 + +wf2 * normal2)

                # 计算法向与相邻面的夹角及其与平均夹角的偏差df
                angle1 = np.arccos(np.dot(node_elem.marching_direction, front1.normal))
                angle2 = np.arccos(np.dot(node_elem.marching_direction, front2.normal))
                avg_angle = (angle1 + angle2) / 2

                # 计算夹角偏差
                df1 = abs(angle1 - avg_angle)
                df2 = abs(angle2 - avg_angle)

                epsilon = 1e-10
                wf1_bar = wf1 * (1 - df1 / (avg_angle + epsilon))
                wf2_bar = wf2 * (1 - df2 / (avg_angle + epsilon))
                wf1 = wf1_bar + wf1 * (1 - (wf1_bar + wf2_bar))
                wf2 = wf2_bar + wf2 * (1 - (wf1_bar + wf2_bar))

                if (
                    np.linalg.norm(
                        np.array(node_elem.marching_direction) - old_direction
                    )
                    < 1e-3
                ):
                    smooth = False

    def laplacian_smooth_point_normals(self):
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

        self.laplacian_smooth_point_normals()

        self.match_parts_with_fronts()

    def compute_front_geometry(self):
        """计算阵面几何信息"""
        # node2front
        self.front_node_list = []
        processed_nodes = set()
        hash_idx_map = {}  # 节点hash值到节点索引的映射
        for front in self.initial_front:
            for i, node_elem in enumerate(front.node_elems):
                if (
                    not isinstance(node_elem, NodeElementALM)
                    and node_elem.hash not in processed_nodes
                ):
                    front.node_elems[i] = NodeElementALM.from_existing_node(node_elem)
                    processed_nodes.add(node_elem.hash)
                    hash_idx_map[node_elem.hash] = front.node_elems[i]
                    self.front_node_list.append(front.node_elems[i])

                    self.node_coords.append(node_elem.coords)
                    self.boundary_nodes.append(node_elem)  # 边界节点
                else:
                    front.node_elems[i] = hash_idx_map[node_elem.hash]

                front.node_elems[i].node2front.append(front)

        self.num_nodes = len(self.front_node_list)

        # 计算node2node
        for front in self.initial_front:
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
                delta = np.radians(node_elem.angle) / (num_multi_direction - 1)
                initial_vectors = normal1

                for i in range(node_elem.num_multi_direction):
                    angle = i * delta
                    rotation_matrix = np.array(
                        [
                            [np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)],
                        ]
                    )
                    rotated_vector = np.dot(rotation_matrix, initial_vectors)
                    node_elem.multi_direction.append(tuple(rotated_vector))

                node_elem.local_step_factor = 1.0
            else:
                node_elem.num_multi_direction = 1
                node_elem.local_step_factor = 1 - np.sign(thetam) * abs(thetam) / pi

        kkk = 1

        # if self.multi_direction:
        #     if node_elem.convex_flag:
        #         node_elem.num_multi_direction = (
        #             int(np.radians(node_elem.angle) / (1.1 * pi / 3)) + 1
        #         )
        #         node_elem.local_step_factor = 1.0
        #     elif node_elem.concav_flag:
        #         node_elem.local_step_factor = 1 - sign(thetam) * abs(thetam) / pi
        # else:
        #     node_elem.num_multi_direction = 1
        #     node_elem.local_step_factor = 1 - sign(thetam) * abs(thetam) / pi

    def match_parts_with_fronts(self):
        """匹配部件和阵面堆"""
        for front in self.initial_front:
            for part in self.part_params:
                if part.name == front.part_name:
                    part.front_heap.append(front)
                    break

import numpy as np
from math import pi
import sys
from pathlib import Path
import heapq

sys.path.append(str(Path(__file__).parent.parent / "utils"))
sys.path.append(str(Path(__file__).parent.parent / "adfront2"))
from geometry_info import (
    NodeElement,
    NodeElementALM,
    Quadrilateral,
    Unstructured_Grid,
    min_distance_between_segments,
    segments_intersect,
)
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
        self.current_step_size = 0  # 当前推进步长
        self.space_grid = None  # 空间查询网格
        self.grid_size = None  # 查询网格尺寸

        self.normal_points = []  # 节点推进方向
        self.normal_fronts = []  # 阵面法向
        self.front_node_list = []  # 当前阵面节点列表
        self.relax_factor = 0.2  # 节点推进方向光滑松弛因子，0-不光滑，1-完全光滑
        self.smooth_iterions = 3  # laplacian光滑次数
        self.quality_threshold = 0.005  # 四边形单元质量阈值，不能小于该值，否则会被删除

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

        # self.draw_prism_cap()

        return self.unstr_grid, self.all_boundary_fronts

    def draw_prism_cap(self):
        for front in self.all_boundary_fronts:
            front.draw_front("r-", self.ax, linewidth=3)

    def construct_unstr_grid(self):
        """构造非结构化网格"""
        self.unstr_grid = Unstructured_Grid(
            self.cell_container, self.node_coords, self.boundary_nodes
        )

        if self.debug_level == 1:
            self.debug_save()

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
            self.debug_save()

    def debug_save(self):
        if self.debug_level < 1:
            return

        self.construct_unstr_grid()
        self.unstr_grid.save_debug_file(f"layer{self.ilayer + 1}")

    def advancing_fronts(self):
        new_interior_list = []  # 新增的边界层法向面，设置为interior
        new_prism_cap_list = []  # 新增的边界层流向面，设置为prism-cap

        # 逐个阵面进行推进
        for front in self.current_part.front_list:
            if front.bc_type == "interior":
                new_interior_list.append(front)  # 未推进的阵面仍然加入到新阵面列表中
                continue

            if front.early_stop_flag:
                new_prism_cap_list.append(front)  # 未推进的阵面仍然加入到新阵面列表中
                continue

            # 逐个节点进行推进，此时生成的均为临时的，只有确定有效后才会加入到真实数据中
            new_cell_nodes = front.node_elems.copy()
            new_node_generated = [None, None]  # 记录新节点是否生成
            temp_num_nodes = self.num_nodes  # 记录当前节点数量
            for i, node_elem in enumerate(front.node_elems):
                if node_elem.corresponding_node is None:
                    # 推进生成一个新点
                    new_node = (
                        np.array(node_elem.coords)
                        + np.array(node_elem.marching_direction)
                        * node_elem.marching_distance
                    )

                    # 创建临时新节点元素
                    new_node_elem = NodeElementALM(
                        coords=new_node.tolist(),
                        idx=temp_num_nodes,
                        bc_type="interior",
                    )

                    temp_num_nodes += 1
                    new_node_generated[i] = new_node_elem
                    new_cell_nodes.append(new_node_elem)
                else:
                    # 当前节点已经推进过了，找出其对应的新节点
                    new_cell_nodes.append(node_elem.corresponding_node)

            # 创建临时新阵面
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

            # 创建新单元
            new_cell = Quadrilateral(
                new_cell_nodes[0],
                new_cell_nodes[1],
                new_cell_nodes[3],
                new_cell_nodes[2],
                self.num_cells,
            )

            # 检查新单元质量，质量不合格则当前front早停
            quality = new_cell.get_skewness()
            if quality < self.quality_threshold:
                front.early_stop_flag = True
                new_prism_cap_list.append(front)
                continue

            # 单元大小、长宽比、full_layer判断早停
            cell_size = new_cell.get_element_size()
            isotropic_size = self.sizing_system.spacing_at(front.center)
            # TODO 限制调整头部和尾部的单元size：[1.2-1.5]
            size_factor = 1.3
            size_condition = cell_size > size_factor * isotropic_size

            # 单元长宽比<1.1
            cell_aspect_ratio = new_cell.get_aspect_ratio2()
            aspect_ratio_condition = cell_aspect_ratio < 1.1

            # 当前层数 > full_layer
            full_layer_condition = self.ilayer >= self.full_layers

            # 长宽比<1.1且当前层数>full_layer，则当前front早停
            if aspect_ratio_condition and full_layer_condition:
                front.early_stop_flag = True
                new_prism_cap_list.append(front)
                continue

            # 单元size>1.5*isotropic_size且当前层数>full_layer，则当前front早停
            if size_condition and full_layer_condition:
                front.early_stop_flag = True
                new_prism_cap_list.append(front)
                continue

            # 邻近检查，检查3个阵面附近是否有其他阵面，若有，则对当前阵面进行早停
            all_fronts = []
            seen = set()
            temp_list = (
                self.current_part.front_list + new_interior_list + new_prism_cap_list
            )
            for tmp_front in temp_list:
                if tmp_front.hash not in seen:
                    seen.add(tmp_front.hash)
                    all_fronts.append(tmp_front)

            check_fronts = [alm_front, new_front1, new_front2]
            for new_front in check_fronts:

                if front.node_ids == (4, 5) or front.node_ids == (61, 62):
                    print("")

                if new_front.node_ids == (926, 886):
                    print("")

                # 长边阵面扩大搜索范围
                # search_range = 2 if new_front.length > safe_distance * 2 else 1
                search_range = 1

                x_coords = [n.coords[0] for n in new_front.node_elems]
                y_coords = [n.coords[1] for n in new_front.node_elems]

                # 扩展搜索范围
                i_min = int((min(x_coords) - self.grid_size) // self.grid_size)
                i_max = int((max(x_coords) + self.grid_size) // self.grid_size)
                j_min = int((min(y_coords) - self.grid_size) // self.grid_size)
                j_max = int((max(y_coords) + self.grid_size) // self.grid_size)

                # 检查相邻网格中的阵面
                seen_candidates = set()
                for i in range(i_min - search_range, i_max + search_range + 1):
                    for j in range(j_min - search_range, j_max + search_range + 1):
                        for candidate in self.space_grid.get((i, j), []):
                            # 避免重复检查
                            if candidate.hash in seen_candidates:
                                continue
                            seen_candidates.add(candidate.hash)

                            if candidate.node_ids == (882, 883):
                                print("")

                            # 若candidate与new_front共点，则跳过
                            if any(
                                id in new_front.node_ids for id in candidate.node_ids
                            ):
                                continue

                            # 邻近阵面检查，new_front与candidate的距离小于safe_distance，则对当前front进行早停
                            # safe_distance通常取为当前推进步长的0.8倍
                            dis = self._fronts_distance(new_front, candidate)
                            safe_distance = 0.5 * min(
                                front.node_elems[0].marching_distance,
                                front.node_elems[1].marching_distance,
                            )

                            if dis < safe_distance:
                                print(
                                    f"邻近信息：阵面{new_front.node_ids}与{candidate.node_ids}距离小于{round(safe_distance,6)}"
                                )
                                front.early_stop_flag = True

                                if self.debug_level >= 1:
                                    self._highlight_intersection(new_front, candidate)
                                break

                            # 精确相交检测，无需再进行，注释掉
                            # if self._segments_intersect(
                            #     new_front.node_elems, candidate.node_elems
                            # ):
                            #     print(
                            #         f"检测到相交：{new_front.node_ids} 与 {candidate.node_ids}"
                            #     )
                            #     front.early_stop_flag = True
                            #     break

                        if front.early_stop_flag:
                            break
                    if front.early_stop_flag:
                        break
                if front.early_stop_flag:
                    break

            if front.early_stop_flag:
                new_prism_cap_list.append(front)
                continue

            # 若没有早停，则更新节点、单元和阵面列表
            # 更新节点：检查新节点是否生成，若生成，则加入到节点列表中
            # 注意corresponding_node也要在此更新，而不是在其他地方更新
            for i in range(2):
                if new_node_generated[i] is not None:
                    front.node_elems[i].corresponding_node = new_node_generated[i]
                    self.node_coords.append(new_cell_nodes[i + 2].coords)
                    self.num_nodes += 1

            # 更新单元
            self.cell_container.append(new_cell)
            self.num_cells += 1

            # 更新阵面列表
            new_prism_cap_list.append(alm_front)
            if self.ax and self.debug_level >= 1:
                alm_front.draw_front("g-", self.ax)

            exists_new1 = any(
                tmp_front.hash == new_front1.hash for tmp_front in new_interior_list
            )
            exists_new2 = any(
                tmp_front.hash == new_front2.hash for tmp_front in new_interior_list
            )

            if not exists_new1:
                new_interior_list.append(new_front1)
                self._add_front_to_space([new_front1])
                if self.ax and self.debug_level >= 1:
                    new_front1.draw_front("g-", self.ax)
            else:
                # 移除相同位置的旧阵面
                new_interior_list = [
                    tmp_front
                    for tmp_front in new_interior_list
                    if tmp_front.hash != new_front1.hash
                ]

            if not exists_new2:
                new_interior_list.append(new_front2)
                self._add_front_to_space([new_front2])
                if self.ax and self.debug_level >= 1:
                    new_front2.draw_front("g-", self.ax)
            else:
                new_interior_list = [
                    tmp_front
                    for tmp_front in new_interior_list
                    if tmp_front.hash != new_front2.hash
                ]

        # 更新part阵面列表
        self.current_part.front_list = []
        for tmp_front in new_prism_cap_list:
            self.current_part.front_list.append(tmp_front)
        for tmp_front in new_interior_list:
            self.current_part.front_list.append(tmp_front)

    def calculate_marching_distance(self):
        """计算节点推进距离"""
        self.current_step_size = 0.0

        if self.current_part.growth_method == "geometric":
            # 计算几何增长距离
            first_height = self.current_part.first_height
            growth_rate = self.current_part.growth_rate
            self.current_step_size = first_height * growth_rate**self.ilayer
        else:
            raise ValueError("未知的步长计算方法！")

        for front in self.current_part.front_list:
            # 计算节点推进距离
            for node in front.node_elems:
                node.marching_distance = self.current_step_size
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
                    self.current_step_size
                    * node.local_step_factor
                    / np.mean([proj1, proj2])
                )  # min(abs(proj1), abs(proj2))

    def visualize_point_normals(self):
        """可视化节点推进方向"""
        if self.ax is None or self.debug_level < 3:
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

            # TODO 应对节点只有一侧有流向阵面，另一侧是法向阵面的情况
            if front1.bc_type == "interior":
                node_elem.marching_direction = tuple(normal2)
                # node_elem.marching_direction = tuple(front2.direction)
                continue
                # normal1 = np.array(front2.direction)
            elif front2.bc_type == "interior":
                node_elem.marching_direction = tuple(normal1)
                # node_elem.marching_direction = tuple(front1.direction)
                continue
                # normal2 = np.array(front1.direction)

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
        # 建立矩形背景网格，便于快速查询
        self._build_space_index(self.current_part.front_list)

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

    def _add_front_to_space(self, fronts):
        """将阵面添加到已有背景网格"""
        for front in fronts:
            # 计算包围盒
            x_min = min(front.node_elems[0].coords[0], front.node_elems[1].coords[0])
            x_max = max(front.node_elems[0].coords[0], front.node_elems[1].coords[0])
            y_min = min(front.node_elems[0].coords[1], front.node_elems[1].coords[1])
            y_max = max(front.node_elems[0].coords[1], front.node_elems[1].coords[1])

            # 计算网格索引
            i_min = int(x_min // self.grid_size)
            i_max = int(x_max // self.grid_size)
            j_min = int(y_min // self.grid_size)
            j_max = int(y_max // self.grid_size)

            # 将阵面注册到覆盖的网格
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    self.space_grid[(i, j)].append(front)

    def _build_space_index(self, fronts):
        """构建空间索引加速相交检测"""
        from collections import defaultdict

        print("构建辅助查询背景网格...")
        # 动态计算网格尺寸（基于当前层推进步长）
        if self.current_part.growth_method == "geometric":
            current_step = self.current_part.first_height * (
                self.current_part.growth_rate**self.ilayer
            )
            self.grid_size = max(current_step * 2.0, 0.1)  # 保持网格尺寸≥0.1
        else:  # 当使用其他增长方式时回退到尺寸场
            self.grid_size = 1.5 * self.sizing_system.global_spacing

        self.space_grid = defaultdict(list)

        for front in fronts:
            # 计算包围盒
            x_min = min(front.node_elems[0].coords[0], front.node_elems[1].coords[0])
            x_max = max(front.node_elems[0].coords[0], front.node_elems[1].coords[0])
            y_min = min(front.node_elems[0].coords[1], front.node_elems[1].coords[1])
            y_max = max(front.node_elems[0].coords[1], front.node_elems[1].coords[1])

            # 计算网格索引
            i_min = int(x_min // self.grid_size)
            i_max = int(x_max // self.grid_size)
            j_min = int(y_min // self.grid_size)
            j_max = int(y_max // self.grid_size)

            # 将阵面注册到覆盖的网格
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    self.space_grid[(i, j)].append(front)

        print(f"全局最大网格尺度：{round(self.sizing_system.global_spacing,6)}")
        print(f"辅助查询网格尺寸：{round(self.grid_size,6)}")
        print(f"辅助查询网格维度：{len(self.space_grid)}\n")

    def _segments_intersect(self, seg1, seg2):
        """精确线段相交检测（排除共端点情况）"""
        p1, p2 = seg1[0].coords, seg1[1].coords
        q1, q2 = seg2[0].coords, seg2[1].coords

        return segments_intersect(p1, p2, q1, q2)

    def _fronts_distance(self, front1, front2):
        """计算两个阵面之间的最小距离"""
        p1 = front1.node_elems[0].coords
        p2 = front1.node_elems[1].coords
        q1 = front2.node_elems[0].coords
        q2 = front2.node_elems[1].coords

        return min_distance_between_segments(p1, p2, q1, q2)

    def _highlight_intersection(self, front1, front2):
        """在调试模式下高亮显示相交阵面"""
        if self.ax:
            front1.draw_front("r--", self.ax, linewidth=2)
            front2.draw_front("m--", self.ax, linewidth=2)
            mid_point = (np.array(front1.center) + np.array(front2.center)) / 2
            self.ax.text(mid_point[0], mid_point[1], "X", color="red", fontsize=14)

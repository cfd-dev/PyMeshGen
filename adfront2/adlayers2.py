import numpy as np
from math import pi, sqrt
import heapq

from geom_toolkit import (
    min_distance_between_segments,
    segments_intersect,
    is_left2d,
    points_equal,
)
from basic_elements import (
    NodeElement,
    NodeElementALM,
    Quadrilateral,
    Unstructured_Grid,
)
from front2d import Front
from message import info, debug, verbose, warning
from utils.timer import TimeSpan
from rtree_space import (
    build_space_index_with_RTree,
    add_elems_to_space_index_with_RTree,
    get_candidate_elements_id,
    build_space_index_with_cartesian_grid,
    add_elems_to_space_index_with_cartesian_grid,
    get_candidate_elements,
)


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

        self.al = 3.0  # TODO: 邻近检查候选阵面搜索范围系数
        self.space_index = None  # 空间索引
        self.space_grid_size = None  # Cartesian查询网格尺寸
        self.front_dict = {}  # 阵面id字典，用于快速查找

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
        self.matching_boundaries = []  # match边界的信息

        self.init_parts_and_connectors_fronts()
        self.initialize_match_boundary()
        self.reorder_node_index_and_front()

    def init_parts_and_connectors_fronts(self):
        """初始化层推进部件和connectors的阵面列表"""
        for part in self.part_params:
            part.match_fronts_with_connectors(self.initial_front_list)
            part.init_part_front_list()
            
    def initialize_match_boundary(self):
        """初始化边界层match部件"""
        num_wall_parts = 0
        wall_part_idx = -1
        for i, part in enumerate(self.part_params):
            if part.PRISM_SWITCH == "wall":
                num_wall_parts += 1
                wall_part_idx = i
                self.max_layers = part.max_layers
                self.full_layers = part.full_layers
                self.first_height = part.first_height
                self.growth_method = part.growth_method
                self.growth_rate = part.growth_rate

        if num_wall_parts == 0:
            raise ValueError("没有找到有效的边界层部件，请检查配置文件.")
        elif num_wall_parts > 1:
            raise ValueError("只支持match一个边界层部件，请检查配置文件.")

        for part in self.part_params:
            if part.PRISM_SWITCH != "match":
                continue
            self.rediscretize_part_to_match(part)

            # 将match部件的front_list添加到wall部件的front_list中
            self.part_params[wall_part_idx].front_list.extend(part.front_list)

        # 删除part_params中的match部件
        self.part_params = [
            part for part in self.part_params if part.PRISM_SWITCH != "match"
        ]

        # 将"wall"部件放在前面，其他部件放在后面
        self.part_params = [
            part for part in self.part_params if part.PRISM_SWITCH == "wall"
        ] + [part for part in self.part_params if part.PRISM_SWITCH != "wall"]

    def rediscretize_part_to_match(self, part):
        """将部件重新离散化以匹配边界层网格"""
        # 暂时认为match部件均为直线，先求出front_list所代表的直线，再对直线重新离散化
        # 如果有几何线的话，可以直接重新运行曲线离散程序，重新生成loop
        # 收集所有端点坐标
        all_points = []
        old_diretion = part.front_list[0].direction
        for front in part.front_list:
            all_points.extend([front.node_elems[0].coords, front.node_elems[1].coords])

        # 找到直线起点终点（任意方向）
        points_array = np.array(all_points)
        centroid = np.mean(points_array, axis=0)
        _, _, v = np.linalg.svd(points_array - centroid)
        direction_vector = v[0]  # 第一主成分方向

        # 计算各点在主方向的投影
        projections = np.dot(points_array - centroid, direction_vector)
        start_idx = np.argmin(projections)
        end_idx = np.argmax(projections)
        start = points_array[start_idx]
        end = points_array[end_idx]

        # 验证共线性
        vec = end - start
        for p in points_array:
            if np.linalg.norm(np.cross(p - start, vec)) > 1e-6:
                raise ValueError("线段不共线，无法进行直线离散化")

        # 找到起点和终点在front_list中对应的node_elem
        # start_node = []
        # end_node = []
        # for front in part.front_list:
        #     if np.array_equal(front.node_elems[0].coords, start):
        #         start_node = front.node_elems[0]
        #     if np.array_equal(front.node_elems[1].coords, start):
        #         start_node = front.node_elems[1]
        #     if np.array_equal(front.node_elems[0].coords, end):
        #         end_node = front.node_elems[0]
        #     if np.array_equal(front.node_elems[1].coords, end):
        #         end_node = front.node_elems[1]

        # 计算直线方向向量
        direction_vector = end - start
        total_length = np.linalg.norm(direction_vector)
        if total_length < 1e-6:
            raise ValueError("线段长度过小，无法进行离散化")
        unit_vector = direction_vector / total_length

        # 推进方向还是
        match_bound = MatchingBoundary(None, None, unit_vector, part.curve_name)
        self.matching_boundaries.append(match_bound)

        # 生成几何增长离散点
        discretized_points = [start]
        cumulative = 0.0
        ilayer = 0
        while True:
            step = self.first_height * (self.growth_rate**ilayer)
            ilayer += 1
            # 检查步长是否超出剩余长度
            if cumulative + step > total_length:
                break  # 提前终止循环

            cumulative += step
            new_point = start + unit_vector * cumulative
            discretized_points.append(new_point)

        # 确保最后一个点正好是终点（考虑浮点精度）
        if np.linalg.norm(discretized_points[-1] - end) > 1e-3:
            discretized_points.append(end)

        # 创建新front列表
        bc_type = part.front_list[0].bc_type
        part.front_list = []
        for i in range(0, len(discretized_points) - 1):
            # 转换为tuple保持坐标类型一致
            p1 = discretized_points[i].tolist()
            p2 = discretized_points[i + 1].tolist()

            node1 = NodeElementALM(
                coords=p1, idx=-1, bc_type=bc_type, match_bound=match_bound
            )
            node2 = NodeElementALM(
                coords=p2, idx=-1, bc_type=bc_type, match_bound=match_bound
            )

            # if i == 0:
            #     node1 = start_node
            # elif i == len(discretized_points) - 2:
            #     node2 = end_node
            if np.dot(unit_vector, old_diretion) < 0:
                part.front_list.append(Front(node2, node1, -1, bc_type, part.part_name))
            else:
                part.front_list.append(Front(node1, node2, -1, bc_type, part.part_name))

    def generate_elements(self):
        """生成边界层网格"""
        timer = TimeSpan("开始生成边界层网格...")
        num_parts = len(self.part_params)
        for i, part in enumerate(self.part_params):
            if part.PRISM_SWITCH != "wall":
                continue

            # 将部件参数设置为当前推进参数
            self.current_part = part
            self.first_height = part.first_height
            self.max_layers = part.max_layers
            self.full_layers = part.full_layers
            self.multi_direction = part.multi_direction
            self.num_prism_cap = len(part.front_list)
            self.ilayer = 0

            info(f"开始生成{part.part_name}的边界层网格...\n")
            while self.ilayer < self.max_layers and self.num_prism_cap > 0:

                info(
                    f"第{i+1}/{num_parts}个部件[{part.part_name}]：第{self.ilayer + 1}层推进中..."
                )

                self.prepare_geometry_info()

                self.visualize_point_normals()

                self.calculate_marching_distance()

                self.advancing_fronts()

                self.show_progress()

                timer.show_to_console(
                    f"第{i}/{num_parts}个部件[{part.part_name}]：第{self.ilayer + 1}层推进完成."
                )

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
        verbose("构建非结构网格数据..., Done.")

        # 汇总所有边界阵面
        self.all_boundary_fronts = []
        for part in self.part_params:
            self.all_boundary_fronts.extend(part.front_list)

        heapq.heapify(self.all_boundary_fronts)
        verbose("构建全局边界阵面..., Done.\n")

    def show_progress(self):
        """显示推进进度"""
        info(f"第{self.ilayer + 1}层推进..., Done.")
        info(f"当前节点数量：{self.num_nodes}")
        info(f"当前单元数量：{self.num_cells} ")

        if self.debug_level >= 2:
            self.debug_save()

    def debug_save(self):
        if self.debug_level < 1:
            return

        self.construct_unstr_grid()
        self.unstr_grid.save_debug_file(f"layer{self.ilayer + 1}")

    def create_new_cell_and_front(self, front):
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

                # 查询是否已经有已有合适点
                search_radius = self.al * node_elem.marching_distance
                candidates = get_candidate_elements_id(
                    front, self.space_index, search_radius
                )

                existed = False
                for f_id in candidates:
                    candidate = self.front_dict.get(f_id)
                    for node_tmp in candidate.node_elems:
                        if points_equal(node_tmp.coords, new_node.tolist(), 1e-3):
                            new_node_elem = node_tmp
                            existed = True
                            break
                    if existed:
                        break

                if not existed:
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
            self.current_part.part_name,
        )
        new_front1 = Front(
            new_cell_nodes[0],
            new_cell_nodes[2],
            -1,
            "interior",
            self.current_part.part_name,
        )
        new_front2 = Front(
            new_cell_nodes[3],
            new_cell_nodes[1],
            -1,
            "interior",
            self.current_part.part_name,
        )

        # if front.node_ids == (7, 5) or front.node_ids == (5, 6):
        #     print("debug")

        # 创建新单元
        new_cell = Quadrilateral(
            new_cell_nodes[0],
            new_cell_nodes[1],
            new_cell_nodes[3],
            new_cell_nodes[2],
            self.num_cells,
        )

        return (
            new_cell,
            alm_front,
            new_front1,
            new_front2,
            new_node_generated,
            new_cell_nodes,
        )

    def geometric_checker(self, front, new_cell):
        # 检查新单元质量，质量不合格则当前front早停
        quality = new_cell.get_skewness()
        if quality < self.quality_threshold:
            return True

        # 当前层数 > full_layer
        full_layer_condition = self.ilayer >= self.full_layers

        # 单元大小、长宽比、full_layer判断早停
        cell_size = new_cell.get_element_size()
        isotropic_size = self.sizing_system.spacing_at(front.center)
        # TODO: 限制调整头部和尾部的单元size：[1.2-1.5]
        size_factor = 1.3
        size_condition = cell_size > size_factor * isotropic_size

        # 单元size>1.5*isotropic_size且当前层数>full_layer，则当前front早停
        if size_condition and full_layer_condition:
            return True

        # 单元长宽比<1.1
        cell_aspect_ratio = new_cell.get_aspect_ratio2()
        aspect_ratio_condition = cell_aspect_ratio < 1.1

        # 长宽比<1.1且当前层数>full_layer，则当前front早停
        if aspect_ratio_condition and full_layer_condition:
            return True

        return False

    def proximity_checker(self, front, check_fronts):
        # 预计算安全距离
        # TODO: safe_distance暂时取为当前推进步长的0.5倍，后续可考虑调整优化
        safe_distance = 0.5 * min(n.marching_distance for n in front.node_elems)
        safe_distance_sq = safe_distance * safe_distance  # 使用平方距离避免开方

        for new_front in check_fronts:
            # 提前计算网格索引范围
            p0, p1 = new_front.node_elems[0].coords, new_front.node_elems[1].coords

            # 搜索范围
            search_radius = self.al * 2.0 * safe_distance
            candidates = get_candidate_elements_id(
                new_front, self.space_index, search_radius
            )

            for f_id in candidates:
                candidate = self.front_dict.get(f_id)
                if candidate is None or candidate.hash == front.hash:
                    continue

                # 若candidate与new_front共点，则跳过
                if any(id in candidate.node_ids for id in new_front.node_ids):
                    continue

                # 若candidate是在front推进方向的反方向，则跳过
                A = front.node_elems[0].coords
                B = front.node_elems[1].coords
                C = candidate.node_elems[0].coords
                D = candidate.node_elems[1].coords

                if not is_left2d(A, B, C) and not is_left2d(A, B, D):
                    continue

                # 邻近阵面检查，new_front与candidate的距离小于safe_distance，则对当前front进行早停
                q0, q1 = candidate.node_elems[0].coords, candidate.node_elems[1].coords
                if self._fast_distance_check(p0, p1, q0, q1, safe_distance_sq):
                    info(
                        f"阵面{front.node_ids}邻近告警：与{candidate.node_ids}距离<{safe_distance:.6f}"
                    )
                    if self.debug_level >= 1:
                        self._highlight_intersection(new_front, candidate)
                    return True

        return False

    def proximity_checker_with_cartesian_index(self, front, check_fronts):
        # 预计算安全距离
        # TODO: safe_distance暂时取为当前推进步长的0.5倍，后续可考虑调整优化
        safe_distance = 0.5 * min(n.marching_distance for n in front.node_elems)
        safe_distance_sq = safe_distance * safe_distance  # 使用平方距离避免开方

        for new_front in check_fronts:
            # 提前计算网格索引范围
            p0, p1 = new_front.node_elems[0].coords, new_front.node_elems[1].coords

            search_radius = self.al * 2.0 * safe_distance
            candidates = get_candidate_elements(
                new_front, self.space_index, self.space_grid_size, search_radius
            )

            for candidate in candidates:
                # 若candidate与new_front共点，则跳过
                if any(id in candidate.node_ids for id in new_front.node_ids):
                    continue

                # 若candidate是在front推进方向的反方向，则跳过
                AB = np.array(front.direction)  # 当前推进的阵面
                AC = np.array(check_fronts[1].direction)  # new_front1
                node0_coords = np.array(front.node_elems[0].coords)
                AE = np.array(candidate.node_elems[0].coords) - node0_coords
                AF = np.array(candidate.node_elems[1].coords) - node0_coords

                if (
                    np.cross(AB, AC) * np.cross(AB, AE) <= 0
                    and np.cross(AB, AC) * np.cross(AB, AF) <= 0
                ):
                    continue

                # 邻近阵面检查，new_front与candidate的距离小于safe_distance，则对当前front进行早停
                q0, q1 = candidate.node_elems[0].coords, candidate.node_elems[1].coords
                if self._fast_distance_check(p0, p1, q0, q1, safe_distance_sq):
                    info(
                        f"阵面{front.node_ids}邻近告警：与{candidate.node_ids}距离<{safe_distance:.6f}"
                    )
                    if self.debug_level >= 1:
                        self._highlight_intersection(new_front, candidate)
                    return True

        return False

    def _fast_distance_check(self, p0, p1, q0, q1, safe_distance_sq):
        """快速距离检查"""
        # 线段端点距离检查
        for p in [p0, p1]:
            for q in [q0, q1]:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                if dx * dx + dy * dy < safe_distance_sq:
                    return True

        # 线段间距离检查
        return min_distance_between_segments(p0, p1, q0, q1) < sqrt(safe_distance_sq)

    def update_front_list(self, check_fronts, new_interior_list, new_prism_cap_list):
        added = []
        front_hashes = {f.hash for f in new_interior_list}
        for chk_fro in check_fronts:
            if chk_fro.bc_type == "prism-cap":
                # prism-cap不会出现重复，必须加入到新阵面列表中
                new_prism_cap_list.append(chk_fro)
                added.append(chk_fro)
            elif chk_fro.bc_type == "interior":
                # interior可能会出现重复，需要检查是否已经存在
                if chk_fro.hash not in front_hashes:
                    new_interior_list.append(chk_fro)
                    added.append(chk_fro)
                else:  # 移除相同位置的旧阵面，此处可能会重新生成new_interior_list
                    new_interior_list = [
                        tmp_fro
                        for tmp_fro in new_interior_list
                        if tmp_fro.hash != chk_fro.hash
                    ]

        # R树索引更新
        add_elems_to_space_index_with_RTree(added, self.space_index, self.front_dict)

        if added and self.ax and self.debug_level >= 1:
            for fro in added:
                fro.draw_front("g-", self.ax)

        # 注意此处必须返回值，并且使用更新过的返回值
        return new_interior_list, new_prism_cap_list

    def update_front_list_old(
        self, check_fronts, new_interior_list, new_prism_cap_list
    ):
        added = []
        front_hashes = {f.hash for f in new_interior_list}
        for chk_fro in check_fronts:
            if chk_fro.bc_type == "prism-cap":
                # prism-cap不会出现重复，必须加入到新阵面列表中
                new_prism_cap_list.append(chk_fro)
                added.append(chk_fro)
            elif chk_fro.bc_type == "interior":
                # interior可能会出现重复，需要检查是否已经存在
                if chk_fro.hash not in front_hashes:
                    new_interior_list.append(chk_fro)
                    added.append(chk_fro)
                else:  # 移除相同位置的旧阵面，此处可能会重新生成new_interior_list
                    new_interior_list = [
                        tmp_fro
                        for tmp_fro in new_interior_list
                        if tmp_fro.hash != chk_fro.hash
                    ]
        # 将新阵面加入到空间查询网格中
        add_elems_to_space_index_with_cartesian_grid(added, self.space_grid_size)

        if added and self.ax and self.debug_level >= 1:
            for fro in added:
                fro.draw_front("g-", self.ax)

        # 注意此处必须返回值，并且使用更新过的返回值
        return new_interior_list, new_prism_cap_list

    def advancing_fronts(self):
        timer = TimeSpan("逐个阵面推进生成单元...")

        new_interior_list = []  # 新增的边界层法向面，设置为interior
        new_prism_cap_list = []  # 新增的边界层流向面，设置为prism-cap
        # 逐个阵面进行推进
        num_old_prism_cap = 0  # 当前层early stop的prism-cap数量

        for front in self.current_part.front_list:
            if front.bc_type != "wall" and front.bc_type != "prism-cap":
                new_interior_list.append(front)  # 未推进的阵面仍然加入到新阵面列表中

        for front in self.current_part.front_list:
            # if front.bc_type == "interior" or front.bc_type == "pressure-far-field":
            # new_interior_list.append(front)  # 未推进的阵面仍然加入到新阵面列表中
            # continue
            if front.bc_type != "wall" and front.bc_type != "prism-cap":
                continue

            if front.early_stop_flag:
                new_prism_cap_list.append(front)  # 未推进的阵面仍然加入到新阵面列表中
                num_old_prism_cap += 1
                continue

            if front.node_ids == [50, 60]:
                print("debug")

            (
                new_cell,  # 新单元Quadrilateral对象，0-1-3-2 顺序
                alm_front,  # 新阵面Front对象，prism-cap，2-3
                new_front1,  # 新阵面Front对象，interior，0-2
                new_front2,  # 新阵面Front对象，interior，3-1
                new_node_generated,  # 新节点NodeElementALM对象，若新生成则为NodeElementALM对象，否则为None
                new_cell_nodes,  # 新单元节点NodeElementALM对象列表，包含新节点和旧节点
            ) = self.create_new_cell_and_front(front)

            # 单元质量、层数、长宽比等早停条件检查
            if self.geometric_checker(front, new_cell):
                front.early_stop_flag = True
                new_prism_cap_list.append(front)
                continue

            # 邻近检查，检查3个阵面附近是否有其他阵面，若有，则对当前阵面进行早停
            check_fronts = [alm_front, new_front1, new_front2]
            if self.proximity_checker(front, check_fronts):
                front.early_stop_flag = True
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
            new_interior_list, new_prism_cap_list = self.update_front_list(
                check_fronts, new_interior_list, new_prism_cap_list
            )

        timer.show_to_console("逐个阵面推进生成单元..., Done.")

        # 下一层需要推进的prism-cap的数量
        self.num_prism_cap = len(new_prism_cap_list) - num_old_prism_cap
        # 更新part阵面列表
        self.current_part.front_list = new_prism_cap_list + new_interior_list
        verbose(f"下一层（第{self.ilayer+2}层）阵面数据更新..., Done.\n")

    def calculate_marching_distance(self):
        """计算节点推进距离"""
        verbose("计算节点推进步长...")
        self.current_step_size = 0.0
        if self.current_part.growth_method == "geometric":
            # 计算几何增长距离
            first_height = self.current_part.first_height
            growth_rate = self.current_part.growth_rate
            self.current_step_size = first_height * growth_rate**self.ilayer
        else:
            raise ValueError("未知的步长计算方法！")

        info(f"第{self.ilayer+1}层推进步长：{self.current_step_size:.6f}")

        for front in self.current_part.front_list:
            # 计算节点推进距离
            for node in front.node_elems:
                if node.idx == 23:
                    print("debug")

                node.marching_distance = self.current_step_size

                if node.bc_type != "wall" and node.bc_type != "prism-cap":
                    continue

                front1, front2 = node.node2front[:2]

                # 节点推进方向与阵面法向的夹角, 节点推进方向投影到面法向
                # proj1 = np.dot(node.marching_direction, front1.normal)
                # proj2 = np.dot(node.marching_direction, front2.normal)

                if not (
                    (front1.bc_type == "wall" or front1.bc_type == "prism-cap")
                    and (front2.bc_type == "wall" or front2.bc_type == "prism-cap")
                ):
                    continue

                proj1 = np.dot(node.marching_direction, front1.normal)
                proj2 = np.dot(node.marching_direction, front2.normal)

                if (
                    proj1 * proj2 < 0
                    and front1.bc_type != "interior"
                    and front2.bc_type != "interior"
                ):
                    warning(
                        f"node{node.idx}推进方向与相邻阵面法向夹角大于90°，可能出现质量差单元！"
                    )

                # 节点推进距离
                node.marching_distance = (
                    self.current_step_size
                    * node.local_step_factor
                    / np.mean([proj1, proj2])
                )  # min(abs(proj1), abs(proj2))

        verbose("计算节点推进步长..., Done.\n")

    def visualize_point_normals(self):
        """可视化节点推进方向"""
        if self.ax is None or self.debug_level < 3:
            return

        for node in self.front_node_list:
            if node.matching_boundary is not None:
                continue
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

        verbose("计算节点初始推进方向...")
        for node_elem in self.front_node_list:
            if len(node_elem.node2front) < 2:
                # raise ValueError("节点必须连接至少2个阵面！")
                continue

            if node_elem.idx == 50 or node_elem.idx == 38:
                print("debug")

            # 对于凸角点，在此不计算，也不光滑
            if len(node_elem.marching_direction) > 1:
                continue

            front1, front2 = node_elem.node2front[:2]
            normal1 = np.array(front1.normal)
            normal2 = np.array(front2.normal)

            # TODO: 应对节点只有一侧有流向阵面，另一侧是法向阵面或其他边界阵面的情况
            # front1是其他阵面，而front2是流向阵面
            if (front1.bc_type != "wall" and front1.bc_type != "prism-cap") and (
                front2.bc_type == "wall" or front2.bc_type == "prism-cap"
            ):
                node_elem.marching_direction = normal2
                continue
                # normal1 = np.array(front2.direction)
            # front2是其他阵面，而front1是流向阵面
            elif (front2.bc_type != "wall" and front2.bc_type != "prism-cap") and (
                front1.bc_type == "wall" or front1.bc_type == "prism-cap"
            ):
                node_elem.marching_direction = normal1
                continue
                # normal2 = np.array(front1.direction)
            elif (front2.bc_type != "wall" and front2.bc_type != "prism-cap") and (
                front1.bc_type != "wall" and front1.bc_type != "prism-cap"
            ):
                warning(f"节点{node_elem.idx}2个相邻阵面为其他阵面！")
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

        verbose("计算节点初始推进方向..., Done.\n")

    def laplacian_smooth_normals(self):
        """拉普拉斯平滑节点推进方向"""
        # FIXME: 边界层推进方向不够光滑，待检查debug
        verbose("节点推进方向光滑....")
        for node_elem in self.front_node_list:
            num_neighbors = len(node_elem.node2node)
            if num_neighbors < 2:
                continue

            if node_elem.idx == 35 or node_elem.idx == 38:
                print("debug")
            iteration = 0
            while iteration < self.smooth_iterions:
                summation = np.zeros_like(np.array(node_elem.marching_direction))
                for neighbor in node_elem.node2node:
                    if neighbor.matching_boundary:
                        neighbor.marching_direction = (
                            neighbor.matching_boundary.direction_vector
                        )
                    summation += np.array(neighbor.marching_direction)

                new_direction = (1 - self.relax_factor) * np.array(
                    node_elem.marching_direction
                ) + self.relax_factor / num_neighbors * summation

                node_elem.marching_direction = tuple(
                    new_direction / np.linalg.norm(new_direction)
                )
                iteration += 1

        verbose("节点推进方向光滑..., Done.\n")

    def prepare_geometry_info(self):
        """准备几何信息"""
        self.compute_front_geometry()

        self.compute_point_normals()

        self.laplacian_smooth_normals()

    def reorder_node_index_and_front(self):
        """初始化节点"""
        node_count = 0
        processed_nodes = set()
        hash_idx_map = {}  # 节点hash值到节点索引的映射
        self.all_boundary_fronts = []
        for part in self.part_params:
            self.all_boundary_fronts.extend(part.front_list)

            for front in part.front_list:
                front.node_ids = []
                for node_elem in front.node_elems:
                    if node_elem.idx == 28 or node_elem.idx == 14:
                        print("debug")

                    if node_elem.hash not in processed_nodes:
                        node_elem.idx = node_count
                        hash_idx_map[node_elem.hash] = node_elem.idx
                        self.boundary_nodes.append(node_elem)
                        self.node_coords.append(node_elem.coords)
                        processed_nodes.add(node_elem.hash)
                        node_count += 1
                    else:
                        node_elem.idx = hash_idx_map[node_elem.hash]

                    front.node_ids.append(node_elem.idx)

        self.num_nodes = len(self.node_coords)

    def reconstruct_node2front(self):
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

    def reconstruct_node2node(self):
        """重构node2node关系"""
        verbose("重构node2node关系...")
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

        verbose("重构node2node关系..., Done.\n")

    def compute_corner_attributes(self):
        """计算阵面间夹角、凹凸标记、局部步长因子、多方向推进数量"""
        verbose("计算阵面间夹角、凹凸标记、局部步长因子、多方向推进数量...")
        for node_elem in self.front_node_list:
            if len(node_elem.node2front) < 2:
                # raise ValueError(f"节点{node_elem.idx}的邻接阵面数量小于2！")
                continue

            front1, front2 = node_elem.node2front[:2]

            # 只有当相邻阵面为wall或者prism-cap类型才需要计算凹凸角信息
            if not (
                (front1.bc_type == "wall" or front1.bc_type == "prism-cap")
                and (front2.bc_type == "wall" or front2.bc_type == "prism-cap")
            ):
                continue

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

        verbose("计算阵面间夹角、凹凸标记、局部步长因子、多方向推进数量..., Done.\n")

    def build_front_rtree(self):
        verbose("构建辅助查询R-Tree...")
        # self.front_dict, self.space_index = build_space_index_with_RTree(
        #     self.current_part.front_list
        # )

        self.all_boundary_fronts = []
        for part in self.part_params:
            self.all_boundary_fronts.extend(part.front_list)

        self.front_dict, self.space_index = build_space_index_with_RTree(
            self.all_boundary_fronts
        )

        verbose(f"R树索引构建完成，包含{len(self.front_dict)}个阵面")

    def build_front_cartesian_space_index(self):
        """构建辅助查询Cartesian背景网格"""
        verbose("构建辅助查询背景网格...")
        # 动态计算网格尺寸（基于当前层推进步长）
        if self.current_part.growth_method == "geometric":
            current_step = (
                self.current_part.first_height
                * self.current_part.growth_rate**self.ilayer
            )
            self.space_grid_size = max(current_step * 2.0, 0.1)  # 保持网格尺寸≥0.1
        else:  # 当使用其他增长方式时回退到尺寸场
            self.space_grid_size = 1.5 * self.sizing_system.global_spacing

        self.space_index = build_space_index_with_cartesian_grid(
            self.current_part.front_list, self.space_grid_size
        )

        verbose(f"全局最大网格尺度：{self.sizing_system.global_spacing:.3f}")
        verbose(f"辅助查询网格尺寸：{self.space_grid_size:.3f}")
        verbose(f"辅助查询网格数量：{len(self.space_index)}\n")

    def compute_front_geometry(self):
        """计算阵面几何信息"""
        verbose("计算物面几何信息...")

        # 建立RTree，便于快速查询
        self.build_front_rtree()

        # 计算node2front
        self.reconstruct_node2front()

        # 计算node2node
        self.reconstruct_node2node()

        # 计算阵面间夹角、凹凸标记、局部步长因子、多方向推进数量
        self.compute_corner_attributes()

        verbose("计算物面几何信息..., Done.\n")


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


class MatchingBoundary:
    def __init__(self, start_node, end_node, direction_vector, part_name):
        self.start_node = start_node
        self.end_node = end_node
        self.part_name = part_name

        if direction_vector is None:
            self.direction_vector = np.array(end_node.coords) - np.array(
                start_node.coords
            )
            self.direction_vector /= np.linalg.norm(self.direction_vector)
            self.direction_vector = self.direction_vector.tolist()
        else:
            self.direction_vector = direction_vector

import numpy as np
from math import pi, sqrt
import heapq
from collections import deque

from utils.geom_toolkit import (
    min_distance_between_segments,
    segments_intersect,
    is_left2d,
    points_equal,
    fast_distance_check,
)
from data_structure.basic_elements import (
    NodeElement,
    NodeElementALM,
    Triangle,
    Quadrilateral,
)
from data_structure.unstructured_grid import Unstructured_Grid
from data_structure.front2d import Front
from utils.message import info, debug, verbose, warning
from utils.timer import TimeSpan
from data_structure.rtree_space import (
    build_space_index_with_RTree,
    add_elems_to_space_index_with_RTree,
    get_candidate_elements_id,
    build_space_index_with_cartesian_grid,
    add_elems_to_space_index_with_cartesian_grid,
    get_candidate_elements,
)
from adfront2.multi_direction import MultiDirectionManager


class Adlayers2:
    def __init__(self, boundary_front, sizing_system, param_obj=None, visual_obj=None):
        self.ax = visual_obj.ax
        self.debug_level = (
            param_obj.debug_level
        )  # 调试级别，0-不输出，1-输出基本信息，2-输出详细信息

        # 层推进全局参数
        self.max_layers = 3  # 最大推进层数
        self.full_layers = 0  # 完整推进层数
        self.growth_rate = 1.2  # 网格高度增长比例
        self.growth_method = "geometric"  # 网格高度增长方法
        self.max_size = 1e6  # 最大网格尺寸
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
        
        # 多方向推进管理器
        self.multi_direction_manager = None

        self.init_parts_and_connectors_fronts()
        self.initialize_match_boundary()
        self.reorder_node_index_and_front()

    def init_parts_and_connectors_fronts(self):
        """初始化层推进部件和connectors的阵面列表"""
        for part in self.part_params:
            part.match_fronts_with_connectors(self.initial_front_list)
            part.init_part_front_list()

    def match_boundary_exists(self):
        """检查是否存在match边界"""
        for part in self.part_params:
            for conn in part.connectors:
                if getattr(conn, "is_match", False):
                    return True
        return False

    def initialize_match_boundary(self):
        """初始化边界层match部件"""
        if not self.match_boundary_exists():
            return

        num_wall_parts = 0
        matched_wall_part = []
        for part in self.part_params:
            if part.part_params.PRISM_SWITCH == "wall":
                num_wall_parts += 1
                matched_wall_part = part

        if num_wall_parts == 0:
            raise ValueError("没有找到有效的边界层部件，请检查配置文件.")
        elif num_wall_parts > 1:
            raise ValueError("只支持match一个边界层部件，请检查配置文件.")

        for part in self.part_params:
            for conn in part.connectors:
                if getattr(conn, "is_match", False):
                    conn.rediscretize_conn_to_match_wall(matched_wall_part)
                    # conn的阵面列表发生变化，重新初始化part的阵面列表
                    part.init_part_front_list()

    def collect_all_boundary_fronts(self):
        """收集所有边界阵面"""
        self.all_boundary_fronts = []
        for part in self.part_params:
            self.all_boundary_fronts.extend(part.front_list)

    def reorder_node_index_and_front(self):
        """对所有节点进行重编号"""
        node_count = 0
        front_count = 0
        processed_nodes = set()
        node_dict = {}  # 节点hash值到节点的映射
        for part in self.part_params:
            for front in part.front_list:
                front.idx = front_count
                front_count += 1
                front.node_ids = []
                for i, node_elem in enumerate(front.node_elems):
                    if node_elem.hash not in processed_nodes:
                        front.node_elems[i].idx = node_count
                        node_dict[node_elem.hash] = front.node_elems[i]

                        self.boundary_nodes.append(front.node_elems[i])
                        self.node_coords.append(node_elem.coords)
                        processed_nodes.add(node_elem.hash)
                        node_count += 1
                    else:
                        front.node_elems[i] = node_dict[node_elem.hash]

                    front.node_ids.append(front.node_elems[i].idx)

            part.init_part_front_list()

        self.num_nodes = len(self.node_coords)

    def set_current_part(self, part):
        """设置当前推进参数"""
        self.current_part = part
        self.first_height = part.part_params.first_height
        self.growth_method = part.part_params.growth_method
        self.growth_rate = part.part_params.growth_rate
        self.max_layers = part.part_params.max_layers
        self.full_layers = part.part_params.full_layers
        self.multi_direction = part.part_params.multi_direction
        self.num_prism_cap = len(part.front_list)

    def generate_elements(self):
        """生成边界层网格"""
        timer = TimeSpan()
        num_parts = len(self.part_params)
        for i, part in enumerate(self.part_params):
            if part.part_params.PRISM_SWITCH != "wall":
                continue

            # 将部件参数设置为当前推进参数
            self.set_current_part(part)

            info(f"开始生成{part.part_name}的边界层网格...\n")

            self.ilayer = 0

            # 初始化多方向管理器
            if self.multi_direction:
                self.multi_direction_manager = MultiDirectionManager(self)

            while self.ilayer < self.max_layers and self.num_prism_cap > 0:
                self._process_single_layer(i, num_parts)

                timer.show_to_console(
                    f"第{i+1}/{num_parts}个部件[{part.part_name}]：第{self.ilayer + 1}层推进完成."
                )

                self.ilayer += 1

        # 后处理：消除虚拟点
        if self.multi_direction and self.multi_direction_manager:
            self.multi_direction_manager.post_process()

        self.construct_unstr_grid()

        self.debug_save()

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
        self.collect_all_boundary_fronts()

        heapq.heapify(self.all_boundary_fronts)
        verbose("构建全局边界阵面..., Done.\n")

    def show_progress(self):
        """显示推进进度"""
        info(f"第{self.ilayer + 1}层推进..., Done.")
        info(f"当前节点数量：{self.num_nodes}")
        info(f"当前单元数量：{self.num_cells} ")

        if self.debug_level >= 2:
            self.debug_save()

    def _process_single_layer(self, part_index, num_parts):
        """执行单层推进完整流程"""
        info(
            f"第{part_index+1}/{num_parts}个部件[{self.current_part.part_name}]：第{self.ilayer + 1}层推进中..."
        )
        start_cell_idx = self.num_cells

        # 阶段1：基础几何与方向
        self.prepare_geometry_info()

        self.log_multi_direction_debug_summary()

        # 阶段2：多方向流程（初始化→方向光滑→步长缩放）
        self.apply_multi_direction_workflow()

        # 阶段3：步长计算与推进
        self.visualize_point_normals()
        self.calculate_marching_distance()

        self.advancing_fronts()
        
        self.log_first_layer_cell_summary(start_cell_idx)

        self.show_progress()

    def debug_save(self):
        if self.debug_level < 1:
            return

        self.construct_unstr_grid()
        self.unstr_grid.save_debug_file(f"layer{self.ilayer + 1}")

    @staticmethod
    def _extract_direction_vector(direction_value):
        """提取二维推进方向向量，多方向列表默认取当前有效方向（第一个）"""
        try:
            vec = np.asarray(direction_value, dtype=float)
        except (TypeError, ValueError):
            return None

        if vec.ndim == 0:
            return None
        if vec.ndim >= 2:
            if vec.shape[0] == 0:
                return None
            vec = vec[0]

        vec = np.asarray(vec, dtype=float).reshape(-1)
        if vec.size < 2:
            return None
        return vec[:2]

    def _node_direction(self, node_elem):
        """获取节点当前有效推进方向，无效时回退为零向量"""
        vec = self._extract_direction_vector(node_elem.marching_direction)
        if vec is None:
            return np.zeros(2)
        return vec

    def _get_or_create_corresponding_node(self, node_elem):
        """获取节点对应推进点；若不存在则创建并回写到全局节点列表"""
        if node_elem.corresponding_node is not None:
            return node_elem.corresponding_node

        new_coords = (
            np.array(node_elem.coords)
            + self._node_direction(node_elem) * node_elem.marching_distance
        )
        new_node = NodeElementALM(
            coords=new_coords.tolist(),
            idx=self.num_nodes,
            bc_type="interior",
        )
        new_node.strandline_start_node = node_elem
        self.node_coords.append(new_coords.tolist())
        self.num_nodes += 1
        node_elem.corresponding_node = new_node
        return new_node

    def _finalize_new_fronts(self, source_front, new_fronts, new_interior_list, new_prism_cap_list):
        """统一处理新阵面的层数、节点关联与全局阵面更新"""
        new_layer_count = source_front.layer_count + 1
        for new_front in new_fronts:
            new_front.layer_count = new_layer_count
            for node in new_front.node_elems:
                node.node2front.append(new_front)

        self.update_front_list_globally(
            new_fronts,
            new_interior_list,
            new_prism_cap_list,
        )

    def _advance_virtual_front(self, front, new_interior_list, new_prism_cap_list):
        """处理零长度虚拟阵面推进，返回是否已处理（含跳过）"""
        if front.length >= 1e-12:
            return False

        node1, node2 = front.node_elems
        is_virt1 = getattr(node1, "is_virtual_point", False)
        is_virt2 = getattr(node2, "is_virtual_point", False)

        if not (is_virt1 or is_virt2):
            debug(f"[虚拟阵面跳过] 阵面{front.node_ids}长度为0且非虚拟阵面，跳过推进")
            return True

        if self.multi_direction_manager is None:
            debug(f"[虚拟阵面跳过] 阵面{front.node_ids}无多方向管理器，跳过推进")
            return True

        # 对零长度虚拟阵面仍按“四边形推进→有效点去重”处理：
        # cell = unique([ValidPoint(node1), ValidPoint(node2), node2_new, node1_new], stable)
        node1_new = self._get_or_create_corresponding_node(node1)
        node2_new = self._get_or_create_corresponding_node(node2)

        if points_equal(node1_new.coords, node2_new.coords, 1e-12):
            debug(f"[虚拟阵面跳过] 阵面{front.node_ids}推进后新点重合，跳过")
            return True

        node1_real = self.multi_direction_manager.get_real_point(node1)
        node2_real = self.multi_direction_manager.get_real_point(node2)
        if node1_real is None:
            node1_real = node1
        if node2_real is None:
            node2_real = node2

        ordered_nodes = [node1_real, node2_real, node2_new, node1_new]
        cell_nodes = []
        seen = set()
        for node in ordered_nodes:
            if node.idx in seen:
                continue
            seen.add(node.idx)
            cell_nodes.append(node)

        if len(cell_nodes) == 4:
            new_cell = Quadrilateral(
                cell_nodes[0],
                cell_nodes[1],
                cell_nodes[2],
                cell_nodes[3],
                part_name="interior-blayers",
                idx=self.num_cells,
            )
        elif len(cell_nodes) == 3:
            new_cell = Triangle(
                cell_nodes[0],
                cell_nodes[1],
                cell_nodes[2],
                part_name="interior-blayers",
                idx=self.num_cells,
            )
        else:
            warning(
                f"[虚拟阵面跳过] 阵面{front.node_ids}有效点不足3个，无法构成单元"
            )
            return True

        new_cell.layer = self.ilayer + 1
        self.cell_container.append(new_cell)
        self.num_cells += 1

        virtual_alm_front = Front(
            node1_new,
            node2_new,
            -1,
            "prism-cap",
            self.current_part.part_name,
        )
        virtual_new_front1 = Front(
            node1,
            node1_new,
            -1,
            "interior",
            self.current_part.part_name,
        )
        virtual_new_front2 = Front(
            node2_new,
            node2,
            -1,
            "interior",
            self.current_part.part_name,
        )

        self._finalize_new_fronts(
            front,
            [virtual_alm_front, virtual_new_front1, virtual_new_front2],
            new_interior_list,
            new_prism_cap_list,
        )

        debug(
            f"[虚拟阵面推进] 阵面{front.node_ids}创建{type(new_cell).__name__}并更新拓扑"
        )
        return True

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
                    + self._node_direction(node_elem)
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
                        # 搜索范围内是否有点与新点重合
                        if points_equal(node_tmp.coords, new_node.tolist(), 1e-6):
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
                    # 串线追踪：特殊点从自身起链，其余节点继承已有起点
                    if node_elem.strandline_start_node is not None:
                        new_node_elem.strandline_start_node = node_elem.strandline_start_node
                    elif node_elem.convex_flag or getattr(node_elem, "is_virtual_point", False):
                        new_node_elem.strandline_start_node = node_elem
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

        new_cell = Quadrilateral(
            new_cell_nodes[0],
            new_cell_nodes[1],
            new_cell_nodes[3],
            new_cell_nodes[2],
            part_name="interior-blayers",
            idx=self.num_cells,
        )
        new_cell.layer = self.ilayer + 1

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

                # 若candidate与new_front共坐标点（虚拟点与真实点重合），则跳过
                shared_coords = any(
                    points_equal(n1.coords, n2.coords, 1e-8)
                    for n1 in new_front.node_elems
                    for n2 in candidate.node_elems
                )
                if shared_coords:
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
                if fast_distance_check(p0, p1, q0, q1, safe_distance_sq):
                    verbose(
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

                # 若candidate与new_front共坐标点（虚拟点与真实点重合），则跳过
                shared_coords = any(
                    points_equal(n1.coords, n2.coords, 1e-8)
                    for n1 in new_front.node_elems
                    for n2 in candidate.node_elems
                )
                if shared_coords:
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
                if fast_distance_check(p0, p1, q0, q1, safe_distance_sq):
                    info(
                        f"阵面{front.node_ids}邻近告警：与{candidate.node_ids}距离<{safe_distance:.6f}"
                    )
                    if self.debug_level >= 1:
                        self._highlight_intersection(new_front, candidate)
                    return True

        return False


    def update_front_list_globally(
        self, check_fronts, new_interior_list, new_prism_cap_list
    ):
        """更新全局所有part的front_list，通过间接更新new_interior_list和new_prism_cap_list，
        以及直接更新其他part的front_list来实现"""
        added = []
        # 需要在全部阵面中查找是否有重复的阵面，如有，则删除
        front_hashes = {f.hash for f in new_interior_list}
        for chk_fro in check_fronts:
            if chk_fro.bc_type == "prism-cap":
                # prism-cap不会出现重复，必须加入到新阵面列表中
                new_prism_cap_list.append(chk_fro)
                added.append(chk_fro)
            elif chk_fro.bc_type == "interior":
                # interior可能会出现重复，需要检查是否已经存在
                # 首先检查其在各个part的front_list中是否存在，若存在，则将其删除
                found1 = False
                for part in self.part_params:
                    original_len = len(part.front_list)
                    # 使用列表推导式过滤当前part的front_list
                    part.front_list = [
                        tmp_fro
                        for tmp_fro in part.front_list
                        if tmp_fro.hash != chk_fro.hash
                    ]
                    # 如果当前part的front_list被修改过，说明在当前part找到了chk_fro，直接跳出循环
                    if len(part.front_list) < original_len:
                        found1 = True
                        break

                # 由于当前层生成的new_interior_list还没有加入到部件阵面中去，因此此处单独对其进行搜索
                found2 = False
                if chk_fro.hash in front_hashes:
                    found2 = True

                if found2:
                    # 原地修改new_interior_list，无需返回值
                    new_interior_list[:] = [
                        tmp_fro
                        for tmp_fro in new_interior_list
                        if tmp_fro.hash != chk_fro.hash
                    ]

                if not found1 and not found2:
                    new_interior_list.append(chk_fro)
                    added.append(chk_fro)

        # R树索引更新
        add_elems_to_space_index_with_RTree(added, self.space_index, self.front_dict)

        if added and self.ax and self.debug_level >= 1:
            for fro in added:
                fro.draw_front("g-", self.ax)

    def update_front_list_locally(
        self, check_fronts, new_interior_list, new_prism_cap_list
    ):
        """只更新当前part的front_list，通过间接更新new_interior_list和new_prism_cap_list来实现"""
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
                    new_interior_list[:] = [
                        tmp_fro
                        for tmp_fro in new_interior_list
                        if tmp_fro.hash != chk_fro.hash
                    ]

        # R树索引更新，默认方式
        add_elems_to_space_index_with_RTree(added, self.space_index, self.front_dict)

        # cartesian空间索引更新
        # add_elems_to_space_index_with_cartesian_grid(added, self.space_grid_size)

        if added and self.ax and self.debug_level >= 1:
            for fro in added:
                fro.draw_front("g-", self.ax)

    def advancing_fronts(self):
        timer = TimeSpan("逐个阵面推进生成单元...")

        new_interior_list = []  # 新增的边界层法向面，设置为 interior
        new_prism_cap_list = []  # 新增的边界层流向面，设置为 prism-cap
        num_old_prism_cap = 0  # 当前层 early stop 的 prism-cap 数量

        # 逐个阵面进行推进
        for front in self.current_part.front_list:
            if front.bc_type == "interior":
                new_interior_list.append(front)  # 未推进的阵面仍然加入到新阵面列表中
                continue

            if front.early_stop_flag:
                new_prism_cap_list.append(front)  # 未推进的阵面仍然加入到新阵面列表中
                num_old_prism_cap += 1
                continue
            
            # 处理零长度阵面（虚拟阵面推进/跳过）
            if self._advance_virtual_front(front, new_interior_list, new_prism_cap_list):
                continue

            # 检查相邻阵面的层数差，若超过 2 则当前阵面早停
            if self._check_neighbor_layer_difference(front):
                front.early_stop_flag = True
                new_prism_cap_list.append(front)
                verbose(f"[早停] 阵面{front.node_ids}因相邻阵面层数差>2 而早停，当前层数：{front.layer_count}")
                continue

            (
                new_cell,  # 新单元 Quadrilateral 对象，0-1-3-2 顺序
                alm_front,  # 新阵面 Front 对象，prism-cap，2-3
                new_front1,  # 新阵面 Front 对象，interior，0-2
                new_front2,  # 新阵面 Front 对象，interior，3-1
                new_node_generated,  # 新节点 NodeElementALM 对象，若新生成则为 NodeElementALM 对象，否则为 None
                new_cell_nodes,  # 新单元节点 NodeElementALM 对象列表，包含新节点和旧节点
            ) = self.create_new_cell_and_front(front)

            # 单元质量、层数、长宽比等早停条件检查
            if self.geometric_checker(front, new_cell):
                front.early_stop_flag = True
                new_prism_cap_list.append(front)
                continue

            # 邻近检查，检查 3 个阵面附近是否有其他阵面，若有，则对当前阵面进行早停
            check_fronts = [alm_front, new_front1, new_front2]
            if self.proximity_checker(front, check_fronts):
                front.early_stop_flag = True
                new_prism_cap_list.append(front)
                continue

            # 若没有早停，则更新节点、单元和阵面列表
            # 更新节点：检查新节点是否生成，若生成，则加入到节点列表中
            # 注意 corresponding_node 也要在此更新，而不是在其他地方更新
            for i in range(2):
                if new_node_generated[i] is not None:
                    front.node_elems[i].corresponding_node = new_node_generated[i]
                    self.node_coords.append(new_cell_nodes[i + 2].coords)
                    self.num_nodes += 1

            # 更新单元
            self.cell_container.append(new_cell)
            self.num_cells += 1
            
            self._finalize_new_fronts(
                front,
                check_fronts,
                new_interior_list,
                new_prism_cap_list,
            )

        timer.show_to_console("逐个阵面推进生成单元..., Done.")

        # 下一层需要推进的 prism-cap 的数量
        self.num_prism_cap = len(new_prism_cap_list) - num_old_prism_cap
        # 更新 part 阵面列表
        self.current_part.front_list = new_prism_cap_list + new_interior_list
        verbose(f"下一层（第{self.ilayer+2}层）阵面数据更新..., Done.\n")

    def _check_neighbor_layer_difference(self, front):
        """检查相邻 prism_cap 阵面的层数差，若超过 2 则返回 True（需要早停）
        
        相邻 prism_cap 的定义：两个 prism_cap 可以通过任意数量的 interior 阵面链相连。
        使用 BFS 算法搜索所有可达的 prism_cap。
        
        Args:
            front: 当前待检查的阵面（prism_cap 类型）
        """
        max_layer_diff = 2  # 最大允许层数差

        neighbor_prism_caps = set()  # 存储相邻 prism_cap 的 hash 值
        neighbor_layers = set()  # 存储相邻 prism_cap 的层数
        neighbor_fronts_info = []
        
        visited_nodes = set()  # 已访问的节点 hash
        queue = deque()  # BFS 队列
        
        # 从当前 prism_cap 的两个节点开始 BFS
        for node in front.node_elems:
            visited_nodes.add(node.hash)
            queue.append(node)
        
        # BFS 搜索所有通过 interior 阵面相连的节点
        while queue:
            current_node = queue.popleft()
            
            # 遍历当前节点所属的所有阵面
            for neighbor_front in current_node.node2front:
                # 跳过自身阵面
                if neighbor_front.hash == front.hash:
                    continue
                
                # 只通过 interior 阵面传播
                if neighbor_front.bc_type != "interior":
                    continue
                
                # 找到这个 interior 阵面的另一个节点
                other_node = None
                for n in neighbor_front.node_elems:
                    if n.hash != current_node.hash:
                        other_node = n
                        break
                
                if other_node is None:
                    continue
                
                # 检查另一个节点是否属于其他 prism_cap
                for other_front in other_node.node2front:
                    if other_front.bc_type == "prism-cap" and other_front.hash != front.hash:
                        # 找到相邻的 prism_cap（不排除当前层新生成的）
                        if other_front.hash not in neighbor_prism_caps:
                            neighbor_prism_caps.add(other_front.hash)
                            neighbor_layers.add(other_front.layer_count)
                            neighbor_fronts_info.append((other_front.node_ids, other_front.layer_count))
                
                # 继续通过 interior 阵面传播
                if other_node.hash not in visited_nodes:
                    visited_nodes.add(other_node.hash)
                    queue.append(other_node)
        
        # 如果没有相邻的 prism_cap，不需要检查
        if not neighbor_prism_caps:
            debug(f"[层数差检查] 阵面{front.node_ids} (层={front.layer_count}): 无相邻prism_cap阵面")
            return False

        # 检查层数差
        current_layer = front.layer_count
        debug(f"[层数差检查] 阵面{front.node_ids} (层={current_layer}): 相邻prism_cap阵面信息={neighbor_fronts_info}")
        
        for neighbor_layer in neighbor_layers:
            if abs(current_layer - neighbor_layer) > max_layer_diff:
                debug(f"[层数差检查] 阵面{front.node_ids} 层数差超限：当前层={current_layer}, 相邻层={neighbor_layer}, 差值={abs(current_layer - neighbor_layer)}")
                return True

        return False

    def is_wall_front(self, front):
        return front.bc_type == "wall" or front.bc_type == "prism-cap"

    def calculate_marching_distance(self):
        """计算节点推进距离"""
        verbose("计算节点推进步长...")
        self.current_step_size = 0.0
        if self.growth_method == "geometric":
            # 计算几何增长距离
            first_height = self.first_height
            growth_rate = self.growth_rate
            self.current_step_size = first_height * growth_rate**self.ilayer
        else:
            raise ValueError("未知的步长计算方法！")

        info(f"第{self.ilayer+1}层推进步长：{self.current_step_size:.6f}")

        for front in self.current_part.front_list:
            for node in front.node_elems:
                # 多方向推进：步长缩放已由 calculate_step_scale_coeff 处理
                if self.multi_direction:
                    node.marching_distance = self.current_step_size * node.local_step_factor
                    continue

                node.marching_distance = self.current_step_size
                # 如果只有一个相邻wall阵面
                # 不考虑推进方向倾斜的影响
                if len(node.node2front) < 2:
                    continue
                
                # 虚拟点需要特殊处理
                if getattr(node, 'is_virtual_point', False):
                    continue
                
                front1, front2 = node.node2front[:2]
                # 如果相邻阵面中只有一个wall阵面
                # 不考虑推进方向倾斜的影响
                if not (self.is_wall_front(front1) and self.is_wall_front(front2)):
                    continue
                
                # 节点推进方向与阵面法向的夹角, 节点推进方向投影到面法向
                direction = self._node_direction(node)
                proj1 = np.dot(direction, front1.normal)
                proj2 = np.dot(direction, front2.normal)

                
                if proj1 * proj2 < 0:
                    warning(
                        f"node{node.idx}推进方向与相邻阵面法向夹角大于90°，可能出现质量差单元！"
                    )
                
                mean_proj = np.mean([proj1, proj2])
                if abs(mean_proj) < 1e-10:
                    warning(f"node{node.idx}推进方向投影过小，保持默认推进步长。")
                    continue

                # 节点推进距离
                node.marching_distance = (
                    self.current_step_size
                    * node.local_step_factor
                    / mean_proj
                )  # min(abs(proj1), abs(proj2))

        verbose("计算节点推进步长..., Done.\n")

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

        verbose("计算节点初始推进方向...")
        for node_elem in self.front_node_list:
            if node_elem.matching_boundary:
                node_elem.marching_direction = (
                    node_elem.matching_boundary.marching_vector
                )
                continue

            if len(node_elem.node2front) < 2:
                verbose(
                    f"节点{node_elem.idx}在当前 part 中只有 1 个相邻阵面，通常为 match point，不计算推进方向！"
                )
                continue

            # 对于第一层凸角多方向点，在此不计算，也不光滑
            if isinstance(node_elem.marching_direction, list) and len(node_elem.marching_direction) > 1:
                continue

            front1, front2 = node_elem.node2front[:2]
            normal1 = np.array(front1.normal)
            normal2 = np.array(front2.normal)

            # 处理节点只有一侧有流向阵面 (prism-cap)，另一侧是法向阵面 (interior) 的情况
            if front1.bc_type == "interior" and front2.bc_type not in ("interior", "match"):
                # interior 阵面不使用其法向，只使用流向阵面的法向
                # 如果 interior 阵面有方向向量，则使用该方向
                if hasattr(front1, 'direction') and front1.direction is not None:
                    normal1 = np.array(front1.direction)
                else:
                    # 使用流向阵面的法向作为推进方向
                    node_elem.marching_direction = tuple(normal2)
                    continue
            elif front2.bc_type == "interior" and front1.bc_type not in ("interior", "match"):
                # interior 阵面不使用其法向，只使用流向阵面的法向
                if hasattr(front2, 'direction') and front2.direction is not None:
                    normal2 = np.array(front2.direction)
                else:
                    # 使用流向阵面的法向作为推进方向
                    node_elem.marching_direction = tuple(normal1)
                    continue

            # 推进方向不做角度加权，直接采用相邻阵面法向的等权平均
            avg_direction = (normal1 + normal2) / 2.0
            norm = np.linalg.norm(avg_direction)
            if norm > 1e-6:
                node_elem.marching_direction = tuple(avg_direction / norm)
            else:
                # 如果平均方向接近零，说明两个法向相反，使用第一个法向
                node_elem.marching_direction = tuple(normal1)

        verbose("计算节点初始推进方向..., Done.\n")

    def laplacian_smooth_normals(self):
        """拉普拉斯平滑节点推进方向 - 修复版本"""
        verbose("节点推进方向光滑....")
        
        # 第一次遍历：收集所有节点的初始方向
        initial_directions = {}
        for node_elem in self.front_node_list:
            initial_directions[node_elem.idx] = np.array(node_elem.marching_direction)
        
        # 进行多次光滑迭代
        for iteration in range(self.smooth_iterions):
            verbose(f"  光滑迭代 {iteration + 1}/{self.smooth_iterions}...")
            
            new_directions = {}
            
            for node_elem in self.front_node_list:
                # 边界匹配节点保持原始方向
                if node_elem.matching_boundary:
                    new_directions[node_elem.idx] = np.array(
                        node_elem.matching_boundary.marching_vector
                    )
                    continue

                num_neighbors = len(node_elem.node2node)
                if num_neighbors < 2:
                    # 相邻节点不足，保持初始方向
                    new_directions[node_elem.idx] = initial_directions[node_elem.idx]
                    continue

                # 计算加权平均方向
                current_dir_raw = node_elem.marching_direction
                if isinstance(current_dir_raw, list):
                    if len(current_dir_raw) > 0:
                        current_dir = np.array(current_dir_raw[0], dtype=float)
                    else:
                        continue
                else:
                    current_dir = np.array(current_dir_raw, dtype=float)
                
                # 确保 current_dir 是一维的
                if current_dir.ndim > 1:
                    current_dir = current_dir.flatten()
                
                # 根据当前方向的维度初始化 summation
                summation = np.zeros_like(current_dir)
                total_weight = 0.0

                current_pos = np.array(node_elem.coords)

                for neighbor in node_elem.node2node:
                    neighbor_dir_raw = neighbor.marching_direction
                    
                    # 处理 marching_direction 是列表的情况（多方向）
                    if isinstance(neighbor_dir_raw, list):
                        if len(neighbor_dir_raw) > 0:
                            # 取第一个方向
                            neighbor_dir = np.array(neighbor_dir_raw[0], dtype=float)
                        else:
                            continue
                    else:
                        neighbor_dir = np.array(neighbor_dir_raw, dtype=float)
                    
                    # 确保维度一致
                    if neighbor_dir.ndim > 1:
                        neighbor_dir = neighbor_dir.flatten()
                    
                    if len(neighbor_dir) != len(current_dir):
                        # 如果维度不一致，跳过该邻居
                        continue

                    # 计算几何权重：基于距离和角度差异
                    neighbor_pos = np.array(neighbor.coords)
                    distance = float(np.linalg.norm(current_pos - neighbor_pos))

                    # 避免除零
                    if distance < 1e-10:
                        distance = 1e-10

                    # 距离权重：距离越近影响越大
                    distance_weight = 1.0 / distance

                    # 角度权重：方向差异越小影响越大
                    dot_product = float(np.dot(current_dir, neighbor_dir))
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    angle = float(np.arccos(dot_product))

                    # 角度权重：使用高斯函数，角度差异越小权重越大
                    angle_weight = float(np.exp(-angle * angle / (2 * 0.5 * 0.5)))

                    # 综合权重
                    weight = distance_weight * angle_weight

                    summation += weight * neighbor_dir
                    total_weight += weight

                if total_weight > 1e-10:
                    # 计算新的推进方向
                    new_direction = (1 - self.relax_factor) * current_dir + \
                                   self.relax_factor * (summation / total_weight)

                    # 归一化
                    norm = np.linalg.norm(new_direction)
                    if norm > 1e-10:
                        new_direction /= norm

                    new_directions[node_elem.idx] = new_direction
                else:
                    new_directions[node_elem.idx] = initial_directions[node_elem.idx]
            
            # 更新所有节点的方向
            for node_elem in self.front_node_list:
                if node_elem.idx in new_directions:
                    # 对于凸点且需要多方向推进的节点，保持其方向列表不变
                    if node_elem.convex_flag and node_elem.num_multi_direction > 1:
                        # 凸点保持多方向列表，只更新第一个方向
                        if isinstance(node_elem.marching_direction, list):
                            node_elem.marching_direction[0] = tuple(new_directions[node_elem.idx])
                        else:
                            # 如果已经被转换成tuple，重新构造列表
                            node_elem.marching_direction = [tuple(new_directions[node_elem.idx])]
                    else:
                        node_elem.marching_direction = tuple(new_directions[node_elem.idx])
                    initial_directions[node_elem.idx] = new_directions[node_elem.idx]

        verbose("节点推进方向光滑..., Done.\n")

    def prepare_geometry_info(self):
        """准备几何信息"""
        self.compute_front_geometry()

        self.compute_point_normals()

        self.laplacian_smooth_normals()

    def apply_multi_direction_workflow(self):
        """多方向推进流程：初始化、方向光滑、步长缩放"""
        if not self.multi_direction or self.multi_direction_manager is None:
            return

        self.multi_direction_manager.run_layer_workflow(
            self.front_node_list,
            self.ilayer,
            smooth_iterations=self.smooth_iterions,
            relax_factor=self.relax_factor,
        )
        if self.ilayer == 0:
            debug(
                f"[多方向调试] 首层初始化后虚拟点数量: "
                f"{len(self.multi_direction_manager.virtual_points)}"
            )

    def log_multi_direction_debug_summary(self):
        """输出首层多方向关键调试信息"""
        if not self.multi_direction or self.ilayer != 0:
            return

        wall_front_count = sum(
            1 for front in self.current_part.front_list if front.bc_type == "wall"
        )
        prism_cap_count = sum(
            1 for front in self.current_part.front_list if front.bc_type == "prism-cap"
        )
        convex_nodes = [node for node in self.front_node_list if node.convex_flag]
        multi_nodes = [node for node in convex_nodes if node.num_multi_direction > 1]

        direction_hist = {}
        for node in multi_nodes:
            direction_hist[node.num_multi_direction] = (
                direction_hist.get(node.num_multi_direction, 0) + 1
            )

        debug(
            f"[多方向调试] 物面线段数量: wall={wall_front_count}, prism-cap={prism_cap_count}"
        )
        debug(
            f"[多方向调试] 凸角节点数量: {len(convex_nodes)}, "
            f"多方向节点数量: {len(multi_nodes)}"
        )

        if direction_hist:
            hist_text = ", ".join(
                [f"{k}方向:{v}个" for k, v in sorted(direction_hist.items())]
            )
            debug(f"[多方向调试] 多方向数量分布: {hist_text}")
        else:
            debug("[多方向调试] 多方向数量分布: 无")

        if multi_nodes:
            details = ", ".join(
                [
                    f"id={node.idx},角度={node.angle:.2f}°,方向数={node.num_multi_direction}"
                    for node in sorted(multi_nodes, key=lambda n: n.angle, reverse=True)
                ]
            )
            verbose(f"[多方向调试] 凸角角度详情: {details}")

    def log_first_layer_cell_summary(self, start_cell_idx):
        """输出首层新增单元数量和类型"""
        if self.ilayer != 0:
            return

        tri_count = 0
        quad_count = 0
        other_count = 0
        for cell_idx in range(start_cell_idx, self.num_cells):
            if isinstance(self.cell_container, dict):
                cell = self.cell_container.get(cell_idx)
            else:
                if cell_idx >= len(self.cell_container):
                    continue
                cell = self.cell_container[cell_idx]
            if cell is None:
                continue
            if isinstance(cell, Triangle):
                tri_count += 1
            elif isinstance(cell, Quadrilateral):
                quad_count += 1
            else:
                other_count += 1

        debug(
            f"[首层统计] 新增单元总数: {self.num_cells - start_cell_idx}, "
            f"四边形: {quad_count}, 三角形: {tri_count}, 其他: {other_count}"
        )
        sample_quads = []
        for cell_idx in range(start_cell_idx, self.num_cells):
            if isinstance(self.cell_container, dict):
                cell = self.cell_container.get(cell_idx)
            else:
                if cell_idx >= len(self.cell_container):
                    continue
                cell = self.cell_container[cell_idx]
            if not isinstance(cell, Quadrilateral):
                continue
            sample_quads.append((cell_idx, list(cell.node_ids)))
            if len(sample_quads) >= 10:
                break

        if sample_quads:
            verbose("[首层统计] 四边形样本(最多10个):")
            for cell_idx, node_ids in sample_quads:
                verbose(f"[首层统计]   单元{cell_idx}: 节点{node_ids}")
                coord_text = []
                for node_id in node_ids:
                    if 0 <= node_id < len(self.node_coords):
                        coord = self.node_coords[node_id]
                        if len(coord) >= 2:
                            coord_text.append(
                                f"{node_id}:({coord[0]:.6f}, {coord[1]:.6f})"
                            )
                if coord_text:
                    verbose(f"[首层统计]     坐标: {'; '.join(coord_text)}")

    def reconstruct_node2front(self):
        self.front_node_list = []
        processed_nodes = set()
        node_dict = {}  # 节点hash值到节点索引的映射
        for front in self.current_part.front_list:
            for i, node_elem in enumerate(front.node_elems):
                if node_elem.hash not in processed_nodes:
                    if not isinstance(node_elem, NodeElementALM):
                        # 将所有节点均转换为NodeElementALM类型
                        front.node_elems[i] = NodeElementALM.from_existing_node(
                            node_elem
                        )

                    processed_nodes.add(node_elem.hash)
                    node_dict[node_elem.hash] = front.node_elems[i]

                    # 为方便对节点进行遍历，收集所有节点
                    self.front_node_list.append(front.node_elems[i])
                    
                    # 清空节点的 node2front 列表，以便重新构建
                    front.node_elems[i].node2front = []
                else:
                    # 处理过的节点，直接取hash值对应的NodeElementALM对象
                    front.node_elems[i] = node_dict[node_elem.hash]

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
                verbose(
                    f"节点{node_elem.idx}，坐标{node_elem.coords} 在当前part内的邻接阵面数量小于2，不计算凹凸角信息！"
                )
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
            if isinstance(cross, np.ndarray):
                cross = cross[2] if cross.size >= 3 else cross.item()
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
            if self.multi_direction and node_elem.convex_flag and self.ilayer == 0:
                num_multi_direction = (
                    int(np.radians(node_elem.angle) / (1.1 * pi / 3)) + 1
                )
                # 仅对明显凸角启用多方向；平滑曲线小角度维持单方向
                node_elem.num_multi_direction = max(1, num_multi_direction)

                debug(
                    f"[凸角检测] 节点{node_elem.idx} 角度={node_elem.angle:.1f}° "
                    f"多方向数量={node_elem.num_multi_direction}"
                )

                if node_elem.num_multi_direction > 1:
                    delta = np.radians(node_elem.angle) / (
                        node_elem.num_multi_direction - 1
                    )
                    initial_vectors = normal1
    
                    # 将 marching_direction 设置为列表
                    node_elem.marching_direction = []
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

                    debug(
                        f"[多方向] 节点{node_elem.idx} 方向数量={len(node_elem.marching_direction)}"
                    )
                    node_elem.local_step_factor = 1.0
                else:
                    node_elem.local_step_factor = 1 - np.sign(thetam) * abs(thetam) / pi
            else:
                node_elem.num_multi_direction = 1
                # 第二层及以后，凸点不再分裂多方向，方向退化为单一向量
                if isinstance(node_elem.marching_direction, list):
                    if len(node_elem.marching_direction) > 0:
                        node_elem.marching_direction = tuple(node_elem.marching_direction[0])
                    else:
                        node_elem.marching_direction = tuple(normal1)

                if self.multi_direction and node_elem.convex_flag:
                    node_elem.local_step_factor = 1.0
                else:
                    node_elem.local_step_factor = 1 - np.sign(thetam) * abs(thetam) / pi

        verbose("计算阵面间夹角、凹凸标记、局部步长因子、多方向推进数量..., Done.\n")

    def build_front_rtree(self):
        verbose("构建辅助查询R-Tree...")
        # self.front_dict, self.space_index = build_space_index_with_RTree(
        #     self.current_part.front_list
        # )

        # 以下采用全局所有阵面构建R树，而不是只采用当前部件的阵面来构建，因为在
        # 多部件情况下，可能存在多个部件的阵面相交的情况，此时需要使用全局所有
        self.collect_all_boundary_fronts()

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
    def __init__(self, start_node, end_node, marching_vector, part_name):
        self.start_node = start_node
        self.end_node = end_node
        self.part_name = part_name

        if marching_vector is None:
            self.marching_vector = np.array(end_node.coords) - np.array(
                start_node.coords
            )
            self.marching_vector /= np.linalg.norm(self.direction_vector)
            self.marching_vector = self.marching_vector.tolist()
        else:
            self.marching_vector = marching_vector

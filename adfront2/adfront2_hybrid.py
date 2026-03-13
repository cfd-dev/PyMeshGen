import heapq
import numpy as np
from math import sqrt
from data_structure.front2d import Front
from adfront2 import Adfront2
from utils.message import info, debug, verbose, warning, error
from data_structure.basic_elements import NodeElement, LineSegment, Triangle, Quadrilateral
from utils.timer import TimeSpan
from utils.geom_toolkit import (
    calculate_angle,
    is_left2d,
    quadrilateral_area,
    fast_distance_check,
    point_to_segment_distance,
)
from optimize.mesh_quality import triangle_shape_quality, quadrilateral_quality2


class Adfront2Hybrid(Adfront2):
    """基于阵面推进法生成四边形网格(DEMO)，还有多处TODO，要进一步调试以达到最佳效果
    主要步骤：
    1. 生成2个新点，并在候选点列表中尝试选择2个点，构成一个质量最高，且满足要求的四边形
    2. 若无法构成合适的四边形，则退回生成三角形的逻辑，生成一个三角形
    3. 更新四边形单元数据结构
    4. 更新三角形单元数据结构
    5. 重复1-4，直到所有阵面被处理完毕
    参考文献：
    [1]陈建军,郑建靖,季廷炜,等.前沿推进曲面四边形网格生成算法[J].计算力学学报,2011,28(05):779-784.
    """

    def __init__(
        self,
        boundary_front,
        sizing_system,
        node_coords=None,
        param_obj=None,
        visual_obj=None,
    ):
        super().__init__(
            boundary_front,
            sizing_system,
            node_coords,
            param_obj,
            visual_obj,
        )

        # TODO discount取多少合适？quality_criterion取多少合适？proximity_tol取多少合适？
        self.al = 0.8  # 在几倍范围内搜索，对于四边形网格生成，al=0.8，对于三角形网格生成，al=3.0
        self.discount = 0.8  # Pbest质量折扣系数，discount越小，选择Pbest的概率越小，已测试该值为0.8时，效果较好
        self.mesh_type = 3  # 1-三角形，2-直角三角形，3-三角形/四边形混合（在生成混合网格的对象中，默认为3）
        self.quality_criterion = 0.5  # 四边形质量阈值，低于这个质量的四边形将被舍弃，已测试该值为0.5时，效果较好
        # 阵面与节点邻近的距离与当地网格步长的比例，小于该比例将返回邻近True
        self.proximity_tol = 0.5 # 已测试该值为0.5时，效果较好
        self.progress_interval = 50
        self.sort_front = False  # 是否对阵面进行排序，四边形网格生成时默认不排序
        self.bbox = None

        self.initialize_data()

    def initialize_data(self):
        """初始化数据"""
        for front in self.front_list:
            front.al = self.al

        # 根据初始front_list计算边界框
        min_x = min(front.node_elems[0].coords[0] for front in self.front_list)
        max_x = max(front.node_elems[1].coords[0] for front in self.front_list)
        min_y = min(front.node_elems[0].coords[1] for front in self.front_list)
        max_y = max(front.node_elems[1].coords[1] for front in self.front_list)
        self.bbox = (min_x, min_y, max_x, max_y)

    def calculate_gap_criterion(self):
        """计算不同部件阵面之间的最小距离"""
        # 判断是否要排序推进，如果已经开启排序推进，则不再判断
        if self.sort_front:
            return

        # 每间隔一定单元数判断一次是否要开启排序推进
        if self.num_cells % self.progress_interval != 0:
            return

        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]

        # 当距离小于0.15倍计算域大小时，开启阵面排序
        safe_distance_sq = 0.15 * (width + height) / 2
        safe_distance_sq *= safe_distance_sq

        for front1 in self.front_list:
            for front2 in self.front_list:
                if front1.part_name != front2.part_name:
                    p0 = front1.node_elems[0].coords
                    p1 = front1.node_elems[1].coords
                    q0 = front2.node_elems[0].coords
                    q1 = front2.node_elems[1].coords
                    if fast_distance_check(p0, p1, q0, q1, safe_distance_sq):
                        info("部件之间阵面最小距离较小，按阵面长短顺序推进生成...")
                        self.sort_front = True
                        return

    def get_base_front(self):
        if self.sort_front:
            heapq.heapify(self.front_list)
            self.base_front = heapq.heappop(self.front_list)
        else:
            self.base_front = self.front_list.pop(0)

    def generate_elements(self):
        timer = TimeSpan("开始推进生成三角形/四边形混合网格...")
        while self.front_list:
            # 对当前阵面构建node2front，要在弹出base_front前完成
            self.reconstruct_node2front()

            self.calculate_gap_criterion()

            self.get_base_front()

            spacing = self.sizing_system.spacing_at(self.base_front.center)

            if (
                self.base_front.node_ids == [131, 132]
                or self.base_front.node_ids == [239, 214]
                or self.base_front.node_ids == [183, 184]
            ):
                # self.debug_save()
                kkk = 0

            self.add_new_points_for_quad(spacing)

            self.search_candidates(self.base_front.al * spacing)

            # self.debug_draw()

            self.select_point_for_quad(spacing)

            self.add_new_point_for_tri(spacing)

            self.select_point_for_tri()

            self.update_data_quad()

            self.update_data_tri()

            self.show_progress()

        self.construct_unstr_grid()

        timer.show_to_console("完成三角形/四边形混合网格生成.")

        return self.unstr_grid

    def add_new_point_for_tri(self, spacing):
        """如果四边形生成失败，尝试生成三角形"""
        if self.pselected is not None:
            return

        self.mesh_type = 1
        self.add_new_point(spacing)

    def select_point_for_tri(self):
        """尝试选择点，生成三角形"""
        # 如果已经选择了点，直接返回
        if self.pselected is not None:
            return

        self.select_point()

        # if self.pselected is None:
        #     info("找不到合适的候选点，按阵面长短顺序推进生成...")
        #     heapq.heapify(self.front_list)
        #     self.sort_front = True

    def update_data_tri(self):
        if self.pselected is None:
            return

        # 如果pselected是一个列表，且长度为2，则直接返回
        if isinstance(self.pselected, list) and len(self.pselected) == 2:
            return

        # 更新节点
        self.update_nodes()

        # 更新阵面
        new_front1 = Front(
            node_elem1=self.base_front.node_elems[0],
            node_elem2=self.pselected,
            idx=-1,
            bc_type="interior",
            part_name=self.base_front.part_name,
            al=self.al,
        )
        new_front2 = Front(
            node_elem1=self.pselected,
            node_elem2=self.base_front.node_elems[1],
            idx=-1,
            bc_type="interior",
            part_name=self.base_front.part_name,
            al=self.al,
        )

        # 更新阵面
        self.update_fronts([new_front1, new_front2])

        # 更新单元
        new_cell = Triangle(
            p1=self.base_front.node_elems[0],
            p2=self.base_front.node_elems[1],
            p3=self.pselected,
            part_name="interior-triangle",
            idx=self.num_cells,
        )

        self.update_cells(new_cell)

    def update_data_quad(self):
        if (
            self.pselected is None  # 如果pselected是None
            or (not isinstance(self.pselected, list))  # 或者pselected不是一个列表
            or len(self.pselected) != 2  # 或者长度不为2
        ):
            return

        # 更新节点
        self.update_nodes()

        # 更新阵面
        new_front1 = Front(
            node_elem1=self.base_front.node_elems[0],
            node_elem2=self.pselected[0],
            idx=-1,
            bc_type="interior",
            part_name=self.base_front.part_name,
            al=self.al,
        )
        new_front2 = Front(
            node_elem1=self.pselected[1],
            node_elem2=self.base_front.node_elems[1],
            idx=-1,
            bc_type="interior",
            part_name=self.base_front.part_name,
            al=self.al,
        )
        new_front3 = Front(
            node_elem1=self.pselected[0],
            node_elem2=self.pselected[1],
            idx=-1,
            bc_type="interior",
            part_name=self.base_front.part_name,
            al=self.al,
        )

        self.update_fronts([new_front1, new_front2, new_front3])

        # 更新单元
        new_cell = Quadrilateral(
            p1=self.base_front.node_elems[0],
            p2=self.base_front.node_elems[1],
            p3=self.pselected[1],
            p4=self.pselected[0],
            part_name="interior",
            idx=self.num_cells,
        )

        self.update_cells(new_cell)

    def select_point_for_quad(self, spacing):
        # 预计算基准点坐标
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords

        # 候选节点质量评估参数
        quality_criterion = self.quality_criterion
        discount = self.discount

        # 生成所有候选节点对（排除相同节点）
        node_pairs = [
            (n1, n2)
            for n1 in self.node_candidates
            for n2 in self.node_candidates
            if n1 != n2
        ]

        # 统一处理四种候选情况
        candidate_sources = [
            # 常规候选对
            (node_pairs, 1.0),
            # pbest[1] 组合
            ([(n, self.pbest[1]) for n in self.node_candidates], discount),
            # pbest[0] 组合
            ([(self.pbest[0], n) for n in self.node_candidates], discount),
            # pbest对
            ([(self.pbest[0], self.pbest[1])], discount**2),
        ]

        # 统一处理质量计算
        scored_candidates = []
        for candidates, discount in candidate_sources:
            for elem1, elem2 in candidates:
                quality = quadrilateral_quality2(p0, p1, elem2.coords, elem1.coords)
                discounted_quality = quality * discount
                if discounted_quality > quality_criterion:
                    scored_candidates.append((discounted_quality, elem1, elem2))

        # 按质量降序排序（质量高的在前）
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        self.pselected = None
        self.best_flag = [False] * 2
        for quality, node_elem1, node_elem2 in scored_candidates:
            if not (
                is_left2d(p0, p1, node_elem1.coords)
                and is_left2d(p0, p1, node_elem2.coords)
            ):
                continue

            if self.is_cross_quad(node_elem1, node_elem2):
                continue

            if self.proximity_check(
                node_elem1, node_elem2, self.proximity_tol * spacing
            ):
                continue

            if self.size_too_big(node_elem1, node_elem2, spacing):
                continue

            self.pselected = [node_elem1, node_elem2]
            break

    def size_too_big(self, node_elem1, node_elem2, spacing):
        """检查当前新增的阵面是否满足长度要求，或者待新增的单元是否满足面积要求"""
        # TODO 是否要判断新增单元尺寸？如何考虑？待完善
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords

        ratio = 2.0
        # line1 = LineSegment(node_elem1.coords, node_elem2.coords)
        # line2 = LineSegment(node_elem1.coords, p0)
        # line3 = LineSegment(node_elem2.coords, p1)
        # if (
        #     (line1.length > ratio * spacing)
        #     or (line2.length > ratio * spacing)
        #     or (line3.length > ratio * spacing)
        # ):
        #     return True

        # 计算新增面积
        ratio = 1.3 * 1.3
        if (
            quadrilateral_area(p0, p1, node_elem2.coords, node_elem1.coords)
            > ratio * spacing * spacing
        ):
            return True

        return False

    def proximity_check(self, node_elem1, node_elem2, distance):
        """检查当前新增的阵面是否过于靠近已有节点"""
        # 其他点到待新增阵面的距离不能太近
        edges = [
            (node_elem1, node_elem2),
            (self.base_front.node_elems[0], node_elem1),
            (self.base_front.node_elems[1], node_elem2),
        ]

        for edge in edges:
            p0 = edge[0]
            p1 = edge[1]

            for node in self.node_candidates:
                if node.idx in [p0.idx, p1.idx]:  # 当前阵面自身的节点不参与判断
                    continue
                # node到edge的距离
                dis = point_to_segment_distance(node.coords, p0.coords, p1.coords)
                if dis < distance:
                    return True

        # 待新增点到其他阵面的距离不能太近
        nodes = [node_elem1, node_elem2]
        new_fronts = {
            (node_elem1.idx, self.base_front.node_elems[0].idx),
            (self.base_front.node_elems[1].idx, node_elem2.idx),
            (node_elem2.idx, node_elem1.idx),
        }
        for node in nodes:
            for front in self.front_candidates:
                # 如果当前阵面是待新增阵面，则不参与判断
                if tuple(front.node_ids) in new_fronts:
                    continue

                # 如果阵面中含有待新增点，则不参与判断
                if node.idx in front.node_ids:
                    continue

                # node到front的距离
                dis = point_to_segment_distance(
                    node.coords,
                    front.node_elems[0].coords,
                    front.node_elems[1].coords,
                )
                if dis < distance:
                    return True

        return False

    def is_cross_quad(self, node_elem0, node_elem1):
        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]

        line1 = LineSegment(node_elem0, p0)
        line2 = LineSegment(node_elem1, p1)
        line3 = LineSegment(node_elem0, node_elem1)

        for front in self.front_candidates:
            front_line = LineSegment(front.node_elems[0], front.node_elems[1])
            if (
                front_line.is_intersect(line1)
                or front_line.is_intersect(line2)
                or front_line.is_intersect(line3)
            ):
                return True

        cell_to_add = Quadrilateral(p0, p1, node_elem1, node_elem0)

        for existed_cell in self.cell_candidates:
            if len(existed_cell.node_ids) == 3 and cell_to_add.is_intersect_triangle(
                existed_cell
            ):
                return True
            elif len(existed_cell.node_ids) == 4 and cell_to_add.is_intersect_quad(
                existed_cell
            ):
                return True

        return False

    def add_new_points_for_quad(self, spacing):
        """计算四边形的2个新顶点"""
        theta1 = 120
        theta2 = 200

        l = self.base_front.length
        d0 = spacing
        # TODO 步长如何计算合适？
        # d = min(1.25 * l, max(0.8 * l, d0))  # 文献方法
        d = d0  # 直接取为当地步长

        # 计算新顶点的坐标
        self.pbest = []
        temp_num_nodes = self.num_nodes
        normal_vec = np.array(self.base_front.normal)
        for base_p in self.base_front.node_elems:
            neighbor1, neighbor2 = base_p.node2front
            # 左侧阵面的另一个点
            pa = next(node for node in neighbor1.node_elems if node.hash != base_p.hash)
            # 右侧阵面的另一个点
            pb = next(node for node in neighbor2.node_elems if node.hash != base_p.hash)

            # 从右侧阵面旋转到左侧阵面的角度，也即阵面间夹角
            angle = calculate_angle(pb.coords, base_p.coords, pa.coords)

            if angle < theta1:
                # base_p是基准阵面的左端点，则对应pa-->base_p(base_p0)-->pb(base_p1)
                if base_p.hash == self.base_front.node_elems[0].hash:
                    pnew_elem = pa
                # base_p是基准阵面的右端点，则对应pa(base_p0)-->base_p(base_p1)-->pb
                elif base_p.hash == self.base_front.node_elems[1].hash:
                    pnew_elem = pb

            elif angle > theta2:
                pnew = np.array(base_p.coords) + normal_vec * d
                pnew_elem = NodeElement(
                    coords=pnew.tolist(),
                    idx=temp_num_nodes,
                    part_name="interior",
                    bc_type="interior",
                )

                temp_num_nodes += 1
            else:
                # 角平分线
                angle = np.radians(angle) / 2
                rotation_matrix = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                # 角平分线方向向量
                rotated_vector = np.dot(rotation_matrix, neighbor2.direction)
                pnew = np.array(base_p.coords) + rotated_vector * d

                pnew_elem = NodeElement(
                    coords=pnew.tolist(),
                    idx=temp_num_nodes,
                    part_name="interior",
                    bc_type="interior",
                )

                temp_num_nodes += 1

            self.pbest.append(pnew_elem)

    def reconstruct_node2front(self):
        """重构node2front列表，按照先左侧阵面，后右侧阵面的顺序存储"""
        num_neighbors = 2
        for front in self.front_list:
            for node_elem in front.node_elems:
                if len(node_elem.node2front) == 0:
                    node_elem.node2front = [None] * num_neighbors  # 预分配2个位置

            # node_elem在front中是起点，则front是在后面
            for i, node_elem in enumerate(front.node_elems):
                node_elem.node2front[(i + 1) % num_neighbors] = front

        # 检查每个节点的邻阵面数量是否为2
        for front in self.front_list:
            for node_elem in front.node_elems:
                if not all(node_elem.node2front):
                    raise ValueError(f"节点 {node_elem.idx} 的邻阵面数量不足2")

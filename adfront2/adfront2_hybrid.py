import heapq
import numpy as np
from math import sqrt
from front2d import Front
from adfront2 import Adfront2
from message import info, debug, verbose, warning, error
from basic_elements import NodeElement, LineSegment, Triangle, Quadrilateral
from utils.timer import TimeSpan
from geom_toolkit import (
    triangle_quality,
    calculate_angle,
    is_left2d,
    quadrilateral_quality2,
    quadrilateral_area,
    _fast_distance_check,
    point_to_segment_distance,
)


class Adfront2Hybrid(Adfront2):
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

        self.al = 0.8  # 在几倍范围内搜索
        self.discount = 0.9  # Pbest质量系数，discount越小，选择Pbest的概率越小
        self.mesh_type = 3  # 1-三角形，2-直角三角形，3-三角形/四边形混合

    def generate_elements(self):
        timer = TimeSpan("开始推进生成三角形/四边形混合网格...")
        while self.front_list:
            # 对当前阵面构建node2front，要在弹出base_front前完成
            self.reconstruct_node2front()
            # 依次弹出阵面
            self.base_front = self.front_list.pop(0)

            spacing = self.sizing_system.spacing_at(self.base_front.center)

            if self.base_front.node_ids == [237, 238]:
                # self.debug_save()
                kkk = 0

            self.add_new_points_for_quad(spacing)

            self.search_candidates(self.base_front.al * spacing)

            self.debug_draw()

            self.select_point_for_quad(spacing)

            self.add_new_point_for_tri(spacing)

            self.select_point_for_tri()

            self.update_cells_quad()

            self.update_cells_tri()

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

        # 预计算基准点坐标
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords

        # 存储带质量的候选节点元组 (质量, 节点)
        scored_candidates = []
        # 遍历所有候选节点计算质量
        for node_elem in self.node_candidates:
            quality = triangle_quality(p0, p1, node_elem.coords)
            if quality > 0:
                scored_candidates.append((quality, node_elem))

        pbest_quality = (
            triangle_quality(p0, p1, self.pbest.coords) * self.discount
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

        if self.pselected == None:
            warning(
                f"阵面{self.base_front.node_ids}候选点列表中没有合适的点，扩大搜索范围！"
            )
            self.base_front.al *= 1.2
            self.debug_save()

        return self.pselected

    def update_cells_tri(self):
        if self.pselected is None:
            return

        if isinstance(self.pselected, list) and len(self.pselected) == 2:
            return

        # 更新节点
        if self.pselected is not None:
            node_hash = self.pselected.hash
            if node_hash not in self.node_hash_list:
                self.node_hash_list.add(node_hash)
                self.node_coords.append(self.pselected.coords)
                self.pselected.idx = self.num_nodes
                self.add_elems_to_space_index(
                    [self.pselected], self.space_index_node, self.node_dict
                )

                self.num_nodes += 1

        # 更新阵面
        new_front1 = Front(
            self.base_front.node_elems[0],
            self.pselected,
            -1,
            "interior",
            "internal",
        )
        new_front2 = Front(
            self.pselected,
            self.base_front.node_elems[1],
            -1,
            "interior",
            "internal",
        )

        # 判断front_list中是否存在new_front，若不存在，
        # 则将其压入front_list，若已存在，则将其从front_list中删除
        new_fronts = [new_front1, new_front2]
        front_hashes = {f.hash for f in self.front_list}
        for chk_fro in new_fronts:
            if chk_fro.hash not in front_hashes:
                # heapq.heappush(self.front_list, chk_fro)
                self.front_list.append(chk_fro)

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

        # heapq.heapify(self.front_list)  # 重新堆化

        # 更新单元
        new_cell = Triangle(
            p1=self.base_front.node_elems[0],
            p2=self.base_front.node_elems[1],
            p3=self.pselected,
            part_name="interior",
            idx=self.num_cells,
        )

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

    def update_cells_quad(self):
        if (
            self.pselected is None
            or (not isinstance(self.pselected, list))
            or len(self.pselected) != 2
        ):
            return

        # 更新节点，如果pselected是一个列表，且长度为2，此时将pselected添加到node_coords中，同时更新num_nodes
        for i, node in enumerate(self.pselected):
            node_hash = node.hash
            if node_hash not in self.node_hash_list:
                self.node_hash_list.add(node_hash)
                self.node_coords.append(node.coords)
                self.pselected[i].idx = self.num_nodes
                self.add_elems_to_space_index(
                    [node], self.space_index_node, self.node_dict
                )

                self.num_nodes += 1

        # 更新阵面
        new_front1 = Front(
            self.base_front.node_elems[0],
            self.pselected[0],
            -1,
            "interior",
            "internal",
        )
        new_front2 = Front(
            self.pselected[1],
            self.base_front.node_elems[1],
            -1,
            "interior",
            "internal",
        )
        new_front3 = Front(
            self.pselected[0],
            self.pselected[1],
            -1,
            "interior",
            "internal",
        )
        # 判断front_list中是否存在new_front，若不存在，
        # 则将其压入front_list，若已存在，则将其从front_list中删除
        new_fronts = [new_front1, new_front2, new_front3]
        front_hashes = {f.hash for f in self.front_list}
        for chk_fro in new_fronts:
            if chk_fro.hash not in front_hashes:
                self.front_list.append(chk_fro)

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

        # 更新单元
        new_cell = Quadrilateral(
            p1=self.base_front.node_elems[0],
            p2=self.base_front.node_elems[1],
            p3=self.pselected[1],
            p4=self.pselected[0],
            part_name="interior",
            idx=self.num_cells,
        )

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

    def select_point_for_quad(self, spacing):
        # 预计算基准点坐标
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords

        # 候选节点质量评估参数
        quality_criterion = 0.5
        discount = self.discount
        scored_candidates = []

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
        for candidates, weight in candidate_sources:
            for elem1, elem2 in candidates:
                quality = quadrilateral_quality2(p0, p1, elem2.coords, elem1.coords)
                weighted_quality = quality * weight
                if weighted_quality > quality_criterion:
                    scored_candidates.append((weighted_quality, elem1, elem2))

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

            # 计算新增边长
            line1 = LineSegment(node_elem1.coords, node_elem2.coords)
            line2 = LineSegment(node_elem1.coords, p0)
            line3 = LineSegment(node_elem2.coords, p1)
            if (
                (line1.length > 2.0 * spacing)
                or (line2.length > 2.0 * spacing)
                or (line3.length > 2.0 * spacing)
            ):
                continue

            if self.proximity_check(node_elem1, node_elem2, 0.3 * spacing):
                continue

            if (
                sqrt(quadrilateral_area(p0, p1, node_elem2.coords, node_elem1.coords))
                > 2.0 * spacing
            ):
                continue

            self.pselected = [node_elem1, node_elem2]
            break

    def proximity_check(self, node_elem1, node_elem2, distance):
        """检查当前新增的阵面是否过于靠近已有节点"""
        # 待新增阵面
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

    def is_cross_quad(self, node_elem0, node_elem1):
        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]

        line1 = LineSegment(node_elem0, p0)
        line2 = LineSegment(node_elem1, p1)
        line3 = LineSegment(node_elem0, node_elem1)

        flag = False
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

        return flag

    def add_new_points_for_quad(self, spacing):
        """计算四边形的2个新顶点"""
        theta1 = 120
        theta2 = 200

        l = self.base_front.length
        d0 = spacing
        d = min(1.25 * l, max(0.8 * l, d0))

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

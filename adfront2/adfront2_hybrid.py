#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于阵面推进法的三角形/四边形混合网格生成器

主要步骤：
1. 对每个基准阵面，优先尝试生成四边形
2. 若无法构成合适的四边形，则退回生成三角形的逻辑
3. 更新四边形/三角形单元数据结构
4. 重复 1-3，直到所有阵面被处理完毕

参考文献：
[1] 陈建军，郑建靖，季廷炜，等。前沿推进曲面四边形网格生成算法 [J].计算力学学报，2011,28(05):779-784.
[2] Lo S H. A new mesh generation scheme for arbitrary planar domains[J]. 
    International Journal for Numerical Methods in Engineering, 1985, 21(8): 1403-1426.
"""

import heapq
import numpy as np
from math import sqrt, radians, cos, sin
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
    calculate_distance,
    is_convex_quad,
)
from optimize.mesh_quality import triangle_shape_quality, quadrilateral_quality2


class Adfront2Hybrid(Adfront2):
    """
    基于阵面推进法的混合网格生成器
    
    核心策略：
    1. 推进过程中优先直接生成四边形
    2. 只有在无法生成四边形时才退化为三角形
    3. 无需后处理合并步骤
    
    改进内容（相比初始版本）：
    1. 优化的四边形生成策略 - 更严格的几何质量控制（凸性检查、尺寸检查）
    2. 角点处理优化 - 更好的锐角/钝角区域处理
    3. 尺寸过渡控制 - 更平滑的网格尺寸过渡
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

        # === 推进参数 ===
        self.al = 1.0  # 搜索半径系数（混合网格）
        self.discount = 0.85  # Pbest 质量折扣系数

        # === 质量控制参数 ===
        self.quality_criterion = 0.4  # 四边形质量阈值（0-1，越高越严格）
        self.proximity_tol = 0.6  # 邻近检查容差
        self.size_ratio_tol = 1.5  # 尺寸比例容差

        # === 角点处理参数 ===
        self.sharp_angle_tol = 75.0  # 锐角阈值（度）
        self.obtuse_angle_tol = 150.0  # 钝角阈值（度）

        # === 其他参数 ===
        self.mesh_type = 3  # 1-三角形，2-直角三角形，3-三角形/四边形混合
        self.progress_interval = 50
        self.sort_front = False  # 是否对阵面进行排序
        self.bbox = None

        self.initialize_data()

    def initialize_data(self):
        """初始化数据"""
        for front in self.front_list:
            front.al = self.al

        # 计算边界框
        min_x = min(front.node_elems[0].coords[0] for front in self.front_list)
        max_x = max(front.node_elems[1].coords[0] for front in self.front_list)
        min_y = min(front.node_elems[0].coords[1] for front in self.front_list)
        max_y = max(front.node_elems[1].coords[1] for front in self.front_list)
        self.bbox = (min_x, min_y, max_x, max_y)

        verbose(f"计算域边界：x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")

    def calculate_gap_criterion(self):
        """计算不同部件阵面之间的最小距离，决定是否开启排序"""
        if self.sort_front:
            return

        if self.num_cells % self.progress_interval != 0:
            return

        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]

        # 当距离小于 0.15 倍计算域大小时，开启阵面排序
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
        """获取基准阵面"""
        if self.sort_front:
            heapq.heapify(self.front_list)
            self.base_front = heapq.heappop(self.front_list)
        else:
            self.base_front = self.front_list.pop(0)

    def generate_elements(self):
        """主推进循环"""
        timer = TimeSpan("开始推进生成三角形/四边形混合网格...")

        while self.front_list:
            # 重构邻接关系
            self.reconstruct_node2front()

            # 检查是否需要排序
            self.calculate_gap_criterion()

            # 获取基准阵面
            self.get_base_front()

            spacing = self.sizing_system.spacing_at(self.base_front.center)

            # 尝试生成四边形
            self.add_new_points_for_quad(spacing)
            self.search_candidates(self.base_front.al * spacing)
            self.select_point_for_quad(spacing)

            # 回退到三角形生成（子函数内部有判断）
            self.add_new_point_for_tri(spacing)
            self.select_point_for_tri()

            # 更新数据（子函数内部有判断）
            self.update_data_quad()
            self.update_data_tri()

            self.show_progress()

        self.construct_unstr_grid()
        timer.show_to_console("完成三角形/四边形混合网格生成.")

        return self.unstr_grid

    # =========================================================================
    # 四边形生成相关方法
    # =========================================================================

    def add_new_points_for_quad(self, spacing):
        """
        计算四边形的 2 个新顶点（理想点）

        策略：
        1. 锐角区域：使用邻接阵面的另一个点
        2. 钝角区域：使用法向方向
        3. 普通区域：使用角平分线方向
        """
        theta_sharp = self.sharp_angle_tol
        theta_obtuse = self.obtuse_angle_tol

        l = self.base_front.length
        d0 = spacing

        # 步长计算：考虑当地步长和阵面长度
        d = min(1.2 * l, max(0.9 * l, d0))

        self.pbest = []
        temp_num_nodes = self.num_nodes
        normal_vec = np.array(self.base_front.normal)

        for base_p in self.base_front.node_elems:
            neighbor1, neighbor2 = base_p.node2front

            # 获取邻接阵面的另一个点
            pa = next(node for node in neighbor1.node_elems if node.hash != base_p.hash)
            pb = next(node for node in neighbor2.node_elems if node.hash != base_p.hash)

            # 阵面间夹角
            angle = calculate_angle(pb.coords, base_p.coords, pa.coords)

            if angle < theta_sharp:
                # 锐角区域：使用邻接点
                if base_p.hash == self.base_front.node_elems[0].hash:
                    pnew_elem = pa
                else:
                    pnew_elem = pb

            elif angle > theta_obtuse:
                # 钝角区域：使用法向
                pnew = np.array(base_p.coords) + normal_vec * d * 0.9
                pnew_elem = NodeElement(
                    coords=pnew.tolist(),
                    idx=temp_num_nodes,
                    part_name="interior",
                    bc_type="interior",
                )
                temp_num_nodes += 1
            else:
                # 普通区域：角平分线方向
                half_angle = radians(angle) / 2
                rotation_matrix = np.array(
                    [[cos(half_angle), -sin(half_angle)],
                     [sin(half_angle), cos(half_angle)]]
                )
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

    def select_point_for_quad(self, spacing):
        """
        选择最佳的四边形顶点组合

        策略：
        1. 评估所有候选节点对的质量
        2. 应用几何约束（左手法则、交叉检查、邻近检查、尺寸检查、凸性检查）
        3. 选择质量最高且满足约束的组合
        """
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords

        quality_criterion = self.quality_criterion
        discount = self.discount

        # 生成候选节点对
        node_pairs = [
            (n1, n2)
            for n1 in self.node_candidates
            for n2 in self.node_candidates
            if n1.idx != n2.idx
        ]

        # 四种候选来源
        candidate_sources = [
            (node_pairs, 1.0),  # 常规候选对
            ([(n, self.pbest[1]) for n in self.node_candidates], discount),
            ([(self.pbest[0], n) for n in self.node_candidates], discount),
            ([(self.pbest[0], self.pbest[1])], discount ** 2),
        ]

        # 计算质量并评分
        scored_candidates = []
        for candidates, disc in candidate_sources:
            for elem1, elem2 in candidates:
                quality = quadrilateral_quality2(p0, p1, elem2.coords, elem1.coords)
                discounted_quality = quality * disc
                if discounted_quality > quality_criterion:
                    scored_candidates.append((discounted_quality, elem1, elem2))

        # 按质量降序排序
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        self.pselected = None
        for quality, node_elem1, node_elem2 in scored_candidates:
            # 几何约束检查
            if not (is_left2d(p0, p1, node_elem1.coords) and
                    is_left2d(p0, p1, node_elem2.coords)):
                continue

            if self.is_cross_quad(node_elem1, node_elem2):
                continue

            if self.proximity_check(node_elem1, node_elem2, self.proximity_tol * spacing):
                continue

            if self.size_too_big(node_elem1, node_elem2, spacing):
                continue

            # 额外检查：四边形凸性
            if not is_convex_quad(p0, p1, node_elem2.coords, node_elem1.coords):
                continue

            self.pselected = [node_elem1, node_elem2]
            break

    def is_cross_quad(self, node_elem0, node_elem1):
        """检查四边形是否与现有几何交叉"""
        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]

        line1 = LineSegment(node_elem0, p0)
        line2 = LineSegment(node_elem1, p1)
        line3 = LineSegment(node_elem0, node_elem1)

        for front in self.front_candidates:
            front_line = LineSegment(front.node_elems[0], front.node_elems[1])
            if (front_line.is_intersect(line1) or
                front_line.is_intersect(line2) or
                front_line.is_intersect(line3)):
                return True

        cell_to_add = Quadrilateral(p0, p1, node_elem1, node_elem0)

        for existed_cell in self.cell_candidates:
            if len(existed_cell.node_ids) == 3:
                if cell_to_add.is_intersect_triangle(existed_cell):
                    return True
            elif len(existed_cell.node_ids) == 4:
                if cell_to_add.is_intersect_quad(existed_cell):
                    return True

        return False

    def proximity_check(self, node_elem1, node_elem2, distance):
        """检查新增阵面是否过于靠近已有节点"""
        edges = [
            (node_elem1, node_elem2),
            (self.base_front.node_elems[0], node_elem1),
            (self.base_front.node_elems[1], node_elem2),
        ]

        for edge in edges:
            p0, p1 = edge[0], edge[1]
            for node in self.node_candidates:
                if node.idx in [p0.idx, p1.idx]:
                    continue
                dis = point_to_segment_distance(node.coords, p0.coords, p1.coords)
                if dis < distance:
                    return True

        # 检查新点到其他阵面的距离
        nodes = [node_elem1, node_elem2]
        new_fronts = {
            (node_elem1.idx, self.base_front.node_elems[0].idx),
            (self.base_front.node_elems[1].idx, node_elem2.idx),
            (node_elem2.idx, node_elem1.idx),
        }

        for node in nodes:
            for front in self.front_candidates:
                if tuple(front.node_ids) in new_fronts:
                    continue
                if node.idx in front.node_ids:
                    continue
                dis = point_to_segment_distance(
                    node.coords,
                    front.node_elems[0].coords,
                    front.node_elems[1].coords,
                )
                if dis < distance:
                    return True

        return False

    def size_too_big(self, node_elem1, node_elem2, spacing):
        """检查单元尺寸是否过大"""
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords

        # 面积检查
        area_ratio = 1.5
        if quadrilateral_area(p0, p1, node_elem2.coords, node_elem1.coords) > \
           area_ratio * spacing * spacing:
            return True

        # 边长检查
        edge_lengths = [
            calculate_distance(node_elem1.coords, node_elem2.coords),
            calculate_distance(node_elem1.coords, p0),
            calculate_distance(node_elem2.coords, p1),
        ]

        for length in edge_lengths:
            if length > self.size_ratio_tol * spacing:
                return True

        return False

    def update_data_quad(self):
        """更新四边形相关数据"""
        if (self.pselected is None or
            not isinstance(self.pselected, list) or
            len(self.pselected) != 2):
            return

        # 更新节点
        self.update_nodes()

        # 更新阵面
        new_fronts = [
            Front(
                node_elem1=self.base_front.node_elems[0],
                node_elem2=self.pselected[0],
                idx=-1,
                bc_type="interior",
                part_name=self.base_front.part_name,
                al=self.al,
            ),
            Front(
                node_elem1=self.pselected[1],
                node_elem2=self.base_front.node_elems[1],
                idx=-1,
                bc_type="interior",
                part_name=self.base_front.part_name,
                al=self.al,
            ),
            Front(
                node_elem1=self.pselected[0],
                node_elem2=self.pselected[1],
                idx=-1,
                bc_type="interior",
                part_name=self.base_front.part_name,
                al=self.al,
            ),
        ]

        self.update_fronts(new_fronts)

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

    # =========================================================================
    # 三角形生成相关方法
    # =========================================================================

    def add_new_point_for_tri(self, spacing):
        """生成三角形的理想点"""
        if self.pselected is not None:
            return

        self.mesh_type = 1
        self.add_new_point(spacing)

    def select_point_for_tri(self):
        """选择三角形顶点"""
        if self.pselected is not None:
            return

        self.select_point()

    def update_data_tri(self):
        """更新三角形相关数据"""
        if self.pselected is None:
            return

        # 如果 pselected 是一个列表，且长度为 2，则直接返回
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

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def reconstruct_node2front(self):
        """重构节点的邻接阵面信息"""
        num_neighbors = 2
        for front in self.front_list:
            for node_elem in front.node_elems:
                if len(node_elem.node2front) == 0:
                    node_elem.node2front = [None] * num_neighbors

            # node_elem 在 front 中是起点，则 front 是在后面
            for i, node_elem in enumerate(front.node_elems):
                node_elem.node2front[(i + 1) % num_neighbors] = front

        # 检查每个节点的邻阵面数量是否为 2
        for front in self.front_list:
            for node_elem in front.node_elems:
                if not all(node_elem.node2front):
                    warning(f"节点 {node_elem.idx} 的邻阵面数量不足 2")

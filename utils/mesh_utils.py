#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网格工具函数
提供网格操作的工具函数，包括单元合并等功能
"""

import copy
import heapq
from itertools import combinations

from data_structure.basic_elements import Triangle, Quadrilateral
from optimize import mesh_quality
from utils.message import info
from utils.timer import TimeSpan
from utils import geom_toolkit as geom_tool
from data_structure.unstructured_grid import Unstructured_Grid


def merge_triangles_to_quads(unstr_grid):
    """
    合并相邻的三角形单元，形成四边形单元。
    
    合并条件：
    1. 两个三角形必须共享一条边
    2. 合并后的四边形是凸多边形
    3. 四边形质量高于合并前三角形质量中位数
    
    Args:
        unstr_grid: Unstructured_Grid 对象，包含网格数据
        
    Returns:
        Unstructured_Grid: 合并后的新网格对象（不修改原始网格）
    """
    timer = TimeSpan("开始合并三角形为四边形...")
    
    # 创建网格的深拷贝，避免修改原始网格
    new_grid = copy.deepcopy(unstr_grid)
    
    node_coords = new_grid.node_coords
    edge_map = {}
    merge_candidates = []

    # 构建边到单元的映射
    for cell_idx, cell in enumerate(new_grid.cell_container):
        if not isinstance(cell, Triangle):
            continue
        for i, j in combinations(sorted(cell.node_ids), 2):
            edge = (i, j)
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append(cell_idx)

    # 寻找可合并的三角形对
    for edge, cells in edge_map.items():
        if len(cells) != 2:
            continue

        cell1, cell2 = cells
        tri1 = new_grid.cell_container[cell1]
        tri2 = new_grid.cell_container[cell2]

        # 获取四个顶点
        common = set(tri1.node_ids) & set(tri2.node_ids)
        if len(common) != 2:
            continue

        a, b = sorted(common)
        c = list(set(tri1.node_ids) - common)[0]
        d = list(set(tri2.node_ids) - common)[0]

        # 凸性检查
        if not geom_tool.is_convex(a, c, b, d, node_coords):
            continue

        # 计算质量增益
        tri1.init_metrics()
        tri2.init_metrics()
        tri_quality = (tri1.quality + tri2.quality) / 2
        quad_quality = mesh_quality.quadrilateral_quality2(
            node_coords[a], node_coords[c], node_coords[b], node_coords[d]
        )

        # 质量提升判断（使用最小堆保存优质候选）
        heapq.heappush(merge_candidates, (-quad_quality, (cell1, cell2, a, b, c, d)))

    # 按质量从高到低处理合并
    merged = set()
    num_merged = 0
    while merge_candidates:
        _, (cell1_idx, cell2_idx, a, b, c, d) = heapq.heappop(merge_candidates)

        # 跳过已处理单元
        if cell1_idx in merged or cell2_idx in merged:
            continue

        # 确保新创建的四边形法向指向z轴正方向
        if not geom_tool.is_left2d(node_coords[a], node_coords[b], node_coords[d]):
            a, c, b, d = a, d, b, c

        # 创建新四边形
        new_quad = Quadrilateral(
            node_coords[a],
            node_coords[c],
            node_coords[b],
            node_coords[d],
            "interior",
            len(new_grid.cell_container),
            [a, c, b, d],
        )

        # 替换原单元
        new_grid.cell_container[cell1_idx] = new_quad
        new_grid.cell_container[cell2_idx] = None  # 标记删除

        merged.update([cell1_idx, cell2_idx])
        num_merged += 1

    # 清理被删除的单元
    new_grid.cell_container = [c for c in new_grid.cell_container if c is not None]
    
    # 更新网格统计信息
    new_grid.update_counts()

    info(f"成功合并{num_merged}对三角形为四边形")
    timer.show_to_console("四边形合并完成.")
    return new_grid

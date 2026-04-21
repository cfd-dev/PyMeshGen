#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网格工具函数
提供网格操作的工具函数，包括单元合并等功能
"""

import copy
import heapq
from collections import defaultdict, deque
from types import SimpleNamespace

from data_structure.basic_elements import Triangle, Quadrilateral
from optimize import mesh_quality
from utils.message import info
from utils.timer import TimeSpan
from utils import geom_toolkit as geom_tool
Q_MORPH_FILL_QUALITY_FLOOR = 0.15
Q_MORPH_FILL_QUALITY_RATIO = 0.7


def deduplicate_grid_cells(unstructured_grid):
    """Remove duplicate cells identified by the same node set."""
    cells = unstructured_grid.cells
    if not cells:
        return 0

    unique_cells = []
    seen = set()
    removed = 0
    for cell in cells:
        key = tuple(sorted(int(node) for node in cell))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique_cells.append(list(cell))

    if removed > 0:
        unstructured_grid.set_cells(unique_cells)
    return removed


def _build_triangle_edge_map(cell_container):
    edge_map = defaultdict(list)
    for cell_idx, cell in enumerate(cell_container):
        if not isinstance(cell, Triangle):
            continue
        node_ids = list(cell.node_ids)
        for idx in range(len(node_ids)):
            edge = tuple(sorted((node_ids[idx], node_ids[(idx + 1) % len(node_ids)])))
            edge_map[edge].append(cell_idx)
    return edge_map


def _compute_triangle_front_depths(edge_map):
    tri_neighbors = defaultdict(set)
    tri_depths = {}
    queue = deque()

    for edge_cells in edge_map.values():
        if len(edge_cells) == 1:
            tri_idx = edge_cells[0]
            if tri_idx not in tri_depths:
                tri_depths[tri_idx] = 0
                queue.append(tri_idx)
        elif len(edge_cells) == 2:
            tri_a, tri_b = edge_cells
            tri_neighbors[tri_a].add(tri_b)
            tri_neighbors[tri_b].add(tri_a)

    while queue:
        tri_idx = queue.popleft()
        for neighbor_idx in tri_neighbors[tri_idx]:
            if neighbor_idx in tri_depths:
                continue
            tri_depths[neighbor_idx] = tri_depths[tri_idx] + 1
            queue.append(neighbor_idx)

    return tri_depths


def _build_triangle_pair_candidate(
    new_grid, cell1_idx, cell2_idx, edge_map, node_container, tri_depths
):
    node_coords = new_grid.node_coords
    tri1 = new_grid.cell_container[cell1_idx]
    tri2 = new_grid.cell_container[cell2_idx]

    tri1_nodes = set(tri1.node_ids)
    tri2_nodes = set(tri2.node_ids)
    if len(tri1_nodes) != 3 or len(tri2_nodes) != 3:
        return None

    common = tri1_nodes & tri2_nodes
    if len(common) != 2:
        return None

    a, b = sorted(common)
    tri1_unique = list(tri1_nodes - common)
    tri2_unique = list(tri2_nodes - common)
    if len(tri1_unique) != 1 or len(tri2_unique) != 1:
        return None

    c = tri1_unique[0]
    d = tri2_unique[0]
    if c == d:
        return None

    if not geom_tool.is_convex(a, c, b, d, node_coords):
        return None

    try:
        quad_nodes = tuple(
            geom_tool.sort_quadrilateral_nodes([a, c, b, d], node_container)
        )
    except ValueError:
        return None

    tri1.init_metrics()
    tri2.init_metrics()
    quad_quality = mesh_quality.quadrilateral_quality2(
        *(node_coords[idx] for idx in quad_nodes)
    )
    if quad_quality <= 0.0:
        return None

    quad_edges = [
        tuple(sorted((quad_nodes[idx], quad_nodes[(idx + 1) % 4]))) for idx in range(4)
    ]
    boundary_edge_count = sum(
        1 for edge in quad_edges if len(edge_map.get(edge, [])) <= 1
    )

    return {
        "cell1_idx": cell1_idx,
        "cell2_idx": cell2_idx,
        "quad_nodes": quad_nodes,
        "quad_quality": quad_quality,
        "tri_quality_min": min(tri1.quality, tri2.quality),
        "tri_quality_mean": (tri1.quality + tri2.quality) / 2.0,
        "boundary_edge_count": boundary_edge_count,
        "depth_min": min(
            tri_depths.get(cell1_idx, float("inf")),
            tri_depths.get(cell2_idx, float("inf")),
        ),
        "depth_max": max(
            tri_depths.get(cell1_idx, float("inf")),
            tri_depths.get(cell2_idx, float("inf")),
        ),
    }


def _collect_triangle_pair_candidates(new_grid):
    node_container = [SimpleNamespace(coords=coords) for coords in new_grid.node_coords]
    edge_map = _build_triangle_edge_map(new_grid.cell_container)
    tri_depths = _compute_triangle_front_depths(edge_map)
    candidates = []

    for edge_cells in edge_map.values():
        if len(edge_cells) != 2:
            continue
        candidate = _build_triangle_pair_candidate(
            new_grid,
            edge_cells[0],
            edge_cells[1],
            edge_map,
            node_container,
            tri_depths,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


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
    merge_candidates = []

    for candidate in _collect_triangle_pair_candidates(new_grid):
        heapq.heappush(
            merge_candidates,
            (
                -candidate["quad_quality"],
                (
                    candidate["cell1_idx"],
                    candidate["cell2_idx"],
                    candidate["quad_nodes"],
                ),
            ),
        )

    # 按质量从高到低处理合并
    merged = set()
    num_merged = 0
    while merge_candidates:
        _, (cell1_idx, cell2_idx, quad_nodes) = heapq.heappop(merge_candidates)

        # 跳过已处理单元
        if cell1_idx in merged or cell2_idx in merged:
            continue

        # 创建新四边形
        new_quad = Quadrilateral(
            *(node_coords[idx] for idx in quad_nodes),
            "interior",
            len(new_grid.cell_container),
            list(quad_nodes),
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


def q_morph_triangles_to_quads(unstr_grid):
    """
    使用 q-morph 风格的边界向内推进策略，将三角形优先转换为质量更稳定的四边形。
    """
    timer = TimeSpan("开始q-morph风格四边形合并...")
    new_grid = copy.deepcopy(unstr_grid)
    node_coords = new_grid.node_coords
    front_candidates = []

    for candidate in _collect_triangle_pair_candidates(new_grid):
        if candidate["quad_quality"] + 1e-12 < candidate["tri_quality_min"]:
            continue

        heapq.heappush(
            front_candidates,
            (
                -candidate["boundary_edge_count"],
                candidate["depth_min"],
                candidate["depth_max"],
                -candidate["quad_quality"],
                (
                    candidate["cell1_idx"],
                    candidate["cell2_idx"],
                    candidate["quad_nodes"],
                ),
            ),
        )

    merged = set()
    front_merged = 0
    while front_candidates:
        _, _, _, _, (cell1_idx, cell2_idx, quad_nodes) = heapq.heappop(
            front_candidates
        )

        if cell1_idx in merged or cell2_idx in merged:
            continue

        new_quad = Quadrilateral(
            *(node_coords[idx] for idx in quad_nodes),
            "interior",
            len(new_grid.cell_container),
            list(quad_nodes),
        )
        new_grid.cell_container[cell1_idx] = new_quad
        new_grid.cell_container[cell2_idx] = None
        merged.update([cell1_idx, cell2_idx])
        front_merged += 1

    new_grid.cell_container = [c for c in new_grid.cell_container if c is not None]
    new_grid.update_counts()

    fill_candidates = []
    for candidate in _collect_triangle_pair_candidates(new_grid):
        if candidate["quad_quality"] < Q_MORPH_FILL_QUALITY_FLOOR:
            continue
        if (
            candidate["quad_quality"] + 1e-12
            < Q_MORPH_FILL_QUALITY_RATIO * candidate["tri_quality_min"]
        ):
            continue

        heapq.heappush(
            fill_candidates,
            (
                -candidate["quad_quality"],
                -candidate["boundary_edge_count"],
                candidate["depth_min"],
                (
                    candidate["cell1_idx"],
                    candidate["cell2_idx"],
                    candidate["quad_nodes"],
                ),
            ),
        )

    fill_merged = 0
    while fill_candidates:
        _, _, _, (cell1_idx, cell2_idx, quad_nodes) = heapq.heappop(fill_candidates)

        cell1 = new_grid.cell_container[cell1_idx]
        cell2 = new_grid.cell_container[cell2_idx]
        if cell1 is None or cell2 is None:
            continue
        if not isinstance(cell1, Triangle) or not isinstance(cell2, Triangle):
            continue

        new_quad = Quadrilateral(
            *(node_coords[idx] for idx in quad_nodes),
            "interior",
            len(new_grid.cell_container),
            list(quad_nodes),
        )
        new_grid.cell_container[cell1_idx] = new_quad
        new_grid.cell_container[cell2_idx] = None
        fill_merged += 1

    new_grid.cell_container = [c for c in new_grid.cell_container if c is not None]
    new_grid.update_counts()

    info(
        f"q-morph风格成功合并{front_merged + fill_merged}对三角形为四边形"
        f"（前沿阶段{front_merged}对，补齐阶段{fill_merged}对）"
    )
    timer.show_to_console("q-morph风格四边形合并完成.")
    return new_grid

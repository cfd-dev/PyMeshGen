"""
网格拓扑分析工具

提供边分类、边界环追踪、欧拉示性数、三角形质量批量评估等
纯数值函数。不依赖 sfmesh 或 OCC，仅依赖 NumPy。

从原 sfmesh/geom_utils.py 和 sfmesh/mesh_quality.py 提取合并。
"""

from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter
import numpy as np

from utils.message import info


# ============================================================================
# 边分类与边界环追踪
# ============================================================================

def classify_edges(triangles):
    """
    统计网格边的使用情况，分类为边界边、内部边和非流形边。

    Args:
        triangles: 三角形列表，每个元素需有 node_ids 属性返回 [id0, id1, id2]

    Returns:
        (edge_count, boundary_edges, interior_edges, non_manifold_edges)
    """
    edge_count: Dict[frozenset, int] = Counter()
    for tri in triangles:
        nids = tri.node_ids
        for k in range(3):
            e = frozenset([nids[k], nids[(k + 1) % 3]])
            edge_count[e] += 1

    boundary_edges = {e: cnt for e, cnt in edge_count.items() if cnt == 1}
    interior_edges = {e: cnt for e, cnt in edge_count.items() if cnt == 2}
    non_manifold_edges = {e: cnt for e, cnt in edge_count.items() if cnt > 2}
    return edge_count, boundary_edges, interior_edges, non_manifold_edges


def trace_boundary_loops(edge_count: Dict[frozenset, int]) -> List[List[int]]:
    """从边计数中追踪所有边界闭环。"""
    adj: Dict[int, List[int]] = {}
    for eh, cnt in edge_count.items():
        if cnt == 1:
            n0, n1 = tuple(eh)
            adj.setdefault(n0, []).append(n1)
            adj.setdefault(n1, []).append(n0)

    visited: Set[int] = set()
    loops = []
    for start in adj:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev = None
        current = start
        max_steps = len(adj) + 1
        for _ in range(max_steps):
            neighbors = adj.get(current, [])
            nexts = [n for n in neighbors if n != prev]
            if not nexts:
                break
            next_node = nexts[0]
            if next_node == start:
                break
            if next_node in visited:
                break
            loop.append(next_node)
            visited.add(next_node)
            prev, current = current, next_node
        loops.append(loop)
    return loops


def euler_characteristic(num_nodes: int, num_edges: int, num_faces: int) -> int:
    """计算欧拉示性数 χ = V - E + F。"""
    return num_nodes - num_edges + num_faces


def build_edge_triangle_map(triangles) -> Dict[frozenset, List[int]]:
    """从三角形列表构建边 → 三角形索引的映射。"""
    edge_map: Dict[frozenset, List[int]] = {}
    for tri_idx, tri in enumerate(triangles):
        ids = tri.node_ids
        for k in range(3):
            edge = frozenset([ids[k], ids[(k + 1) % 3]])
            edge_map.setdefault(edge, []).append(tri_idx)
    return edge_map


# ============================================================================
# 批量网格质量评估（原 SurfaceMeshQuality.evaluate_mesh）
# ============================================================================

def _safe_histogram(data: np.ndarray, bins: int = 10) -> List[int]:
    """安全的直方图计算，处理空数据、NaN、Inf 和零范围"""
    clean = data[np.isfinite(data)]
    if len(clean) == 0:
        return [0] * bins
    try:
        return np.histogram(clean, bins=bins)[0].tolist()
    except ValueError:
        return [len(clean)] + [0] * (bins - 1)


def evaluate_mesh_quality(
    triangles,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    批量评估网格质量（向量化实现）。

    接受任何有 nodes[k].coords 属性的三角形对象列表。

    Args:
        triangles: 三角形列表 (每个元素需有 nodes[i].coords, i=0,1,2)
        verbose: 是否输出详细报告

    Returns:
        评估结果字典
    """
    if not triangles:
        return {'num_triangles': 0}

    n = len(triangles)

    # 批量提取坐标 (N, 3, 3)
    coords = np.array(
        [[tri.nodes[k].coords for k in range(3)] for tri in triangles],
        dtype=np.float64
    )

    # 批量边长
    e0 = coords[:, 1] - coords[:, 0]
    e1 = coords[:, 2] - coords[:, 1]
    e2 = coords[:, 0] - coords[:, 2]

    len0 = np.linalg.norm(e0, axis=1)
    len1 = np.linalg.norm(e1, axis=1)
    len2 = np.linalg.norm(e2, axis=1)

    # 批量面积 (Heron)
    semi = (len0 + len1 + len2) / 2.0
    area_sq = semi * (semi - len0) * (semi - len1) * (semi - len2)
    area_sq = np.maximum(area_sq, 0.0)
    areas = np.sqrt(area_sq)

    # 批量形状质量: 4√3·A / Σl²
    sum_len_sq = len0**2 + len1**2 + len2**2
    qualities = np.where(
        sum_len_sq > 1e-30,
        4.0 * np.sqrt(3) * areas / sum_len_sq,
        0.0
    )
    qualities = np.clip(qualities, 0.0, 1.0)

    # 批量长宽比: l_max² / (4√3·A)
    max_edge_sq = np.maximum.reduce([len0**2, len1**2, len2**2])
    denom_ar = 4.0 * np.sqrt(3) * areas
    aspect_ratios = np.where(denom_ar > 1e-30, max_edge_sq / denom_ar, np.inf)

    # 批量角度
    cos_a0 = np.clip(
        np.sum((-e2) * e0, axis=1) / (len2 * len0 + 1e-30), -1.0, 1.0)
    cos_a1 = np.clip(
        np.sum((-e0) * e1, axis=1) / (len0 * len1 + 1e-30), -1.0, 1.0)
    cos_a2 = np.clip(
        np.sum((-e1) * e2, axis=1) / (len1 * len2 + 1e-30), -1.0, 1.0)
    all_angles = np.degrees(np.arccos(
        np.stack([cos_a0, cos_a1, cos_a2], axis=1)))
    min_angles = np.min(all_angles, axis=1)
    max_angles = np.max(all_angles, axis=1)

    poor_mask = qualities < 0.3
    poor_count = int(np.sum(poor_mask))

    fin_mask = np.isfinite(aspect_ratios)
    result: Dict[str, Any] = {
        'num_triangles': n,
        'quality_mean': float(np.mean(qualities)),
        'quality_min': float(np.min(qualities)),
        'quality_max': float(np.max(qualities)),
        'quality_std': float(np.std(qualities)),
        'aspect_ratio_mean': float(np.mean(aspect_ratios[fin_mask])) if np.any(fin_mask) else 0.0,
        'aspect_ratio_max': float(np.max(aspect_ratios[fin_mask])) if np.any(fin_mask) else float('inf'),
        'min_angle_mean': float(np.mean(min_angles)),
        'min_angle_min': float(np.min(min_angles)),
        'max_angle_mean': float(np.mean(max_angles)),
        'max_angle_max': float(np.max(max_angles)),
        'total_area': float(np.sum(areas)),
        'area_mean': float(np.mean(areas)),
        'quality_histogram': _safe_histogram(qualities, bins=10),
        'poor_quality_count': poor_count,
        'poor_quality_ratio': poor_count / n,
    }

    if verbose:
        info(f"[质量] 平均={result['quality_mean']:.4f}, "
             f"最小={result['quality_min']:.4f}, "
             f"低质量(<0.3)={result['poor_quality_count']}")

    return result


# ============================================================================
# 网格拓扑验证（原 validate_mesh_topology）
# ============================================================================

def validate_mesh_topology(triangles, verbose: bool = True) -> dict:
    """
    验证网格拓扑完整性并输出统计信息。

    检查项目：边分类 / 边界闭环 / 欧拉示性数 / 三角形质量统计

    Args:
        triangles: 三角形列表
        verbose: 是否输出详细信息

    Returns:
        包含所有拓扑指标的字典
    """
    num_faces = len(triangles)
    if num_faces == 0:
        return {'num_faces': 0}

    edge_count, boundary_edges, interior_edges, non_manifold_edges = \
        classify_edges(triangles)
    num_edges = len(edge_count)

    all_node_ids: Set[int] = set()
    for tri in triangles:
        all_node_ids.update(tri.node_ids)
    num_nodes = len(all_node_ids)

    loops = trace_boundary_loops(edge_count)
    chi = euler_characteristic(num_nodes, num_edges, num_faces)

    quality_stats = evaluate_mesh_quality(triangles, verbose=False)

    result = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_faces': num_faces,
        'boundary_edges': len(boundary_edges),
        'interior_edges': len(interior_edges),
        'non_manifold_edges': len(non_manifold_edges),
        'num_boundary_loops': len(loops),
        'boundary_loop_sizes': [len(l) for l in loops],
        'euler_characteristic': chi,
        'quality_mean': quality_stats.get('quality_mean', 0.0),
        'quality_min': quality_stats.get('quality_min', 0.0),
        'poor_quality_count': quality_stats.get('poor_quality_count', 0),
    }

    if verbose:
        info(f"[拓扑] 节点={num_nodes}, 边={num_edges}, 面={num_faces}")
        info(f"[拓扑] 边界边={len(boundary_edges)}, 内部边={len(interior_edges)}, "
             f"非流形边={len(non_manifold_edges)}")
        info(f"[拓扑] 边界环={len(loops)}: {[len(l) for l in loops]}")
        info(f"[拓扑] 欧拉示性数={chi}")
        if non_manifold_edges:
            info(f"[拓扑] 警告: 存在 {len(non_manifold_edges)} 条非流形边")
        if chi != 1 and chi != 2:
            info(f"[拓扑] 警告: 欧拉示性数异常 (预期 1 或 2, 实际 {chi})")
        info(f"[质量] 平均={quality_stats.get('quality_mean', 0):.4f}, "
             f"最小={quality_stats.get('quality_min', 0):.4f}, "
             f"低质量(<0.3)={quality_stats.get('poor_quality_count', 0)}")

    return result

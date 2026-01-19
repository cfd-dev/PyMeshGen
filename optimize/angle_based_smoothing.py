"""
Angle-based smoothing algorithm implementation for mesh optimization.

This module implements the angleBasedSmoothing and GetMe methods from the C++ version,
adapted for the PyMeshGen project's data structures.
"""

import numpy as np
from math import acos, cos, sin
from utils.geom_toolkit import triangle_area
from utils.timer import TimeSpan
from utils.message import info, error
from .mesh_quality import triangle_shape_quality


def angle_based_smoothing(unstr_grid, iterations=1):
    """
    Angle-based smoothing algorithm implementation.
    
    This algorithm moves each interior node to a position calculated based on
    the angles formed by its neighboring nodes, which helps preserve geometric features.
    
    Args:
        unstr_grid: Unstructured_Grid object containing the mesh
        iterations: Number of smoothing iterations to perform (default: 1)
    
    Returns:
        Unstructured_Grid: The smoothed grid
    """
    timer = TimeSpan("开始基于角度平滑优化...")

    # 使用基于边的顶点邻接关系作为一环邻居，
    # 而 init_node2node_by_cell 会把同一单元内的非边相邻节点也加入（例如四边形对角点）。
    # 这会破坏角度排序并导致平滑结果错误，因此这里强制使用基于边的邻接关系。
    unstr_grid.init_node2node()
    # 将邻居按极角排序成环，匹配 C++ 中围绕顶点的遍历顺序
    unstr_grid.cyclic_node2node()

    # Get boundary nodes
    boundary_nodes = set(unstr_grid.boundary_nodes_list) if hasattr(unstr_grid, 'boundary_nodes_list') else set()
    
    node_coords = np.array(unstr_grid.node_coords, dtype=float)
    
    for iteration in range(iterations):
        info(f"基于角度平滑第 {iteration + 1} 次迭代")
        
        for node_id in range(len(node_coords)):
            # 边界点保持不动
            if node_id in boundary_nodes:
                continue  # Skip boundary nodes
            
            # 取当前节点的一环邻居（已被 cyclic_node2node 排序成闭环）
            if node_id not in unstr_grid.node2node:
                continue
                
            neighbor_ids = unstr_grid.node2node[node_id]
            # 至少需要 3 个邻居才能构成连续三元组
            if not neighbor_ids or len(neighbor_ids) < 3:
                continue
            
            # 计算新位置：对每个连续三元组计算旋转贡献并求平均
            new_x, new_y = 0.0, 0.0
            count = 0
            
            # 遍历每个邻居作为 vv_it（C++），并构造 vvp/vvm/vvn
            num_neighbors = len(neighbor_ids)
            for i in range(num_neighbors):
                # 三个连续邻居（对应 C++ 的 vvp_it / vvm_it / vvn_it）
                vvp_idx = neighbor_ids[i]  # vv_it
                vvm_idx = neighbor_ids[(i + 1) % num_neighbors]  # vv_it + 1
                vvn_idx = neighbor_ids[(i + 2) % num_neighbors]  # vv_it + 2
                
                # 坐标获取（当前点与三个相邻点）
                curr_node_pos = node_coords[node_id]  # Current node position
                vvp_pos = node_coords[vvp_idx]  # Previous neighbor
                vvm_pos = node_coords[vvm_idx]  # Current neighbor
                vvn_pos = node_coords[vvn_idx]  # Next neighbor
                
                # 以 vvm 为旋转中心构造向量（与 smoothing.cpp 一致）
                v = curr_node_pos - vvm_pos  # Vector from vvm to current node
                v1 = vvp_pos - vvm_pos  # Vector from vvm to vvp
                v2 = vvn_pos - vvm_pos  # Vector from vvm to vvn
                
                # 归一化向量以计算夹角
                v_norm = np.linalg.norm(v)
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                # 与 C++ 行为一致：遇到退化情况则跳过该三元组
                if v_norm == 0 or v1_norm == 0 or v2_norm == 0:
                    continue
                
                v_unit = v / v_norm
                v1_unit = v1 / v1_norm
                v2_unit = v2 / v2_norm
                
                # 计算点乘并转成夹角
                dot_v1_v = np.dot(v1_unit, v_unit)
                dot_v2_v = np.dot(v2_unit, v_unit)
                
                # 数值安全：夹角计算前裁剪到 [-1, 1]
                dot_v1_v = np.clip(dot_v1_v, -1.0, 1.0)
                dot_v2_v = np.clip(dot_v2_v, -1.0, 1.0)
                
                # beta 角：对应 -(0.5 * (acos(v1|v) - acos(v2|v)))
                beta = -0.5 * (acos(dot_v1_v) - acos(dot_v2_v))
                
                # 将当前点绕 vvm 旋转 beta，得到该三元组贡献
                dx = curr_node_pos[0] - vvm_pos[0]
                dy = curr_node_pos[1] - vvm_pos[1]
                
                new_x_contrib = vvm_pos[0] + dx * cos(beta) - dy * sin(beta)
                new_y_contrib = vvm_pos[1] + dx * sin(beta) + dy * cos(beta)
                
                new_x += new_x_contrib
                new_y += new_y_contrib
                count += 1
            
            # 取平均得到新位置（与 C++ 中 cog/valence 一致）
            if count > 0:
                node_coords[node_id][0] = new_x / count
                node_coords[node_id][1] = new_y / count
    
    # Update the grid with new coordinates
    unstr_grid.node_coords = node_coords.tolist()
    
    timer.show_to_console("基于角度的平滑完成.")
    return unstr_grid


def getme_method(unstr_grid, iterations=1):
    """
    GetMe (Geometric Element Transfer and Mesh Enhancement) method implementation.
    
    This method uses quality-based weighting to determine optimal node positions
    based on quality of surrounding triangles.
    
    Args:
        unstr_grid: Unstructured_Grid object containing the mesh
        iterations: Number of smoothing iterations to perform (default: 1)
    
    Returns:
        Unstructured_Grid: The smoothed grid
    """
    timer = TimeSpan("开始GetMe方法优化...")
    
    # Make sure node2node connectivity is initialized
    if not hasattr(unstr_grid, 'node2node') or unstr_grid.node2node is None:
        unstr_grid.cyclic_node2node()
    
    # Get boundary nodes
    boundary_nodes = set(unstr_grid.boundary_nodes_list) if hasattr(unstr_grid, 'boundary_nodes_list') else set()
    
    node_coords = np.array(unstr_grid.node_coords, dtype=float)
    
    for iteration in range(iterations):
        info(f"GetMe方法第 {iteration + 1} 次迭代")
        
        new_coords = node_coords.copy()
        
        for node_id in range(len(node_coords)):
            if node_id in boundary_nodes:
                continue  # Skip boundary nodes
            
            # Get neighboring nodes (ring of nodes around current node)
            if node_id >= len(unstr_grid.node2node):
                continue
                
            neighbor_ids = unstr_grid.node2node[node_id]
            if not neighbor_ids or len(neighbor_ids) < 2:  # Need at least 2 neighbors
                continue
            
            # Add the first neighbor at the end to close the ring
            extended_neighbor_ids = neighbor_ids + [neighbor_ids[0]] if neighbor_ids else neighbor_ids
            
            new_x, new_y = 0.0, 0.0
            total_weight = 0.0
            
            # Process each triangle formed with consecutive neighbors
            for i in range(len(extended_neighbor_ids) - 1):
                # Get three points: current node, current neighbor, next neighbor
                # Use 2D coordinates for GetMe transformation (preserve original z if present)
                p1 = node_coords[node_id][:2]  # Current node
                p2 = node_coords[extended_neighbor_ids[i]][:2]  # Current neighbor
                p3 = node_coords[extended_neighbor_ids[i + 1]][:2]  # Next neighbor
                
                # Calculate quality of the triangle before transformation
                quality_before = _calculate_triangle_quality(p1, p2, p3)
                area_before = _calculate_triangle_area(p1, p2, p3)
                
                if area_before <= 0 or quality_before <= 0:
                    continue
                
                # Apply transformation to get new triangle (match smoothing.cpp formulas)
                y1 = np.array([
                    (p1[0] + p3[0]) / 2.0 + np.sqrt(3.0) / 2.0 * (p1[1] - p3[1]),
                    (p1[1] + p3[1]) / 2.0 + np.sqrt(3.0) / 2.0 * (p3[0] - p1[0]),
                ])
                y2 = np.array([
                    (p2[0] + p1[0]) / 2.0 + np.sqrt(3.0) / 2.0 * (p2[1] - p1[1]),
                    (p2[1] + p1[1]) / 2.0 + np.sqrt(3.0) / 2.0 * (p1[0] - p2[0]),
                ])
                y3 = np.array([
                    (p3[0] + p2[0]) / 2.0 + np.sqrt(3.0) / 2.0 * (p3[1] - p2[1]),
                    (p3[1] + p2[1]) / 2.0 + np.sqrt(3.0) / 2.0 * (p2[0] - p3[0]),
                ])
                z1 = np.array([
                    (y2[0] + y1[0]) / 2.0 + np.sqrt(3.0) / 2.0 * (y2[1] - y1[1]),
                    (y2[1] + y1[1]) / 2.0 + np.sqrt(3.0) / 2.0 * (y1[0] - y2[0]),
                ])
                z2 = np.array([
                    (y3[0] + y2[0]) / 2.0 + np.sqrt(3.0) / 2.0 * (y3[1] - y2[1]),
                    (y3[1] + y2[1]) / 2.0 + np.sqrt(3.0) / 2.0 * (y2[0] - y3[0]),
                ])
                z3 = np.array([
                    (y1[0] + y3[0]) / 2.0 + np.sqrt(3.0) / 2.0 * (y1[1] - y3[1]),
                    (y1[1] + y3[1]) / 2.0 + np.sqrt(3.0) / 2.0 * (y3[0] - y1[0]),
                ])
                
                # Calculate quality after transformation
                quality_after = _calculate_triangle_quality(z1, z2, z3)
                area_after = _calculate_triangle_area(z1, z2, z3)
                
                if area_after <= 0 or quality_after <= 0:
                    continue
                
                # Calculate new point based on quality ratio (use original centroid as in C++)
                centroid_before = (p1 + p2 + p3) / 3.0
                scale_factor = np.sqrt(area_before / area_after) if area_after > 0 else 1.0
                new_point = centroid_before + scale_factor * (z1 - centroid_before)
                
                # Calculate weight based on quality improvement
                weight = quality_after / quality_before if quality_before > 0 else 1.0
                
                new_x += new_point[0] * weight
                new_y += new_point[1] * weight
                total_weight += weight
            
            if total_weight > 0:
                new_coords[node_id][0] = new_x / total_weight
                new_coords[node_id][1] = new_y / total_weight
        
        # Update coordinates for next iteration
        node_coords = new_coords
    
    # Update the grid with new coordinates
    unstr_grid.node_coords = node_coords.tolist()
    
    timer.show_to_console("基于GetMe方法的平滑完成.")
    return unstr_grid


def _calculate_triangle_quality(p1, p2, p3):
    """
    Calculate the quality of a triangle using the standard formula.
    This function uses the existing triangle_shape_quality function from mesh_quality module.
    """
    return triangle_shape_quality(p1, p2, p3)


def _calculate_triangle_area(p1, p2, p3):
    """Calculate the area of a triangle using the existing function from geom_toolkit."""
    return triangle_area(p1, p2, p3)


# GUI可调用的函数接口
def smooth_mesh_angle_based(mesh_data, iterations=1):
    """
    GUI-friendly interface for angle-based smoothing.
    
    Args:
        mesh_data: The mesh data to smooth
        iterations: Number of smoothing iterations (default: 1)
    
    Returns:
        Smoothed mesh data
    """
    try:
        result = angle_based_smoothing(mesh_data, iterations)
        info(f"基于角度的平滑完成，迭代次数: {iterations}")
        return result
    except Exception as e:
        error(f"基于角度的平滑失败: {str(e)}")
        raise


def smooth_mesh_getme(mesh_data, iterations=1):
    """
    GUI-friendly interface for GetMe method smoothing.
    
    Args:
        mesh_data: The mesh data to smooth
        iterations: Number of smoothing iterations (default: 1)
    
    Returns:
        Smoothed mesh data
    """
    try:
        result = getme_method(mesh_data, iterations)
        info(f"基于GetMe方法的平滑完成，迭代次数: {iterations}")
        return result
    except Exception as e:
        error(f"基于GetMe方法的平滑失败: {str(e)}")
        raise

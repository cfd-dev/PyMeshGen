"""
Angle-based smoothing algorithm implementation for mesh optimization.

This module implements the angleBasedSmoothing and GetMe methods from the C++ version,
adapted for the PyMeshGen project's data structures.
"""

import numpy as np
from math import acos, cos, sin
from utils.geom_toolkit import calculate_distance, triangle_area
from data_structure.basic_elements import Triangle
from utils.timer import TimeSpan
from utils.message import info, debug, warning, error
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
    
    # Make sure node2node connectivity is initialized
    if not hasattr(unstr_grid, 'node2node') or unstr_grid.node2node is None:
        unstr_grid.cyclic_node2node()
    
    # Get boundary nodes
    boundary_nodes = set(unstr_grid.boundary_nodes_list) if hasattr(unstr_grid, 'boundary_nodes_list') else set()
    
    node_coords = np.array(unstr_grid.node_coords, dtype=float)
    original_coords = node_coords.copy()
    
    for iteration in range(iterations):
        info(f"基于角度平滑第 {iteration + 1} 次迭代")
        
        new_coords = node_coords.copy()
        
        for node_id in range(len(node_coords)):
            if node_id in boundary_nodes:
                continue  # Skip boundary nodes
            
            # Get neighboring nodes (ring of nodes around current node)
            if node_id >= len(unstr_grid.node2node):
                continue
                
            neighbor_ids = unstr_grid.node2node[node_id]
            if not neighbor_ids or len(neighbor_ids) < 3:  # Need at least 3 neighbors for meaningful smoothing
                continue
            
            # Calculate the new position based on angle-based algorithm
            new_x, new_y = 0.0, 0.0
            count = 0
            
            # Process each neighbor triplet to calculate angle-based position
            num_neighbors = len(neighbor_ids)
            for i in range(num_neighbors):
                # Get three consecutive neighbors (vvp_it, vvm_it, vvn_it in C++)
                vvp_idx = neighbor_ids[i]  # Previous neighbor
                vvm_idx = neighbor_ids[(i + 1) % num_neighbors]  # Current neighbor  
                vvn_idx = neighbor_ids[(i + 2) % num_neighbors]  # Next neighbor
                
                # Get coordinates
                curr_node_pos = node_coords[node_id]  # Current node position
                vvp_pos = node_coords[vvp_idx]  # Previous neighbor
                vvm_pos = node_coords[vvm_idx]  # Current neighbor
                vvn_pos = node_coords[vvn_idx]  # Next neighbor
                
                # Calculate vectors
                v = curr_node_pos - vvm_pos  # Vector from vvm to current node
                v1 = vvp_pos - vvm_pos  # Vector from vvm to vvp
                v2 = vvn_pos - vvm_pos  # Vector from vvm to vvn
                
                # Normalize vectors
                v_norm = np.linalg.norm(v)
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v_norm == 0 or v1_norm == 0 or v2_norm == 0:
                    continue
                
                v_unit = v / v_norm
                v1_unit = v1 / v1_norm
                v2_unit = v2 / v2_norm
                
                # Calculate dot products
                dot_v1_v = np.dot(v1_unit, v_unit)
                dot_v2_v = np.dot(v2_unit, v_unit)
                
                # Clamp values to valid range for acos
                dot_v1_v = np.clip(dot_v1_v, -1.0, 1.0)
                dot_v2_v = np.clip(dot_v2_v, -1.0, 1.0)
                
                # Calculate angle beta
                beta = -0.5 * (acos(dot_v1_v) - acos(dot_v2_v))
                
                # Calculate new position components
                dx = curr_node_pos[0] - vvm_pos[0]
                dy = curr_node_pos[1] - vvm_pos[1]
                
                new_x_contrib = vvm_pos[0] + dx * cos(beta) - dy * sin(beta)
                new_y_contrib = vvm_pos[1] + dx * sin(beta) + dy * cos(beta)
                
                new_x += new_x_contrib
                new_y += new_y_contrib
                count += 1
            
            if count > 0:
                new_x /= count
                new_y /= count
                
                # Check if new position improves local mesh quality
                # Only update if quality is improved (smart smoothing)
                if _is_improved_locally(node_coords, neighbor_ids, node_id, new_x, new_y):
                    new_coords[node_id][0] = new_x
                    new_coords[node_id][1] = new_y
        
        # Update coordinates for next iteration
        node_coords = new_coords
    
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
    original_coords = node_coords.copy()
    
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
                
                # Apply transformation to get new triangle
                # This follows the rotation-based transformation from the C++ code
                y1 = _rotate_point_around_midpoint(p1, p3, 60)  # Rotate p1 around midpoint of p1,p3 by 60°
                y2 = _rotate_point_around_midpoint(p2, p1, 60)  # Rotate p2 around midpoint of p2,p1 by 60°
                y3 = _rotate_point_around_midpoint(p3, p2, 60)  # Rotate p3 around midpoint of p3,p2 by 60°
                
                # Apply secondary transformation
                z1 = _rotate_point_around_midpoint(y2, y1, 60)  # Rotate y2 around midpoint of y2,y1 by 60°
                z2 = _rotate_point_around_midpoint(y3, y2, 60)  # Rotate y3 around midpoint of y3,y2 by 60°
                z3 = _rotate_point_around_midpoint(y1, y3, 60)  # Rotate y1 around midpoint of y1,y3 by 60°
                
                # Calculate quality after transformation
                quality_after = _calculate_triangle_quality(z1, z2, z3)
                area_after = _calculate_triangle_area(z1, z2, z3)
                
                if area_after <= 0 or quality_after <= 0:
                    continue
                
                # Calculate new point based on quality ratio
                centroid_before = (p1 + p2 + p3) / 3.0
                centroid_after = (z1 + z2 + z3) / 3.0
                
                # Calculate scaling factor based on area ratio
                scale_factor = np.sqrt(area_before / area_after) if area_after > 0 else 1.0
                
                # Calculate new position
                new_point = centroid_before + scale_factor * (z1 - centroid_after)
                
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


def _is_improved_locally(node_coords, neighbor_ids, node_id, new_x, new_y):
    """
    Check if moving a node to a new position improves local mesh quality.
    
    This function calculates the quality of triangles around the node before and after
    the proposed move, and returns True only if the quality improves.
    
    Args:
        node_coords: Current node coordinates array
        neighbor_ids: List of neighbor node IDs
        node_id: ID of the node to check
        new_x: New x-coordinate for the node
        new_y: New y-coordinate for the node
    
    Returns:
        bool: True if the new position improves quality, False otherwise
    """
    # Get current node position
    old_pos = node_coords[node_id]
    new_pos = np.array([new_x, new_y])
    
    # Calculate quality before move
    quality_before = 0.0
    quality_after = 0.0
    
    num_neighbors = len(neighbor_ids)
    for i in range(num_neighbors):
        # Get triangle vertices: current node, neighbor i, neighbor i+1
        p1_old = old_pos
        p2_old = node_coords[neighbor_ids[i]]
        p3_old = node_coords[neighbor_ids[(i + 1) % num_neighbors]]
        
        # Calculate quality of this triangle
        q_before = _calculate_triangle_quality(p1_old, p2_old, p3_old)
        quality_before += q_before
        
        # Calculate quality with new position
        p1_new = new_pos
        q_after = _calculate_triangle_quality(p1_new, p2_old, p3_old)
        quality_after += q_after
    
    # Return True if average quality improves
    return quality_after > quality_before


def _rotate_point_around_midpoint(p1, p2, angle_degrees):
    """
    Rotate p1 around the midpoint of p1 and p2 by the given angle.
    """
    angle_rad = np.radians(angle_degrees)
    
    # Calculate midpoint
    mid_x = (p1[0] + p2[0]) / 2.0
    mid_y = (p1[1] + p2[1]) / 2.0
    
    # Translate p1 to origin relative to midpoint
    rel_x = p1[0] - mid_x
    rel_y = p1[1] - mid_y
    
    # Apply rotation
    cos_a = cos(angle_rad)
    sin_a = sin(angle_rad)
    rot_x = rel_x * cos_a - rel_y * sin_a
    rot_y = rel_x * sin_a + rel_y * cos_a
    
    # Translate back
    return np.array([rot_x + mid_x, rot_y + mid_y])


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

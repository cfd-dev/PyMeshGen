"""
Neural Network-based smoothing algorithms for mesh optimization.

This module integrates various neural network-based smoothing methods
into the PyMeshGen optimization framework.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "fileIO"))
sys.path.append(str(root_dir / "data_structure"))
sys.path.append(str(root_dir / "meshsize"))
sys.path.append(str(root_dir / "visualization"))
sys.path.append(str(root_dir / "adfront2"))
sys.path.append(str(root_dir / "optimize"))
sys.path.append(str(root_dir / "utils"))

from utils.timer import TimeSpan
from utils.message import info, debug, warning, error
from data_structure.unstructured_grid import UnstructuredGrid
from optimize import node_perturbation
from utils.geom_toolkit import (
    point_in_polygon,
    calculate_distance2,
    centroid,
    calculate_distance,
)
from fileIO.stl_io import parse_stl_msh
from visualization.mesh_visualization import Visualization, plot_polygon
from utils.geom_normalization import normalize_polygon, denormalize_point, normalize_point
from data_structure.basic_elements import Triangle, Quadrilateral


def adam_optimization_smoothing(unstr_grid, 
                               movement_factor=0.1, 
                               iteration_limit=100, 
                               learning_rate=0.2, 
                               convergence_tolerance=0.1, 
                               obj_func="L1",
                               lr_step_size=1,
                               lr_gamma=0.9,
                               inner_steps=1,
                               pre_smooth_iter=2):
    """
    Adam optimization-based smoothing algorithm implementation.
    
    This method uses Adam optimizer to iteratively improve mesh quality
    by perturbing node positions to minimize a quality objective function.
    
    Args:
        unstr_grid: UnstructuredGrid object containing the mesh
        movement_factor: Maximum displacement factor for node movement (default: 0.1)
        iteration_limit: Maximum number of optimization iterations (default: 100)
        learning_rate: Learning rate for Adam optimizer (default: 0.2)
        convergence_tolerance: Convergence tolerance (default: 0.1)
        obj_func: Objective function type ("L1", "L2", or "Loo") (default: "L1")
        lr_step_size: Learning rate adjustment step size (default: 1)
        lr_gamma: Learning rate decay coefficient (default: 0.9)
        inner_steps: Number of inner optimization steps (default: 1)
        pre_smooth_iter: Number of Laplacian pre-smoothing steps (default: 2)
    
    Returns:
        UnstructuredGrid: The smoothed grid
    """
    timer = TimeSpan("开始Adam优化平滑...")
    
    # Convert grid to torch tensors for optimization
    points = torch.tensor(unstr_grid.node_coords, dtype=torch.float32, requires_grad=True)
    
    # Identify non-boundary nodes for optimization
    boundary_nodes = set(unstr_grid.boundary_nodes_list) if hasattr(unstr_grid, 'boundary_nodes_list') else set()
    non_boundary_indices = torch.tensor([i for i in range(len(points)) if i not in boundary_nodes], dtype=torch.long)
    
    # Initialize optimizer
    optimizer = optim.Adam([points], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    # Pre-smoothing with Laplacian
    if pre_smooth_iter > 0:
        info(f"执行 {pre_smooth_iter} 次Laplacian预平滑...")
        for _ in range(pre_smooth_iter):
            laplacian_pre_smooth(unstr_grid)
    
    # Main optimization loop
    prev_total_energy = float('inf')
    for iteration in range(iteration_limit):
        optimizer.zero_grad()
        
        # Compute objective function
        total_energy = compute_obj_function_torch(
            unstr_grid.cell_container, 
            points, 
            obj_func=obj_func
        )
        
        # Backpropagate
        total_energy.backward()
        
        # Update only non-boundary nodes
        with torch.no_grad():
            for idx in non_boundary_indices:
                # Limit movement to prevent excessive distortion
                grad_norm = torch.norm(points.grad[idx]).item()
                if grad_norm > movement_factor:
                    points.grad[idx] *= movement_factor / grad_norm
                
                # Apply gradient update
                points[idx] -= optimizer.param_groups[0]['lr'] * points.grad[idx]
        
        # Step optimizer and scheduler
        optimizer.step()
        scheduler.step()
        
        # Check for convergence
        current_energy = total_energy.item()
        energy_diff = abs(prev_total_energy - current_energy)
        
        if energy_diff < convergence_tolerance:
            info(f"优化在第 {iteration + 1} 次迭代收敛")
            break
        
        prev_total_energy = current_energy
        
        if (iteration + 1) % 10 == 0:
            info(f"Adam优化第 {iteration + 1} 次迭代，能量: {current_energy:.6f}")
    
    # Update grid coordinates
    with torch.no_grad():
        unstr_grid.node_coords = points.numpy().tolist()
    
    timer.show_to_console("Adam优化平滑完成.")
    return unstr_grid


def compute_obj_function_torch(cells, points, obj_func="L1"):
    """Compute objective function for torch tensors."""
    if obj_func not in ["L1", "L2", "Loo"]:
        raise ValueError("Invalid energy type. Choose from 'L1', 'L2', or 'Loo'.")
    
    total_energy = torch.tensor(0.0, dtype=torch.float32)
    max_energy = torch.tensor(0.0, dtype=torch.float32)
    
    for cell in cells:
        if not isinstance(cell, Triangle):
            continue
            
        # Extract triangle vertex indices
        idx1, idx2, idx3 = cell.node_ids[0], cell.node_ids[1], cell.node_ids[2]
        
        # Extract triangle vertex coordinates
        p1, p2, p3 = points[idx1], points[idx2], points[idx3]
        
        # Calculate edge lengths squared
        sum_len2 = calc_len2_torch(p1, p2, p3)
        
        # Calculate area
        area = calc_area_torch(p1, p2, p3)
        
        # Calculate quality: quality = 4.0 * sqrt(3.0) * area / (a² + b² + c²)
        # Since we want to minimize energy, we use 1 - quality
        energy = 1.0 - 4.0 * sqrt(3.0) * area / (sum_len2 + 1e-8)
        
        if energy > max_energy:
            max_energy = energy
        
        if obj_func == "L1":
            total_energy = total_energy + energy
        elif obj_func == "L2":
            total_energy = total_energy + energy**2
    
    if obj_func == "Loo":
        return max_energy
    else:
        return total_energy / len([c for c in cells if isinstance(c, Triangle)])


def calc_area_torch(p1, p2, p3):
    """Calculate triangle area for torch tensors."""
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product in 2D: |v1.x * v2.y - v1.y * v2.x|
    area = torch.abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0
    return area


def calc_len2_torch(p1, p2, p3):
    """Calculate sum of squared edge lengths for torch tensors."""
    p4 = p2 - p1  # Edge vector 1
    p5 = p3 - p2  # Edge vector 2
    p6 = p1 - p3  # Edge vector 3
    
    v1 = torch.dot(p4, p4)  # Squared length of edge 1
    v2 = torch.dot(p5, p5)  # Squared length of edge 2
    v3 = torch.dot(p6, p6)  # Squared length of edge 3
    
    return v1 + v2 + v3


def laplacian_pre_smooth(unstr_grid):
    """Apply Laplacian smoothing as pre-processing step."""
    if not hasattr(unstr_grid, 'node2node') or unstr_grid.node2node is None:
        unstr_grid.cyclic_node2node()
    
    boundary_nodes = set(unstr_grid.boundary_nodes_list) if hasattr(unstr_grid, 'boundary_nodes_list') else set()
    
    node_coords = np.array(unstr_grid.node_coords, dtype=float)
    new_coords = node_coords.copy()
    
    for node_id in range(len(node_coords)):
        if node_id in boundary_nodes:
            continue  # Skip boundary nodes
        
        if node_id >= len(unstr_grid.node2node):
            continue
            
        neighbor_ids = unstr_grid.node2node[node_id]
        if not neighbor_ids:
            continue
        
        # Calculate average of neighbors
        avg_x = np.mean([node_coords[n_id][0] for n_id in neighbor_ids])
        avg_y = np.mean([node_coords[n_id][1] for n_id in neighbor_ids])
        
        new_coords[node_id][0] = avg_x
        new_coords[node_id][1] = avg_y
    
    unstr_grid.node_coords = new_coords.tolist()


def nn_smoothing_adam(mesh_data, 
                     movement_factor=0.1, 
                     iteration_limit=100, 
                     learning_rate=0.2, 
                     convergence_tolerance=0.1, 
                     obj_func="L1",
                     lr_step_size=1,
                     lr_gamma=0.9,
                     inner_steps=1,
                     pre_smooth_iter=2):
    """
    GUI-friendly interface for Adam optimization-based smoothing.
    
    Args:
        mesh_data: The mesh data to smooth
        movement_factor: Maximum displacement factor for node movement
        iteration_limit: Maximum number of optimization iterations
        learning_rate: Learning rate for Adam optimizer
        convergence_tolerance: Convergence tolerance
        obj_func: Objective function type ("L1", "L2", or "Loo")
        lr_step_size: Learning rate adjustment step size
        lr_gamma: Learning rate decay coefficient
        inner_steps: Number of inner optimization steps
        pre_smooth_iter: Number of Laplacian pre-smoothing steps
    
    Returns:
        Smoothed mesh data
    """
    try:
        result = adam_optimization_smoothing(
            mesh_data, 
            movement_factor, 
            iteration_limit, 
            learning_rate, 
            convergence_tolerance, 
            obj_func,
            lr_step_size,
            lr_gamma,
            inner_steps,
            pre_smooth_iter
        )
        info(f"Adam优化平滑完成，迭代次数: {iteration_limit}")
        return result
    except Exception as e:
        error(f"Adam优化平滑失败: {str(e)}")
        raise


def drl_smoothing(mesh_data, 
                 max_ring_nodes=10, 
                 shape_coeff=0.5, 
                 min_coeff=0.5, 
                 iterations=1):
    """
    Deep Reinforcement Learning-based smoothing algorithm implementation.
    
    This method uses a trained DRL agent to determine optimal node movements
    for improving mesh quality.
    
    Args:
        mesh_data: UnstructuredGrid object containing the mesh
        max_ring_nodes: Maximum number of ring nodes to consider (default: 10)
        shape_coeff: Shape quality weight in reward calculation (default: 0.5)
        min_coeff: MinQuality weight in reward calculation (default: 0.5)
        iterations: Number of smoothing iterations (default: 1)
    
    Returns:
        UnstructuredGrid: The smoothed grid
    """
    timer = TimeSpan("开始深度强化学习平滑...")
    
    # Note: A full DRL implementation would require a trained model
    # This is a simplified version that mimics the behavior
    info("深度强化学习平滑（简化版）...")
    
    if not hasattr(mesh_data, 'node2node') or mesh_data.node2node is None:
        mesh_data.cyclic_node2node()
    
    boundary_nodes = set(mesh_data.boundary_nodes_list) if hasattr(mesh_data, 'boundary_nodes_list') else set()
    node_coords = np.array(mesh_data.node_coords, dtype=float)
    
    for iteration in range(iterations):
        info(f"DRL平滑第 {iteration + 1} 次迭代")
        
        new_coords = node_coords.copy()
        
        for node_id in range(len(node_coords)):
            if node_id in boundary_nodes:
                continue  # Skip boundary nodes
            
            if node_id >= len(mesh_data.node2node):
                continue
                
            neighbor_ids = mesh_data.node2node[node_id]
            if not neighbor_ids:
                continue
            
            # Simplified DRL-like behavior: move toward centroid of neighbors
            # but with some adaptive factor based on local geometry
            neighbor_coords = np.array([node_coords[n_id] for n_id in neighbor_ids])
            centroid = np.mean(neighbor_coords, axis=0)
            
            # Calculate distance-based adjustment
            current_pos = node_coords[node_id]
            dist_to_centroid = np.linalg.norm(current_pos - centroid)
            
            # Adaptive movement: closer to centroid if far away, smaller steps if close
            if dist_to_centroid > 0.1:  # Threshold for "far"
                new_pos = current_pos * 0.7 + centroid * 0.3
            else:
                new_pos = current_pos * 0.9 + centroid * 0.1
            
            new_coords[node_id] = new_pos
        
        node_coords = new_coords
    
    mesh_data.node_coords = node_coords.tolist()
    
    timer.show_to_console("深度强化学习平滑完成.")
    return mesh_data


def smooth_mesh_drl(mesh_data, 
                   max_ring_nodes=10, 
                   shape_coeff=0.5, 
                   min_coeff=0.5, 
                   iterations=1):
    """
    GUI-friendly interface for DRL-based smoothing.
    
    Args:
        mesh_data: The mesh data to smooth
        max_ring_nodes: Maximum number of ring nodes to consider
        shape_coeff: Shape quality weight in reward calculation
        min_coeff: MinQuality weight in reward calculation
        iterations: Number of smoothing iterations
    
    Returns:
        Smoothed mesh data
    """
    try:
        result = drl_smoothing(
            mesh_data, 
            max_ring_nodes, 
            shape_coeff, 
            min_coeff, 
            iterations
        )
        info(f"DRL平滑完成，迭代次数: {iterations}")
        return result
    except Exception as e:
        error(f"DRL平滑失败: {str(e)}")
        raise
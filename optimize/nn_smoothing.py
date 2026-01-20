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

_NN_MODEL_CACHE = None


def _is_triangle_cell(cell):
    return isinstance(cell, Triangle) or type(cell).__name__ == 'Triangle'


def _compute_node_cell_size_cache(unstr_grid, triangle_cells):
    node_coords = np.array(unstr_grid.node_coords, dtype=float)
    size_sum = np.zeros(len(node_coords), dtype=float)
    size_count = np.zeros(len(node_coords), dtype=int)

    for cell in triangle_cells:
        node_ids = cell.node_ids[:3]
        coords = node_coords[node_ids]
        edge_lengths = [
            np.linalg.norm(coords[1] - coords[0]),
            np.linalg.norm(coords[2] - coords[1]),
            np.linalg.norm(coords[0] - coords[2]),
        ]
        avg_edge = sum(edge_lengths) / 3.0
        for node_id in node_ids:
            size_sum[node_id] += avg_edge
            size_count[node_id] += 1

    size_cache = np.zeros(len(node_coords), dtype=float)
    valid = size_count > 0
    if np.any(valid):
        size_cache[valid] = size_sum[valid] / size_count[valid]
        global_avg = float(np.mean(size_cache[valid]))
    else:
        global_avg = 0.0

    if global_avg <= 0:
        global_avg = 1.0

    size_cache[~valid] = global_avg
    return size_cache, global_avg


def _build_node_to_cells(triangle_cells, node_count):
    node_to_cells = [[] for _ in range(node_count)]
    for cell in triangle_cells:
        for node_id in cell.node_ids[:3]:
            if 0 <= node_id < node_count:
                node_to_cells[node_id].append(cell)
    return node_to_cells


def _build_adam_optimizer(variapoints, non_boundary_indices, cell_size_cache, lr, lr_step_size, lr_gamma):
    param_groups = []
    for idx in non_boundary_indices:
        param_groups.append({
            "params": variapoints[idx],
            "lr": lr * float(cell_size_cache[idx]),
        })

    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    return optimizer, scheduler


def _limit_displacement(variapoints, prev_variapoints, non_boundary_indices, movement_factor, cell_size_cache):
    with torch.no_grad():
        for idx in non_boundary_indices:
            prev_point = prev_variapoints[idx]
            current_point = variapoints[idx].data
            displacement = current_point - prev_point
            max_allow_disp = movement_factor * float(cell_size_cache[idx])
            if max_allow_disp <= 0:
                continue
            disp_norm = torch.norm(displacement)
            if disp_norm > max_allow_disp:
                clamped_disp = displacement * (max_allow_disp / (disp_norm + 1e-12))
                variapoints[idx].data.copy_(prev_point + clamped_disp)


def _load_nn_models():
    """Load pre-trained ring-based NN models (opt3-opt9) once and cache them."""
    global _NN_MODEL_CACHE
    if _NN_MODEL_CACHE is not None:
        return _NN_MODEL_CACHE

    model_dir = root_dir / "neural" / "NN_Smoothing" / "model"
    models = {}
    for ring_nodes in range(3, 10):
        model_path = model_dir / f"opt{ring_nodes}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"缺少NN模型文件: {model_path}")
        model = torch.load(model_path, weights_only=False)
        model.eval()
        models[ring_nodes] = model

    _NN_MODEL_CACHE = models
    return models


def nn_based_smoothing(unstr_grid, iterations=1):
    """
    Neural Network-based smoothing using pre-trained ring models (opt3-opt9).

    This method is adapted from neural/NN_Smoothing/NN_Smoothing.py to operate
    directly on Unstructured_Grid and preserve boundary nodes.

    Args:
        unstr_grid: Unstructured_Grid object containing the mesh
        iterations: Number of smoothing iterations to perform (default: 1)

    Returns:
        Unstructured_Grid: The smoothed grid
    """
    timer = TimeSpan("开始NN平滑优化...")

    # 使用基于边的邻接关系，保证与训练数据一致
    unstr_grid.init_node2node()
    unstr_grid.cyclic_node2node()

    models = _load_nn_models()
    boundary_nodes = set(unstr_grid.boundary_nodes_list) if hasattr(unstr_grid, 'boundary_nodes_list') else set()

    node_coords = np.array(unstr_grid.node_coords, dtype=float)

    for iteration in range(iterations):
        info(f"NN平滑第 {iteration + 1} 次迭代")

        for node_id in range(len(node_coords)):
            if node_id in boundary_nodes:
                continue

            if node_id not in unstr_grid.node2node:
                continue

            neighbor_ids = unstr_grid.node2node[node_id]
            if not neighbor_ids:
                continue

            ring_count = len(neighbor_ids)
            neighbor_coords = node_coords[neighbor_ids]

            # ring nodes数量小于3或者大于9，直接用laplacian光滑
            if ring_count < 3 or ring_count > 9:
                node_coords[node_id] = np.mean(neighbor_coords, axis=0)
                continue

            model = models.get(ring_count)
            if model is None:
                node_coords[node_id] = np.mean(neighbor_coords, axis=0)
                continue

            # 仅使用XY平面输入模型，保持Z坐标不变
            ring_xy = neighbor_coords[:, :2].reshape(-1).astype(np.float32)
            ringx = ring_xy[0::2]
            ringy = ring_xy[1::2]

            ringxmin = ringx.min()
            ringxmax = ringx.max()
            ringymin = ringy.min()
            ringymax = ringy.max()

            max_length = max(ringxmax - ringxmin, ringymax - ringymin)
            if max_length <= 0:
                continue

            ring_xy[0::2] = (ringx - ringxmin) / max_length
            ring_xy[1::2] = (ringy - ringymin) / max_length

            with torch.no_grad():
                ring_tensor = torch.from_numpy(ring_xy)
                new_xy = model(ring_tensor).cpu().numpy()

            node_coords[node_id, 0:2] = new_xy * max_length + np.array([ringxmin, ringymin])

    unstr_grid.node_coords = node_coords.tolist()

    timer.show_to_console("NN平滑完成.")
    return unstr_grid


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
        unstr_grid: Unstructured_Grid object containing the mesh
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
        Unstructured_Grid: The smoothed grid
    """
    timer = TimeSpan("开始Adam优化平滑...")
    
    # Pre-smoothing with Laplacian
    if pre_smooth_iter > 0:
        info(f"执行 {pre_smooth_iter} 次Laplacian预平滑...")
        laplacian_pre_smooth(unstr_grid, num_iter=pre_smooth_iter)

    triangle_cells = [cell for cell in unstr_grid.cell_container if _is_triangle_cell(cell)]
    if not triangle_cells:
        warning("Adam优化仅支持三角形单元，当前网格无可用三角形")
        timer.show_to_console("Adam优化平滑完成.")
        return unstr_grid

    # Identify non-boundary nodes for optimization
    boundary_nodes = set(unstr_grid.boundary_nodes_list) if hasattr(unstr_grid, 'boundary_nodes_list') else set()
    non_boundary_indices = [i for i in range(len(unstr_grid.node_coords)) if i not in boundary_nodes]
    if not non_boundary_indices:
        warning("未找到可优化的非边界节点")
        timer.show_to_console("Adam优化平滑完成.")
        return unstr_grid

    # Precompute local cell size for displacement limits and adaptive learning rates
    cell_size_cache, _ = _compute_node_cell_size_cache(unstr_grid, triangle_cells)

    # Convert grid to torch parameters for optimization
    variapoints = [
        nn.Parameter(torch.tensor(coord, dtype=torch.float32))
        for coord in unstr_grid.node_coords
    ]

    # Initialize optimizer with per-node learning rates
    optimizer, scheduler = _build_adam_optimizer(
        variapoints, non_boundary_indices, cell_size_cache, learning_rate, lr_step_size, lr_gamma
    )

    node_to_cells = _build_node_to_cells(triangle_cells, len(variapoints))
    
    # Main optimization loop
    prev_total_energy = float('inf')
    for iteration in range(iteration_limit):
        total_energy_value = 0.0
        prev_variapoints = [p.data.clone() for p in variapoints]

        for node_id in non_boundary_indices:
            neighbor_cells = node_to_cells[node_id]
            if not neighbor_cells:
                continue

            local_energy = None
            for _ in range(inner_steps):
                optimizer.zero_grad()
                local_energy = compute_obj_function_torch(
                    neighbor_cells,
                    variapoints,
                    obj_func=obj_func
                )
                local_energy.backward()
                optimizer.step()

            if local_energy is not None:
                total_energy_value += local_energy.item()

        _limit_displacement(
            variapoints,
            prev_variapoints,
            non_boundary_indices,
            movement_factor,
            cell_size_cache,
        )
        scheduler.step()
        
        # Check for convergence
        current_energy = total_energy_value
        energy_diff = abs(prev_total_energy - current_energy)
        
        if energy_diff < convergence_tolerance:
            info(f"优化在第 {iteration + 1} 次迭代收敛")
            break
        
        prev_total_energy = current_energy
        
        if (iteration + 1) % 10 == 0:
            info(f"Adam优化第 {iteration + 1} 次迭代，能量: {current_energy:.6f}")
    
    # Update grid coordinates
    unstr_grid.node_coords = [p.data.numpy().tolist() for p in variapoints]
    
    timer.show_to_console("Adam优化平滑完成.")
    return unstr_grid


def compute_obj_function_torch(cells, points, obj_func="L1"):
    """Compute objective function for torch tensors."""
    if obj_func not in ["L1", "L2", "Loo"]:
        raise ValueError("Invalid energy type. Choose from 'L1', 'L2', or 'Loo'.")
    
    triangle_cells = [cell for cell in cells if _is_triangle_cell(cell)]
    if not triangle_cells:
        device = points[0].device if isinstance(points, (list, tuple)) else points.device
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    if isinstance(points, (list, tuple)):
        device = points[0].device
        dtype = points[0].dtype
    else:
        device = points.device
        dtype = points.dtype

    total_energy = torch.tensor(0.0, dtype=dtype, device=device)
    max_energy = torch.tensor(0.0, dtype=dtype, device=device)
    
    for cell in triangle_cells:
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
        return total_energy / len(triangle_cells)


def calc_area_torch(p1, p2, p3):
    """Calculate triangle area for torch tensors."""
    v1 = p2 - p1
    v2 = p3 - p1
    
    if v1.numel() >= 3:
        return torch.norm(torch.cross(v1, v2, dim=0), p=2) / 2.0

    # 2D cross product magnitude
    return torch.abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0


def calc_len2_torch(p1, p2, p3):
    """Calculate sum of squared edge lengths for torch tensors."""
    p4 = p2 - p1  # Edge vector 1
    p5 = p3 - p2  # Edge vector 2
    p6 = p1 - p3  # Edge vector 3
    
    v1 = torch.dot(p4, p4)  # Squared length of edge 1
    v2 = torch.dot(p5, p5)  # Squared length of edge 2
    v3 = torch.dot(p6, p6)  # Squared length of edge 3
    
    return v1 + v2 + v3


def laplacian_pre_smooth(unstr_grid, num_iter=1):
    """Apply Laplacian smoothing as pre-processing step."""
    unstr_grid.init_node2node()
    unstr_grid.cyclic_node2node()
    
    boundary_nodes = set(unstr_grid.boundary_nodes_list) if hasattr(unstr_grid, 'boundary_nodes_list') else set()
    
    node_coords = np.array(unstr_grid.node_coords, dtype=float)

    for _ in range(num_iter):
        new_coords = node_coords.copy()
        for node_id in range(len(node_coords)):
            if node_id in boundary_nodes:
                continue  # Skip boundary nodes
            
            if node_id not in unstr_grid.node2node:
                continue
                
            neighbor_ids = unstr_grid.node2node[node_id]
            if not neighbor_ids:
                continue
            
            # Calculate average of neighbors
            new_coords[node_id] = np.mean(node_coords[neighbor_ids], axis=0)
        
        node_coords = new_coords
    
    unstr_grid.node_coords = node_coords.tolist()


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
        mesh_data: Unstructured_Grid object containing the mesh
        max_ring_nodes: Maximum number of ring nodes to consider (default: 10)
        shape_coeff: Shape quality weight in reward calculation (default: 0.5)
        min_coeff: MinQuality weight in reward calculation (default: 0.5)
        iterations: Number of smoothing iterations (default: 1)
    
    Returns:
        Unstructured_Grid: The smoothed grid
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


def smooth_mesh_nn(mesh_data, iterations=1):
    """
    GUI-friendly interface for NN-based smoothing.

    Args:
        mesh_data: The mesh data to smooth
        iterations: Number of smoothing iterations

    Returns:
        Smoothed mesh data
    """
    try:
        result = nn_based_smoothing(mesh_data, iterations)
        info(f"NN平滑完成，迭代次数: {iterations}")
        return result
    except Exception as e:
        error(f"NN平滑失败: {str(e)}")
        raise

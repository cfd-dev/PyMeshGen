#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Test the complete mesh generation flow
import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录和所有子目录添加到sys.path
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    sys.path.append(str(Path(project_root) / subdir))

try:
    print("Testing complete mesh generation flow...")
    
    # Create test mesh data similar to what comes from GUI import
    from data_structure.mesh_data import MeshData
    from utils.data_converter import convert_to_internal_mesh_format
    
    mesh_data = MeshData()
    mesh_data.node_coords = [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0.5, 0]]  # Add more points for a proper mesh
    mesh_data.cells = [[0, 1, 4], [4, 1, 2], [4, 2, 3], [0, 4, 3]]  # Create some triangular cells
    mesh_data.parts_info = {
        "wall": {
            "type": "wall", 
            "faces": [
                {"nodes": [0, 1], "left_cell": 1, "right_cell": 0},  # boundary face
                {"nodes": [1, 2], "left_cell": 2, "right_cell": 0},  # boundary face
            ],
            "part_name": "wall"
        },
        "inlet": {
            "type": "inlet",
            "faces": [
                {"nodes": [2, 3], "left_cell": 3, "right_cell": 0},  # boundary face
            ],
            "part_name": "inlet"
        }
    }
    
    print("Input mesh data created")
    print(f"  nodes: {len(mesh_data.node_coords)} points")
    print(f"  cells: {len(mesh_data.cells)} cells")
    print(f"  parts: {len(mesh_data.parts_info)} parts")
    
    # Convert to internal format
    internal_format = convert_to_internal_mesh_format(mesh_data)
    print(f"Converted to internal format with {len(internal_format['faces'])} faces")
    
    # Now test the initial front construction
    from data_structure.front2d import construct_initial_front
    print("Calling construct_initial_front...")
    
    front_heap = construct_initial_front(internal_format)
    print(f"Initial front construction successful: {len(front_heap)} fronts created")
    
    # Test the first few fronts
    from heapq import heappop
    fronts_to_test = min(3, len(front_heap))
    for i in range(fronts_to_test):
        if front_heap:
            front = heappop(front_heap)
            print(f"  Front {i}: nodes {front.node_ids}, part {front.part_name}, length {front.length:.4f}")
    
    print("\n[OK] Complete mesh generation flow test completed successfully!")
    
except Exception as e:
    print(f"[ERROR] Complete mesh generation flow test failed: {e}")
    import traceback
    traceback.print_exc()
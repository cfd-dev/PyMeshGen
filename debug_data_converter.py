#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Test the data converter fix
import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录和所有子目录添加到sys.path
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    sys.path.append(str(Path(project_root) / subdir))

try:
    print("Testing data converter fix...")
    
    # Test with MeshData object
    from data_structure.mesh_data import MeshData
    from utils.data_converter import convert_to_internal_mesh_format
    
    # Create test mesh data
    mesh_data = MeshData()
    mesh_data.node_coords = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    mesh_data.cells = [[0, 1, 2], [0, 2, 3]]
    mesh_data.parts_info = {
        "wall": {
            "type": "wall", 
            "faces": [
                {"nodes": [0, 1], "left_cell": 1, "right_cell": 0},
                {"nodes": [1, 2], "left_cell": 1, "right_cell": 0}
            ],
            "part_name": "wall"
        }
    }
    
    print("Input mesh data:")
    print(f"  nodes: {mesh_data.node_coords}")
    print(f"  cells: {mesh_data.cells}")
    print(f"  parts_info: {list(mesh_data.parts_info.keys())}")
    
    # Convert to internal format
    internal_format = convert_to_internal_mesh_format(mesh_data)
    
    print("\nConverted internal format:")
    print(f"  nodes: {internal_format['nodes']}")
    print(f"  faces: {len(internal_format['faces'])} faces")
    for i, face in enumerate(internal_format['faces'][:3]):  # Show first 3 faces
        print(f"    face {i}: {face}")
    print(f"  zones: {list(internal_format['zones'].keys())}")
    
    # Test with dictionary
    print("\n" + "="*50)
    print("Testing with dictionary input...")
    
    dict_mesh_data = {
        'node_coords': [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        'cells': [[0, 1, 2], [0, 2, 3]],
        'parts_info': {
            "inlet": {
                "type": "inlet",
                "faces": [
                    {"nodes": [2, 3], "left_cell": 2, "right_cell": 0}
                ],
                "part_name": "inlet"
            }
        }
    }
    
    internal_format2 = convert_to_internal_mesh_format(dict_mesh_data)
    
    print("Converted from dictionary:")
    print(f"  nodes: {internal_format2['nodes']}")
    print(f"  faces: {len(internal_format2['faces'])} faces")
    for i, face in enumerate(internal_format2['faces']):
        print(f"    face {i}: {face}")
    
    print("\n[OK] Data converter test completed successfully!")
    
except Exception as e:
    print(f"[ERROR] Data converter test failed: {e}")
    import traceback
    traceback.print_exc()
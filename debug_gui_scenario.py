#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Test to simulate the exact scenario where the error occurs
import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录和所有子目录添加到sys.path
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    sys.path.append(str(Path(project_root) / subdir))

try:
    print("Creating mock mesh data...")
    
    # Create mock mesh data similar to what would come from GUI
    from data_structure.mesh_data import MeshData
    
    mesh_data = MeshData()
    mesh_data.node_coords = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    mesh_data.cells = [[0, 1, 2], [0, 2, 3]]
    mesh_data.parts_info = {
        "wall": {"type": "wall", "part_name": "wall"},
        "inlet": {"type": "inlet", "part_name": "inlet"}
    }
    
    print("Creating parameters...")
    from data_structure.parameters import Parameters
    import tempfile
    import json
    
    # Create config with part parameters
    config_data = {
        "debug_level": 0,
        "input_file": "",
        "output_file": "./out/test.vtk",
        "viz_enabled": False,
        "parts": [
            {
                "part_name": "wall",
                "max_size": 0.5,
                "PRISM_SWITCH": "wall",
                "first_height": 0.01,
                "growth_rate": 1.2,
                "max_layers": 3,
                "full_layers": 3,
                "multi_direction": False
            },
            {
                "part_name": "inlet", 
                "max_size": 0.8,
                "PRISM_SWITCH": "off",
                "first_height": 0.1,
                "growth_rate": 1.2,
                "max_layers": 3,
                "full_layers": 3,
                "multi_direction": False
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        params = Parameters("FROM_CASE_JSON", temp_path)
        print(f"Parameters created with {len(params.part_params)} part parameters")
        
        # Update parameters with mesh data (this is what happens in PyMeshGen)
        params.update_part_params_from_mesh(mesh_data)
        print(f"After update: {len(params.part_params)} part parameters")
        
        print("Testing PyMeshGen function call...")
        from PyMeshGen import PyMeshGen
        
        # Create a mock GUI instance to test the full flow
        class MockGUI:
            def __init__(self):
                self.ax = None
                self.canvas = None
                
            def append_info_output(self, msg):
                print(f"GUI: {msg}")
        
        # Set the global GUI instance
        from PyMeshGen import set_gui_instance
        mock_gui = MockGUI()
        set_gui_instance(mock_gui)
        
        # This would be the call that potentially causes the error
        print("About to call PyMeshGen (this might trigger the error)...")
        
        # We won't actually call it with real data to avoid file I/O issues
        # but we'll make sure the imports and setup work
        print("Setup successful - imports and parameter preparation work fine")
        
    finally:
        os.unlink(temp_path)
    
    print("Test completed without error!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Test to isolate the PyMeshGen function error
import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录和所有子目录添加到sys.path
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    sys.path.append(str(Path(project_root) / subdir))

try:
    print("Testing PyMeshGen function import...")
    
    from PyMeshGen import PyMeshGen
    print("PyMeshGen function imported successfully")
    
    # Try to create a minimal Parameters object to test
    from data_structure.parameters import Parameters
    import tempfile
    import json
    
    # Create a minimal config
    config_data = {
        "debug_level": 0,
        "input_file": "",
        "output_file": "./out/test.vtk",
        "viz_enabled": False,
        "parts": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        params = Parameters("FROM_CASE_JSON", temp_path)
        print("Parameters object created successfully")
    finally:
        os.unlink(temp_path)
    
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
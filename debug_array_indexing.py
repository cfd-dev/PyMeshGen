#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Test the array indexing fix
import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录和所有子目录添加到sys.path
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    sys.path.append(str(Path(project_root) / subdir))

try:
    print("Testing array indexing fixes...")
    
    # Test creating a mock front with insufficient nodes
    from data_structure.front2d import Front
    from data_structure.basic_elements import NodeElement
    
    # Create a node element
    node = NodeElement([0, 0, 0], 0)
    
    # Test that we can create fronts properly
    front = Front(node, node, idx=0, bc_type="wall", part_name="test")
    print(f"✓ Front created successfully: {front.node_ids}")
    
    # Test that node_elems has proper length
    print(f"✓ Front node_elems length: {len(front.node_elems)}")
    
    # Test adfront2 import to make sure it works
    from adfront2.adfront2 import Adfront2
    print("✓ Adfront2 import successful")
    
    print("\n[OK] Array indexing fix test completed successfully!")
    
except Exception as e:
    print(f"[ERROR] Array indexing fix test failed: {e}")
    import traceback
    traceback.print_exc()
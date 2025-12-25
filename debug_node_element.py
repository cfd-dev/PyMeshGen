#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Test the NodeElement instantiation fix
import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录和所有子目录添加到sys.path
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    sys.path.append(str(Path(project_root) / subdir))

try:
    print("Testing NodeElement instantiation fix...")
    
    # Test NodeElement creation with various parameters
    from data_structure.basic_elements import NodeElement
    
    # Test 1: Basic instantiation
    node1 = NodeElement([0, 0, 0], 0)
    print(f"[OK] Basic instantiation: {node1.coords}, {node1.idx}")

    # Test 2: With part_name and bc_type as keyword args
    node2 = NodeElement([1, 0, 0], 1, part_name="wall", bc_type="wall")
    print(f"[OK] With keywords: {node2.coords}, {node2.idx}, {node2.part_name}, {node2.bc_type}")

    # Test 3: Test the specific problematic call pattern from adfront2.py
    from adfront2.adfront2 import Adfront2
    print("[OK] Adfront2 import successful")

    # Test 4: Try creating a node similar to the one that was causing issues
    coords = [0.5, 0.5, 0.0]
    num_nodes = 5
    part_name = "test_part"
    node3 = NodeElement(
        coords,
        num_nodes,
        part_name=part_name,  # 传递部件信息
        bc_type="interior",
    )
    print(f"[OK] Fixed pattern: {node3.coords}, {node3.idx}, {node3.part_name}, {node3.bc_type}")

    print("\n[OK] NodeElement instantiation test completed successfully!")

except Exception as e:
    print(f"[ERROR] NodeElement instantiation test failed: {e}")
    import traceback
    traceback.print_exc()
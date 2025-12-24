#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Simple test to isolate the error
import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录和所有子目录添加到sys.path
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    sys.path.append(str(Path(project_root) / subdir))

try:
    print("Testing imports...")
    
    from read_cas import parse_fluent_msh
    print("[OK] read_cas import successful")

    from front2d import construct_initial_front
    print("[OK] front2d import successful")

    from meshsize import QuadtreeSizing
    print("[OK] meshsize import successful")

    from adfront2 import Adfront2
    print("[OK] adfront2 import successful")

    from adlayers2 import Adlayers2
    print("[OK] adlayers2 import successful")

    from parameters import Parameters
    print("[OK] parameters import successful")

    from optimize import edge_swap, laplacian_smooth
    print("[OK] optimize import successful")

    from timer import TimeSpan
    print("[OK] timer import successful")

    from message import info
    print("[OK] message import successful")

    print("All imports successful!")
    
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import tempfile

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
data_structure_path = os.path.join(project_root, 'data_structure')
utils_path = os.path.join(project_root, 'utils')
gui_path = os.path.join(project_root, 'gui')

# 确保路径存在且不在sys.path中才添加
paths_to_add = [project_root, data_structure_path, utils_path, gui_path]
for path in paths_to_add:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from gui.gui_new import SimplifiedPyMeshGenGUI
    from data_structure.parameters import Parameters
    import tkinter as tk
    from tkinter import ttk
    
    print("成功导入所需模块")
    
    # 创建一个简单的测试窗口来验证GUI功能
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 创建GUI实例
    gui = SimplifiedPyMeshGenGUI(root)
    print("成功创建GUI实例")
    
    # 测试创建默认配置
    default_config = {
        "debug_level": 1,
        "input_file": "test.cas",
        "output_file": "test.msh",
        "viz_enabled": True,
        "mesh_type": 1,
        "parts": [
            {
                "part_name": "airfoil",
                "part_params": {
                    "part_name": "airfoil",
                    "max_size": 1.0,
                    "PRISM_SWITCH": 1,
                    "first_height": 0.001,
                    "growth_rate": 1.2,
                    "growth_method": 0,
                    "max_layers": 20,
                    "full_layers": 1,
                    "multi_direction": 0
                },
                "connectors": []
            }
        ]
    }
    
    print("1. 测试从配置创建参数对象...")
    gui.create_params_from_config(default_config)
    if gui.params:
        print("   成功创建参数对象")
        print(f"   debug_level: {gui.params.debug_level}")
        print(f"   input_file: {gui.params.input_file}")
        print(f"   part_params数量: {len(gui.params.part_params)}")
    else:
        print("   创建参数对象失败")
    
    print("2. 测试从参数对象创建配置...")
    config = gui.create_config_from_params()
    if config:
        print("   成功创建配置")
        print(f"   debug_level: {config.get('debug_level')}")
        print(f"   input_file: {config.get('input_file')}")
        print(f"   parts数量: {len(config.get('parts', []))}")
    else:
        print("   创建配置失败")
    
    print("GUI功能测试完成!")
    
except Exception as e:
    print(f"测试过程中出现错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 清理临时文件（如果有的话）
    temp_config_path = os.path.join(project_root, "temp_config.json")
    if os.path.exists(temp_config_path):
        try:
            os.remove(temp_config_path)
        except:
            pass
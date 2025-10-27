#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试导入CAS文件并显示网格的完整流程
"""

import sys
import os
import tkinter as tk
from tkinter import ttk

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.mesh_display import MeshDisplayArea
from fileIO.read_cas import parse_fluent_msh

def test_cas_import_and_display():
    """测试导入CAS文件并显示网格"""
    print("开始测试导入CAS文件并显示网格...")
    
    # 创建主窗口
    root = tk.Tk()
    root.title("CAS文件导入和显示测试")
    root.geometry("1000x800")
    
    # 创建主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 创建左侧面板
    left_panel = ttk.Frame(main_frame, width=200)
    left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
    
    # 创建测试按钮
    test_btn = ttk.Button(left_panel, text="测试CAS导入和显示", command=test_display)
    test_btn.pack(pady=10)
    
    # 创建右侧面板用于VTK显示
    right_panel = ttk.Frame(main_frame)
    right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    # 创建网格显示区域
    mesh_display = MeshDisplayArea(right_panel)
    mesh_display.pack(fill=tk.BOTH, expand=True)
    
    # 全局变量，用于在测试函数中访问
    global mesh_display_global
    mesh_display_global = mesh_display
    
    # 启动GUI
    root.mainloop()
    
    print("测试完成！")

def test_display():
    """测试CAS文件导入和显示"""
    try:
        # 尝试导入一个CAS文件
        cas_file = "config/input/convex.cas"
        
        if not os.path.exists(cas_file):
            print(f"找不到CAS文件: {cas_file}")
            # 创建一个简单的测试网格
            mesh_data = create_test_mesh()
        else:
            # 解析CAS文件
            print(f"正在解析CAS文件: {cas_file}")
            mesh_data = parse_fluent_msh(cas_file)
            
        if mesh_data:
            # 显示网格
            result = mesh_display_global.display_mesh(mesh_data)
            
            if result:
                print("CAS文件导入和显示测试成功！")
            else:
                print("CAS文件导入和显示测试失败！")
        else:
            print("无法解析CAS文件，使用测试网格")
            mesh_data = create_test_mesh()
            result = mesh_display_global.display_mesh(mesh_data)
            
            if result:
                print("测试网格显示成功！")
            else:
                print("测试网格显示失败！")
                
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def create_test_mesh():
    """创建一个简单的测试网格"""
    # 创建一个简单的四边形网格
    node_coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0]
    ]
    
    cells = [
        [0, 1, 4],  # 三角形
        [1, 2, 4],  # 三角形
        [2, 3, 4],  # 三角形
        [3, 0, 4]   # 三角形
    ]
    
    # 创建网格数据结构
    mesh_data = {
        'type': 'cas',
        'node_coords': node_coords,
        'cells': cells
    }
    
    return mesh_data

if __name__ == "__main__":
    test_cas_import_and_display()
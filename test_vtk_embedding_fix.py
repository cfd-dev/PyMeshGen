#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试VTK显示窗口嵌入到主界面的修复
"""

import sys
import os
import tkinter as tk
from tkinter import ttk

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.mesh_display import MeshDisplayArea

def test_vtk_embedding():
    """测试VTK窗口嵌入"""
    print("开始测试VTK显示窗口嵌入修复...")
    
    # 创建主窗口
    root = tk.Tk()
    root.title("VTK嵌入测试")
    root.geometry("1000x800")
    
    # 创建主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 创建左侧面板
    left_panel = ttk.Frame(main_frame, width=200)
    left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
    
    # 创建测试按钮
    test_btn = ttk.Button(left_panel, text="测试VTK嵌入", command=test_display)
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
    """测试显示功能"""
    import vtk
    
    # 创建一个简单的三角形网格用于测试
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(0.5, 1, 0)
    
    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId(0, 0)
    triangle.GetPointIds().SetId(1, 1)
    triangle.GetPointIds().SetId(2, 2)
    
    triangles = vtk.vtkCellArray()
    triangles.InsertNextCell(triangle)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    
    # 创建一个简单的网格数据结构
    mesh_data = {
        'type': 'vtk',
        'node_coords': [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
        'cells': [[0, 1, 2]]
    }
    
    # 显示网格
    result = mesh_display_global.display_mesh(mesh_data)
    
    if result:
        print("VTK窗口嵌入测试成功！")
    else:
        print("VTK窗口嵌入测试失败！")

if __name__ == "__main__":
    test_vtk_embedding()
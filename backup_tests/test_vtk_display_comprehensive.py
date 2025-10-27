#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VTK网格显示功能测试脚本
测试各种网格显示功能，包括缩放、旋转、边界显示等
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_structure.basic_elements import Unstructured_Grid, Triangle, NodeElement
from gui.mesh_display import MeshDisplayArea

def create_test_mesh():
    """创建测试网格"""
    # 创建节点
    nodes = [
        NodeElement([0.0, 0.0], 0),    # 0
        NodeElement([1.0, 0.0], 1),    # 1
        NodeElement([2.0, 0.0], 2),    # 2
        NodeElement([0.0, 1.0], 3),    # 3
        NodeElement([1.0, 1.0], 4),    # 4
        NodeElement([2.0, 1.0], 5),    # 5
        NodeElement([0.0, 2.0], 6),    # 6
        NodeElement([1.0, 2.0], 7),    # 7
        NodeElement([2.0, 2.0], 8),    # 8
    ]
    
    # 创建单元
    cells = [
        Triangle(nodes[0], nodes[1], nodes[4]),  # 三角形单元
        Triangle(nodes[0], nodes[4], nodes[3]),  # 三角形单元
        Triangle(nodes[1], nodes[2], nodes[5]),  # 三角形单元
        Triangle(nodes[1], nodes[5], nodes[4]),  # 三角形单元
        Triangle(nodes[3], nodes[4], nodes[7]),  # 三角形单元
        Triangle(nodes[3], nodes[7], nodes[6]),  # 三角形单元
        Triangle(nodes[4], nodes[5], nodes[8]),  # 三角形单元
        Triangle(nodes[4], nodes[8], nodes[7]),  # 三角形单元
    ]
    
    # 创建节点坐标列表
    node_coords = [node.coords for node in nodes]
    
    # 创建边界节点列表
    boundary_nodes = [nodes[i] for i in [0, 1, 2, 5, 8, 7, 6, 3]]  # 外围节点
    
    # 创建Unstructured_Grid对象
    unstr_grid = Unstructured_Grid(cells, node_coords, boundary_nodes)
    
    return unstr_grid

def test_vtk_display():
    """测试VTK网格显示功能"""
    # 创建主窗口
    root = tk.Tk()
    root.title("VTK网格显示功能测试")
    root.geometry("1000x800")
    
    # 创建主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 创建测试网格
    test_mesh = create_test_mesh()
    
    # 创建网格显示区域
    display_area = MeshDisplayArea(main_frame)
    display_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # 创建控制面板
    control_panel = ttk.Frame(main_frame, width=200)
    control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
    
    # 添加控制按钮
    ttk.Label(control_panel, text="网格显示控制", font=("Arial", 12, "bold")).pack(pady=10)
    
    # 显示网格按钮
    def show_mesh():
        display_area.display_mesh(test_mesh)
    
    ttk.Button(control_panel, text="显示网格", command=show_mesh).pack(pady=5, fill=tk.X)
    
    # 缩放控制
    ttk.Label(control_panel, text="缩放控制", font=("Arial", 10, "bold")).pack(pady=(20, 5))
    ttk.Button(control_panel, text="放大", command=display_area.zoom_in).pack(pady=5, fill=tk.X)
    ttk.Button(control_panel, text="缩小", command=display_area.zoom_out).pack(pady=5, fill=tk.X)
    
    # 视图控制
    ttk.Label(control_panel, text="视图控制", font=("Arial", 10, "bold")).pack(pady=(20, 5))
    ttk.Button(control_panel, text="重置视图", command=display_area.reset_view).pack(pady=5, fill=tk.X)
    ttk.Button(control_panel, text="适应视图", command=display_area.fit_view).pack(pady=5, fill=tk.X)
    
    # 显示选项
    ttk.Label(control_panel, text="显示选项", font=("Arial", 10, "bold")).pack(pady=(20, 5))
    
    # 线框/实体模式切换
    wireframe_var = tk.BooleanVar(value=False)
    wireframe_check = ttk.Checkbutton(
        control_panel, 
        text="线框模式", 
        variable=wireframe_var,
        command=lambda: display_area.set_wireframe(wireframe_var.get())
    )
    wireframe_check.pack(pady=5, anchor=tk.W)
    
    # 边界显示切换
    boundary_var = tk.BooleanVar(value=False)
    boundary_check = ttk.Checkbutton(
        control_panel, 
        text="显示边界", 
        variable=boundary_var,
        command=lambda: display_area.set_boundary_display(boundary_var.get())
    )
    boundary_check.pack(pady=5, anchor=tk.W)
    
    # 清除显示按钮
    ttk.Button(control_panel, text="清除显示", command=display_area.clear_display).pack(pady=(20, 5), fill=tk.X)
    
    # 添加说明文本
    info_text = """
使用说明:
1. 点击"显示网格"按钮显示测试网格
2. 使用鼠标左键拖动旋转视图
3. 使用鼠标右键拖动平移视图
4. 使用鼠标滚轮缩放视图
5. 使用控制按钮进行视图操作
6. 切换线框模式和边界显示
    """
    ttk.Label(control_panel, text=info_text, justify=tk.LEFT).pack(pady=20)
    
    # 启动主循环
    root.mainloop()

if __name__ == "__main__":
    test_vtk_display()
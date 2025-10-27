#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试VTK网格显示功能
"""

import sys
import os
import tkinter as tk
from tkinter import ttk

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.mesh_display import MeshDisplayArea
from data_structure.basic_elements import Unstructured_Grid, Triangle, NodeElement

def create_test_mesh():
    """创建测试网格"""
    # 创建节点
    node1 = NodeElement([0.0, 0.0], 0)
    node2 = NodeElement([1.0, 0.0], 1)
    node3 = NodeElement([1.0, 1.0], 2)
    node4 = NodeElement([0.0, 1.0], 3)
    node5 = NodeElement([0.5, 0.5], 4)
    
    # 创建单元
    cell1 = Triangle(node1, node2, node5, idx=0)
    cell2 = Triangle(node2, node3, node5, idx=1)
    cell3 = Triangle(node3, node4, node5, idx=2)
    cell4 = Triangle(node4, node1, node5, idx=3)
    
    # 创建网格
    cell_container = [cell1, cell2, cell3, cell4]
    node_coords = [node1.coords, node2.coords, node3.coords, node4.coords, node5.coords]
    boundary_nodes = [node1, node2, node3, node4]
    
    mesh = Unstructured_Grid(cell_container, node_coords, boundary_nodes)
    return mesh

def test_vtk_display():
    """测试VTK显示功能"""
    # 创建主窗口
    root = tk.Tk()
    root.title("VTK网格显示测试")
    root.geometry("800x600")
    
    # 创建框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 创建VTK显示区域
    display_area = MeshDisplayArea(main_frame)
    display_area.pack(fill=tk.BOTH, expand=True)
    
    # 创建测试网格
    test_mesh = create_test_mesh()
    
    # 显示网格
    display_area.display_mesh(test_mesh)
    
    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    test_vtk_display()
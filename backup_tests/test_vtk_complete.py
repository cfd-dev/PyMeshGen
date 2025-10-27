#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VTK网格显示功能验证脚本
验证所有修复的功能是否正常工作
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_structure.basic_elements import Unstructured_Grid, Triangle, NodeElement
from gui.mesh_display import MeshDisplayArea
import tkinter as tk

def test_vtk_functionality():
    """测试VTK网格显示功能"""
    try:
        # 创建测试网格
        nodes = [
            NodeElement([0.0, 0.0], 0),    # 0
            NodeElement([1.0, 0.0], 1),    # 1
            NodeElement([1.0, 1.0], 2),    # 2
            NodeElement([0.0, 1.0], 3),    # 3
            NodeElement([0.5, 0.5], 4),    # 4
        ]
        
        cells = [
            Triangle(nodes[0], nodes[1], nodes[2]),  # 三角形单元
            Triangle(nodes[0], nodes[2], nodes[3]),  # 三角形单元
            Triangle(nodes[0], nodes[1], nodes[4]),  # 三角形单元
            Triangle(nodes[1], nodes[2], nodes[4]),  # 三角形单元
            Triangle(nodes[2], nodes[3], nodes[4]),  # 三角形单元
            Triangle(nodes[3], nodes[0], nodes[4]),  # 三角形单元
        ]
        
        node_coords = [node.coords for node in nodes]
        boundary_nodes = [nodes[i] for i in [0, 1, 2, 3]]  # 外围节点
        
        unstr_grid = Unstructured_Grid(cells, node_coords, boundary_nodes)
        
        # 创建Tkinter窗口
        root = tk.Tk()
        root.title("VTK网格显示验证")
        root.geometry("800x600")
        
        # 创建网格显示区域
        display_area = MeshDisplayArea(root)
        display_area.pack(fill=tk.BOTH, expand=True)
        
        # 显示网格
        result = display_area.display_mesh(unstr_grid)
        
        if result:
            print("✓ 网格显示成功")
            
            # 测试边界显示切换
            display_area.toggle_boundary_display()
            print("✓ 边界显示切换成功")
            
            # 测试线框显示切换
            display_area.toggle_wireframe()
            print("✓ 线框显示切换成功")
            
            # 测试视图控制
            display_area.zoom_in()
            print("✓ 放大功能成功")
            
            display_area.zoom_out()
            print("✓ 缩小功能成功")
            
            display_area.reset_view()
            print("✓ 重置视图成功")
            
            display_area.fit_view()
            print("✓ 适应视图成功")
            
            # 设置定时器关闭窗口
            root.after(3000, root.destroy)
            # 运行主循环
            root.mainloop()
            return True
        else:
            print("✗ 网格显示失败")
            return False
            
    except Exception as e:
        print(f"✗ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始VTK网格显示功能验证...")
    success = test_vtk_functionality()
    if success:
        print("\n所有VTK网格显示功能验证成功！")
    else:
        print("\nVTK网格显示功能验证失败！")
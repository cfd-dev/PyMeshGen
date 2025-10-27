#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：VTK网格显示功能
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from data_structure.basic_elements import Unstructured_Grid, Triangle, NodeElement
from gui.mesh_display import MeshDisplayArea
import tkinter as tk


class TestVTKDisplay(unittest.TestCase):
    """VTK网格显示功能测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试网格
        self.nodes = [
            NodeElement([0.0, 0.0], 0),    # 0
            NodeElement([1.0, 0.0], 1),    # 1
            NodeElement([1.0, 1.0], 2),    # 2
            NodeElement([0.0, 1.0], 3),    # 3
            NodeElement([0.5, 0.5], 4),    # 4
        ]
        
        self.cells = [
            Triangle(self.nodes[0], self.nodes[1], self.nodes[2]),  # 三角形单元
            Triangle(self.nodes[0], self.nodes[2], self.nodes[3]),  # 三角形单元
            Triangle(self.nodes[0], self.nodes[1], self.nodes[4]),  # 三角形单元
            Triangle(self.nodes[1], self.nodes[2], self.nodes[4]),  # 三角形单元
            Triangle(self.nodes[2], self.nodes[3], self.nodes[4]),  # 三角形单元
            Triangle(self.nodes[3], self.nodes[0], self.nodes[4]),  # 三角形单元
        ]
        
        self.node_coords = [node.coords for node in self.nodes]
        self.boundary_nodes = [self.nodes[i] for i in [0, 1, 2, 3]]  # 外围节点
        
        self.unstr_grid = Unstructured_Grid(self.cells, self.node_coords, self.boundary_nodes)
        
        # 创建Tkinter窗口（不显示）
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏窗口
        
        # 创建网格显示区域
        self.display_area = MeshDisplayArea(self.root)
    
    def tearDown(self):
        """测试后的清理工作"""
        # 关闭Tkinter窗口
        self.root.destroy()
    
    def test_mesh_display(self):
        """测试网格显示功能"""
        # 显示网格
        result = self.display_area.display_mesh(self.unstr_grid)
        
        # 验证显示结果
        self.assertTrue(result, "网格显示应该成功")
    
    def test_boundary_display_toggle(self):
        """测试边界显示切换功能"""
        # 先显示网格
        self.display_area.display_mesh(self.unstr_grid)
        
        # 测试边界显示切换
        try:
            self.display_area.toggle_boundary_display()
            self.display_area.toggle_boundary_display()
            # 如果没有抛出异常，则测试通过
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"边界显示切换失败: {str(e)}")
    
    def test_wireframe_toggle(self):
        """测试线框显示切换功能"""
        # 先显示网格
        self.display_area.display_mesh(self.unstr_grid)
        
        # 测试线框显示切换
        try:
            self.display_area.toggle_wireframe()
            self.display_area.toggle_wireframe()
            # 如果没有抛出异常，则测试通过
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"线框显示切换失败: {str(e)}")
    
    def test_view_controls(self):
        """测试视图控制功能"""
        # 先显示网格
        self.display_area.display_mesh(self.unstr_grid)
        
        # 测试各种视图控制功能
        try:
            self.display_area.zoom_in()
            self.display_area.zoom_out()
            self.display_area.reset_view()
            self.display_area.fit_view()
            # 如果没有抛出异常，则测试通过
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"视图控制功能失败: {str(e)}")
    
    def test_clear_display(self):
        """测试清除显示功能"""
        # 先显示网格
        self.display_area.display_mesh(self.unstr_grid)
        
        # 测试清除显示
        try:
            self.display_area.clear_display()
            # 如果没有抛出异常，则测试通过
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"清除显示失败: {str(e)}")


if __name__ == "__main__":
    unittest.main()
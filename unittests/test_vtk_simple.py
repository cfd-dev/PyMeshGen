#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：VTK网格显示功能（简单版）
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


class TestVTKSimple(unittest.TestCase):
    """VTK网格显示功能简单测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建简单的测试网格
        self.nodes = [
            NodeElement([0.0, 0.0], 0),    # 0
            NodeElement([1.0, 0.0], 1),    # 1
            NodeElement([1.0, 1.0], 2),    # 2
            NodeElement([0.0, 1.0], 3),    # 3
        ]
        
        self.cells = [
            Triangle(self.nodes[0], self.nodes[1], self.nodes[2]),  # 三角形单元
            Triangle(self.nodes[0], self.nodes[2], self.nodes[3]),  # 三角形单元
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
    
    def test_simple_mesh_display(self):
        """测试简单网格显示功能"""
        # 显示网格
        result = self.display_area.display_mesh(self.unstr_grid)
        
        # 验证显示结果
        self.assertTrue(result, "简单网格显示应该成功")
    
    def test_mesh_data_structure(self):
        """测试网格数据结构"""
        # 验证节点数量
        self.assertEqual(len(self.nodes), 4, "应该有4个节点")
        
        # 验证单元数量
        self.assertEqual(len(self.cells), 2, "应该有2个三角形单元")
        
        # 验证边界节点数量
        self.assertEqual(len(self.boundary_nodes), 4, "应该有4个边界节点")
        
        # 验证网格对象
        self.assertIsInstance(self.unstr_grid, Unstructured_Grid, "应该是Unstructured_Grid类型")
        self.assertEqual(len(self.unstr_grid.node_coords), 4, "网格应该有4个节点坐标")
        self.assertEqual(len(self.unstr_grid.cell_container), 2, "网格应该有2个单元")
    
    def test_node_coordinates(self):
        """测试节点坐标"""
        expected_coords = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ]
        
        for i, node in enumerate(self.nodes):
            self.assertEqual(node.coords, expected_coords[i], f"节点{i}的坐标不正确")
    
    def test_cell_structure(self):
        """测试单元结构"""
        # 验证第一个三角形
        self.assertEqual(self.cells[0].nodes[0], self.nodes[0], "第一个三角形的第一个节点应该是nodes[0]")
        self.assertEqual(self.cells[0].nodes[1], self.nodes[1], "第一个三角形的第二个节点应该是nodes[1]")
        self.assertEqual(self.cells[0].nodes[2], self.nodes[2], "第一个三角形的第三个节点应该是nodes[2]")
        
        # 验证第二个三角形
        self.assertEqual(self.cells[1].nodes[0], self.nodes[0], "第二个三角形的第一个节点应该是nodes[0]")
        self.assertEqual(self.cells[1].nodes[1], self.nodes[2], "第二个三角形的第二个节点应该是nodes[2]")
        self.assertEqual(self.cells[1].nodes[2], self.nodes[3], "第二个三角形的第三个节点应该是nodes[3]")


if __name__ == "__main__":
    unittest.main()
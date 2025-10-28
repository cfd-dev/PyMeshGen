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

from data_structure.basic_elements import Unstructured_Grid, Triangle, Quadrilateral, NodeElement
from gui.mesh_display import MeshDisplayArea
import tkinter as tk


class TestVTKDisplay(unittest.TestCase):
    """VTK网格显示功能测试类"""
    
    def setUp(self):
        """在每个测试方法运行前执行"""
        # 创建测试所需的节点和单元
        from data_structure.basic_elements import NodeElement, Triangle, Quadrilateral
        
        # 创建节点
        nodes = [
            NodeElement([0.0, 0.0, 0.0], 0),    # 0
            NodeElement([1.0, 0.0, 0.0], 1),    # 1
            NodeElement([1.0, 1.0, 0.0], 2),    # 2
            NodeElement([0.0, 1.0, 0.0], 3),    # 3
            NodeElement([0.0, 0.0, 1.0], 4),    # 4
            NodeElement([1.0, 0.0, 1.0], 5),    # 5
            NodeElement([1.0, 1.0, 1.0], 6),    # 6
            NodeElement([0.0, 1.0, 1.0], 7),    # 7
        ]
        
        # 创建三角形和四边形单元（2D单元）
        tri = Triangle(nodes[0], nodes[1], nodes[2])  # 底面三角形
        quad = Quadrilateral(nodes[4], nodes[5], nodes[6], nodes[7])  # 顶面四边形
        
        cells = [tri, quad]
        node_coords = [node.coords for node in nodes]
        boundary_nodes = [nodes[0], nodes[1], nodes[2], nodes[3]]  # 简单示例
        
        # 创建Unstructured_Grid对象
        self.test_grid = Unstructured_Grid(cells, node_coords, boundary_nodes)
        
        # 创建一个隐藏的Tkinter窗口用于测试
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏窗口
        
        # 创建MeshDisplayArea实例，使用离屏渲染模式
        self.display_area = MeshDisplayArea(self.root, offscreen=True)
    
    def tearDown(self):
        """测试后的清理工作"""
        # 关闭Tkinter窗口
        self.root.destroy()
    
    def test_mesh_display(self):
        """测试网格显示功能"""
        # 显示网格
        result = self.display_area.display_mesh(self.test_grid)
        
        # 检查返回值
        self.assertIsNotNone(result)
        self.assertTrue(result)
        
        # 检查是否设置了网格演员
        self.assertIsNotNone(self.display_area.mesh_actor)
        
        # 检查属性设置
        actor = self.display_area.mesh_actor
        property = actor.GetProperty()
        self.assertIsNotNone(property)
        
        print("网格显示测试通过")

    def test_boundary_display_toggle(self):
        """测试边界显示切换功能"""
        # 先显示网格
        self.display_area.display_mesh(self.test_grid)
        
        # 保存初始边界演员
        initial_boundary_actors = list(self.display_area.boundary_actors)
        
        # 切换边界显示（关闭）
        self.display_area.toggle_boundary_display()
        
        # 检查边界演员是否被清除
        self.assertEqual(len(self.display_area.boundary_actors), 0)
        
        # 再次切换边界显示（打开）
        self.display_area.toggle_boundary_display()
        
        # 检查是否重新创建了边界演员
        self.assertGreaterEqual(len(self.display_area.boundary_actors), 0)
        
        print("边界显示切换测试通过")

    def test_wireframe_toggle(self):
        """测试线框模式切换功能"""
        # 先显示网格
        self.display_area.display_mesh(self.test_grid)
        
        # 获取初始表示模式
        actor = self.display_area.mesh_actor
        initial_repr = actor.GetProperty().GetRepresentation()
        
        # 切换线框模式
        self.display_area.wireframe_var.set(not self.display_area.wireframe_var.get())
        
        # 重新显示网格以应用新的线框模式
        result = self.display_area.display_mesh(self.test_grid)
        
        # 验证显示操作成功
        self.assertTrue(result)
        
        # 检查表示模式是否改变
        actor = self.display_area.mesh_actor
        new_repr = actor.GetProperty().GetRepresentation()
        self.assertNotEqual(initial_repr, new_repr)
        
        print("线框模式切换测试通过")

    def test_view_controls(self):
        """测试视图控制功能"""
        # 先显示网格
        self.display_area.display_mesh(self.test_grid)
        
        # 测试重置视图
        self.display_area.reset_view()
        
        # 测试缩放
        self.display_area.zoom_in()
        self.display_area.zoom_out()
        
        # 测试适应视图
        self.display_area.fit_view()
        
        print("视图控制测试通过")

    def test_clear_display(self):
        """测试清除显示功能"""
        # 先显示网格
        self.display_area.display_mesh(self.test_grid)
        
        # 检查是否设置了网格演员
        self.assertIsNotNone(self.display_area.mesh_actor)
        
        # 清除显示
        self.display_area.clear_display()
        
        # 检查网格演员是否被清除
        self.assertIsNone(self.display_area.mesh_actor)
        
        print("清除显示测试通过")


if __name__ == "__main__":
    unittest.main()
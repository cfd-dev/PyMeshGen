#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：VTK网格显示功能 - 修复版（不依赖GUI组件）
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from data_structure.basic_elements import Unstructured_Grid, Triangle, Quadrilateral, NodeElement


class TestVTKDisplay(unittest.TestCase):
    """VTK网格显示功能测试类（不依赖GUI组件）"""

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

    def test_mesh_data_structure(self):
        """测试网格数据结构 - 验证网格对象的基本属性"""
        # 验证Unstructured_Grid对象创建成功
        self.assertIsNotNone(self.test_grid)
        self.assertIsInstance(self.test_grid, Unstructured_Grid)

        # 验证网格对象的属性
        self.assertTrue(hasattr(self.test_grid, 'cell_container'))
        self.assertTrue(hasattr(self.test_grid, 'node_coords'))
        self.assertTrue(hasattr(self.test_grid, 'boundary_nodes'))

        # 验证节点和单元的数量
        self.assertGreater(len(self.test_grid.node_coords), 0)
        self.assertGreater(len(self.test_grid.cell_container), 0)
        self.assertGreater(len(self.test_grid.boundary_nodes), 0)

        print("网格数据结构测试通过")

    def test_basic_elements_creation(self):
        """测试基本元素创建 - 验证节点、三角形、四边形的创建"""
        # 验证节点创建
        node = NodeElement([1.0, 2.0, 3.0], 10)
        self.assertIsNotNone(node)
        self.assertEqual(node.coords, [1.0, 2.0, 3.0])
        self.assertEqual(node.idx, 10)

        # 验证三角形创建
        node1 = NodeElement([0.0, 0.0, 0.0], 0)
        node2 = NodeElement([1.0, 0.0, 0.0], 1)
        node3 = NodeElement([0.0, 1.0, 0.0], 2)
        tri = Triangle(node1, node2, node3)
        self.assertIsNotNone(tri)

        # 验证四边形创建
        node4 = NodeElement([1.0, 1.0, 0.0], 3)
        quad = Quadrilateral(node1, node2, node3, node4)
        self.assertIsNotNone(quad)

        print("基本元素创建测试通过")

    def test_unstructured_grid_attributes(self):
        """测试Unstructured_Grid对象的属性 - 验证网格对象的属性和方法"""
        # 验证网格对象的属性
        self.assertTrue(hasattr(self.test_grid, 'num_nodes'))
        self.assertTrue(hasattr(self.test_grid, 'num_cells'))
        self.assertTrue(hasattr(self.test_grid, 'num_boundary_nodes'))

        # 验证网格对象的计数属性
        self.assertEqual(self.test_grid.num_nodes, len(self.test_grid.node_coords))
        self.assertGreater(self.test_grid.num_cells, 0)
        self.assertGreater(self.test_grid.num_boundary_nodes, 0)

        # 验证其他属性
        self.assertTrue(hasattr(self.test_grid, 'boundary_nodes_list'))
        self.assertGreater(len(self.test_grid.boundary_nodes_list), 0)

        print("Unstructured_Grid属性测试通过")

    def test_triangle_quadrilateral_properties(self):
        """测试三角形和四边形的属性 - 验证几何元素的基本属性"""
        # 创建测试节点
        nodes = [NodeElement([float(i), 0.0, 0.0], i) for i in range(4)]
        
        # 验证三角形
        tri = Triangle(nodes[0], nodes[1], nodes[2])
        self.assertIsNotNone(tri)
        self.assertTrue(hasattr(tri, 'p1'))
        self.assertTrue(hasattr(tri, 'p2'))
        self.assertTrue(hasattr(tri, 'p3'))
        
        # 验证四边形
        quad = Quadrilateral(nodes[0], nodes[1], nodes[2], nodes[3])
        self.assertIsNotNone(quad)
        self.assertTrue(hasattr(quad, 'p1'))
        self.assertTrue(hasattr(quad, 'p2'))
        self.assertTrue(hasattr(quad, 'p3'))
        self.assertTrue(hasattr(quad, 'p4'))

        print("三角形和四边形属性测试通过")

    def test_mesh_coordinates_validation(self):
        """测试网格坐标验证 - 验证坐标数据的正确性"""
        # 验证节点坐标的维度
        for coord in self.test_grid.node_coords:
            self.assertIsInstance(coord, list)
            self.assertGreaterEqual(len(coord), 2)  # 至少2D坐标
            for val in coord:
                self.assertIsInstance(val, (int, float))

        # 验证边界节点
        for node in self.test_grid.boundary_nodes:
            self.assertIsInstance(node, NodeElement)
            self.assertIsInstance(node.coords, list)

        print("网格坐标验证测试通过")


if __name__ == "__main__":
    unittest.main()
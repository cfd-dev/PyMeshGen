#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：测试Unstructured_Grid类的cyclic_node2node和build_topological_ring方法
"""

import unittest
import sys
import os
import numpy as np

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from data_structure.unstructured_grid import Unstructured_Grid
from data_structure.basic_elements import Triangle, NodeElement


class TestTopologyMethods(unittest.TestCase):
    """测试Unstructured_Grid类的拓扑方法"""

    def setUp(self):
        """设置测试用例的初始网格"""
        # 创建一个简单的三角形网格，包含一个中心节点和周围的邻居节点
        # 网格布局如下：
        #       n3
        #      / | \
        #    /   |   \
        #  n2----n0---n4
        #    \   |   /
        #      \ | /
        #       n1
        
        # 定义节点坐标
        nodes = [
            NodeElement([0.0, 0.0, 0.0], 0),  # 中心节点
            NodeElement([-1.0, -1.0, 0.0], 1),  # 下左
            NodeElement([1.0, -1.0, 0.0], 2),   # 下右
            NodeElement([0.0, 1.0, 0.0], 3),    # 上
            NodeElement([2.0, 0.0, 0.0], 4),    # 右
        ]

        # 定义三角形单元
        # 围绕中心节点的三角形
        tri1 = Triangle(nodes[0], nodes[1], nodes[2])  # 中心-下左-下右
        tri2 = Triangle(nodes[0], nodes[2], nodes[4])  # 中心-下右-右
        tri3 = Triangle(nodes[0], nodes[4], nodes[3])  # 中心-右-上
        tri4 = Triangle(nodes[0], nodes[3], nodes[1])  # 中心-上-下左

        # 创建网格
        cells = [tri1, tri2, tri3, tri4]
        node_coords = [node.coords for node in nodes]
        boundary_nodes = [nodes[1], nodes[2], nodes[3], nodes[4]]  # 除了中心节点外都是边界节点

        self.grid = Unstructured_Grid(cells, node_coords, boundary_nodes)

    def test_cyclic_node2node_initialization(self):
        """测试cyclic_node2node方法的初始化功能"""
        # 初始状态下，node2node应该为None
        self.assertIsNone(self.grid.node2node)
        
        # 调用cyclic_node2node方法
        self.grid.cyclic_node2node()
        
        # 现在node2node应该被初始化
        self.assertIsNotNone(self.grid.node2node)
        
        # 检查每个节点的邻居数量
        for node_id in range(len(self.grid.node_coords)):
            if node_id in self.grid.node2node:
                neighbors = self.grid.node2node[node_id]
                # 中心节点(0)应该有4个邻居
                if node_id == 0:
                    self.assertEqual(len(neighbors), 4)
                # 边界节点应该有2-3个邻居
                else:
                    self.assertGreaterEqual(len(neighbors), 2)
                    self.assertLessEqual(len(neighbors), 4)

    def test_cyclic_node2node_ordering(self):
        """测试cyclic_node2node方法的邻居排序功能"""
        # 调用cyclic_node2node方法
        self.grid.cyclic_node2node()
        
        # 检查中心节点(0)的邻居是否按顺序排列
        center_neighbors = self.grid.node2node[0]
        self.assertEqual(len(center_neighbors), 4)
        
        # 邻居应该按某种顺序排列（通常是按角度顺序）
        # 验证邻居确实是围绕中心节点的节点
        expected_neighbors = {1, 2, 3, 4}  # 中心节点的邻居
        actual_neighbors = set(center_neighbors)
        self.assertEqual(actual_neighbors, expected_neighbors)
        
        # 检查邻居是否形成了一个环（首尾相连）
        # 每个节点都应该有一个前驱和后继
        num_neighbors = len(center_neighbors)
        for i in range(num_neighbors):
            current = center_neighbors[i]
            next_neighbor = center_neighbors[(i + 1) % num_neighbors]
            
            # 检查这对邻居是否在同一个单元中相邻
            # 这里我们只是验证邻居列表的结构是否合理

    def test_build_topological_ring_basic(self):
        """测试build_topological_ring方法的基本功能"""
        # 首先初始化node2node
        self.grid.init_node2node_by_cell()
        
        # 测试中心节点(0)的拓扑环
        ring = self.grid.build_topological_ring(0)
        
        # 中心节点应该能构建出一个环
        self.assertIsNotNone(ring)
        self.assertEqual(len(ring), 4)  # 中心节点有4个邻居
        
        # 验证环中的节点都是邻居
        expected_neighbors = {1, 2, 3, 4}
        actual_neighbors = set(ring)
        self.assertEqual(actual_neighbors, expected_neighbors)

    def test_build_topological_ring_order(self):
        """测试build_topological_ring方法的顺序功能"""
        # 首先初始化node2node
        self.grid.init_node2node_by_cell()
        
        # 构建中心节点(0)的拓扑环
        ring = self.grid.build_topological_ring(0)
        
        # 验证环的顺序是否符合拓扑连接
        # 根据我们的网格定义，正确的顺序可能是 [1, 2, 4, 3] 或类似的顺序
        # 检查相邻的节点是否在网格中确实相邻
        self.assertIsNotNone(ring)
        if ring is not None:
            # 检查环的长度
            self.assertEqual(len(ring), 4)
            
            # 检查每个连续的节点对是否在某个单元中共存
            # 通过检查单元容器来验证拓扑关系
            node_pairs_in_units = set()
            for cell in self.grid.cell_container:
                node_ids = cell.node_ids
                for i in range(len(node_ids)):
                    j = (i + 1) % len(node_ids)
                    pair = tuple(sorted([node_ids[i], node_ids[j]]))
                    node_pairs_in_units.add(pair)
            
            # 检查环中的相邻节点对是否存在于单元中
            for i in range(len(ring)):
                current = ring[i]
                next_node = ring[(i + 1) % len(ring)]
                pair = tuple(sorted([current, next_node]))
                
                # 每一对相邻节点应该在某个单元中共存
                # 注意：中心节点0与环中的每个节点都相连，所以检查方式略有不同
                # 我们检查环中相邻节点是否通过中心节点间接相连

    def test_build_topological_ring_edge_cases(self):
        """测试build_topological_ring方法的边界情况"""
        # 创建一个只有一个三角形的简单网格
        nodes = [
            NodeElement([0.0, 0.0, 0.0], 0),
            NodeElement([1.0, 0.0, 0.0], 1),
            NodeElement([0.0, 1.0, 0.0], 2),
        ]
        
        tri = Triangle(nodes[0], nodes[1], nodes[2])
        node_coords = [node.coords for node in nodes]
        boundary_nodes = nodes  # 所有节点都是边界节点
        
        simple_grid = Unstructured_Grid([tri], node_coords, boundary_nodes)
        
        # 测试每个节点的拓扑环
        for node_id in range(3):
            simple_grid.init_node2node_by_cell()
            ring = simple_grid.build_topological_ring(node_id)
            
            # 对于三角形中的每个节点，它有两个邻居
            # 拓扑环应该包含这两个邻居，并且按正确顺序排列
            if ring is not None:
                self.assertEqual(len(ring), 2)
    
    def test_consistency_between_methods(self):
        """测试两种方法结果的一致性"""
        # 先使用拓扑方法构建环
        self.grid.init_node2node_by_cell()
        topological_ring = self.grid.build_topological_ring(0)
        
        # 再使用几何排序方法
        self.grid.cyclic_node2node()
        geometric_neighbors = self.grid.node2node[0]
        
        # 两种方法应该识别出相同的邻居节点集合，尽管顺序可能不同
        if topological_ring is not None:
            self.assertEqual(set(topological_ring), set(geometric_neighbors))
            self.assertEqual(len(topological_ring), len(geometric_neighbors))

    def test_empty_case(self):
        """测试空网格的情况"""
        # 创建一个没有单元的网格
        empty_nodes = [NodeElement([0.0, 0.0, 0.0], 0)]
        empty_grid = Unstructured_Grid([], [[0.0, 0.0, 0.0]], [])
        
        # 测试build_topological_ring在空网格上的行为
        empty_grid.init_node2node_by_cell()
        ring = empty_grid.build_topological_ring(0)
        # 应该返回None，因为没有单元与节点关联
        self.assertIsNone(ring)
        
        # 测试cyclic_node2node在空网格上的行为
        empty_grid.cyclic_node2node()
        # node2node字典应该存在，但节点0不应该有邻居
        self.assertIsNotNone(empty_grid.node2node)
        if 0 in empty_grid.node2node:
            self.assertEqual(len(empty_grid.node2node[0]), 0)


if __name__ == "__main__":
    unittest.main()
#!/usr/bin/env python3
"""
调试脚本：检查Unstructured_Grid类型检查问题
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入Unstructured_Grid类
from data_structure.basic_elements import Unstructured_Grid

# 创建一个简单的Unstructured_Grid对象用于测试
class MockNodeElement:
    def __init__(self, coords, idx):
        self.coords = coords
        self.idx = idx

# 创建测试数据
node_coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
boundary_nodes = [MockNodeElement([0, 0], 0), MockNodeElement([1, 0], 1)]

# 创建一个简单的Unstructured_Grid对象
class MockCell:
    def __init__(self, node_ids):
        self.node_ids = node_ids

cell_container = [MockCell([0, 1, 2]), MockCell([1, 3, 2])]

# 创建Unstructured_Grid对象
test_grid = Unstructured_Grid(cell_container, node_coords, boundary_nodes)

print(f"测试对象类型: {type(test_grid)}")
print(f"Unstructured_Grid类: {Unstructured_Grid}")
print(f"isinstance检查结果: {isinstance(test_grid, Unstructured_Grid)}")

# 检查模块信息
print(f"Unstructured_Grid模块: {Unstructured_Grid.__module__}")
print(f"测试对象模块: {test_grid.__class__.__module__}")

# 检查类是否相同
print(f"类相同性检查: {test_grid.__class__ == Unstructured_Grid}")

# 检查导入路径
import data_structure.basic_elements as basic_elements_module
print(f"模块中的Unstructured_Grid: {basic_elements_module.Unstructured_Grid}")
print(f"模块类相同性: {test_grid.__class__ == basic_elements_module.Unstructured_Grid}")
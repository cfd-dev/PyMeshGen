#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import unittest
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from data_structure.basic_elements import Unstructured_Grid
    # 直接从文件导入，而不是通过包
    from visualization.mesh_visualization import visualize_mesh_2d, visualize_unstr_grid_2d
    import matplotlib.pyplot as plt
    
    class TestVisualization(unittest.TestCase):
        """测试可视化模块的功能"""
        
        def setUp(self):
            """测试前的准备工作"""
            # 创建测试用的网格数据
            self.test_nodes = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0]
            ]
            
            self.test_elements = [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4]
            ]
            
            # 创建简化版的Unstructured_Grid对象（使用空列表作为占位符）
            self.grid = Unstructured_Grid([], self.test_nodes, [])
            
            # 创建字典格式的网格数据
            self.grid_dict = {
                "nodes": self.test_nodes,
                "zones": {
                    "zone_1": {
                        "type": "faces",
                        "bc_type": "wall",
                        "data": [
                            {"nodes": [0, 1]},  # 使用0-based索引
                            {"nodes": [1, 2]},
                            {"nodes": [2, 3]},
                            {"nodes": [3, 0]}
                        ]
                    }
                }
            }
        
        def test_unstructured_grid_visualization(self):
            """测试Unstructured_Grid对象的可视化"""
            # 验证网格对象创建成功
            self.assertIsNotNone(self.grid)
            self.assertEqual(len(self.grid.node_coords), 5)
            
            # 测试visualize_unstr_grid_2d函数处理Unstructured_Grid对象
            try:
                # 创建一个临时的图形对象用于测试
                fig, ax = plt.subplots()
                visualize_unstr_grid_2d(self.grid, ax)
                plt.close(fig)  # 关闭图形以释放资源
                # 如果没有抛出异常，则认为测试通过
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"visualize_unstr_grid_2d函数处理Unstructured_Grid对象时出现错误: {e}")
        
        def test_dict_grid_visualization(self):
            """测试字典格式网格数据的可视化"""
            # 验证网格字典创建成功
            self.assertIsNotNone(self.grid_dict)
            self.assertIn('nodes', self.grid_dict)
            self.assertIn('zones', self.grid_dict)
            
            # 测试visualize_mesh_2d函数处理字典格式网格数据
            try:
                # 创建一个临时的图形对象用于测试
                fig, ax = plt.subplots()
                visualize_mesh_2d(self.grid_dict, ax)
                plt.close(fig)  # 关闭图形以释放资源
                # 如果没有抛出异常，则认为测试通过
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"visualize_mesh_2d函数处理字典格式网格数据时出现错误: {e}")

    if __name__ == "__main__":
        unittest.main()

except ImportError as e:
    print(f"导入模块失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"测试过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
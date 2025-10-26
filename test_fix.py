#!/usr/bin/env python3
"""测试修复后的mesh_visualization.py"""

import sys
sys.path.append('.')

from data_structure.basic_elements import Unstructured_Grid, Triangle, NodeElement
from visualization.mesh_visualization import visualize_mesh_2d
import numpy as np

def test_unstructured_grid():
    """测试Unstructured_Grid对象的可视化"""
    print("测试Unstructured_Grid对象的可视化...")
    
    # 创建一个简单的Unstructured_Grid对象
    node_coords = [
        [0.0, 0.0],
        [1.0, 0.0], 
        [1.0, 1.0],
        [0.0, 1.0]
    ]
    
    # 创建三角形单元
    cell_container = [
        Triangle([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], node_ids=[0, 1, 2]),
        Triangle([0.0, 0.0], [1.0, 1.0], [0.0, 1.0], node_ids=[0, 2, 3])
    ]
    
    # 创建边界节点
    boundary_nodes = [
        NodeElement([0.0, 0.0], 0, bc_type="wall"),
        NodeElement([1.0, 0.0], 1, bc_type="wall"),
        NodeElement([1.0, 1.0], 2, bc_type="wall"),
        NodeElement([0.0, 1.0], 3, bc_type="wall")
    ]
    
    # 创建Unstructured_Grid对象
    grid = Unstructured_Grid(cell_container, node_coords, boundary_nodes)
    
    try:
        # 测试visualize_mesh_2d函数
        visualize_mesh_2d(grid)
        print("✓ Unstructured_Grid对象可视化测试通过")
        return True
    except Exception as e:
        print(f"✗ Unstructured_Grid对象可视化测试失败: {e}")
        return False

def test_dict_grid():
    """测试字典格式网格数据的可视化"""
    print("测试字典格式网格数据的可视化...")
    
    # 创建一个简单的字典格式网格数据
    grid_dict = {
        "nodes": [
            [0.0, 0.0],
            [1.0, 0.0], 
            [1.0, 1.0],
            [0.0, 1.0]
        ],
        "zones": {
            "zone_1": {
                "type": "faces",
                "bc_type": "wall",
                "data": [
                    {"nodes": [1, 2]},
                    {"nodes": [2, 3]},
                    {"nodes": [3, 4]},
                    {"nodes": [4, 1]}
                ]
            }
        }
    }
    
    try:
        # 测试visualize_mesh_2d函数
        visualize_mesh_2d(grid_dict)
        print("✓ 字典格式网格数据可视化测试通过")
        return True
    except Exception as e:
        print(f"✗ 字典格式网格数据可视化测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试mesh_visualization.py修复...")
    
    # 运行测试
    test1_passed = test_unstructured_grid()
    test2_passed = test_dict_grid()
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！修复成功！")
    else:
        print("\n❌ 部分测试失败，需要进一步调试")
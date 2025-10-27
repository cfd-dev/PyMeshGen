#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：CAS文件导入功能
"""

import unittest
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fileIO.read_cas import parse_cas_to_unstr_grid, parse_fluent_msh, reconstruct_mesh_from_cas
from data_structure.basic_elements import Unstructured_Grid


class TestCASImport(unittest.TestCase):
    """CAS文件导入功能测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 获取测试文件路径
        self.test_files = [
            "config/input/quad.cas",
            "config/input/concave.cas",
            "config/input/convex.cas"
        ]
        
        # 过滤存在的测试文件
        self.existing_files = [f for f in self.test_files if os.path.exists(f)]
    
    def test_parse_cas_to_unstr_grid(self):
        """测试parse_cas_to_unstr_grid函数"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                with self.subTest(file=test_file):
                    # 使用parse_cas_to_unstr_grid直接转换
                    unstr_grid = parse_cas_to_unstr_grid(test_file)
                    
                    # 验证基本属性
                    self.assertIsInstance(unstr_grid, Unstructured_Grid)
                    self.assertGreater(len(unstr_grid.node_coords), 0)
                    self.assertGreater(len(unstr_grid.cell_container), 0)
                    
                    # 验证节点坐标维度
                    for i in range(min(3, len(unstr_grid.node_coords))):
                        coords = unstr_grid.node_coords[i]
                        self.assertEqual(len(coords), 3, f"节点{i}坐标应该是3D的")
    
    def test_parse_fluent_msh_and_reconstruct(self):
        """测试parse_fluent_msh和reconstruct_mesh_from_cas函数"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                # 分步解析
                cas_data = parse_fluent_msh(test_file)
                unstr_grid = reconstruct_mesh_from_cas(cas_data)
                
                # 验证基本属性
                self.assertIsInstance(unstr_grid, Unstructured_Grid)
                self.assertGreater(len(unstr_grid.node_coords), 0)
                self.assertGreater(len(unstr_grid.cell_container), 0)
    
    def test_methods_consistency(self):
        """测试两种导入方法的结果一致性"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                # 方法1: 直接转换
                unstr_grid1 = parse_cas_to_unstr_grid(test_file)
                
                # 方法2: 分步解析
                cas_data = parse_fluent_msh(test_file)
                unstr_grid2 = reconstruct_mesh_from_cas(cas_data)
                
                # 验证结果一致性
                self.assertEqual(len(unstr_grid1.node_coords), len(unstr_grid2.node_coords))
                self.assertEqual(len(unstr_grid1.cell_container), len(unstr_grid2.cell_container))
                self.assertEqual(len(unstr_grid1.boundary_nodes), len(unstr_grid2.boundary_nodes))
    
    def test_vtk_conversion(self):
        """测试VTK文件转换功能"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                # 导入CAS文件
                unstr_grid = parse_cas_to_unstr_grid(test_file)
                
                # 转换为VTK文件
                vtk_file = test_file.replace('.cas', '_test.vtk')
                unstr_grid.save_to_vtkfile(vtk_file)
                
                # 验证VTK文件是否创建
                self.assertTrue(os.path.exists(vtk_file))
                
                # 从VTK文件加载
                from fileIO.vtk_io import parse_vtk_msh
                loaded_grid = parse_vtk_msh(vtk_file)
                
                # 验证加载的网格
                self.assertIsInstance(loaded_grid, Unstructured_Grid)
                self.assertEqual(len(unstr_grid.node_coords), len(loaded_grid.node_coords))
                self.assertEqual(len(unstr_grid.cell_container), len(loaded_grid.cell_container))
                
                # 清理测试文件
                if os.path.exists(vtk_file):
                    os.remove(vtk_file)


if __name__ == "__main__":
    unittest.main()
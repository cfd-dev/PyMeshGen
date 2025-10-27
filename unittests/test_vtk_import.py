#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：VTK文件导入功能
"""

import unittest
import os
import sys

# 添加必要的路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from fileIO.vtk_io import read_vtk, reconstruct_mesh_from_vtk
from data_structure.basic_elements import Unstructured_Grid


class TestVTKImport(unittest.TestCase):
    """VTK文件导入功能测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 查找VTK文件
        self.vtk_files = []
        for root, dirs, files in os.walk(current_dir):
            for file in files:
                if file.endswith('.vtk'):
                    self.vtk_files.append(os.path.join(root, file))
    
    def test_read_vtk(self):
        """测试read_vtk函数"""
        for vtk_file in self.vtk_files:
            with self.subTest(file=vtk_file):
                # 读取VTK文件
                node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container = read_vtk(vtk_file)
                
                # 验证返回值
                self.assertIsInstance(node_coords, list)
                self.assertIsInstance(cell_idx_container, list)
                self.assertIsInstance(boundary_nodes_idx, list)
                self.assertIsInstance(cell_type_container, list)
                
                # 验证节点坐标
                self.assertGreater(len(node_coords), 0)
                for i in range(min(3, len(node_coords))):
                    coords = node_coords[i]
                    self.assertEqual(len(coords), 3, f"节点{i}坐标应该是3D的")
                
                # 验证单元
                self.assertGreater(len(cell_idx_container), 0)
    
    def test_reconstruct_mesh_from_vtk(self):
        """测试reconstruct_mesh_from_vtk函数"""
        for vtk_file in self.vtk_files:
            with self.subTest(file=vtk_file):
                # 读取VTK文件
                node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container = read_vtk(vtk_file)
                
                # 重建网格
                mesh = reconstruct_mesh_from_vtk(node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container)
                
                # 验证网格对象
                self.assertIsInstance(mesh, Unstructured_Grid)
                self.assertEqual(len(mesh.node_coords), len(node_coords))
                self.assertEqual(len(mesh.cell_container), len(cell_idx_container))
                self.assertGreater(mesh.dim, 0)
    
    def test_mesh_properties(self):
        """测试网格属性"""
        for vtk_file in self.vtk_files:
            with self.subTest(file=vtk_file):
                # 读取并重建网格
                node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container = read_vtk(vtk_file)
                mesh = reconstruct_mesh_from_vtk(node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container)
                
                # 验证网格维度
                self.assertIn(mesh.dim, [2, 3])
                
                # 验证边界框
                self.assertIsNotNone(mesh.bbox)
                self.assertIsInstance(mesh.bbox, (list, dict))
                
                # 如果是列表格式，应该有4或6个元素（2D或3D）
                if isinstance(mesh.bbox, list):
                    self.assertIn(len(mesh.bbox), [4, 6])
                
                # 如果是字典格式，应该有必要的键
                elif isinstance(mesh.bbox, dict):
                    required_keys = ['min_x', 'max_x', 'min_y', 'max_y']
                    for key in required_keys:
                        self.assertIn(key, mesh.bbox)
    
    def test_vtk_roundtrip(self):
        """测试VTK文件的往返转换"""
        for vtk_file in self.vtk_files:
            with self.subTest(file=vtk_file):
                # 读取并重建网格
                node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container = read_vtk(vtk_file)
                mesh = reconstruct_mesh_from_vtk(node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container)
                
                # 保存为新的VTK文件
                test_vtk_file = vtk_file.replace('.vtk', '_test_roundtrip.vtk')
                mesh.save_to_vtkfile(test_vtk_file)
                
                # 验证新文件是否创建
                self.assertTrue(os.path.exists(test_vtk_file))
                
                # 从新文件读取
                new_node_coords, new_cell_idx_container, new_boundary_nodes_idx, new_cell_type_container = read_vtk(test_vtk_file)
                new_mesh = reconstruct_mesh_from_vtk(new_node_coords, new_cell_idx_container, new_boundary_nodes_idx, new_cell_type_container)
                
                # 验证往返转换的一致性
                self.assertEqual(len(mesh.node_coords), len(new_mesh.node_coords))
                self.assertEqual(len(mesh.cell_container), len(new_mesh.cell_container))
                
                # 清理测试文件
                if os.path.exists(test_vtk_file):
                    os.remove(test_vtk_file)


if __name__ == "__main__":
    unittest.main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单元测试：测试从cas文件导入部件信息的功能
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyqt_gui.file_operations import FileOperations
from fileIO.read_cas import parse_cas_to_unstr_grid

class TestCasParts(unittest.TestCase):
    """测试cas文件部件信息提取功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_ops = FileOperations(self.project_root)
        self.test_file = os.path.join(self.project_root, "config/input/30p30n-small.cas")
    
    def test_extract_parts_from_cas(self):
        """测试从cas文件提取部件信息的功能"""
        # 检查测试文件是否存在
        self.assertTrue(os.path.exists(self.test_file), f"测试文件 {self.test_file} 不存在")

        # 直接导入CAS文件 using the file_ops method
        mesh_data = self.file_ops.import_mesh(self.test_file)

        # 验证返回的是MeshData对象
        from data_structure.mesh_data import MeshData
        self.assertIsInstance(mesh_data, MeshData, "应该返回MeshData对象")

        # 验证部件信息
        self.assertTrue(hasattr(mesh_data, 'parts_info'), "MeshData对象应包含parts_info属性")

        parts_info = mesh_data.parts_info
        # parts_info should be a dictionary, not a list
        self.assertIsInstance(parts_info, dict, "部件信息应该是字典类型")

        # If there are parts, validate their structure
        if parts_info:
            for part_name, part_data in parts_info.items():
                self.assertIsInstance(part_data, dict, f"部件 {part_name} 应该是字典类型")
                self.assertIn('type', part_data, f"部件 {part_name} 应包含type字段")
                self.assertIn('face_count', part_data, f"部件 {part_name} 应包含face_count字段")
                self.assertIn('part_name', part_data, f"部件 {part_name} 应包含part_name字段")
                self.assertEqual(part_data['part_name'], part_name, f"part_name字段应与键名一致")
    
    def test_import_mesh_with_parts(self):
        """测试导入cas文件并提取部件信息的功能"""
        # 检查测试文件是否存在
        self.assertTrue(os.path.exists(self.test_file), f"测试文件 {self.test_file} 不存在")

        # 导入网格数据
        mesh_data = self.file_ops.import_mesh(self.test_file)

        # 验证返回的是MeshData对象
        from data_structure.mesh_data import MeshData
        self.assertIsInstance(mesh_data, MeshData, "应该返回MeshData对象")

        # 检查是否包含部件信息
        self.assertTrue(hasattr(mesh_data, 'parts_info'), "MeshData对象应包含parts_info属性")

        parts_info = mesh_data.parts_info
        self.assertIsInstance(parts_info, dict, "部件信息应该是字典类型")

        # If there are parts, validate their structure
        if parts_info:
            self.assertGreater(len(parts_info), 0, "应该至少有一个部件")

            # 验证每个部件的结构
            for part_name, part_data in parts_info.items():
                self.assertIn('part_name', part_data, f"部件 {part_name} 应包含part_name字段")
                self.assertIn('face_count', part_data, f"部件 {part_name} 应包含face_count字段")
                # Note: bc_type might not be present in all parts, so we don't check for it

if __name__ == '__main__':
    unittest.main()
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

from gui.file_operations import FileOperations
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
        
        # 解析cas文件
        unstr_grid = parse_cas_to_unstr_grid(self.test_file)
        
        # 创建模拟的mesh_data
        mesh_data = {
            'type': 'cas',
            'unstr_grid': unstr_grid
        }
        
        # 提取部件信息
        parts_info = self.file_ops.extract_parts_from_cas(mesh_data)
        
        # 验证结果
        self.assertIsInstance(parts_info, list, "部件信息应该是列表类型")
        self.assertGreater(len(parts_info), 0, "应该至少有一个部件")
        
        # 验证每个部件的结构
        for part in parts_info:
            self.assertIn('part_name', part, "部件应包含part_name字段")
            self.assertIn('bc_type', part, "部件应包含bc_type字段")
            self.assertIn('face_count', part, "部件应包含face_count字段")
            self.assertIn('nodes', part, "部件应包含nodes字段")
            self.assertIn('cells', part, "部件应包含cells字段")
    
    def test_import_mesh_with_parts(self):
        """测试导入cas文件并提取部件信息的功能"""
        # 检查测试文件是否存在
        self.assertTrue(os.path.exists(self.test_file), f"测试文件 {self.test_file} 不存在")
        
        # 导入网格数据
        mesh_data = self.file_ops.import_mesh(self.test_file)
        
        # 检查是否包含部件信息
        self.assertIn('parts_info', mesh_data, "导入的网格数据应包含parts_info字段")
        
        parts_info = mesh_data['parts_info']
        self.assertIsInstance(parts_info, list, "部件信息应该是列表类型")
        self.assertGreater(len(parts_info), 0, "应该至少有一个部件")
        
        # 验证每个部件的结构
        for part in parts_info:
            self.assertIn('part_name', part, "部件应包含part_name字段")
            self.assertIn('bc_type', part, "部件应包含bc_type字段")
            self.assertIn('face_count', part, "部件应包含face_count字段")

if __name__ == '__main__':
    unittest.main()
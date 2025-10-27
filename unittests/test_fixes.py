#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：验证修复后的功能
"""

import unittest
import os
import sys

# 添加路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(current_dir, 'fileIO'))
sys.path.append(current_dir)


class TestFixes(unittest.TestCase):
    """修复功能测试类"""
    
    def test_vtk_import(self):
        """测试VTK文件导入功能"""
        try:
            # 尝试导入read_vtk函数
            from vtk_io import read_vtk
            self.assertTrue(True, "VTK文件读取器导入成功")
        except ImportError as e:
            self.fail(f"VTK文件读取器导入失败: {e}")
    
    def test_cas_import(self):
        """测试CAS文件导入功能"""
        try:
            from fileIO.read_cas import parse_cas_to_unstr_grid
            self.assertTrue(True, "CAS文件读取器导入成功")
        except Exception as e:
            self.fail(f"CAS文件读取器导入失败: {e}")
    
    def test_icons(self):
        """测试图标文件是否存在"""
        icon_dir = os.path.join(current_dir, "gui", "icons")
        required_icons = [
            "new.png", "open.png", "save.png", 
            "import.png", "export.png", 
            "generate.png", "display.png", "clear.png"
        ]
        
        missing_icons = []
        for icon in required_icons:
            icon_path = os.path.join(icon_dir, icon)
            if not os.path.exists(icon_path):
                missing_icons.append(icon)
        
        self.assertEqual(len(missing_icons), 0, 
                         f"缺失的图标文件: {missing_icons}")
    
    def test_gui_import(self):
        """测试GUI模块导入"""
        try:
            from gui.gui_main import SimplifiedPyMeshGenGUI
            self.assertTrue(True, "GUI模块导入成功")
        except Exception as e:
            self.fail(f"GUI模块导入失败: {e}")
    
    def test_file_operations(self):
        """测试文件操作功能"""
        try:
            from gui.file_operations import FileOperations
            self.assertTrue(True, "文件操作模块导入成功")
        except Exception as e:
            self.fail(f"文件操作模块导入失败: {e}")
    
    def test_vtk_file_operations(self):
        """测试VTK文件操作功能"""
        try:
            from fileIO.vtk_io import read_vtk
            from gui.file_operations import FileOperations
            
            # 查找VTK文件
            vtk_files = []
            for root, dirs, files in os.walk(current_dir):
                for file in files:
                    if file.endswith('.vtk'):
                        vtk_files.append(os.path.join(root, file))
            
            if not vtk_files:
                self.skipTest("未找到VTK文件，跳过测试")
            
            # 测试第一个VTK文件
            vtk_file = vtk_files[0]
            
            # 创建FileOperations实例
            file_ops = FileOperations(current_dir)
            
            # 导入VTK文件
            mesh_data = file_ops.import_mesh(vtk_file)
            
            # 验证返回的数据结构
            self.assertIsInstance(mesh_data, dict)
            self.assertEqual(mesh_data.get('type'), 'vtk')
            self.assertGreater(mesh_data.get('num_points', 0), 0)
            self.assertGreater(mesh_data.get('num_cells', 0), 0)
            
            # 验证节点坐标
            node_coords = mesh_data.get('node_coords', [])
            self.assertGreater(len(node_coords), 0)
            self.assertEqual(len(node_coords[0]), 3)
            
        except Exception as e:
            self.fail(f"VTK文件操作测试失败: {e}")
    
    def test_cas_file_operations(self):
        """测试CAS文件操作功能"""
        try:
            from fileIO.read_cas import parse_cas_to_unstr_grid
            from gui.file_operations import FileOperations
            
            # 查找CAS文件
            cas_files = []
            for root, dirs, files in os.walk(current_dir):
                for file in files:
                    if file.endswith('.cas'):
                        cas_files.append(os.path.join(root, file))
            
            if not cas_files:
                self.skipTest("未找到CAS文件，跳过测试")
            
            # 测试第一个CAS文件
            cas_file = cas_files[0]
            
            # 创建FileOperations实例
            file_ops = FileOperations(current_dir)
            
            # 导入CAS文件
            mesh_data = file_ops.import_mesh(cas_file)
            
            # 验证返回的数据结构
            self.assertIsInstance(mesh_data, dict)
            self.assertEqual(mesh_data.get('type'), 'cas')
            
            # 验证节点坐标
            node_coords = mesh_data.get('node_coords', [])
            self.assertGreater(len(node_coords), 0)
            self.assertEqual(len(node_coords[0]), 3)
            
        except Exception as e:
            self.fail(f"CAS文件操作测试失败: {e}")


if __name__ == "__main__":
    unittest.main()
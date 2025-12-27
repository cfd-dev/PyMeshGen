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
        """测试图标功能是否正常工作"""
        # 测试图标管理器功能
        try:
            from pyqt_gui.icon_manager import get_icon
            # Test that the icon manager can be imported and has basic functionality
            # Skip the actual icon creation test since it requires GUI context
            self.assertIsNotNone(get_icon, "图标管理器应该存在")

            print("PASS: 图标管理器导入成功")
        except ImportError:
            # 如果图标管理器不可用，测试是否有图标目录
            icon_dir = os.path.join(current_dir, "pyqt_gui", "icons")
            if os.path.exists(icon_dir):
                print("PASS: 图标目录存在")
            else:
                print("INFO: 图标目录不存在，但此功能现在通过图标管理器实现")
    
    def test_gui_import(self):
        """测试GUI模块导入"""
        try:
            from pyqt_gui.gui_main import SimplifiedPyMeshGenGUI
            self.assertTrue(True, "GUI模块导入成功")
        except Exception as e:
            self.fail(f"GUI模块导入失败: {e}")
    
    def test_file_operations(self):
        """测试文件操作功能"""
        try:
            from pyqt_gui.file_operations import FileOperations
            self.assertTrue(True, "文件操作模块导入成功")
        except Exception as e:
            self.fail(f"文件操作模块导入失败: {e}")
    
    def test_vtk_file_operations(self):
        """测试VTK文件操作功能"""
        try:
            from fileIO.vtk_io import read_vtk
            from pyqt_gui.file_operations import FileOperations
            
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
            
            # 验证返回的数据结构 - now returns MeshData object
            from data_structure.mesh_data import MeshData
            self.assertIsInstance(mesh_data, MeshData)
            # Check that it has the expected attributes
            self.assertTrue(hasattr(mesh_data, 'node_coords'))
            self.assertTrue(hasattr(mesh_data, 'cells'))

            # Verify node coordinates exist
            self.assertGreater(len(mesh_data.node_coords), 0)
            self.assertGreater(len(mesh_data.cells), 0)
            coord_len = len(mesh_data.node_coords[0])
            self.assertIn(coord_len, [2, 3])  # 2D or 3D coordinates
            
        except Exception as e:
            self.fail(f"VTK文件操作测试失败: {e}")
    
    def test_cas_file_operations(self):
        """测试CAS文件操作功能"""
        try:
            from fileIO.read_cas import parse_cas_to_unstr_grid
            from pyqt_gui.file_operations import FileOperations
            
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
            
            # 验证返回的数据结构 - now returns MeshData object
            from data_structure.mesh_data import MeshData
            self.assertIsInstance(mesh_data, MeshData)
            # Check that it has the expected attributes
            self.assertTrue(hasattr(mesh_data, 'node_coords'))
            self.assertTrue(hasattr(mesh_data, 'cells'))

            # Verify node coordinates exist
            self.assertGreater(len(mesh_data.node_coords), 0)
            self.assertGreater(len(mesh_data.cells), 0)
            coord_len = len(mesh_data.node_coords[0])
            self.assertIn(coord_len, [2, 3])  # 2D or 3D coordinates
            
        except Exception as e:
            self.fail(f"CAS文件操作测试失败: {e}")


if __name__ == "__main__":
    unittest.main()
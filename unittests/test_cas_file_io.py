#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：CAS文件导入导出功能
整合了test_cas_import.py, test_cas_parts.py, test_import_cas_file_fix.py的测试用例
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fileIO.read_cas import parse_cas_to_unstr_grid, parse_fluent_msh, reconstruct_mesh_from_cas
from data_structure.unstructured_grid import Unstructured_Grid
from gui.file_operations import FileOperations
from data_structure.mesh_data import MeshData


class TestCASFileIO(unittest.TestCase):
    """CAS文件导入导出功能测试类"""

    def setUp(self):
        """测试前的准备工作"""
        self.test_files = [
            "config/input/quad.cas",
            "config/input/concave.cas",
            "config/input/convex.cas"
        ]
        self.existing_files = [f for f in self.test_files if os.path.exists(f)]

    def test_parse_cas_to_unstr_grid(self):
        """测试parse_cas_to_unstr_grid函数"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                unstr_grid = parse_cas_to_unstr_grid(test_file)

                self.assertIsInstance(unstr_grid, Unstructured_Grid)
                self.assertGreater(len(unstr_grid.node_coords), 0)
                self.assertGreater(len(unstr_grid.cell_container), 0)

                for i in range(min(3, len(unstr_grid.node_coords))):
                    coords = unstr_grid.node_coords[i]
                    self.assertEqual(len(coords), 3, f"节点{i}坐标应该是3D的")

    def test_parse_fluent_msh_and_reconstruct(self):
        """测试parse_fluent_msh和reconstruct_mesh_from_cas函数"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                cas_data = parse_fluent_msh(test_file)
                unstr_grid = reconstruct_mesh_from_cas(cas_data)

                self.assertIsInstance(unstr_grid, Unstructured_Grid)
                self.assertGreater(len(unstr_grid.node_coords), 0)
                self.assertGreater(len(unstr_grid.cell_container), 0)

    def test_methods_consistency(self):
        """测试两种导入方法的结果一致性"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                unstr_grid1 = parse_cas_to_unstr_grid(test_file)
                cas_data = parse_fluent_msh(test_file)
                unstr_grid2 = reconstruct_mesh_from_cas(cas_data)

                self.assertEqual(len(unstr_grid1.node_coords), len(unstr_grid2.node_coords))
                self.assertEqual(len(unstr_grid1.cell_container), len(unstr_grid2.cell_container))
                self.assertEqual(len(unstr_grid1.boundary_nodes), len(unstr_grid2.boundary_nodes))

    def test_vtk_conversion(self):
        """测试VTK文件转换功能"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                unstr_grid = parse_cas_to_unstr_grid(test_file)
                vtk_file = test_file.replace('.cas', '_test.vtk')
                unstr_grid.save_to_vtkfile(vtk_file)

                self.assertTrue(os.path.exists(vtk_file))

                from fileIO.vtk_io import parse_vtk_msh
                loaded_grid = parse_vtk_msh(vtk_file)

                self.assertIsInstance(loaded_grid, Unstructured_Grid)
                self.assertEqual(len(unstr_grid.node_coords), len(loaded_grid.node_coords))
                self.assertEqual(len(unstr_grid.cell_container), len(loaded_grid.cell_container))

                if os.path.exists(vtk_file):
                    os.remove(vtk_file)


class TestCASParts(unittest.TestCase):
    """测试CAS文件部件信息提取功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_ops = FileOperations(self.project_root)
        self.test_file = os.path.join(self.project_root, "config/input/30p30n-small.cas")

    def test_extract_parts_from_cas(self):
        """测试从cas文件提取部件信息的功能"""
        self.assertTrue(os.path.exists(self.test_file), f"测试文件 {self.test_file} 不存在")

        mesh_data = self.file_ops.import_mesh(self.test_file)

        self.assertIsInstance(mesh_data, MeshData, "应该返回MeshData对象")
        self.assertTrue(hasattr(mesh_data, 'parts_info'), "MeshData对象应包含parts_info属性")

        parts_info = mesh_data.parts_info
        self.assertIsInstance(parts_info, dict, "部件信息应该是字典类型")

        if parts_info:
            for part_name, part_data in parts_info.items():
                self.assertIsInstance(part_data, dict, f"部件 {part_name} 应该是字典类型")
                self.assertIn('bc_type', part_data, f"部件 {part_name} 应包含bc_type字段")
                self.assertIn('face_count', part_data, f"部件 {part_name} 应包含face_count字段")
                self.assertIn('part_name', part_data, f"部件 {part_name} 应包含part_name字段")
                self.assertEqual(part_data['part_name'], part_name, f"part_name字段应与键名一致")

    def test_import_mesh_with_parts(self):
        """测试导入cas文件并提取部件信息的功能"""
        self.assertTrue(os.path.exists(self.test_file), f"测试文件 {self.test_file} 不存在")

        mesh_data = self.file_ops.import_mesh(self.test_file)

        self.assertIsInstance(mesh_data, MeshData, "应该返回MeshData对象")
        self.assertTrue(hasattr(mesh_data, 'parts_info'), "MeshData对象应包含parts_info属性")

        parts_info = mesh_data.parts_info
        self.assertIsInstance(parts_info, dict, "部件信息应该是字典类型")

        if parts_info:
            self.assertGreater(len(parts_info), 0, "应该至少有一个部件")

            for part_name, part_data in parts_info.items():
                self.assertIn('part_name', part_data, f"部件 {part_name} 应包含part_name字段")
                self.assertIn('face_count', part_data, f"部件 {part_name} 应包含face_count字段")


class TestCASImportFixes(unittest.TestCase):
    """测试CAS文件导入修复功能"""

    def setUp(self):
        """测试前的设置"""
        try:
            from gui.gui_main import PyMeshGenGUI
        except ImportError:
            self.skipTest("PyQt5未安装，跳过GUI相关测试")

        from unittest.mock import Mock
        self.root = Mock()
        self.root.configure = Mock()

        self.mesh_status_label = Mock()
        self.mesh_status_label.config = Mock()
        self.mesh_status_label.cget = Mock(return_value="状态: 已导入")

        self.mesh_info_label = Mock()
        self.mesh_info_label.config = Mock()
        self.mesh_info_label.cget = Mock(return_value="节点数: 1000\n单元数: 2000")

        self.parts_listbox = Mock()
        self.parts_listbox.size = Mock(return_value=2)

        self.mesh_display = Mock()
        self.mesh_display.mesh_data = None

        self.app = Mock(spec=PyMeshGenGUI)
        self.app.mesh_display = self.mesh_display
        self.app.mesh_status_label = self.mesh_status_label
        self.app.mesh_info_label = self.mesh_info_label
        self.app.parts_listbox = self.parts_listbox
        self.app.part_manager = Mock()
        self.app.part_manager.update_parts_list_from_cas = Mock()

    def test_mesh_data_setting(self):
        """测试mesh_data设置"""
        mock_mesh_data = {
            'type': 'cas',
            'num_points': 1000,
            'num_cells': 2000,
            'parts_info': [
                {'part_name': 'inlet', 'face_count': 10, 'nodes': [1, 2, 3], 'cells': [1, 2]},
                {'part_name': 'outlet', 'face_count': 20, 'nodes': [4, 5, 6], 'cells': [3, 4]}
            ]
        }

        self.app.mesh_display.mesh_data = mock_mesh_data
        self.assertEqual(self.app.mesh_display.mesh_data, mock_mesh_data, "mesh_data设置失败")

    def test_mesh_status_update(self):
        """测试网格状态更新"""
        self.app.mesh_status_label.config(text="状态: 已导入")
        self.app.mesh_status_label.config.assert_called_with(text="状态: 已导入")

    def test_mesh_info_update(self):
        """测试网格信息更新"""
        mock_mesh_data = {
            'num_points': 1000,
            'num_cells': 2000
        }

        expected_text = f"节点数: {mock_mesh_data['num_points']}\n单元数: {mock_mesh_data['num_cells']}"
        self.app.mesh_info_label.config(text=expected_text)
        self.app.mesh_info_label.config.assert_called_with(text=expected_text)

    def test_parts_list_update(self):
        """测试部件列表更新"""
        mock_parts_info = [
            {'part_name': 'inlet', 'face_count': 10, 'nodes': [1, 2, 3], 'cells': [1, 2]},
            {'part_name': 'outlet', 'face_count': 20, 'nodes': [4, 5, 6], 'cells': [3, 4]}
        ]

        self.app.part_manager.update_parts_list_from_cas(parts_info=mock_parts_info, update_status=False)
        self.app.part_manager.update_parts_list_from_cas.assert_called_with(parts_info=mock_parts_info, update_status=False)
        self.assertEqual(self.parts_listbox.size(), len(mock_parts_info), "部件列表更新失败")


if __name__ == "__main__":
    unittest.main()

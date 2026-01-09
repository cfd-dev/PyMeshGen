#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：meshio 网格导入功能
测试使用 meshio 库导入各种网格文件格式
"""

import unittest
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

# 添加 meshio 到 sys.path
meshio_path = Path(current_dir) / "3rd_party" / "meshio" / "src"
if meshio_path.exists():
    sys.path.insert(0, str(meshio_path))

from gui.file_operations import FileOperations


class TestMeshIOImport(unittest.TestCase):
    """meshio 网格导入功能测试类"""

    def setUp(self):
        """测试前的准备工作"""
        self.project_root = current_dir
        self.file_ops = FileOperations(self.project_root)
        
        # 定义测试文件列表
        self.test_files = [
            "unittests/test_files/naca0012.vtk",
            "3rd_party/meshio/tests/meshes/msh/insulated-4.1.msh",
        ]

    def test_vtk_import(self):
        """测试 VTK 文件导入"""
        file_path = os.path.join(self.project_root, "unittests/test_files/naca0012.vtk")
        
        if not os.path.exists(file_path):
            self.skipTest(f"测试文件不存在: {file_path}")
        
        mesh_data = self.file_ops.import_mesh(file_path)
        
        # 验证导入成功
        self.assertIsNotNone(mesh_data, "导入 VTK 文件失败")
        
        # 验证基本属性
        self.assertEqual(mesh_data.mesh_type, "vtk", "网格类型不正确")
        self.assertGreater(mesh_data.num_points, 0, "节点数量应该大于0")
        self.assertGreater(mesh_data.num_cells, 0, "单元数量应该大于0")
        
        # 验证节点坐标
        self.assertIsNotNone(mesh_data.node_coords, "节点坐标不能为空")
        self.assertGreater(len(mesh_data.node_coords), 0, "节点坐标列表不能为空")
        
        # 验证单元数据
        self.assertIsNotNone(mesh_data.cells, "单元数据不能为空")
        self.assertGreater(len(mesh_data.cells), 0, "单元列表不能为空")
        
        # 验证 VTK PolyData 已创建
        self.assertTrue(
            hasattr(mesh_data, 'vtk_poly_data') and mesh_data.vtk_poly_data is not None,
            "VTK PolyData 未创建"
        )
        
        # 验证点数据
        if hasattr(mesh_data, 'point_data'):
            self.assertIsInstance(mesh_data.point_data, dict, "点数据应该是字典类型")
        
        # 验证单元数据
        if hasattr(mesh_data, 'cell_data_dict'):
            self.assertIsInstance(mesh_data.cell_data_dict, dict, "单元数据应该是字典类型")

    def test_msh_import(self):
        """测试 MSH 文件导入"""
        file_path = os.path.join(self.project_root, "3rd_party/meshio/tests/meshes/msh/insulated-4.1.msh")
        
        if not os.path.exists(file_path):
            self.skipTest(f"测试文件不存在: {file_path}")
        
        mesh_data = self.file_ops.import_mesh(file_path)
        
        # 验证导入成功
        self.assertIsNotNone(mesh_data, "导入 MSH 文件失败")
        
        # 验证基本属性
        self.assertEqual(mesh_data.mesh_type, "msh", "网格类型不正确")
        self.assertGreater(mesh_data.num_points, 0, "节点数量应该大于0")
        self.assertGreater(mesh_data.num_cells, 0, "单元数量应该大于0")
        
        # 验证节点坐标
        self.assertIsNotNone(mesh_data.node_coords, "节点坐标不能为空")
        self.assertGreater(len(mesh_data.node_coords), 0, "节点坐标列表不能为空")
        
        # 验证单元数据
        self.assertIsNotNone(mesh_data.cells, "单元数据不能为空")
        self.assertGreater(len(mesh_data.cells), 0, "单元列表不能为空")
        
        # 验证 VTK PolyData 已创建
        self.assertTrue(
            hasattr(mesh_data, 'vtk_poly_data') and mesh_data.vtk_poly_data is not None,
            "VTK PolyData 未创建"
        )

    def test_meshio_fallback(self):
        """测试 meshio 失败时的回退机制"""
        # 测试不存在的文件格式
        file_path = os.path.join(self.project_root, "unittests/test_files/nonexistent.xyz")
        
        if os.path.exists(file_path):
            self.skipTest(f"测试文件已存在: {file_path}")
        
        # 应该返回 None 或抛出异常，而不是崩溃
        mesh_data = self.file_ops.import_mesh(file_path)
        self.assertIsNone(mesh_data, "不存在的文件应该返回 None")

    def test_vtk_poly_data_creation(self):
        """测试 VTK PolyData 对象的创建"""
        file_path = os.path.join(self.project_root, "unittests/test_files/naca0012.vtk")
        
        if not os.path.exists(file_path):
            self.skipTest(f"测试文件不存在: {file_path}")
        
        mesh_data = self.file_ops.import_mesh(file_path)
        
        # 验证 VTK PolyData 对象
        self.assertIsNotNone(mesh_data.vtk_poly_data, "VTK PolyData 未创建")
        
        # 验证 VTK PolyData 的基本属性
        vtk_poly = mesh_data.vtk_poly_data
        self.assertGreater(vtk_poly.GetNumberOfPoints(), 0, "VTK PolyData 应该包含点")
        self.assertGreater(vtk_poly.GetNumberOfCells(), 0, "VTK PolyData 应该包含单元")


if __name__ == '__main__':
    unittest.main()

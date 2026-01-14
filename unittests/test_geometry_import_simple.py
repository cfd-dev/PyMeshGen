#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何导入功能单元测试
测试STEP、IGES、STL文件的导入和可视化
"""

import os
import sys
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))


class TestGeometryImport(unittest.TestCase):
    """几何导入功能测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.meshes_dir = current_dir / "examples" / "cad"
        cls.test_files = {
            'step': [],
            'iges': [],
            'stl': []
        }

        # 查找测试文件
        if cls.meshes_dir.exists():
            for file in cls.meshes_dir.iterdir():
                file_path = cls.meshes_dir / file
                if file_path.is_file() and file_path.stat().st_size > 100:
                    if file.suffix in ['.step', '.stp']:
                        cls.test_files['step'].append(file_path)
                    elif file.suffix in ['.iges', '.igs']:
                        cls.test_files['iges'].append(file_path)
                    elif file.suffix == '.stl':
                        cls.test_files['stl'].append(file_path)

    def setUp(self):
        """每个测试方法前的初始化"""
        # 检查依赖
        try:
            from fileIO.geometry_io import (
                import_geometry_file,
                get_shape_statistics,
                get_shape_bounding_box
            )
            self.geometry_io_available = True
        except ImportError:
            self.geometry_io_available = False

        try:
            from fileIO.occ_to_vtk import (
                shape_to_vtk_polydata,
                create_shape_actor
            )
            self.occ_to_vtk_available = True
        except ImportError:
            self.occ_to_vtk_available = False

        try:
            import vtk
            self.vtk_available = True
        except ImportError:
            self.vtk_available = False

    def test_geometry_import_module_available(self):
        """测试几何导入模块是否可用"""
        self.assertTrue(self.geometry_io_available,
                       "几何导入模块加载失败，请确保已安装pythonocc-core")

    def test_occ_to_vtk_module_available(self):
        """测试OCC到VTK转换模块是否可用"""
        self.assertTrue(self.occ_to_vtk_available,
                       "OCC到VTK转换模块加载失败，请确保已安装VTK")

    def test_vtk_library_available(self):
        """测试VTK库是否可用"""
        self.assertTrue(self.vtk_available,
                       "VTK库未安装")

    def test_meshes_directory_exists(self):
        """测试测试文件目录是否存在"""
        self.assertTrue(self.meshes_dir.exists(),
                       f"测试文件目录不存在: {self.meshes_dir}")

    def test_test_files_exist(self):
        """测试测试文件是否存在"""
        total_files = (len(self.test_files['step']) +
                      len(self.test_files['iges']) +
                      len(self.test_files['stl']))
        self.assertGreater(total_files, 0,
                          "未找到任何测试文件")

    def test_step_file_import(self):
        """测试STEP文件导入"""
        if not self.geometry_io_available:
            self.skipTest("几何导入模块不可用")

        if not self.test_files['step']:
            self.skipTest("未找到STEP测试文件")

        from fileIO.geometry_io import import_geometry_file

        step_file = self.test_files['step'][0]
        try:
            shape = import_geometry_file(str(step_file))
            self.assertIsNotNone(shape,
                                f"STEP文件导入失败: {step_file.name}")
        except Exception as e:
            self.fail(f"STEP文件导入异常: {e}")

    def test_step_file_statistics(self):
        """测试STEP文件统计信息"""
        if not self.geometry_io_available:
            self.skipTest("几何导入模块不可用")

        if not self.test_files['step']:
            self.skipTest("未找到STEP测试文件")

        from fileIO.geometry_io import import_geometry_file, get_shape_statistics

        step_file = self.test_files['step'][0]
        try:
            shape = import_geometry_file(str(step_file))
            stats = get_shape_statistics(shape)

            self.assertIsInstance(stats, dict,
                                 "统计信息应该是字典类型")
            self.assertIn('num_vertices', stats,
                         "统计信息应包含顶点数")
            self.assertIn('num_edges', stats,
                         "统计信息应包含边数")
            self.assertIn('num_faces', stats,
                         "统计信息应包含面数")
            self.assertIn('num_solids', stats,
                         "统计信息应包含实体数")
            self.assertIn('bounding_box', stats,
                         "统计信息应包含边界框")

            self.assertGreater(stats['num_vertices'], 0,
                             "顶点数应大于0")
            self.assertGreater(stats['num_edges'], 0,
                             "边数应大于0")
            self.assertGreater(stats['num_faces'], 0,
                             "面数应大于0")
        except Exception as e:
            self.fail(f"获取STEP文件统计信息异常: {e}")

    def test_step_file_bounding_box(self):
        """测试STEP文件边界框"""
        if not self.geometry_io_available:
            self.skipTest("几何导入模块不可用")

        if not self.test_files['step']:
            self.skipTest("未找到STEP测试文件")

        from fileIO.geometry_io import import_geometry_file, get_shape_bounding_box

        step_file = self.test_files['step'][0]
        try:
            shape = import_geometry_file(str(step_file))
            bbox_min, bbox_max = get_shape_bounding_box(shape)

            self.assertIsInstance(bbox_min, tuple,
                                 "最小边界点应该是元组类型")
            self.assertIsInstance(bbox_max, tuple,
                                 "最大边界点应该是元组类型")
            self.assertEqual(len(bbox_min), 3,
                           "最小边界点应包含3个坐标")
            self.assertEqual(len(bbox_max), 3,
                           "最大边界点应包含3个坐标")

            self.assertLess(bbox_min[0], bbox_max[0],
                          "X坐标最小值应小于最大值")
            self.assertLess(bbox_min[1], bbox_max[1],
                          "Y坐标最小值应小于最大值")
            self.assertLess(bbox_min[2], bbox_max[2],
                          "Z坐标最小值应小于最大值")
        except Exception as e:
            self.fail(f"获取STEP文件边界框异常: {e}")

    def test_step_file_vtk_conversion(self):
        """测试STEP文件VTK转换"""
        if not self.geometry_io_available or not self.occ_to_vtk_available:
            self.skipTest("几何导入模块或VTK转换模块不可用")

        if not self.test_files['step']:
            self.skipTest("未找到STEP测试文件")

        from fileIO.geometry_io import import_geometry_file
        from fileIO.occ_to_vtk import shape_to_vtk_polydata

        step_file = self.test_files['step'][0]
        try:
            shape = import_geometry_file(str(step_file))
            polydata = shape_to_vtk_polydata(shape, mesh_quality=1.0)

            self.assertIsNotNone(polydata,
                                "VTK转换失败")
            points = polydata.GetPoints()
            polys = polydata.GetPolys()

            self.assertIsNotNone(points,
                                "VTK点数据不应为空")
            self.assertIsNotNone(polys,
                                "VTK面数据不应为空")
            self.assertGreater(points.GetNumberOfPoints(), 0,
                             "VTK点数应大于0")
            self.assertGreater(polys.GetNumberOfCells(), 0,
                             "VTK面数应大于0")
        except Exception as e:
            self.fail(f"STEP文件VTK转换异常: {e}")

    def test_iges_file_import(self):
        """测试IGES文件导入"""
        if not self.geometry_io_available:
            self.skipTest("几何导入模块不可用")

        if not self.test_files['iges']:
            self.skipTest("未找到IGES测试文件")

        from fileIO.geometry_io import import_geometry_file

        iges_file = self.test_files['iges'][0]
        try:
            shape = import_geometry_file(str(iges_file))
            self.assertIsNotNone(shape,
                                f"IGES文件导入失败: {iges_file.name}")
        except Exception as e:
            self.fail(f"IGES文件导入异常: {e}")

    def test_iges_file_statistics(self):
        """测试IGES文件统计信息"""
        if not self.geometry_io_available:
            self.skipTest("几何导入模块不可用")

        if not self.test_files['iges']:
            self.skipTest("未找到IGES测试文件")

        from fileIO.geometry_io import import_geometry_file, get_shape_statistics

        iges_file = self.test_files['iges'][0]
        try:
            shape = import_geometry_file(str(iges_file))
            stats = get_shape_statistics(shape)

            self.assertIsInstance(stats, dict,
                                 "统计信息应该是字典类型")
            self.assertIn('num_vertices', stats,
                         "统计信息应包含顶点数")
            self.assertIn('num_edges', stats,
                         "统计信息应包含边数")
            self.assertIn('num_faces', stats,
                         "统计信息应包含面数")
        except Exception as e:
            self.fail(f"获取IGES文件统计信息异常: {e}")

    def test_iges_file_vtk_conversion(self):
        """测试IGES文件VTK转换"""
        if not self.geometry_io_available or not self.occ_to_vtk_available:
            self.skipTest("几何导入模块或VTK转换模块不可用")

        if not self.test_files['iges']:
            self.skipTest("未找到IGES测试文件")

        from fileIO.geometry_io import import_geometry_file
        from fileIO.occ_to_vtk import shape_to_vtk_polydata

        iges_file = self.test_files['iges'][0]
        try:
            shape = import_geometry_file(str(iges_file))
            polydata = shape_to_vtk_polydata(shape, mesh_quality=1.0)

            self.assertIsNotNone(polydata,
                                "VTK转换失败")
            points = polydata.GetPoints()
            polys = polydata.GetPolys()

            self.assertIsNotNone(points,
                                "VTK点数据不应为空")
            self.assertIsNotNone(polys,
                                "VTK面数据不应为空")
            self.assertGreater(points.GetNumberOfPoints(), 0,
                             "VTK点数应大于0")
            self.assertGreater(polys.GetNumberOfCells(), 0,
                             "VTK面数应大于0")
        except Exception as e:
            self.fail(f"IGES文件VTK转换异常: {e}")

    def test_stl_file_import(self):
        """测试STL文件导入"""
        if not self.geometry_io_available:
            self.skipTest("几何导入模块不可用")

        if not self.test_files['stl']:
            self.skipTest("未找到STL测试文件")

        from fileIO.geometry_io import import_geometry_file

        stl_file = self.test_files['stl'][0]
        try:
            shape = import_geometry_file(str(stl_file))
            self.assertIsNotNone(shape,
                                f"STL文件导入失败: {stl_file.name}")
        except Exception as e:
            self.fail(f"STL文件导入异常: {e}")

    def test_stl_file_statistics(self):
        """测试STL文件统计信息"""
        if not self.geometry_io_available:
            self.skipTest("几何导入模块不可用")

        if not self.test_files['stl']:
            self.skipTest("未找到STL测试文件")

        from fileIO.geometry_io import import_geometry_file, get_shape_statistics

        stl_file = self.test_files['stl'][0]
        try:
            shape = import_geometry_file(str(stl_file))
            stats = get_shape_statistics(shape)

            self.assertIsInstance(stats, dict,
                                 "统计信息应该是字典类型")
            self.assertIn('num_vertices', stats,
                         "统计信息应包含顶点数")
            self.assertIn('num_faces', stats,
                         "统计信息应包含面数")
        except Exception as e:
            self.fail(f"获取STL文件统计信息异常: {e}")

    def test_stl_file_vtk_conversion(self):
        """测试STL文件VTK转换"""
        if not self.geometry_io_available or not self.occ_to_vtk_available:
            self.skipTest("几何导入模块或VTK转换模块不可用")

        if not self.test_files['stl']:
            self.skipTest("未找到STL测试文件")

        from fileIO.geometry_io import import_geometry_file
        from fileIO.occ_to_vtk import shape_to_vtk_polydata

        stl_file = self.test_files['stl'][0]
        try:
            shape = import_geometry_file(str(stl_file))
            polydata = shape_to_vtk_polydata(shape, mesh_quality=1.0)

            self.assertIsNotNone(polydata,
                                "VTK转换失败")
            points = polydata.GetPoints()
            polys = polydata.GetPolys()

            self.assertIsNotNone(points,
                                "VTK点数据不应为空")
            self.assertIsNotNone(polys,
                                "VTK面数据不应为空")
            self.assertGreater(points.GetNumberOfPoints(), 0,
                             "VTK点数应大于0")
            self.assertGreater(polys.GetNumberOfCells(), 0,
                             "VTK面数应大于0")
        except Exception as e:
            self.fail(f"STL文件VTK转换异常: {e}")


if __name__ == '__main__':
    unittest.main()

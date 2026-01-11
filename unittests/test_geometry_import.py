#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：几何文件导入功能
测试STEP、IGES、STL等格式的几何文件导入和OCC Shape到VTK的转换
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

try:
    from fileIO.geometry_io import (
        import_geometry_file,
        import_step_file,
        import_iges_file,
        import_stl_file,
        get_shape_statistics,
        get_shape_bounding_box,
        extract_vertices_from_shape,
        extract_edges_from_shape,
        extract_faces_from_shape
    )
    from fileIO.occ_to_vtk import (
        shape_to_vtk_polydata,
        shape_edges_to_vtk_polydata,
        shape_vertices_to_vtk_polydata,
        create_shape_actor,
        compute_shape_normals,
        smooth_shape_mesh,
        decimate_shape_mesh
    )
    GEOMETRY_IMPORT_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入几何导入模块: {e}")
    GEOMETRY_IMPORT_AVAILABLE = False

try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


class TestGeometryImport(unittest.TestCase):
    """几何文件导入功能测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类的准备工作"""
        if not GEOMETRY_IMPORT_AVAILABLE:
            cls.skipTest("几何导入模块不可用，请确保已安装pythonocc-core或OCP")

        # 查找测试文件
        cls.test_files = {
            'step': [],
            'iges': [],
            'stl': []
        }

        # 搜索meshes目录下的几何文件
        meshes_dir = os.path.join(current_dir, "meshes")
        if os.path.exists(meshes_dir):
            for file in os.listdir(meshes_dir):
                file_path = os.path.join(meshes_dir, file)
                if os.path.getsize(file_path) > 100:
                    if file.endswith('.step') or file.endswith('.stp'):
                        cls.test_files['step'].append(file_path)
                    elif file.endswith('.iges') or file.endswith('.igs'):
                        cls.test_files['iges'].append(file_path)
                    elif file.endswith('.stl'):
                        cls.test_files['stl'].append(file_path)

        print(f"\n找到测试文件:")
        print(f"  STEP文件: {len(cls.test_files['step'])} 个")
        print(f"  IGES文件: {len(cls.test_files['iges'])} 个")
        print(f"  STL文件: {len(cls.test_files['stl'])} 个")

    def setUp(self):
        """每个测试用例前的准备工作"""
        self.shapes = {}

    def test_import_step_file(self):
        """测试STEP文件导入"""
        if not self.test_files['step']:
            self.skipTest("没有找到STEP测试文件")

        for step_file in self.test_files['step'][:2]:  # 只测试前2个文件
            try:
                print(f"\n测试导入STEP文件: {os.path.basename(step_file)}")
                shape = import_step_file(step_file)
                self.assertIsNotNone(shape, f"导入STEP文件失败: {step_file}")
                self.shapes[step_file] = shape

                # 获取统计信息
                stats = get_shape_statistics(shape)
                print(f"  顶点数: {stats['num_vertices']}")
                print(f"  边数: {stats['num_edges']}")
                print(f"  面数: {stats['num_faces']}")
                print(f"  实体数: {stats['num_solids']}")

                # 验证统计信息
                self.assertGreaterEqual(stats['num_vertices'], 0)
                self.assertGreaterEqual(stats['num_edges'], 0)
                self.assertGreaterEqual(stats['num_faces'], 0)
                self.assertGreaterEqual(stats['num_solids'], 0)

            except Exception as e:
                self.fail(f"导入STEP文件 {step_file} 时发生错误: {str(e)}")

    def test_import_iges_file(self):
        """测试IGES文件导入"""
        if not self.test_files['iges']:
            self.skipTest("没有找到IGES测试文件")

        for iges_file in self.test_files['iges'][:2]:  # 只测试前2个文件
            try:
                print(f"\n测试导入IGES文件: {os.path.basename(iges_file)}")
                shape = import_iges_file(iges_file)
                self.assertIsNotNone(shape, f"导入IGES文件失败: {iges_file}")
                self.shapes[iges_file] = shape

                # 获取统计信息
                stats = get_shape_statistics(shape)
                print(f"  顶点数: {stats['num_vertices']}")
                print(f"  边数: {stats['num_edges']}")
                print(f"  面数: {stats['num_faces']}")

                # 验证统计信息
                self.assertGreaterEqual(stats['num_vertices'], 0)
                self.assertGreaterEqual(stats['num_edges'], 0)
                self.assertGreaterEqual(stats['num_faces'], 0)

            except Exception as e:
                self.fail(f"导入IGES文件 {iges_file} 时发生错误: {str(e)}")

    def test_import_stl_file(self):
        """测试STL文件导入"""
        if not self.test_files['stl']:
            self.skipTest("没有找到STL测试文件")

        for stl_file in self.test_files['stl'][:2]:  # 只测试前2个文件
            try:
                print(f"\n测试导入STL文件: {os.path.basename(stl_file)}")
                shape = import_stl_file(stl_file)
                self.assertIsNotNone(shape, f"导入STL文件失败: {stl_file}")
                self.shapes[stl_file] = shape

                # 获取统计信息
                stats = get_shape_statistics(shape)
                print(f"  顶点数: {stats['num_vertices']}")
                print(f"  边数: {stats['num_edges']}")
                print(f"  面数: {stats['num_faces']}")

                # 验证统计信息
                self.assertGreaterEqual(stats['num_vertices'], 0)
                self.assertGreaterEqual(stats['num_edges'], 0)
                self.assertGreaterEqual(stats['num_faces'], 0)

            except Exception as e:
                self.fail(f"导入STL文件 {stl_file} 时发生错误: {str(e)}")

    def test_import_geometry_file_auto_detect(self):
        """测试自动检测文件格式的导入"""
        # 测试STEP文件
        if self.test_files['step']:
            step_file = self.test_files['step'][0]
            print(f"\n测试自动检测导入STEP文件: {os.path.basename(step_file)}")
            shape = import_geometry_file(step_file)
            self.assertIsNotNone(shape, f"自动检测导入STEP文件失败: {step_file}")

        # 测试IGES文件
        if self.test_files['iges']:
            iges_file = self.test_files['iges'][0]
            print(f"\n测试自动检测导入IGES文件: {os.path.basename(iges_file)}")
            shape = import_geometry_file(iges_file)
            self.assertIsNotNone(shape, f"自动检测导入IGES文件失败: {iges_file}")

        # 测试STL文件
        if self.test_files['stl']:
            stl_file = self.test_files['stl'][0]
            print(f"\n测试自动检测导入STL文件: {os.path.basename(stl_file)}")
            shape = import_geometry_file(stl_file)
            self.assertIsNotNone(shape, f"自动检测导入STL文件失败: {stl_file}")

    def test_get_shape_bounding_box(self):
        """测试获取形状边界框"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:3]:
            try:
                print(f"\n测试获取边界框: {os.path.basename(file_path)}")
                min_point, max_point = get_shape_bounding_box(shape)

                print(f"  最小点: ({min_point[0]:.2f}, {min_point[1]:.2f}, {min_point[2]:.2f})")
                print(f"  最大点: ({max_point[0]:.2f}, {max_point[1]:.2f}, {max_point[2]:.2f})")

                # 验证边界框
                self.assertIsNotNone(min_point)
                self.assertIsNotNone(max_point)
                self.assertEqual(len(min_point), 3)
                self.assertEqual(len(max_point), 3)

                # 验证最大点大于最小点
                self.assertGreater(max_point[0], min_point[0])
                self.assertGreater(max_point[1], min_point[1])
                self.assertGreater(max_point[2], min_point[2])

            except Exception as e:
                self.fail(f"获取边界框时发生错误: {str(e)}")

    def test_extract_vertices_from_shape(self):
        """测试从形状中提取顶点"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:2]:
            try:
                print(f"\n测试提取顶点: {os.path.basename(file_path)}")
                vertices = extract_vertices_from_shape(shape)

                print(f"  提取到 {len(vertices)} 个顶点")

                # 验证顶点
                self.assertIsInstance(vertices, list)
                self.assertGreater(len(vertices), 0)

                # 验证顶点坐标格式
                for vertex in vertices[:5]:  # 只检查前5个顶点
                    self.assertEqual(len(vertex), 3)
                    self.assertIsInstance(vertex[0], (int, float))
                    self.assertIsInstance(vertex[1], (int, float))
                    self.assertIsInstance(vertex[2], (int, float))

            except Exception as e:
                self.fail(f"提取顶点时发生错误: {str(e)}")

    def test_extract_edges_from_shape(self):
        """测试从形状中提取边"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:2]:
            try:
                print(f"\n测试提取边: {os.path.basename(file_path)}")
                edges = extract_edges_from_shape(shape)

                print(f"  提取到 {len(edges)} 条边")

                # 验证边
                self.assertIsInstance(edges, list)

                # 验证边上的点
                for edge in edges[:3]:  # 只检查前3条边
                    self.assertIsInstance(edge, list)
                    self.assertGreater(len(edge), 0)

                    for point in edge[:5]:  # 只检查每条边的前5个点
                        self.assertEqual(len(point), 3)

            except Exception as e:
                self.fail(f"提取边时发生错误: {str(e)}")


class TestOCCToVTK(unittest.TestCase):
    """OCC Shape到VTK转换测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类的准备工作"""
        if not GEOMETRY_IMPORT_AVAILABLE or not VTK_AVAILABLE:
            cls.skipTest("几何导入模块或VTK不可用")

        # 查找测试文件
        cls.test_files = []
        meshes_dir = os.path.join(current_dir, "meshes")
        if os.path.exists(meshes_dir):
            for file in os.listdir(meshes_dir):
                file_path = os.path.join(meshes_dir, file)
                if os.path.getsize(file_path) > 100:
                    if file.endswith('.step') or file.endswith('.stp') or \
                       file.endswith('.iges') or file.endswith('.igs') or \
                       file.endswith('.stl'):
                        cls.test_files.append(file_path)

        # 只使用前3个文件进行测试
        cls.test_files = cls.test_files[:3]

    def setUp(self):
        """每个测试用例前的准备工作"""
        self.shapes = {}
        for file_path in self.test_files:
            try:
                shape = import_geometry_file(file_path)
                self.shapes[file_path] = shape
            except:
                pass

    def test_shape_to_vtk_polydata(self):
        """测试将OCC Shape转换为VTK PolyData"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:2]:
            try:
                print(f"\n测试Shape到VTK PolyData转换: {os.path.basename(file_path)}")
                polydata = shape_to_vtk_polydata(shape, mesh_quality=1.0)

                self.assertIsNotNone(polydata, "转换失败，返回None")

                # 验证PolyData
                points = polydata.GetPoints()
                polys = polydata.GetPolys()

                print(f"  点数: {points.GetNumberOfPoints()}")
                print(f"  面数: {polys.GetNumberOfCells()}")

                self.assertGreater(points.GetNumberOfPoints(), 0, "PolyData中没有点")
                self.assertGreater(polys.GetNumberOfCells(), 0, "PolyData中没有面")

            except Exception as e:
                self.fail(f"Shape到VTK PolyData转换时发生错误: {str(e)}")

    def test_shape_edges_to_vtk_polydata(self):
        """测试将OCC Shape的边转换为VTK PolyData"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:2]:
            try:
                print(f"\n测试Shape边到VTK PolyData转换: {os.path.basename(file_path)}")
                polydata = shape_edges_to_vtk_polydata(shape, sample_rate=0.1)

                self.assertIsNotNone(polydata, "转换失败，返回None")

                # 验证PolyData
                points = polydata.GetPoints()
                lines = polydata.GetLines()

                print(f"  点数: {points.GetNumberOfPoints()}")
                print(f"  线数: {lines.GetNumberOfCells()}")

                self.assertGreater(points.GetNumberOfPoints(), 0, "PolyData中没有点")

            except Exception as e:
                self.fail(f"Shape边到VTK PolyData转换时发生错误: {str(e)}")

    def test_shape_vertices_to_vtk_polydata(self):
        """测试将OCC Shape的顶点转换为VTK PolyData"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:2]:
            try:
                print(f"\n测试Shape顶点到VTK PolyData转换: {os.path.basename(file_path)}")
                polydata = shape_vertices_to_vtk_polydata(shape)

                self.assertIsNotNone(polydata, "转换失败，返回None")

                # 验证PolyData
                points = polydata.GetPoints()
                verts = polydata.GetVerts()

                print(f"  点数: {points.GetNumberOfPoints()}")
                print(f"  顶点单元数: {verts.GetNumberOfCells()}")

                self.assertGreater(points.GetNumberOfPoints(), 0, "PolyData中没有点")

            except Exception as e:
                self.fail(f"Shape顶点到VTK PolyData转换时发生错误: {str(e)}")

    def test_create_shape_actor(self):
        """测试创建VTK Actor"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:1]:
            try:
                print(f"\n测试创建VTK Actor: {os.path.basename(file_path)}")

                # 测试不同显示模式
                display_modes = ['surface', 'wireframe', 'points', 'surface_with_edges']

                for mode in display_modes:
                    print(f"  显示模式: {mode}")
                    actor = create_shape_actor(
                        shape,
                        mesh_quality=1.0,
                        display_mode=mode,
                        color=(0.8, 0.8, 0.9),
                        opacity=0.8,
                        edge_color=(0.0, 0.0, 0.0),
                        edge_width=1.0
                    )

                    self.assertIsNotNone(actor, f"创建Actor失败，显示模式: {mode}")

                    # 验证Actor
                    self.assertIsNotNone(actor.GetMapper(), "Actor没有Mapper")

            except Exception as e:
                self.fail(f"创建VTK Actor时发生错误: {str(e)}")

    def test_compute_shape_normals(self):
        """测试计算形状法向量"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:1]:
            try:
                print(f"\n测试计算法向量: {os.path.basename(file_path)}")
                polydata = shape_to_vtk_polydata(shape, mesh_quality=1.0)

                polydata_with_normals = compute_shape_normals(polydata)

                self.assertIsNotNone(polydata_with_normals, "计算法向量失败")

                # 验证法向量
                point_normals = polydata_with_normals.GetPointData().GetNormals()
                cell_normals = polydata_with_normals.GetCellData().GetNormals()

                print(f"  点法向量数: {point_normals.GetNumberOfTuples() if point_normals else 0}")
                print(f"  面法向量数: {cell_normals.GetNumberOfTuples() if cell_normals else 0}")

                self.assertIsNotNone(point_normals, "没有点法向量")
                self.assertGreater(point_normals.GetNumberOfTuples(), 0, "点法向量为空")

            except Exception as e:
                self.fail(f"计算法向量时发生错误: {str(e)}")

    def test_smooth_shape_mesh(self):
        """测试平滑网格"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:1]:
            try:
                print(f"\n测试平滑网格: {os.path.basename(file_path)}")
                polydata = shape_to_vtk_polydata(shape, mesh_quality=1.0)

                smoothed_polydata = smooth_shape_mesh(polydata, iterations=10, relaxation_factor=0.1)

                self.assertIsNotNone(smoothed_polydata, "平滑网格失败")

                # 验证平滑后的网格
                points_before = polydata.GetPoints().GetNumberOfPoints()
                points_after = smoothed_polydata.GetPoints().GetNumberOfPoints()

                print(f"  平滑前点数: {points_before}")
                print(f"  平滑后点数: {points_after}")

                self.assertEqual(points_before, points_after, "平滑后点数发生变化")

            except Exception as e:
                self.fail(f"平滑网格时发生错误: {str(e)}")

    def test_decimate_shape_mesh(self):
        """测试简化网格"""
        if not self.shapes:
            self.skipTest("没有可用的形状对象")

        for file_path, shape in list(self.shapes.items())[:1]:
            try:
                print(f"\n测试简化网格: {os.path.basename(file_path)}")
                polydata = shape_to_vtk_polydata(shape, mesh_quality=1.0)

                target_reduction = 0.5
                decimated_polydata = decimate_shape_mesh(polydata, target_reduction=target_reduction)

                self.assertIsNotNone(decimated_polydata, "简化网格失败")

                # 验证简化后的网格
                cells_before = polydata.GetPolys().GetNumberOfCells()
                cells_after = decimated_polydata.GetPolys().GetNumberOfCells()

                print(f"  简化前面数: {cells_before}")
                print(f"  简化后面数: {cells_after}")
                print(f"  简化比例: {1.0 - cells_after / cells_before:.2f}")

                self.assertLess(cells_after, cells_before, "简化后面数未减少")
                self.assertGreater(cells_after, 0, "简化后面数为0")

            except Exception as e:
                self.fail(f"简化网格时发生错误: {str(e)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

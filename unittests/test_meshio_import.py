#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：meshio网格导入功能
测试使用meshio库导入各种格式的网格文件
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

# 添加 meshio 到 sys.path
meshio_path = os.path.join(current_dir, "3rd_party", "meshio", "src")
if os.path.exists(meshio_path):
    sys.path.insert(0, meshio_path)

from gui.file_operations import FileOperations
from data_structure.mesh_data import MeshData


class TestMeshIOImport(unittest.TestCase):
    """meshio网格导入功能测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 查找测试文件
        self.test_files = []
        self.project_root = current_dir

        # 支持的网格文件扩展名
        supported_extensions = {
            '.vtk', '.vtu', '.vtp', '.vtr', '.vts', '.vti', '.vto', '.stl', '.obj', '.ply',
            '.msh', '.msh2', '.msh4', '.xdmf', '.xmf', '.off', '.med', '.mesh', '.meshb',
            '.bdf', '.fem', '.nas', '.inp', '.e', '.exo', '.ex2', '.su2', '.cgns', '.avs',
            '.vol', '.mdpa', '.h5m', '.f3grid', '.dat', '.tec', '.ugrid', '.ele', '.node',
            '.xml', '.post', '.wkt', '.hmf', '.ply', '.p3d', '.geo', '.m', '.mat'
        }

        # 诊断信息
        self.diagnostic_info = {
            'test_files_dir': 0,
            'meshio_test_dir': 0,
            'formats_found': {},
            'total_files': 0
        }

        # 检查unittests/test_files目录
        test_files_dir = os.path.join(current_dir, "unittests", "test_files")
        if os.path.exists(test_files_dir):
            for file in os.listdir(test_files_dir):
                file_path = os.path.join(test_files_dir, file)
                # 只处理文件，跳过目录
                if not os.path.isfile(file_path):
                    continue
                # 检查文件扩展名
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in supported_extensions or file_ext == '.cas':
                    if os.path.getsize(file_path) > 100:
                        self.test_files.append(file_path)
                        self.diagnostic_info['test_files_dir'] += 1
                        self.diagnostic_info['formats_found'][file_ext] = \
                            self.diagnostic_info['formats_found'].get(file_ext, 0) + 1

        # 检查meshio测试文件目录
        meshio_test_dir = os.path.join(current_dir, "3rd_party", "meshio", "tests", "meshes")
        if os.path.exists(meshio_test_dir):
            for root, dirs, files in os.walk(meshio_test_dir):
                for file in files:
                    # 跳过非网格文件
                    if file.lower().endswith(('.md', '.txt', '.gitignore', '.makefile', 'makefile')):
                        continue

                    file_path = os.path.join(root, file)
                    # 只处理文件，跳过目录
                    if not os.path.isfile(file_path):
                        continue

                    # 检查文件扩展名
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in supported_extensions:
                        if os.path.getsize(file_path) > 100:
                            self.test_files.append(file_path)
                            self.diagnostic_info['meshio_test_dir'] += 1
                            self.diagnostic_info['formats_found'][file_ext] = \
                                self.diagnostic_info['formats_found'].get(file_ext, 0) + 1

        self.diagnostic_info['total_files'] = len(self.test_files)

        # 创建FileOperations实例
        self.file_ops = FileOperations(str(self.project_root))

        # 打印诊断信息
        self._print_diagnostic_info()

    def _print_diagnostic_info(self):
        """打印测试环境诊断信息"""
        print("\n" + "="*70)
        print("测试环境诊断信息")
        print("="*70)
        print(f"项目根目录: {self.project_root}")
        print(f"测试文件总数: {self.diagnostic_info['total_files']}")
        print(f"  - unittests/test_files: {self.diagnostic_info['test_files_dir']} 个文件")
        print(f"  - meshio/tests/meshes: {self.diagnostic_info['meshio_test_dir']} 个文件")
        print("\n发现的格式分布:")
        for fmt, count in sorted(self.diagnostic_info['formats_found'].items()):
            print(f"  {fmt}: {count} 个文件")
        print("="*70 + "\n")

    def test_meshio_import_basic(self):
        """测试基本的meshio导入功能"""
        for test_file in self.test_files:
            with self.subTest(file=test_file):
                file_ext = os.path.splitext(test_file)[1].lower()

                # 跳过CAS格式（meshio不支持）
                if file_ext == '.cas':
                    self.skipTest(f"CAS格式不被meshio支持: {test_file}")

                try:
                    # 使用meshio方法导入
                    mesh_data = self.file_ops.import_mesh(test_file)

                    # 验证导入成功
                    self.assertIsNotNone(mesh_data, f"导入文件失败: {test_file}")
                    self.assertIsInstance(mesh_data, MeshData, f"返回类型错误: {type(mesh_data)}")

                    # 验证基本属性
                    self.assertIsNotNone(mesh_data.file_path, "file_path不能为None")
                    self.assertIsNotNone(mesh_data.mesh_type, "mesh_type不能为None")
                    self.assertGreater(mesh_data.num_points, 0, "节点数量必须大于0")
                    self.assertGreater(mesh_data.num_cells, 0, "单元数量必须大于0")

                    # 验证VTK数据或原始数据
                    if mesh_data.vtk_poly_data is not None:
                        self.assertEqual(mesh_data.vtk_poly_data.GetNumberOfPoints(), mesh_data.num_points,
                                       f"VTK节点数与MeshData节点数不一致: {mesh_data.vtk_poly_data.GetNumberOfPoints()} vs {mesh_data.num_points}")
                    else:
                        # 如果 vtk_poly_data 为 None，验证原始数据
                        self.assertIsNotNone(mesh_data.node_coords, "vtk_poly_data为None时，node_coords不能为None")
                        self.assertIsNotNone(mesh_data.cells, "vtk_poly_data为None时，cells不能为None")
                        self.assertEqual(len(mesh_data.node_coords), mesh_data.num_points,
                                       f"node_coords长度与num_points不一致: {len(mesh_data.node_coords)} vs {mesh_data.num_points}")

                except ImportError as e:
                    if "No module named" in str(e):
                        self.skipTest(f"缺少依赖模块: {e}")
                    else:
                        self.fail(f"导入错误: {e}")
                except (ValueError, TypeError) as e:
                    # 这些通常是真正的错误，不应该跳过
                    self.fail(f"导入失败（数据错误）: {e}")
                except Exception as e:
                    error_msg = str(e).lower()
                    # 只跳过明确的格式不支持错误
                    if any(keyword in error_msg for keyword in ["not supported", "cannot read", "unknown format", "unsupported"]):
                        self.skipTest(f"格式不支持: {e}")
                    else:
                        self.fail(f"导入失败（未预期的错误）: {e}")

    def test_mesh_data_properties(self):
        """测试MeshData对象的属性"""
        for test_file in self.test_files:
            with self.subTest(file=test_file):
                file_ext = os.path.splitext(test_file)[1].lower()

                if file_ext == '.cas':
                    self.skipTest(f"CAS格式不被meshio支持: {test_file}")

                try:
                    mesh_data = self.file_ops.import_mesh(test_file)

                    if mesh_data is None:
                        self.fail(f"导入失败，返回None: {test_file}")

                    # 测试文件路径
                    self.assertTrue(os.path.exists(mesh_data.file_path),
                                  f"文件路径不存在: {mesh_data.file_path}")

                    # 测试网格类型
                    self.assertIsInstance(mesh_data.mesh_type, str, "mesh_type必须是字符串")
                    self.assertEqual(mesh_data.mesh_type, file_ext[1:],
                                   f"网格类型不匹配: {mesh_data.mesh_type} vs {file_ext[1:]}")

                    # 测试节点数据
                    self.assertIsNotNone(mesh_data.node_coords, "node_coords不能为None")
                    self.assertEqual(len(mesh_data.node_coords), mesh_data.num_points,
                                   f"node_coords长度与num_points不一致: {len(mesh_data.node_coords)} vs {mesh_data.num_points}")

                    # 测试单元数据
                    self.assertIsNotNone(mesh_data.cells, "cells不能为None")
                    self.assertEqual(len(mesh_data.cells), mesh_data.num_cells,
                                   f"cells长度与num_cells不一致: {len(mesh_data.cells)} vs {mesh_data.num_cells}")

                except ImportError as e:
                    if "No module named" in str(e):
                        self.skipTest(f"缺少依赖模块: {e}")
                    else:
                        self.fail(f"导入错误: {e}")
                except (ValueError, TypeError) as e:
                    self.fail(f"属性验证失败（数据错误）: {e}")
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ["not supported", "cannot read", "unknown format", "unsupported"]):
                        self.skipTest(f"格式不支持: {e}")
                    else:
                        self.fail(f"属性验证失败（未预期的错误）: {e}")

    def test_vtk_poly_data_structure(self):
        """测试VTK PolyData结构或原始数据结构"""
        for test_file in self.test_files:
            with self.subTest(file=test_file):
                file_ext = os.path.splitext(test_file)[1].lower()

                if file_ext == '.cas':
                    self.skipTest(f"CAS格式不被meshio支持: {test_file}")

                try:
                    mesh_data = self.file_ops.import_mesh(test_file)

                    if mesh_data is None:
                        self.fail(f"导入失败，返回None: {test_file}")

                    vtk_data = mesh_data.vtk_poly_data

                    # 如果 vtk_poly_data 为 None，验证原始数据结构
                    if vtk_data is None:
                        # 验证原始节点数据
                        self.assertIsNotNone(mesh_data.node_coords, "vtk_poly_data为None时，node_coords不能为None")
                        self.assertGreater(len(mesh_data.node_coords), 0, "node_coords长度必须大于0")

                        # 验证原始单元数据
                        self.assertIsNotNone(mesh_data.cells, "vtk_poly_data为None时，cells不能为None")
                        self.assertGreater(len(mesh_data.cells), 0, "cells长度必须大于0")

                        # 验证节点坐标的维度（可以是1D、2D或3D）
                        for i, coord in enumerate(mesh_data.node_coords[:5]):  # 检查前5个节点
                            self.assertIn(len(coord), [1, 2, 3], f"节点{i}的坐标维度应该是1D、2D或3D: {coord}")
                    else:
                        # 验证VTK点数据
                        num_points = vtk_data.GetNumberOfPoints()
                        self.assertGreater(num_points, 0, "VTK点数必须大于0")

                        # 验证VTK单元数据
                        num_cells = vtk_data.GetNumberOfCells()
                        self.assertGreater(num_cells, 0, "VTK单元数必须大于0")

                        # 验证点数据结构
                        points = vtk_data.GetPoints()
                        self.assertIsNotNone(points, "VTK points不能为None")
                        self.assertEqual(points.GetNumberOfPoints(), num_points,
                                       "VTK points数量与GetNumberOfPoints不一致")

                except ImportError as e:
                    if "No module named" in str(e):
                        self.skipTest(f"缺少依赖模块: {e}")
                    else:
                        self.fail(f"导入错误: {e}")
                except (ValueError, TypeError) as e:
                    self.fail(f"数据结构验证失败（数据错误）: {e}")
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ["not supported", "cannot read", "unknown format", "unsupported"]):
                        self.skipTest(f"格式不支持: {e}")
                    else:
                        self.fail(f"数据结构验证失败（未预期的错误）: {e}")

    def test_specific_formats(self):
        """测试特定格式的导入"""
        # 定义要测试的特定格式
        format_tests = {
            '.vtk': ['unittests/test_files/naca0012.vtk'],
            '.msh': ['3rd_party/meshio/tests/meshes/msh/insulated-4.1.msh'],
            '.stl': [],
            '.vtu': [],
        }

        # 统计测试结果
        tested_files = 0
        skipped_formats = []

        for file_ext, test_patterns in format_tests.items():
            for pattern in test_patterns:
                test_file = os.path.join(self.project_root, pattern)

                if not os.path.exists(test_file):
                    skipped_formats.append(f"{file_ext}: {pattern} (文件不存在)")
                    continue

                tested_files += 1
                with self.subTest(file=test_file, format=file_ext):
                    try:
                        mesh_data = self.file_ops.import_mesh(test_file)

                        self.assertIsNotNone(mesh_data, f"导入{file_ext}文件失败")
                        self.assertEqual(mesh_data.mesh_type, file_ext[1:],
                                       f"网格类型应为{file_ext[1:]}")
                        self.assertGreater(mesh_data.num_points, 0,
                                         f"{file_ext}文件应包含节点")
                        self.assertGreater(mesh_data.num_cells, 0,
                                         f"{file_ext}文件应包含单元")

                    except ImportError as e:
                        if "No module named" in str(e):
                            self.skipTest(f"缺少依赖模块: {e}")
                        else:
                            self.fail(f"导入错误: {e}")
                    except (ValueError, TypeError) as e:
                        self.fail(f"特定格式测试失败（数据错误）: {e}")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if any(keyword in error_msg for keyword in ["not supported", "cannot read", "unknown format", "unsupported"]):
                            self.skipTest(f"格式不支持: {e}")
                        else:
                            self.fail(f"特定格式测试失败（未预期的错误）: {e}")

        # 如果没有测试任何文件，发出警告
        if tested_files == 0:
            self.skipTest(f"没有找到可用的测试文件。跳过的格式: {', '.join(skipped_formats)}")


if __name__ == '__main__':
    unittest.main()

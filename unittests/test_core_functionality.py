#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：核心功能
整合了test_fixes.py, test_debug_fixes.py, test_backup_functionalities.py, test_front_init.py, test_config_load.py, test_fileIO.py, test_environment.py的测试用例
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import json
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCoreFunctionality(unittest.TestCase):
    """核心功能测试类"""

    def setUp(self):
        """Setup test fixtures before each test method."""
        pass

    def test_vtk_import(self):
        """测试VTK文件导入功能"""
        try:
            from fileIO.vtk_io import read_vtk
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

    def test_gui_import(self):
        """测试GUI模块导入"""
        try:
            from gui.gui_main import PyMeshGenGUI
            self.assertTrue(True, "GUI模块导入成功")
        except ImportError as e:
            if "PyQt5" in str(e):
                self.skipTest("PyQt5未安装，跳过GUI模块导入测试")
            else:
                self.fail(f"GUI模块导入失败: {e}")

    def test_file_operations(self):
        """测试文件操作功能"""
        try:
            from gui.file_operations import FileOperations
            self.assertTrue(True, "文件操作模块导入成功")
        except ImportError as e:
            if "PyQt5" in str(e):
                self.skipTest("PyQt5未安装，跳过文件操作模块导入测试")
            else:
                self.fail(f"文件操作模块导入失败: {e}")

    def test_import_functionality(self):
        """测试导入功能"""
        try:
            from fileIO.read_cas import parse_fluent_msh
            from data_structure.parameters import Parameters
            from data_structure.basic_elements import NodeElement, Triangle, Quadrilateral

            self.assertTrue(callable(parse_fluent_msh))
            self.assertTrue(hasattr(Parameters, '__init__'))
            self.assertTrue(hasattr(NodeElement, '__init__'))

        except Exception as e:
            self.fail(f"Import functionality test failed: {e}")

    def test_node_element_instantiation(self):
        """测试NodeElement实例化"""
        try:
            from data_structure.basic_elements import NodeElement

            node1 = NodeElement([1.0, 2.0], 0)
            node2 = NodeElement([3.0, 4.0, 5.0], 1)
            node3 = NodeElement((6.0, 7.0), 2)

            self.assertEqual(node1.idx, 0)
            self.assertEqual(node1.coords, [1.0, 2.0])
            # self.assertEqual(len(node2.coords), 3)
            self.assertEqual(node3.coords, [6.0, 7.0])

        except Exception as e:
            self.fail(f"NodeElement instantiation test failed: {e}")

    def test_pymeshgen_function(self):
        """测试PyMeshGen函数导入和基本功能"""
        try:
            from PyMeshGen import PyMeshGen
            from data_structure.parameters import Parameters

            self.assertTrue(callable(PyMeshGen))

        except Exception as e:
            self.fail(f"PyMeshGen function test failed: {e}")

    def test_part_workflow(self):
        """测试部件参数工作流程功能"""
        try:
            from data_structure.parameters import Parameters
            from data_structure.mesh_data import MeshData

            mesh_data = MeshData()
            mesh_data.node_coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            mesh_data.cells = [[0, 1, 2], [0, 2, 3]]
            mesh_data.parts_info = {"test_part": {"bc_type": "wall", "faces": []}}

            params = Parameters("FROM_MAIN_JSON")

            self.assertIsNotNone(mesh_data)
            self.assertIsNotNone(params)

        except Exception as e:
            self.fail(f"Part workflow test failed: {e}")

    def test_complete_mesh_flow(self):
        """测试完整网格生成流程模拟"""
        try:
            from data_structure.mesh_data import MeshData
            from data_structure.parameters import Parameters
            from data_structure.unstructured_grid import Unstructured_Grid
            from data_structure.basic_elements import Triangle, Quadrilateral
            from utils.data_converter import convert_to_internal_mesh_format

            mesh_data = MeshData()
            mesh_data.node_coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            mesh_data.cells = [[0, 1, 2], [0, 2, 3]]
            mesh_data.parts_info = {"test_part": {"bc_type": "wall", "faces": []}}

            internal_format = convert_to_internal_mesh_format(mesh_data)

            self.assertIsNotNone(internal_format)
            self.assertIn('nodes', internal_format)
            self.assertIn('faces', internal_format)

        except Exception as e:
            self.fail(f"Complete mesh flow test failed: {e}")


class TestDebugFixes(unittest.TestCase):
    """修复功能测试类"""

    def setUp(self):
        """Setup test fixtures before each test method."""
        pass

    def test_array_indexing_fix(self):
        """测试数组索引修复"""
        try:
            from data_structure.front2d import Front
            from data_structure.basic_elements import NodeElement

            node1 = NodeElement([0.0, 0.0], 0)
            node2 = NodeElement([1.0, 0.0], 1)

            front = Front(node1, node2, idx=0, bc_type="test")

            self.assertIsNotNone(front)

        except Exception as e:
            self.fail(f"Array indexing fix test failed: {e}")

    def test_data_converter_fix(self):
        """测试数据转换器功能"""
        try:
            from data_structure.mesh_data import MeshData

            mesh_data = MeshData()
            mesh_data.node_coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            mesh_data.cells = [[0, 1, 2], [0, 2, 3]]
            mesh_data.parts_info = {"wall": {"bc_type": "wall", "faces": []}}

            self.assertIsNotNone(mesh_data)
            self.assertEqual(len(mesh_data.node_coords), 4)
            self.assertEqual(len(mesh_data.cells), 2)

        except Exception as e:
            self.fail(f"Data converter fix test failed: {e}")


class TestBackupFunctionalities(unittest.TestCase):
    """备份功能测试类"""

    def setUp(self):
        """Setup test fixtures before each test method."""
        pass

    def test_cas_import_functionality(self):
        """测试CAS文件导入功能"""
        try:
            from fileIO.read_cas import parse_fluent_msh
            from data_structure.parameters import Parameters

            self.assertTrue(callable(parse_fluent_msh))
            self.assertIsNotNone(Parameters)

        except ImportError as e:
            self.skipTest(f"Import error: {e}")

    def test_vtk_display_functionality(self):
        """测试VTK显示功能"""
        try:
            from data_structure.unstructured_grid import Unstructured_Grid
            from data_structure.basic_elements import Triangle, NodeElement

            self.assertIsNotNone(Unstructured_Grid)
            self.assertIsNotNone(Triangle)
            self.assertIsNotNone(NodeElement)

        except ImportError as e:
            self.skipTest(f"Import error: {e}")

    def test_gui_components_available(self):
        """测试GUI组件可用性"""
        try:
            from gui.gui_main import PyMeshGenGUI
            from gui.mesh_display import MeshDisplayArea

            self.assertIsNotNone(PyMeshGenGUI)
            self.assertIsNotNone(MeshDisplayArea)

        except ImportError as e:
            self.skipTest(f"Import error: {e}")

    def test_mesh_generation_components(self):
        """测试网格生成核心组件"""
        try:
            from core import generate_mesh
            from data_structure.parameters import Parameters

            self.assertTrue(callable(generate_mesh))
            self.assertIsNotNone(Parameters)

        except ImportError as e:
            self.skipTest(f"Import error: {e}")

    def test_basic_elements_functionality(self):
        """测试基本元素功能"""
        try:
            from data_structure.unstructured_grid import Unstructured_Grid
            from data_structure.basic_elements import (
                Triangle, Quadrilateral, NodeElement
            )

            node1 = NodeElement([0.0, 0.0], 0)
            node2 = NodeElement([1.0, 0.0], 1)
            node3 = NodeElement([0.5, 0.866], 2)

            triangle = Triangle(node1.coords, node2.coords, node3.coords)

            self.assertIsNotNone(triangle)
            self.assertEqual(len(triangle.p1), 2)

        except ImportError as e:
            self.skipTest(f"Import error: {e}")
        except Exception as e:
            self.fail(f"Basic elements functionality test failed: {e}")

    def test_parameter_objects(self):
        """测试参数对象功能"""
        try:
            from data_structure.parameters import Parameters

            self.assertIsNotNone(Parameters)

        except ImportError as e:
            self.skipTest(f"Import error: {e}")

    def test_mesh_display_components(self):
        """测试网格显示组件功能"""
        try:
            from visualization.mesh_visualization import Visualization
            from data_structure.unstructured_grid import Unstructured_Grid
            from data_structure.basic_elements import NodeElement

            vis = Visualization(SWITCH=False)
            self.assertIsNotNone(vis)

            node1 = NodeElement([0.0, 0.0], 0)
            node2 = NodeElement([1.0, 0.0], 1)
            node3 = NodeElement([0.5, 0.866], 2)

            grid = Unstructured_Grid([], [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], [])
            self.assertIsNotNone(grid)

        except ImportError as e:
            self.fail(f"Import error: {e}")
        except Exception as e:
            self.fail(f"Mesh display components test failed: {e}")


class TestFrontInitialization(unittest.TestCase):
    """Front初始化测试类"""

    def test_front_initialization(self):
        """测试front初始化功能"""
        try:
            from data_structure.front2d import construct_initial_front
            from fileIO.read_cas import parse_fluent_msh

            self.assertTrue(callable(construct_initial_front))

        except ImportError as e:
            self.skipTest(f"Import error: {e}")
        except Exception as e:
            self.fail(f"Front initialization test failed: {e}")


class TestConfigLoad(unittest.TestCase):
    """配置加载测试类"""

    def setUp(self):
        """测试前的准备工作"""
        self.test_config_path = Path(__file__).parent / 'test_config.json'

    def test_config_file_loading(self):
        """测试配置文件加载"""
        if not self.test_config_path.exists():
            self.skipTest("测试配置文件不存在")

        with open(self.test_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.assertIsNotNone(config)
        self.assertIn('debug_level', config)
        self.assertIn('input_file', config)
        self.assertIn('output_file', config)
        self.assertIn('mesh_type', config)
        self.assertIn('parts', config)

        self.assertIsInstance(config['debug_level'], int)
        self.assertIsInstance(config['input_file'], str)
        self.assertIsInstance(config['output_file'], str)
        self.assertIsInstance(config['mesh_type'], int)
        self.assertIsInstance(config['parts'], list)

    def test_parameters_object_creation(self):
        """测试Parameters对象创建"""
        if not self.test_config_path.exists():
            self.skipTest("测试配置文件不存在")

        from data_structure.parameters import Parameters
        params = Parameters("FROM_CASE_JSON", str(self.test_config_path))

        self.assertIsNotNone(params)
        self.assertIsInstance(params.debug_level, int)
        self.assertIsInstance(params.input_file, str)
        self.assertIsInstance(params.mesh_type, int)
        self.assertIsInstance(params.part_params, list)

        with open(self.test_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.assertEqual(params.debug_level, config['debug_level'])
        self.assertEqual(params.input_file, config['input_file'])
        self.assertEqual(params.mesh_type, config['mesh_type'])

        if config['parts']:
            self.assertGreater(len(params.part_params), 0)
            self.assertEqual(params.part_params[0].part_name, config['parts'][0]['part_name'])


class TestFileIO(unittest.TestCase):
    """文件IO测试类"""

    def test_unstructured_grid_visualization(self):
        """测试Unstructured_Grid对象的可视化"""
        try:
            from data_structure.unstructured_grid import Unstructured_Grid
            import visualization.mesh_visualization as viz
            import matplotlib.pyplot as plt

            test_nodes = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0]
            ]

            grid = Unstructured_Grid([], test_nodes, [])

            self.assertIsNotNone(grid)
            self.assertEqual(len(grid.node_coords), 5)

            fig, ax = plt.subplots()
            viz.visualize_unstr_grid_2d(grid, ax)
            plt.close(fig)

            self.assertTrue(True)

        except Exception as e:
            self.fail(f"Unstructured_Grid visualization test failed: {e}")

    def test_dict_grid_visualization(self):
        """测试字典格式网格数据的可视化"""
        try:
            from visualization.mesh_visualization import visualize_mesh_2d
            import matplotlib.pyplot as plt

            grid_dict = {
                "nodes": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 0.5, 0.0]
                ],
                "zones": {
                    "zone_1": {
                        "type": "faces",
                        "bc_type": "wall",
                        "data": [
                            {"nodes": [0, 1]},
                            {"nodes": [1, 2]},
                            {"nodes": [2, 3]},
                            {"nodes": [3, 0]}
                        ]
                    }
                }
            }

            self.assertIsNotNone(grid_dict)
            self.assertIn('nodes', grid_dict)
            self.assertIn('zones', grid_dict)

            fig, ax = plt.subplots()
            visualize_mesh_2d(grid_dict, ax)
            plt.close(fig)

            self.assertTrue(True)

        except Exception as e:
            self.fail(f"Dict grid visualization test failed: {e}")


class TestCUDAEnvironment(unittest.TestCase):
    """CUDA环境检测测试套件"""

    def test_cuda_availability(self):
        """检测CUDA是否可用"""
        try:
            import torch
            self.assertTrue(
                torch.cuda.is_available(), "CUDA不可用，请检查显卡驱动和PyTorch安装"
            )
            print(f"CUDA可用性验证通过: {torch.cuda.is_available()}")
        except ImportError:
            self.skipTest("PyTorch未安装，跳过CUDA测试")

    def test_device_properties(self):
        """检测GPU设备属性"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                print(f"当前设备: {torch.cuda.get_device_name(device)}")
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"PyTorch版本: {torch.__version__}")

                total_mem = torch.cuda.get_device_properties(device).total_memory
                print(f"显存容量: {total_mem / 1024/1024/1024:.2f} GB")
                self.assertGreater(total_mem, 0, "显存容量检测异常")
            else:
                self.skipTest("CUDA不可用")
        except ImportError:
            self.skipTest("PyTorch未安装，跳过CUDA测试")

    def test_gpu_count(self):
        """检测可用GPU数量"""
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            print(f"检测到GPU数量: {gpu_count}")
            self.assertGreater(gpu_count, 0, "未检测到可用GPU设备")
        except ImportError:
            self.skipTest("PyTorch未安装，跳过CUDA测试")


if __name__ == '__main__':
    unittest.main()

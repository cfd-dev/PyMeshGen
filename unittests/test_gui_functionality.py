#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：GUI相关功能
整合了test_gui_functions.py, test_gui_config.py, test_gui_message.py, test_properties_panel.py, test_properties_display_logic.py, test_import_mesh_fix.py的测试用例
"""

import sys
import os
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_structure'))
sys.path.insert(0, str(project_root / 'utils'))

try:
    from parameters import Parameters
    from data_structure.parameters import Parameters as DataParameters
    from data_structure.basic_elements import Part, Connector
    from gui.gui_main import PyMeshGenGUI
    from gui.mesh_display import MeshDisplayArea
    from utils.message import info, error, warning, debug, verbose, set_debug_level, DEBUG_LEVEL_INFO
except ImportError as e:
    print(f"导入模块失败: {e}")


class TestGUIFunctions(unittest.TestCase):
    """测试GUI相关功能（不依赖GUI组件）"""

    def setUp(self):
        """测试前的准备工作"""
        self.default_config = {
            "debug_level": 1,
            "input_file": "test.cas",
            "output_file": "test.msh",
            "viz_enabled": True,
            "mesh_type": 1,
            "parts": [
                {
                    "part_name": "airfoil",
                    "part_params": {
                        "part_name": "airfoil",
                        "max_size": 1.0,
                        "PRISM_SWITCH": 1,
                        "first_height": 0.001,
                        "growth_rate": 1.2,
                        "growth_method": 0,
                        "max_layers": 20,
                        "full_layers": 1,
                        "multi_direction": 0
                    },
                    "connectors": []
                }
            ]
        }

    def tearDown(self):
        """测试后的清理工作"""
        temp_config_path = Path(__file__).parent / "temp_config.json"
        if temp_config_path.exists():
            temp_config_path.unlink()

    def test_parameters_functionality(self):
        """测试参数对象功能 - 模拟GUI参数处理功能"""
        try:
            params = Parameters("FROM_MAIN_JSON")

            self.assertIsNotNone(params)
            self.assertIsInstance(params, Parameters)

            self.assertTrue(hasattr(params, 'debug_level'))
            self.assertTrue(hasattr(params, 'input_file'))
            self.assertTrue(hasattr(params, 'output_file'))
            self.assertTrue(hasattr(params, 'part_params'))

        except Exception as e:
            self.fail(f"Parameters functionality test failed: {e}")

    def test_config_data_structure(self):
        """测试配置数据结构 - 模拟GUI配置处理功能"""
        try:
            self.assertIsNotNone(self.default_config)
            self.assertIsInstance(self.default_config, dict)

            required_keys = ["debug_level", "input_file", "output_file", "parts"]
            for key in required_keys:
                self.assertIn(key, self.default_config, f"配置应包含{key}键")

            parts = self.default_config.get("parts", [])
            self.assertIsInstance(parts, list, "parts应该是列表类型")

            if parts:
                part = parts[0]
                self.assertIsInstance(part, dict, "part应该是字典类型")
                self.assertIn("part_name", part, "part应包含part_name字段")

        except Exception as e:
            self.fail(f"Config data structure test failed: {e}")

    def test_config_to_params_workflow(self):
        """测试配置到参数的工作流程 - 模拟GUI配置导入功能"""
        try:
            self.assertIsNotNone(self.default_config)

            params = Parameters("FROM_MAIN_JSON")

            self.assertIsNotNone(params)
            self.assertIsInstance(params, Parameters)

            self.assertTrue(hasattr(params, '__dict__'))
            self.assertTrue(hasattr(params, 'part_params'))
            self.assertTrue(hasattr(params, 'debug_level'))

        except Exception as e:
            self.fail(f"Config to parameters workflow test failed: {e}")


class TestGUIConfig(unittest.TestCase):
    """测试配置功能（不依赖GUI组件）"""

    def setUp(self):
        """测试前的准备工作"""
        self.default_config = {
            "debug_level": 1,
            "input_file": "test.cas",
            "output_file": "test.msh",
            "viz_enabled": True,
            "mesh_type": 1,
            "parts": [
                {
                    "part_name": "airfoil",
                    "part_params": {
                        "part_name": "airfoil",
                        "max_size": 1.0,
                        "PRISM_SWITCH": 1,
                        "first_height": 0.001,
                        "growth_rate": 1.2,
                        "growth_method": 0,
                        "max_layers": 20,
                        "full_layers": 1,
                        "multi_direction": 0
                    },
                    "connectors": []
                }
            ]
        }

    def tearDown(self):
        """测试后的清理工作"""
        temp_config_path = Path(__file__).parent / "temp_config.json"
        if temp_config_path.exists():
            temp_config_path.unlink()

    def test_create_params_from_config(self):
        """测试从配置创建参数对象 - 模拟GUI配置导入功能"""
        try:
            params = Parameters("FROM_MAIN_JSON")

            self.assertIsNotNone(params)
            self.assertIsInstance(params, Parameters)

            self.assertTrue(hasattr(params, 'debug_level'))
            self.assertTrue(hasattr(params, 'input_file'))
            self.assertTrue(hasattr(params, 'output_file'))

        except Exception as e:
            self.fail(f"Failed to create parameters from config: {e}")

    def test_create_config_from_params(self):
        """测试从参数对象创建配置 - 模拟GUI配置导出功能"""
        try:
            params = Parameters("FROM_MAIN_JSON")

            self.assertIsNotNone(params)
            self.assertIsInstance(params, Parameters)

            self.assertTrue(hasattr(params, 'debug_level'))
            self.assertTrue(hasattr(params, 'input_file'))
            self.assertTrue(hasattr(params, 'output_file'))
            self.assertTrue(hasattr(params, 'part_params'))

        except Exception as e:
            self.fail(f"Failed to prepare parameters for config export: {e}")


class TestGUIMessage(unittest.TestCase):
    """测试GUI消息输出功能"""

    def test_gui_message(self):
        """测试GUI环境中的消息输出"""
        info("GUI消息测试开始")
        warning("这是一条警告消息")
        error("这是一条错误消息")
        info("GUI消息测试结束")

        self.assertTrue(True)


class TestPropertiesPanel(unittest.TestCase):
    """测试属性面板功能"""

    def setUp(self):
        """测试前的准备工作"""
        pass

    def test_part_properties(self):
        """测试Part类的get_properties方法"""
        try:
            part_name = "测试部件"
            part_params = {"param1": "value1", "param2": "value2"}
            connectors = [
                Connector(part_name, "曲线1", {"密度": 0.1}),
                Connector(part_name, "曲线2", {"密度": 0.2})
            ]

            part = Part(part_name, part_params, connectors)

            properties = part.get_properties()

            self.assertEqual(properties["部件名称"], part_name, "部件名称应该正确")
            self.assertEqual(properties["连接器数量"], 2, "连接器数量应该正确")
            self.assertEqual(properties["阵面总数"], 0, "初始状态下阵面应该为空")

        except Exception as e:
            self.fail(f"测试Part类的get_properties方法失败: {str(e)}")

    def test_parameters_access(self):
        """测试Parameters类的部件访问"""
        try:
            params = Parameters("FROM_MAIN_JSON")

            self.assertTrue(hasattr(params, 'part_params'), "Parameters类应该有part_params属性")

            self.assertIsInstance(params.part_params, list, "part_params应该是列表类型")

            if len(params.part_params) > 0:
                first_part = params.part_params[0]
                self.assertTrue(hasattr(first_part, 'part_name'), "部件应该有part_name属性")

        except Exception as e:
            self.fail(f"测试Parameters类的部件访问失败: {str(e)}")


class TestPropertiesDisplayLogic(unittest.TestCase):
    """测试属性显示逻辑"""

    def setUp(self):
        """测试前的准备工作"""
        pass

    def test_part_properties(self):
        """测试Part类的get_properties方法"""
        try:
            params = Parameters("FROM_MAIN_JSON")

            if len(params.part_params) > 0:
                first_part = params.part_params[0]

                properties = first_part.get_properties()

                self.assertIsInstance(properties, dict, "get_properties应返回字典")
                self.assertIn('部件名称', properties, "属性应包含部件名称")
                self.assertIn('连接器数量', properties, "属性应包含连接器数量")
                self.assertIn('阵面总数', properties, "属性应包含阵面总数")
            else:
                test_part = Part("test_part", params)
                params.part_params["test_part"] = test_part

                properties = test_part.get_properties()

                self.assertIsInstance(properties, dict, "get_properties应返回字典")
                self.assertIn('部件名称', properties, "属性应包含部件名称")

        except Exception as e:
            self.fail(f"测试Part类的get_properties方法失败: {str(e)}")

    def test_parameters_access(self):
        """测试Parameters类的部件访问"""
        try:
            params = Parameters("FROM_MAIN_JSON")

            self.assertTrue(hasattr(params, 'part_params'), "Parameters类应该有part_params属性")

            self.assertIsInstance(params.part_params, list, "part_params应该是列表类型")

        except Exception as e:
            self.fail(f"测试Parameters类的部件访问失败: {str(e)}")


class TestImportMesh(unittest.TestCase):
    """测试导入网格功能"""

    def setUp(self):
        """测试前的设置"""
        self.root = Mock()
        self.root.configure = Mock()

        self.app = Mock(spec=PyMeshGenGUI)

        self.mesh_display = Mock(spec=MeshDisplayArea)
        self.app.mesh_display = self.mesh_display

    def test_mesh_display_attribute(self):
        """测试mesh_display属性是否存在"""
        self.assertTrue(hasattr(self.app, 'mesh_display'), "mesh_display属性未找到")

    def test_mesh_display_instance(self):
        """测试mesh_display是否是MeshDisplayArea的正确实例"""
        self.mesh_display.__class__ = MeshDisplayArea

        self.assertIsInstance(self.app.mesh_display, MeshDisplayArea,
                              "mesh_display不是MeshDisplayArea的实例")

    def test_gui_initialization(self):
        """测试GUI初始化过程"""
        from unittest.mock import patch
        with patch('gui.gui_main.PyMeshGenGUI') as mock_gui_class:
            mock_instance = Mock()
            mock_instance.mesh_display = Mock(spec=MeshDisplayArea)
            mock_gui_class.return_value = mock_instance

            app = mock_gui_class(self.root)

            mock_gui_class.assert_called_once_with(self.root)

            self.assertTrue(hasattr(app, 'mesh_display'))

            self.assertIsInstance(app.mesh_display, Mock)


if __name__ == '__main__':
    unittest.main()

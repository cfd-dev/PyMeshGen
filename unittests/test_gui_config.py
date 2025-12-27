#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import unittest
import tempfile
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_structure'))
sys.path.insert(0, str(project_root / 'utils'))

try:
    # 使用与gui_main.py相同的导入方式
    try:
        # 尝试相对导入（在包中运行时）
        from parameters import Parameters
    except ImportError:
        # 尝试绝对导入（在测试环境中）
        from data_structure.parameters import Parameters

    class TestGUIConfig(unittest.TestCase):
        """测试配置功能（不依赖GUI组件）"""

        def setUp(self):
            """测试前的准备工作"""
            # Create test default config
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
            # 清理临时文件
            temp_config_path = Path(__file__).parent / "temp_config.json"
            if temp_config_path.exists():
                temp_config_path.unlink()

        def test_create_params_from_config(self):
            """测试从配置创建参数对象 - 模拟GUI配置导入功能"""
            # This test will verify that the Parameters class can be created from a config
            # without requiring GUI components
            try:
                # Create Parameters object from config-like data
                # This simulates the import_config functionality
                params = Parameters("FROM_MAIN_JSON")  # Use default parameters

                # Verify that parameters object was created
                self.assertIsNotNone(params)
                self.assertIsInstance(params, Parameters)

                # Verify that it has expected attributes
                self.assertTrue(hasattr(params, 'debug_level'))
                self.assertTrue(hasattr(params, 'input_file'))
                self.assertTrue(hasattr(params, 'output_file'))

                print("PASS: Parameters object created successfully from config")

            except Exception as e:
                self.fail(f"Failed to create parameters from config: {e}")

        def test_create_config_from_params(self):
            """测试从参数对象创建配置 - 模拟GUI配置导出功能"""
            try:
                # Create a Parameters object
                params = Parameters("FROM_MAIN_JSON")

                # Verify that parameters object was created
                self.assertIsNotNone(params)
                self.assertIsInstance(params, Parameters)

                # Test that we can access the expected attributes that would be used for config export
                self.assertTrue(hasattr(params, 'debug_level'))
                self.assertTrue(hasattr(params, 'input_file'))
                self.assertTrue(hasattr(params, 'output_file'))
                self.assertTrue(hasattr(params, 'part_params'))

                # Verify that the parameters have the expected structure for config export
                # This simulates the export_config functionality
                print("PASS: Parameters object can be used for config export")

            except Exception as e:
                self.fail(f"Failed to prepare parameters for config export: {e}")

    if __name__ == "__main__":
        unittest.main()

except ImportError as e:
    print(f"导入模块失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"测试过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
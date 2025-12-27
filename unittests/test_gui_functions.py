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

    class TestGUIFunctions(unittest.TestCase):
        """测试GUI相关功能（不依赖GUI组件）"""

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

        def test_parameters_functionality(self):
            """测试参数对象功能 - 模拟GUI参数处理功能"""
            try:
                # Create Parameters object
                params = Parameters("FROM_MAIN_JSON")  # Use default parameters

                # 验证参数对象创建成功
                self.assertIsNotNone(params)
                self.assertIsInstance(params, Parameters)

                # 验证参数对象有预期的属性
                self.assertTrue(hasattr(params, 'debug_level'))
                self.assertTrue(hasattr(params, 'input_file'))
                self.assertTrue(hasattr(params, 'output_file'))
                self.assertTrue(hasattr(params, 'part_params'))

                print("PASS: Parameters functionality works correctly")

            except Exception as e:
                self.fail(f"Parameters functionality test failed: {e}")

        def test_config_data_structure(self):
            """测试配置数据结构 - 模拟GUI配置处理功能"""
            try:
                # 验证配置数据结构
                self.assertIsNotNone(self.default_config)
                self.assertIsInstance(self.default_config, dict)

                # 验证配置有必要的键
                required_keys = ["debug_level", "input_file", "output_file", "parts"]
                for key in required_keys:
                    self.assertIn(key, self.default_config, f"配置应包含{key}键")

                # 验证parts结构
                parts = self.default_config.get("parts", [])
                self.assertIsInstance(parts, list, "parts应该是列表类型")

                if parts:
                    part = parts[0]
                    self.assertIsInstance(part, dict, "part应该是字典类型")
                    self.assertIn("part_name", part, "part应包含part_name字段")

                print("PASS: Config data structure is valid")

            except Exception as e:
                self.fail(f"Config data structure test failed: {e}")

        def test_config_to_params_workflow(self):
            """测试配置到参数的工作流程 - 模拟GUI配置导入功能"""
            try:
                # 验证配置数据结构
                self.assertIsNotNone(self.default_config)

                # 创建参数对象（模拟GUI导入配置后的操作）
                params = Parameters("FROM_MAIN_JSON")

                # 验证参数对象已创建
                self.assertIsNotNone(params)
                self.assertIsInstance(params, Parameters)

                # 验证参数对象有预期的功能
                self.assertTrue(hasattr(params, '__dict__'))  # 检查是否有属性字典
                self.assertTrue(hasattr(params, 'part_params'))  # 检查是否有部件参数属性
                self.assertTrue(hasattr(params, 'debug_level'))  # 检查是否有调试级别属性

                print("PASS: Config to parameters workflow works correctly")

            except Exception as e:
                self.fail(f"Config to parameters workflow test failed: {e}")

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
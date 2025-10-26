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
sys.path.insert(0, str(project_root / 'gui'))

try:
    from gui.gui_new import SimplifiedPyMeshGenGUI
    # 使用与gui_new.py相同的导入方式
    try:
        # 尝试相对导入（在包中运行时）
        from parameters import Parameters
    except ImportError:
        # 尝试绝对导入（在测试环境中）
        from data_structure.parameters import Parameters
    import tkinter as tk
    from tkinter import ttk
    
    class TestGUIFunctions(unittest.TestCase):
        """测试GUI功能"""
        
        def setUp(self):
            """测试前的准备工作"""
            # 创建一个隐藏的tkinter根窗口
            self.root = tk.Tk()
            self.root.withdraw()  # 隐藏主窗口
            
            # 创建GUI实例
            self.gui = SimplifiedPyMeshGenGUI(self.root)
            
            # 创建测试用的默认配置
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
            # 销毁tkinter根窗口
            self.root.destroy()
            
            # 清理临时文件
            temp_config_path = Path(__file__).parent / "temp_config.json"
            if temp_config_path.exists():
                temp_config_path.unlink()
        
        def test_gui_creation(self):
            """测试GUI实例创建"""
            # 验证GUI实例创建成功
            self.assertIsNotNone(self.gui)
            self.assertIsInstance(self.gui, SimplifiedPyMeshGenGUI)
        
        def test_create_params_from_config(self):
            """测试从配置创建参数对象"""
            # 调用方法创建参数对象
            self.gui.create_params_from_config(self.default_config)
            
            # 验证参数对象创建成功
            self.assertIsNotNone(self.gui.params)
            self.assertIsInstance(self.gui.params, Parameters)
            
            # 验证参数值
            self.assertEqual(self.gui.params.debug_level, self.default_config["debug_level"])
            self.assertEqual(self.gui.params.input_file, self.default_config["input_file"])
            self.assertEqual(self.gui.params.mesh_type, self.default_config["mesh_type"])
            
            # 验证部件参数
            if self.default_config["parts"]:
                self.assertGreater(len(self.gui.params.part_params), 0)
                self.assertEqual(self.gui.params.part_params[0].part_name, 
                                self.default_config["parts"][0]["part_name"])
        
        def test_create_config_from_params(self):
            """测试从参数对象创建配置"""
            # 先创建参数对象
            self.gui.create_params_from_config(self.default_config)
            self.assertIsNotNone(self.gui.params)
            
            # 调用方法创建配置
            config = self.gui.create_config_from_params()
            
            # 验证配置创建成功
            self.assertIsNotNone(config)
            self.assertIsInstance(config, dict)
            
            # 验证配置值
            self.assertEqual(config.get("debug_level"), self.default_config["debug_level"])
            self.assertEqual(config.get("input_file"), self.default_config["input_file"])
            self.assertEqual(config.get("mesh_type"), self.default_config["mesh_type"])
            
            # 验证部件配置
            if "parts" in self.default_config:
                self.assertIn("parts", config)
                self.assertGreater(len(config["parts"]), 0)
                self.assertEqual(config["parts"][0]["part_name"], 
                                self.default_config["parts"][0]["part_name"])

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
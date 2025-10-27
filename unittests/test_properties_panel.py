#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：测试属性面板修复的脚本
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的模块
from data_structure.parameters import Parameters
from data_structure.basic_elements import Part, Connector

class TestPropertiesPanel(unittest.TestCase):
    """测试属性面板功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        pass
    
    def test_part_properties(self):
        """测试Part类的get_properties方法"""
        # 创建一个测试部件
        part_name = "测试部件"
        part_params = {"param1": "value1", "param2": "value2"}
        connectors = [
            Connector(part_name, "曲线1", {"密度": 0.1}),
            Connector(part_name, "曲线2", {"密度": 0.2})
        ]
        
        part = Part(part_name, part_params, connectors)
        
        # 测试get_properties方法
        properties = part.get_properties()
        
        # 验证属性是否正确
        self.assertEqual(properties["部件名称"], part_name, "部件名称应该正确")
        self.assertEqual(properties["连接器数量"], 2, "连接器数量应该正确")
        self.assertEqual(properties["阵面总数"], 0, "初始状态下阵面应该为空")
    
    def test_parameters_access(self):
        """测试Parameters类的部件访问"""
        # 创建一个测试参数对象，使用FROM_MAIN_JSON模式
        params = Parameters("FROM_MAIN_JSON")
        
        # 检查part_params属性是否存在
        self.assertTrue(hasattr(params, 'part_params'), "Parameters类应该有part_params属性")
        
        # 检查part_params是否为列表
        self.assertIsInstance(params.part_params, list, "part_params应该是列表类型")
        
        # 验证可以通过self.params.part_params访问部件列表
        if len(params.part_params) > 0:
            first_part = params.part_params[0]
            self.assertTrue(hasattr(first_part, 'part_name'), "部件应该有part_name属性")

if __name__ == '__main__':
    unittest.main()
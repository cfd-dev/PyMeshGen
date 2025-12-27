#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：测试属性面板功能的脚本
直接测试Part类的get_properties方法和属性面板显示逻辑
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestPropertiesDisplayLogic(unittest.TestCase):
    """测试属性显示逻辑"""
    
    def setUp(self):
        """测试前的准备工作"""
        pass
    
    def test_part_properties(self):
        """测试Part类的get_properties方法"""
        try:
            # 导入必要的类
            from data_structure.parameters import Parameters
            from data_structure.basic_elements import Part
            
            # 创建Parameters对象
            params = Parameters("FROM_MAIN_JSON")
            
            # 检查是否有部件
            if len(params.part_params) > 0:
                # 获取第一个部件
                first_part = params.part_params[0]
                
                # 测试get_properties方法
                properties = first_part.get_properties()
                
                # 验证返回的属性
                self.assertIsInstance(properties, dict, "get_properties应返回字典")
                self.assertIn('部件名称', properties, "属性应包含部件名称")
                self.assertIn('连接器数量', properties, "属性应包含连接器数量")
                self.assertIn('阵面总数', properties, "属性应包含阵面总数")
            else:
                # 没有找到部件，创建一个测试部件
                test_part = Part("test_part", params)
                params.part_params["test_part"] = test_part
                
                # 测试get_properties方法
                properties = test_part.get_properties()
                
                # 验证返回的属性
                self.assertIsInstance(properties, dict, "get_properties应返回字典")
                self.assertIn('部件名称', properties, "属性应包含部件名称")
        except Exception as e:
            self.fail(f"测试Part类的get_properties方法失败: {str(e)}")
    
    def test_parameters_access(self):
        """测试Parameters类的部件访问"""
        try:
            # 导入必要的类
            from data_structure.parameters import Parameters
            
            # 创建Parameters对象
            params = Parameters("FROM_MAIN_JSON")
            
            # 检查part_params属性是否存在
            self.assertTrue(hasattr(params, 'part_params'), "Parameters类应该有part_params属性")
            
            # 检查part_params是否为列表
            self.assertIsInstance(params.part_params, list, "part_params应该是列表类型")
        except Exception as e:
            self.fail(f"测试Parameters类的部件访问失败: {str(e)}")

if __name__ == '__main__':
    unittest.main()
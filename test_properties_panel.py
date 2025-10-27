#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试属性面板修复的脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的模块
from data_structure.parameters import Parameters
from data_structure.basic_elements import Part, Connector

def test_part_properties():
    """测试Part类的get_properties方法"""
    print("测试Part类的get_properties方法...")
    
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
    
    print("部件属性:")
    for key, value in properties.items():
        print(f"  {key}: {value}")
    
    # 验证属性是否正确
    assert properties["部件名称"] == part_name
    assert properties["连接器数量"] == 2
    assert properties["阵面总数"] == 0  # 初始状态下阵面为空
    
    print("测试通过！")
    return True

def test_parameters_access():
    """测试Parameters类的部件访问"""
    print("\n测试Parameters类的部件访问...")
    
    # 创建一个测试参数对象，使用FROM_MAIN_JSON模式
    params = Parameters("FROM_MAIN_JSON")
    
    # 检查part_params属性是否存在
    assert hasattr(params, 'part_params'), "Parameters类应该有part_params属性"
    
    # 检查part_params是否为列表
    assert isinstance(params.part_params, list), "part_params应该是列表类型"
    
    print(f"初始部件数量: {len(params.part_params)}")
    
    # 验证可以通过self.params.part_params访问部件列表
    if len(params.part_params) > 0:
        print(f"第一个部件名称: {params.part_params[0].part_name}")
    
    print("测试通过！")
    return True

def test_parameters_part_access():
    """测试Parameters类的部件访问"""
    print("\n=== 测试Parameters类的部件访问 ===")
    
    try:
        # 创建Parameters对象，使用FROM_MAIN_JSON模式
        params = Parameters("FROM_MAIN_JSON")
        
        # 检查是否有部件
        if hasattr(params, 'part_params') and len(params.part_params) > 0:
            print(f"成功访问Parameters对象，共有 {len(params.part_params)} 个部件")
            
            # 测试访问第一个部件
            first_part = params.part_params[0]
            print(f"第一个部件名称: {first_part.part_name}")
            
            # 测试调用get_properties方法
            part_props = first_part.get_properties()
            print(f"第一个部件属性: {part_props}")
            
            print("Parameters类的部件访问测试通过")
            return True
        else:
            print("Parameters对象没有部件或part_params属性不存在")
            return False
    except Exception as e:
        print(f"Parameters类的部件访问测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试属性面板修复...")
    
    try:
        # 测试Part类的get_properties方法
        test_part_properties()
        
        # 测试Parameters类的部件访问
        test_parameters_access()
        
        print("\n所有测试通过！属性面板修复成功。")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试属性面板功能的脚本
直接测试Part类的get_properties方法和GUI的on_part_select方法
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入GUI类
from gui.gui_main import SimplifiedPyMeshGenGUI

def test_part_properties():
    """测试Part类的get_properties方法"""
    print("测试Part类的get_properties方法...")
    
    try:
        # 导入必要的类
        from data_structure.parameters import Parameters, Part
        
        # 创建Parameters对象
        params = Parameters("FROM_MAIN_JSON")
        
        # 检查是否有部件
        if len(params.part_params) > 0:
            # 获取第一个部件
            first_part = params.part_params[0]
            
            # 测试get_properties方法
            properties = first_part.get_properties()
            
            print(f"部件名称: {properties.get('名称', 'N/A')}")
            print(f"参数数量: {properties.get('参数数量', 'N/A')}")
            print(f"连接器数量: {properties.get('连接器数量', 'N/A')}")
            print(f"阵面总数: {properties.get('阵面总数', 'N/A')}")
            
            print("成功: Part类的get_properties方法正常工作")
            return True
        else:
            print("警告: 没有找到部件，创建一个测试部件")
            
            # 创建一个测试部件
            test_part = Part("test_part", params)
            params.part_params["test_part"] = test_part
            
            # 测试get_properties方法
            properties = test_part.get_properties()
            
            print(f"部件名称: {properties.get('名称', 'N/A')}")
            print(f"参数数量: {properties.get('参数数量', 'N/A')}")
            print(f"连接器数量: {properties.get('连接器数量', 'N/A')}")
            print(f"阵面总数: {properties.get('阵面总数', 'N/A')}")
            
            print("成功: Part类的get_properties方法正常工作")
            return True
    except Exception as e:
        print(f"错误: 测试Part类的get_properties方法失败: {str(e)}")
        return False

def test_gui_properties_display():
    """测试GUI属性面板显示功能"""
    print("\n测试GUI属性面板显示功能...")
    
    try:
        # 创建主窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 创建GUI实例
        app = SimplifiedPyMeshGenGUI(root)
        
        # 初始化一个默认配置
        app.new_config()
        
        # 检查params是否已初始化
        if app.params is None:
            print("错误: params未初始化")
            return False
        
        # 确保有部件数据
        if not hasattr(app.params, 'part_params') or len(app.params.part_params) == 0:
            print("警告: 没有部件数据，添加一个测试部件")
            # 添加一个测试部件
            from data_structure.parameters import Part
            test_part = Part("test_part", app.params)
            app.params.part_params["test_part"] = test_part
        
        # 更新部件列表
        app.update_parts_list()
        
        # 选择第一个部件
        if len(app.params.part_params) > 0:
            app.parts_listbox.selection_set(0)
            
            # 触发选择事件
            app.on_part_select(None)
            
            # 检查属性面板是否更新
            props_content = app.props_text.get("1.0", tk.END)
            print(f"属性面板内容:\n{props_content}")
            
            # 检查是否包含部件信息
            if "部件属性" in props_content or "部件类型" in props_content:
                print("成功: 属性面板显示了部件信息")
                return True
            else:
                print("失败: 属性面板未显示部件信息")
                return False
        else:
            print("错误: 没有可选择的部件")
            return False
    except Exception as e:
        print(f"错误: 测试GUI属性面板显示功能失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("开始测试属性面板功能...")
    
    # 测试Part类的get_properties方法
    test1_result = test_part_properties()
    
    # 测试GUI属性面板显示功能
    test2_result = test_gui_properties_display()
    
    # 输出测试结果
    print("\n测试结果:")
    print(f"Part类get_properties方法: {'通过' if test1_result else '失败'}")
    print(f"GUI属性面板显示功能: {'通过' if test2_result else '失败'}")
    
    if test1_result and test2_result:
        print("\n所有测试通过! 属性面板修复成功。")
        return 0
    else:
        print("\n部分测试失败，请检查问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
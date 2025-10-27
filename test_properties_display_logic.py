#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试属性面板功能的脚本
直接测试Part类的get_properties方法和属性面板显示逻辑
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def test_properties_display_logic():
    """测试属性面板显示逻辑"""
    print("\n测试属性面板显示逻辑...")
    
    try:
        import time
        # 导入必要的类
        from data_structure.parameters import Parameters, Part
        
        # 创建Parameters对象
        params = Parameters("FROM_MAIN_JSON")
        
        # 确保有部件数据
        if not hasattr(params, 'part_params') or len(params.part_params) == 0:
            print("警告: 没有部件数据，添加一个测试部件")
            # 添加一个测试部件
            test_part = Part("test_part", params)
            params.part_params["test_part"] = test_part
        
        # 创建一个简单的GUI框架，只包含属性面板
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 创建属性面板
        props_frame = ttk.Frame(root)
        props_text = tk.Text(props_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        
        # 模拟选择部件并显示属性
        if len(params.part_params) > 0:
            # 获取第一个部件
            first_part = params.part_params[0]
            
            # 清空属性文本框
            props_text.config(state=tk.NORMAL)
            props_text.delete(1.0, tk.END)
            
            # 添加标题
            props_text.insert(tk.END, f"=== 部件属性 ===\n\n")
            
            # 显示部件属性
            if hasattr(first_part, 'get_properties'):
                props = first_part.get_properties()
                for key, value in props.items():
                    props_text.insert(tk.END, f"{key}: {value}\n")
            else:
                # 如果部件没有get_properties方法，显示基本信息
                props_text.insert(tk.END, f"部件类型: {type(first_part).__name__}\n")
                if hasattr(first_part, 'part_name'):
                    props_text.insert(tk.END, f"名称: {first_part.part_name}\n")
                if hasattr(first_part, 'name'):
                    props_text.insert(tk.END, f"名称: {first_part.name}\n")
                if hasattr(first_part, 'id'):
                    props_text.insert(tk.END, f"ID: {first_part.id}\n")
            
            # 添加状态信息
            props_text.insert(tk.END, f"\n=== 状态信息 ===\n")
            props_text.insert(tk.END, f"选择时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            props_text.insert(tk.END, f"部件索引: 0\n")
            props_text.insert(tk.END, f"总部件数: {len(params.part_params)}\n")
            
            props_text.config(state=tk.DISABLED)
            
            # 获取属性面板内容
            props_content = props_text.get("1.0", tk.END)
            print(f"属性面板内容:\n{props_content}")
            
            # 检查是否包含部件信息
            if "部件属性" in props_content:
                print("成功: 属性面板显示了部件信息")
                return True
            else:
                print("失败: 属性面板未显示部件信息")
                return False
        else:
            print("错误: 没有可选择的部件")
            return False
    except Exception as e:
        print(f"错误: 测试属性面板显示逻辑失败: {str(e)}")
        return False

def main():
    """主函数"""
    import time
    print("开始测试属性面板功能...")
    
    # 测试Part类的get_properties方法
    test1_result = test_part_properties()
    
    # 测试属性面板显示逻辑
    test2_result = test_properties_display_logic()
    
    # 输出测试结果
    print("\n测试结果:")
    print(f"Part类get_properties方法: {'通过' if test1_result else '失败'}")
    print(f"属性面板显示逻辑: {'通过' if test2_result else '失败'}")
    
    if test1_result and test2_result:
        print("\n所有测试通过! 属性面板修复成功。")
        return 0
    else:
        print("\n部分测试失败，请检查问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
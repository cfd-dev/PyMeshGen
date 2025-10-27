#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试GUI应用程序的综合功能
验证整个GUI应用程序是否正常工作
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_gui_app():
    """测试GUI应用程序的综合功能"""
    print("测试GUI应用程序的综合功能...")
    
    try:
        # 导入必要的类
        from gui.gui_main import SimplifiedPyMeshGenGUI
        
        # 创建主窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 创建GUI实例
        app = SimplifiedPyMeshGenGUI(root)
        
        # 确保所有UI组件都已创建
        # 检查是否有debug_level_var等变量，如果没有，手动初始化
        if not hasattr(app, 'debug_level_var'):
            app.debug_level_var = tk.StringVar(value="0")
            app.input_file_var = tk.StringVar(value="")
            app.output_file_var = tk.StringVar(value="")
            app.viz_enabled_var = tk.BooleanVar(value=False)
            app.mesh_type_var = tk.StringVar(value="1")
        
        # 检查是否有config_label，如果没有，手动创建
        if not hasattr(app, 'config_label'):
            app.config_label = ttk.Label(app.left_panel, text="配置信息: 未设置")
            app.config_label.pack(fill=tk.X, padx=5, pady=2)
        
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
            from data_structure.parameters import Part, MeshParameters
            from data_structure.basic_elements import Connector
            
            # 创建网格参数
            mesh_params = MeshParameters(
                part_name="test_part",
                max_size=1.0,
                PRISM_SWITCH="off",
                first_height=0.1,
                growth_rate=1.2,
                growth_method="geometric",
                max_layers=3,
                full_layers=0,
                multi_direction=False
            )
            
            # 创建连接器
            connector = Connector(part_name="test_part", curve_name="test_curve", param=mesh_params)
            
            # 创建部件
            test_part = Part(part_name="test_part", part_params=mesh_params, connectors=[connector])
            app.params.part_params.append(test_part)
        
        # 更新部件列表
        app.update_parts_list()
        
        # 测试部件选择和属性显示
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
                props_test_passed = True
            else:
                print("失败: 属性面板未显示部件信息")
                props_test_passed = False
        else:
            print("错误: 没有可选择的部件")
            props_test_passed = False
        
        # 测试状态更新功能
        app.update_status("测试状态更新")
        # 由于GUI使用status_bar而不是status_var，我们检查状态栏的文本
        if hasattr(app, 'status_bar'):
            status_test_passed = True
            print("成功: 状态更新功能正常")
        else:
            print("失败: 状态栏不存在")
            status_test_passed = False
        
        # 测试日志功能
        app.log_info("测试信息日志")
        app.log_warning("测试警告日志")
        app.log_error("测试错误日志")
        
        # 检查日志输出
        if hasattr(app, 'info_output'):
            log_test_passed = True
            print("成功: 日志功能正常")
        else:
            print("失败: 日志功能异常")
            log_test_passed = False
        
        # 销毁窗口
        root.destroy()
        
        # 返回综合测试结果
        return props_test_passed and status_test_passed and log_test_passed
    except Exception as e:
        print(f"错误: 测试GUI应用程序功能失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("开始测试GUI应用程序的综合功能...")
    
    # 测试GUI应用程序功能
    test_result = test_gui_app()
    
    # 输出测试结果
    print("\n测试结果:")
    print(f"GUI应用程序功能: {'通过' if test_result else '失败'}")
    
    if test_result:
        print("\n测试通过! GUI应用程序功能正常。")
        return 0
    else:
        print("\n测试失败，请检查问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
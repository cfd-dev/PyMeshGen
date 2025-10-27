#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试属性面板功能的脚本
"""

import sys
import os
import tkinter as tk
from tkinter import ttk

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的模块
from gui.gui_main import SimplifiedPyMeshGenGUI

def test_properties_panel():
    """测试属性面板功能"""
    print("测试属性面板功能...")
    
    # 创建主窗口
    root = tk.Tk()
    root.title("属性面板测试")
    
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
    
    # 模拟选择部件
    if hasattr(app, 'params') and len(app.params.part_params) > 0:
        # 选择第一个部件
        app.selected_part_index = 0
        app.on_part_select()
        
        print("属性面板测试完成")
        print("请检查GUI窗口中的属性面板是否正确显示部件信息")
        
        # 运行主循环
        root.mainloop()
        
        return True
    else:
        print("没有可用的部件进行测试")
        return False

if __name__ == "__main__":
    test_properties_panel()
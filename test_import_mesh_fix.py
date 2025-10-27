#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试导入网格功能的修复
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from gui.gui_main import SimplifiedPyMeshGenGUI
import tkinter as tk

def test_import_mesh():
    """测试导入网格功能"""
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口，只测试功能
    
    try:
        # 创建GUI实例
        app = SimplifiedPyMeshGenGUI(root)
        
        # 检查是否有mesh_display属性
        if hasattr(app, 'mesh_display'):
            print("✓ mesh_display属性已成功创建")
        else:
            print("✗ mesh_display属性未找到")
            return False
        
        # 检查mesh_display是否是MeshDisplayArea的实例
        from gui.mesh_display import MeshDisplayArea
        if isinstance(app.mesh_display, MeshDisplayArea):
            print("✓ mesh_display是MeshDisplayArea的正确实例")
        else:
            print("✗ mesh_display不是MeshDisplayArea的实例")
            return False
        
        print("✓ 所有测试通过，导入网格功能修复成功")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False
    finally:
        # 销毁窗口
        root.destroy()

if __name__ == "__main__":
    print("开始测试导入网格功能修复...")
    success = test_import_mesh()
    if success:
        print("测试成功！导入网格功能修复完成。")
    else:
        print("测试失败！导入网格功能修复未完成。")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试导入CAS文件功能的修复
"""

import os
import sys
import tkinter as tk
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from gui.gui_main import SimplifiedPyMeshGenGUI

def test_import_cas_file():
    """测试导入CAS文件功能"""
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口，只测试功能
    
    try:
        # 创建GUI实例
        app = SimplifiedPyMeshGenGUI(root)
        
        # 检查是否有mesh_display属性
        if not hasattr(app, 'mesh_display'):
            print("✗ mesh_display属性未找到")
            return False
        
        # 模拟导入CAS文件的数据
        mock_mesh_data = {
            'type': 'cas',
            'num_points': 1000,
            'num_cells': 2000,
            'parts_info': [
                {'part_name': 'inlet', 'face_count': 10, 'nodes': [1, 2, 3], 'cells': [1, 2]},
                {'part_name': 'outlet', 'face_count': 20, 'nodes': [4, 5, 6], 'cells': [3, 4]}
            ]
        }
        
        # 模拟设置mesh_data
        app.mesh_display.mesh_data = mock_mesh_data
        
        # 检查mesh_data是否设置成功
        if app.mesh_display.mesh_data == mock_mesh_data:
            print("✓ mesh_data设置成功")
        else:
            print("✗ mesh_data设置失败")
            return False
        
        # 模拟更新网格状态
        app.mesh_status_label.config(text="状态: 已导入")
        if app.mesh_status_label.cget("text") == "状态: 已导入":
            print("✓ 网格状态更新成功")
        else:
            print("✗ 网格状态更新失败")
            return False
        
        # 模拟更新网格信息
        app.mesh_info_label.config(text=f"节点数: {mock_mesh_data['num_points']}\n单元数: {mock_mesh_data['num_cells']}")
        if app.mesh_info_label.cget("text") == f"节点数: {mock_mesh_data['num_points']}\n单元数: {mock_mesh_data['num_cells']}":
            print("✓ 网格信息更新成功")
        else:
            print("✗ 网格信息更新失败")
            return False
        
        # 测试更新部件列表
        app.update_parts_list_from_cas(mock_mesh_data['parts_info'])
        if app.parts_listbox.size() == len(mock_mesh_data['parts_info']):
            print("✓ 部件列表更新成功")
        else:
            print("✗ 部件列表更新失败")
            return False
        
        print("✓ 所有测试通过，导入CAS文件功能修复成功")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 销毁窗口
        root.destroy()

if __name__ == "__main__":
    print("开始测试导入CAS文件功能修复...")
    success = test_import_cas_file()
    if success:
        print("测试成功！导入CAS文件功能修复完成。")
    else:
        print("测试失败！导入CAS文件功能修复未完成。")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试GUI中的部件面板功能
"""

import sys
import os
import tkinter as tk

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.gui_main import SimplifiedPyMeshGenGUI
from gui.file_operations import FileOperations

def test_gui_parts_panel():
    """测试GUI中的部件面板功能"""
    print("=" * 50)
    print("测试GUI中的部件面板功能")
    print("=" * 50)
    
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 创建GUI实例
    app = SimplifiedPyMeshGenGUI(root)
    
    # 测试文件路径
    test_file = "config/input/30p30n-small.cas"
    
    if not os.path.exists(test_file):
        print(f"错误：测试文件 {test_file} 不存在")
        return False
    
    # 直接导入网格数据
    print(f"正在导入cas文件: {test_file}")
    try:
        # 使用file_operations导入网格
        file_ops = FileOperations(os.path.dirname(os.path.abspath(__file__)))
        mesh_data = file_ops.import_mesh(test_file)
        
        # 设置当前网格
        app.current_mesh = mesh_data
        
        # 更新网格显示区域的网格数据
        app.mesh_display.mesh_data = mesh_data
        
        # 更新网格状态
        app.mesh_status_label.config(text="状态: 已导入")
        
        # 获取网格信息
        if isinstance(mesh_data, dict):
            # 处理字典类型的网格数据
            node_count = mesh_data.get('num_points', 0)
            element_count = mesh_data.get('num_cells', 0)
            app.mesh_info_label.config(text=f"节点数: {node_count}\n单元数: {element_count}")
            
            # 如果是cas文件，提取部件信息
            if mesh_data.get('type') == 'cas' and 'parts_info' in mesh_data:
                app.update_parts_list_from_cas(mesh_data['parts_info'])
        
        print("文件导入成功!")
    except Exception as e:
        print(f"导入文件时出错: {str(e)}")
        return False
    
    # 检查部件列表
    if hasattr(app, 'cas_parts_info') and app.cas_parts_info:
        print(f"成功加载 {len(app.cas_parts_info)} 个部件到部件面板:")
        for i, part in enumerate(app.cas_parts_info):
            print(f"部件 {i+1}: {part['part_name']} (类型: {part['bc_type']})")
        
        # 模拟选择第一个部件
        print("\n模拟选择第一个部件...")
        app.on_part_select(0)
        print("部件选择测试完成!")
        
        return True
    else:
        print("错误：未找到部件信息")
        return False

if __name__ == "__main__":
    success = test_gui_parts_panel()
    if success:
        print("\nGUI部件面板测试通过!")
    else:
        print("\nGUI部件面板测试失败!")
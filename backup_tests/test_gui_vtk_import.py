#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试GUI中的VTK文件导入修复
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入必要的模块
from gui.file_operations import FileOperations

def test_gui_vtk_import():
    """测试GUI中的VTK文件导入"""
    try:
        # 创建FileOperations实例
        file_ops = FileOperations(current_dir)
        
        # 查找VTK文件
        vtk_files = []
        for root, dirs, files in os.walk(current_dir):
            for file in files:
                if file.endswith('.vtk'):
                    vtk_files.append(os.path.join(root, file))
        
        if not vtk_files:
            messagebox.showinfo("测试结果", "未找到VTK文件，无法测试")
            return False
        
        # 测试第一个VTK文件
        vtk_file = vtk_files[0]
        print(f"测试文件: {vtk_file}")
        
        # 导入VTK文件
        mesh_data = file_ops.import_mesh(vtk_file)
        
        # 检查返回的数据结构
        if isinstance(mesh_data, dict):
            print("成功导入VTK文件，返回字典类型数据")
            print(f"网格类型: {mesh_data.get('type', 'unknown')}")
            print(f"节点数量: {mesh_data.get('num_points', 0)}")
            print(f"单元数量: {mesh_data.get('num_cells', 0)}")
            
            # 检查节点坐标
            node_coords = mesh_data.get('node_coords', [])
            if node_coords:
                print(f"第一个节点坐标: {node_coords[0]}")
                if len(node_coords[0]) == 3:
                    print("节点坐标包含x, y, z三个值")
                else:
                    print(f"警告: 节点坐标只包含{len(node_coords[0])}个值")
            
            # 获取网格信息
            mesh_info = file_ops.get_mesh_info(mesh_data)
            if mesh_info:
                print("成功获取网格信息")
                print(f"边界框: X({mesh_info['bounds'][0]:.4f}, {mesh_info['bounds'][1]:.4f}), "
                      f"Y({mesh_info['bounds'][2]:.4f}, {mesh_info['bounds'][3]:.4f})")
                if len(mesh_info['bounds']) > 4:
                    print(f"Z({mesh_info['bounds'][4]:.4f}, {mesh_info['bounds'][5]:.4f})")
            
            messagebox.showinfo("测试结果", "VTK文件导入测试成功！\n节点坐标已正确包含x, y, z三个值。")
            return True
        else:
            print(f"导入VTK文件失败，返回类型: {type(mesh_data)}")
            messagebox.showerror("测试结果", f"导入VTK文件失败，返回类型: {type(mesh_data)}")
            return False
            
    except Exception as e:
        error_msg = f"测试VTK文件导入时出错: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        messagebox.showerror("测试结果", error_msg)
        return False

if __name__ == "__main__":
    # 创建简单的GUI窗口
    root = tk.Tk()
    root.title("VTK文件导入测试")
    root.geometry("300x100")
    
    # 添加测试按钮
    test_button = tk.Button(root, text="测试VTK文件导入", command=test_gui_vtk_import)
    test_button.pack(pady=20)
    
    # 运行GUI
    root.mainloop()
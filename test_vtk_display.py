#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试VTK文件导入和显示修复
"""

import os
import sys

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入必要的模块
from gui.file_operations import FileOperations
from gui.mesh_display import MeshDisplayArea
import tkinter as tk

def test_vtk_import_and_display():
    """测试VTK文件导入和显示"""
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
            print("未找到VTK文件，无法测试")
            return False
        
        # 测试第一个VTK文件
        vtk_file = vtk_files[0]
        print(f"测试文件: {vtk_file}")
        
        # 导入VTK文件
        mesh_data = file_ops.import_mesh(vtk_file)
        
        # 检查返回的数据结构
        if isinstance(mesh_data, dict):
            print("✓ 成功导入VTK文件，返回字典类型数据")
            
            # 检查节点坐标
            node_coords = mesh_data.get('node_coords', [])
            if node_coords and len(node_coords[0]) == 3:
                print("✓ 节点坐标包含x, y, z三个值")
                print(f"  第一个节点坐标: {node_coords[0]}")
            else:
                print(f"✗ 节点坐标只包含{len(node_coords[0])}个值")
                return False
            
            # 创建简单的GUI窗口
            root = tk.Tk()
            root.title("VTK文件显示测试")
            root.geometry("800x600")
            
            # 创建网格显示区域
            mesh_display = MeshDisplayArea(root)
            mesh_display.pack(fill=tk.BOTH, expand=True)
            
            # 设置网格数据
            mesh_display.set_mesh_data(mesh_data)
            
            # 测试display_mesh方法
            print("测试display_mesh方法...")
            try:
                # 测试不带参数的调用
                result = mesh_display.display_mesh()
                if result:
                    print("✓ display_mesh()调用成功")
                else:
                    print("✗ display_mesh()调用失败")
                    return False
                
                # 测试带参数的调用
                result = mesh_display.display_mesh(mesh_data)
                if result:
                    print("✓ display_mesh(mesh_data)调用成功")
                else:
                    print("✗ display_mesh(mesh_data)调用失败")
                    return False
                
                print("✅ VTK文件导入和显示修复验证成功！")
                
                # 显示窗口
                root.update()
                print("网格已显示在窗口中")
                
                # 关闭窗口
                root.destroy()
                
                return True
            except Exception as e:
                print(f"✗ 测试display_mesh方法时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"✗ 导入VTK文件失败，返回类型: {type(mesh_data)}")
            return False
            
    except Exception as e:
        print(f"✗ 测试VTK文件导入和显示时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("测试VTK文件导入和显示修复...")
    success = test_vtk_import_and_display()
    if success:
        print("\n所有修复验证通过！")
    else:
        print("\n修复验证失败！")
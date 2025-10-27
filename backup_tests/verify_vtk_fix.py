#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证VTK文件导入修复
"""

import os
import sys

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入必要的模块
from gui.file_operations import FileOperations

def verify_vtk_import_fix():
    """验证VTK文件导入修复"""
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
            print("未找到VTK文件，无法验证")
            return False
        
        # 测试第一个VTK文件
        vtk_file = vtk_files[0]
        print(f"验证文件: {vtk_file}")
        
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
            
            # 获取网格信息
            mesh_info = file_ops.get_mesh_info(mesh_data)
            if mesh_info and len(mesh_info['bounds']) == 6:
                print("✓ 成功获取包含Z轴的网格信息")
                print(f"  边界框: X({mesh_info['bounds'][0]:.4f}, {mesh_info['bounds'][1]:.4f}), "
                      f"Y({mesh_info['bounds'][2]:.4f}, {mesh_info['bounds'][3]:.4f}), "
                      f"Z({mesh_info['bounds'][4]:.4f}, {mesh_info['bounds'][5]:.4f})")
            else:
                print("✗ 网格信息不完整")
                return False
            
            print("\n✅ VTK文件导入修复验证成功！")
            return True
        else:
            print(f"✗ 导入VTK文件失败，返回类型: {type(mesh_data)}")
            return False
            
    except Exception as e:
        print(f"✗ 验证VTK文件导入时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("验证VTK文件导入修复...")
    success = verify_vtk_import_fix()
    if success:
        print("\n所有修复验证通过！")
    else:
        print("\n修复验证失败！")
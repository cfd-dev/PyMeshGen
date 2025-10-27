#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试VTK文件导入修复
"""

import os
import sys

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入必要的模块
from fileIO.vtk_io import read_vtk, reconstruct_mesh_from_vtk

def test_vtk_import():
    """测试VTK文件导入"""
    try:
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
        
        # 使用read_vtk函数读取VTK文件
        node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container = read_vtk(vtk_file)
        
        print(f"成功读取VTK文件")
        print(f"节点数量: {len(node_coords)}")
        print(f"单元数量: {len(cell_idx_container)}")
        
        # 检查节点坐标
        if node_coords:
            print(f"第一个节点坐标: {node_coords[0]}")
            if len(node_coords[0]) == 3:
                print("节点坐标包含x, y, z三个值")
            else:
                print(f"警告: 节点坐标只包含{len(node_coords[0])}个值")
        
        # 使用reconstruct_mesh_from_vtk函数重建网格
        mesh = reconstruct_mesh_from_vtk(node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container)
        
        print(f"成功重建网格对象")
        print(f"网格维度: {mesh.dim}")
        print(f"网格边界框: {mesh.bbox}")
        
        return True
            
    except Exception as e:
        print(f"测试VTK文件导入时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vtk_import()
    if success:
        print("VTK文件导入测试成功")
    else:
        print("VTK文件导入测试失败")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试cas文件导入功能
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fileIO.read_cas import parse_cas_to_unstr_grid, parse_fluent_msh, reconstruct_mesh_from_cas
from data_structure.basic_elements import Unstructured_Grid

def test_cas_import():
    """测试cas文件导入功能"""
    # 测试文件路径
    test_files = [
        "config/input/quad.cas",
        "config/input/concave.cas",
        "config/input/convex.cas"
    ]
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"警告: 测试文件不存在: {test_file}")
            continue
            
        print(f"\n测试文件: {test_file}")
        print("-" * 50)
        
        try:
            # 方法1: 使用parse_cas_to_unstr_grid直接转换
            unstr_grid = parse_cas_to_unstr_grid(test_file)
            print(f"方法1 - parse_cas_to_unstr_grid:")
            print(f"  节点数: {len(unstr_grid.node_coords)}")
            print(f"  单元数: {len(unstr_grid.cell_container)}")
            print(f"  边界节点数: {len(unstr_grid.boundary_nodes)}")
            
            # 检查边界信息
            bc_types = set()
            part_names = set()
            for node in unstr_grid.boundary_nodes:
                if hasattr(node, 'bc_type') and node.bc_type:
                    bc_types.add(node.bc_type)
                if hasattr(node, 'part_name') and node.part_name:
                    part_names.add(node.part_name)
            
            print(f"  边界类型: {bc_types}")
            print(f"  部件名称: {part_names}")
            
            # 方法2: 分步解析
            cas_data = parse_fluent_msh(test_file)
            unstr_grid2 = reconstruct_mesh_from_cas(cas_data)
            print(f"\n方法2 - parse_fluent_msh + reconstruct_mesh_from_cas:")
            print(f"  节点数: {len(unstr_grid2.node_coords)}")
            print(f"  单元数: {len(unstr_grid2.cell_container)}")
            print(f"  边界节点数: {len(unstr_grid2.boundary_nodes)}")
            
            # 验证两种方法结果一致
            nodes_match = len(unstr_grid.node_coords) == len(unstr_grid2.node_coords)
            cells_match = len(unstr_grid.cell_container) == len(unstr_grid2.cell_container)
            boundary_match = len(unstr_grid.boundary_nodes) == len(unstr_grid2.boundary_nodes)
            
            print(f"\n验证结果:")
            print(f"  节点数一致: {nodes_match}")
            print(f"  单元数一致: {cells_match}")
            print(f"  边界节点数一致: {boundary_match}")
            
            # 测试保存为VTK文件
            vtk_file = test_file.replace('.cas', '_test.vtk')
            unstr_grid.save_to_vtkfile(vtk_file)
            print(f"\n已保存为VTK文件: {vtk_file}")
            
            # 测试从VTK文件加载
            from fileIO.vtk_io import parse_vtk_msh
            loaded_grid = parse_vtk_msh(vtk_file)
            print(f"\n从VTK文件加载:")
            print(f"  节点数: {len(loaded_grid.node_coords)}")
            print(f"  单元数: {len(loaded_grid.cell_container)}")
            
            print(f"\n测试成功: {test_file}")
            
        except Exception as e:
            print(f"测试失败: {test_file}")
            print(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_cas_import()
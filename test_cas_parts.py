#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试CAS文件部件信息解析
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fileIO.read_cas import parse_fluent_msh, reconstruct_mesh_from_cas


def test_cas_parts_info():
    """测试CAS文件部件信息解析"""
    cas_file = r"c:\Users\HighOrderMesh\.vscode\PyMeshGen\examples\2d_simple\quad-hybrid.cas"
    
    print("=" * 60)
    print("测试CAS文件部件信息解析")
    print("=" * 60)
    
    # 解析CAS文件
    print("\n1. 解析CAS文件...")
    raw_cas_data = parse_fluent_msh(cas_file)
    print(f"   节点数: {raw_cas_data['node_count']}")
    print(f"   面数: {raw_cas_data['face_count']}")
    print(f"   单元数: {raw_cas_data['cell_count']}")
    print(f"   区域数: {len(raw_cas_data['zones'])}")
    
    # 检查区域信息
    print("\n2. 检查区域信息...")
    for zone_id, zone in raw_cas_data['zones'].items():
        print(f"   {zone_id}:")
        print(f"      类型: {zone['type']}")
        if 'bc_type' in zone:
            print(f"      边界条件: {zone['bc_type']}")
        if 'part_name' in zone:
            print(f"      部件名称: {zone['part_name']}")
        if 'data' in zone:
            print(f"      数据量: {len(zone['data'])}")
    
    # 重建网格
    print("\n3. 重建网格...")
    mesh = reconstruct_mesh_from_cas(raw_cas_data)
    print(f"   网格对象: {mesh}")
    print(f"   节点数: {len(mesh.node_coords)}")
    print(f"   单元数: {len(mesh.cells)}")
    
    # 检查parts_info
    print("\n4. 检查parts_info...")
    if hasattr(mesh, 'parts_info'):
        print(f"   parts_info类型: {type(mesh.parts_info)}")
        print(f"   parts_info内容: {mesh.parts_info}")
        if isinstance(mesh.parts_info, dict):
            print(f"   部件数量: {len(mesh.parts_info)}")
            for part_name, part_data in mesh.parts_info.items():
                print(f"      {part_name}:")
                print(f"         边界条件: {part_data.get('bc_type', 'N/A')}")
                print(f"         面数量: {len(part_data.get('faces', []))}")
    else:
        print("   错误: mesh对象没有parts_info属性")
    
    # 检查boundary_info
    print("\n5. 检查boundary_info...")
    if hasattr(mesh, 'boundary_info'):
        print(f"   boundary_info类型: {type(mesh.boundary_info)}")
        print(f"   boundary_info内容: {mesh.boundary_info}")
        if isinstance(mesh.boundary_info, dict):
            print(f"   边界数量: {len(mesh.boundary_info)}")
            for bc_name, bc_data in mesh.boundary_info.items():
                print(f"      {bc_name}:")
                print(f"         边界条件: {bc_data.get('bc_type', 'N/A')}")
                print(f"         面数量: {len(bc_data.get('faces', []))}")
    else:
        print("   错误: mesh对象没有boundary_info属性")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_cas_parts_info()

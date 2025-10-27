#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试从cas文件导入部件信息的功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.file_operations import FileOperations
from fileIO.read_cas import parse_cas_to_unstr_grid

def test_extract_parts_from_cas():
    """测试从cas文件提取部件信息的功能"""
    print("=" * 50)
    print("测试从cas文件提取部件信息的功能")
    print("=" * 50)
    
    # 创建文件操作对象
    file_ops = FileOperations(os.path.dirname(os.path.abspath(__file__)))
    
    # 测试文件路径
    test_file = "config/input/30p30n-small.cas"
    
    if not os.path.exists(test_file):
        print(f"错误：测试文件 {test_file} 不存在")
        return False
    
    # 解析cas文件
    print(f"正在解析cas文件: {test_file}")
    unstr_grid = parse_cas_to_unstr_grid(test_file)
    
    # 创建模拟的mesh_data
    mesh_data = {
        'type': 'cas',
        'unstr_grid': unstr_grid
    }
    
    # 提取部件信息
    print("正在提取部件信息...")
    parts_info = file_ops.extract_parts_from_cas(mesh_data)
    
    # 显示结果
    print(f"找到 {len(parts_info)} 个部件:")
    for i, part in enumerate(parts_info):
        print(f"部件 {i+1}:")
        print(f"  名称: {part['part_name']}")
        print(f"  边界条件类型: {part['bc_type']}")
        print(f"  面数量: {part['face_count']}")
        print(f"  节点数量: {len(part['nodes'])}")
        print(f"  单元数量: {len(part['cells'])}")
        print()
    
    return True

def test_import_mesh_with_parts():
    """测试导入cas文件并提取部件信息的功能"""
    print("=" * 50)
    print("测试导入cas文件并提取部件信息的功能")
    print("=" * 50)
    
    # 创建文件操作对象
    file_ops = FileOperations(os.path.dirname(os.path.abspath(__file__)))
    
    # 测试文件路径
    test_file = "config/input/30p30n-small.cas"
    
    if not os.path.exists(test_file):
        print(f"错误：测试文件 {test_file} 不存在")
        return False
    
    # 导入网格数据
    print(f"正在导入网格文件: {test_file}")
    mesh_data = file_ops.import_mesh(test_file)
    
    # 检查是否包含部件信息
    if 'parts_info' in mesh_data:
        print("成功提取部件信息!")
        parts_info = mesh_data['parts_info']
        print(f"找到 {len(parts_info)} 个部件:")
        for i, part in enumerate(parts_info):
            print(f"部件 {i+1}: {part['part_name']} (类型: {part['bc_type']})")
    else:
        print("警告：未找到部件信息")
        return False
    
    return True

if __name__ == "__main__":
    # 测试部件信息提取功能
    success1 = test_extract_parts_from_cas()
    print()
    
    # 测试导入网格并提取部件信息功能
    success2 = test_import_mesh_with_parts()
    
    if success1 and success2:
        print("所有测试通过!")
    else:
        print("测试失败!")
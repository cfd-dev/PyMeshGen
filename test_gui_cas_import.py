#!/usr/bin/env python3
"""
测试GUI中导入.cas文件功能
验证"not enough values to unpack (expected 3, got 2)"错误是否已修复
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.file_operations import FileOperations

def test_gui_cas_import():
    """测试GUI中导入.cas文件功能"""
    print("=== 测试GUI中导入.cas文件功能 ===\n")
    
    # 查找.cas文件
    test_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".cas"):
                test_files.append(os.path.join(root, file))
    
    if not test_files:
        print("未找到.cas文件，无法测试")
        return False
    
    # 测试第一个找到的.cas文件
    test_file = test_files[0]
    print(f"测试文件: {test_file}")
    
    try:
        # 创建FileOperations实例
        project_root = os.path.dirname(os.path.abspath(__file__))
        file_ops = FileOperations(project_root)
        
        # 导入.cas文件
        print("正在导入.cas文件...")
        unstr_grid = file_ops.import_mesh(test_file)
        print("✓ .cas文件导入成功")
        
        # 检查节点坐标
        node_coords = unstr_grid['node_coords']  # 修改为字典访问方式
        print(f"节点数量: {len(node_coords)}")
        
        # 检查前几个节点的坐标
        print("前5个节点坐标:")
        for i in range(min(5, len(node_coords))):
            coords = node_coords[i]
            print(f"  节点{i}: {coords}")
            
            # 检查坐标数量
            if len(coords) != 3:
                print(f"✗ 错误: 节点{i}坐标数量不正确，期望3个，实际{len(coords)}个")
                return False
            
            # 检查坐标值
            x, y, z = coords
            print(f"    x={x:.4f}, y={y:.4f}, z={z:.4f}")
        
        print("✓ 所有节点坐标检查通过")
        
        # 检查网格维度 - 从原始Unstructured_Grid对象获取
        original_grid = unstr_grid.get('unstr_grid')
        if original_grid and hasattr(original_grid, 'dim'):
            print(f"网格维度: {original_grid.dim}")
            if original_grid.dim != 3:
                print(f"✗ 错误: 网格维度不正确，期望3，实际{original_grid.dim}")
                return False
            print("✓ 网格维度检查通过")
        else:
            print("无法获取网格维度信息")
        
        # 检查边界框 - 从原始Unstructured_Grid对象获取
        if original_grid and hasattr(original_grid, 'bbox'):
            bbox = original_grid.bbox
            # 检查bbox的格式
            if isinstance(bbox, dict):
                # 如果是字典格式
                print(f"边界框: X({bbox.get('min_x', 0):.4f}, {bbox.get('max_x', 0):.4f}), Y({bbox.get('min_y', 0):.4f}, {bbox.get('max_y', 0):.4f})")
            elif isinstance(bbox, list) and len(bbox) >= 2:
                # 如果是列表格式
                if isinstance(bbox[0], list) or isinstance(bbox[0], tuple):
                    # 二维列表/元组
                    print(f"边界框: X({bbox[0][0]:.4f}, {bbox[0][1]:.4f}), Y({bbox[1][0]:.4f}, {bbox[1][1]:.4f})")
                else:
                    # 一维列表
                    print(f"边界框: X({bbox[0]:.4f}, {bbox[1]:.4f}), Y({bbox[2]:.4f}, {bbox[3]:.4f})")
            else:
                print(f"边界框格式未知: {bbox}")
        else:
            # 计算边界框
            import numpy as np
            coords = np.array(node_coords)
            min_x, min_y, min_z = np.min(coords, axis=0)
            max_x, max_y, max_z = np.max(coords, axis=0)
            print(f"边界框: X({min_x:.4f}, {max_x:.4f}), Y({min_y:.4f}, {max_y:.4f}), Z({min_z:.4f}, {max_z:.4f})")
        
        print("\n=== 所有检查通过 ===")
        print("GUI中导入.cas文件的'not enough values to unpack'错误已修复")
        return True
        
    except Exception as e:
        print(f"✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gui_cas_import()
    sys.exit(0 if success else 1)
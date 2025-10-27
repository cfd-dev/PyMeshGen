#!/usr/bin/env python3
"""
测试GUI中导入和显示.cas文件功能
验证修复后的KeyError: 'nodes'问题
测试导入和显示功能
"""

import os
import sys
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.file_operations import FileOperations
from gui.mesh_display import MeshDisplayArea

def test_cas_import_gui():
    """测试GUI中导入.cas文件功能"""
    print("测试GUI中导入.cas文件功能...")
    
    # 初始化FileOperations
    project_root = os.path.dirname(os.path.abspath(__file__))
    file_ops = FileOperations(project_root)
    
    # 查找测试用的.cas文件
    test_files = ["quad.cas", "concave.cas", "convex.cas"]
    found_files = []
    
    for test_file in test_files:
        # 在当前目录和子目录中查找
        for root, dirs, files in os.walk(project_root):
            if test_file in files:
                found_files.append(os.path.join(root, test_file))
                break
    
    if not found_files:
        print("错误：未找到测试用的.cas文件")
        return False
    
    print(f"找到测试文件: {found_files}")
    
    # 测试每个文件
    all_success = True
    mesh_data_list = []  # 存储导入的网格数据用于显示测试
    
    for file_path in found_files:
        print(f"\n测试文件: {os.path.basename(file_path)}")
        try:
            # 使用FileOperations导入网格
            result = file_ops.import_mesh(file_path)
            
            if result:
                print(f"✓ 成功导入 {os.path.basename(file_path)}")
                mesh_data_list.append((os.path.basename(file_path), result))
                
                # 检查返回的数据结构
                if isinstance(result, dict):
                    print(f"  - 返回类型: 字典")
                    print(f"  - 键: {list(result.keys())}")
                    
                    if 'node_coords' in result:
                        node_coords = result['node_coords']
                        print(f"  - 节点数量: {len(node_coords)}")
                        
                        # 检查节点坐标维度
                        if node_coords:
                            first_coord = node_coords[0]
                            if len(first_coord) == 3:
                                print(f"  - 节点坐标: 3D (x, y, z)")
                            else:
                                print(f"  - 节点坐标: {len(first_coord)}D")
                    
                    if 'unstr_grid' in result:
                        unstr_grid = result['unstr_grid']
                        print(f"  - 网格对象类型: {type(unstr_grid).__name__}")
                        
                        # 检查边界信息
                        if hasattr(unstr_grid, 'boundary_info'):
                            if unstr_grid.boundary_info:
                                print(f"  - 边界区域数量: {len(unstr_grid.boundary_info)}")
                                for zone_name, zone_data in unstr_grid.boundary_info.items():
                                    zone_type = zone_data.get('type', 'unspecified')
                                    face_count = len(zone_data.get('faces', []))
                                    print(f"    * {zone_name}: {zone_type} ({face_count} 个面)")
                            else:
                                print(f"  - 边界区域: 无")
                else:
                    print(f"  - 返回类型: {type(result).__name__}")
            else:
                print(f"✗ 导入失败: {os.path.basename(file_path)}")
                all_success = False
                
        except Exception as e:
            print(f"✗ 导入 {os.path.basename(file_path)} 时出错: {str(e)}")
            all_success = False
    
    # 测试显示功能
    if mesh_data_list and all_success:
        print("\n测试显示功能...")
        try:
            # 创建图形界面
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 创建MeshDisplayArea对象
            mesh_display = MeshDisplayArea(None)  # 父级设为None，因为我们只使用显示功能
            mesh_display.ax = ax
            
            display_success = True
            for filename, mesh_data in mesh_data_list:
                print(f"\n显示文件: {filename}")
                try:
                    # 设置网格数据
                    mesh_display.set_mesh_data(mesh_data)
                    print(f"  ✓ 成功设置网格数据")
                    
                    # 显示网格
                    mesh_display.display_mesh()
                    print(f"  ✓ 成功显示网格")
                    
                    # 保存图像
                    output_file = f"display_{filename.replace('.cas', '.png')}"
                    plt.savefig(output_file)
                    print(f"  ✓ 图像已保存到: {output_file}")
                    
                    # 清除当前显示，准备下一个测试
                    ax.clear()
                    
                except Exception as e:
                    print(f"  ✗ 显示失败: {str(e)}")
                    display_success = False
                    all_success = False
            
            # 关闭图形
            plt.close(fig)
            
            if display_success:
                print("\n✓ 所有文件显示测试通过！")
            
        except Exception as e:
            print(f"\n✗ 显示功能测试失败: {str(e)}")
            all_success = False
    
    return all_success

if __name__ == "__main__":
    success = test_cas_import_gui()
    if success:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败！")
        sys.exit(1)
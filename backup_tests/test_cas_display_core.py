#!/usr/bin/env python3
"""
测试GUI中显示.cas文件功能的核心逻辑
验证修复后的KeyError: 'nodes'问题
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.file_operations import FileOperations
from visualization.mesh_visualization import Visualization

def test_cas_display_core():
    """测试GUI中显示.cas文件功能的核心逻辑"""
    print("测试GUI中显示.cas文件功能的核心逻辑...")
    
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
    
    # 创建Visualization实例
    visual_obj = Visualization()
    
    # 测试每个文件
    all_success = True
    for file_path in found_files:
        print(f"\n测试文件: {os.path.basename(file_path)}")
        try:
            # 使用FileOperations导入网格
            result = file_ops.import_mesh(file_path)
            
            if result:
                print(f"✓ 成功导入 {os.path.basename(file_path)}")
                
                # 检查返回的数据结构
                if isinstance(result, dict):
                    print(f"  - 返回类型: 字典")
                    
                    # 对于.cas文件，需要构造符合plot_mesh期望的字典格式
                    if result.get('type') == 'cas':
                        unstr_grid = result['unstr_grid']
                        grid_dict = {
                            "nodes": result['node_coords'],
                            "zones": {}
                        }
                        
                        # 如果有边界信息，添加到zones中
                        if hasattr(unstr_grid, 'boundary_info') and unstr_grid.boundary_info:
                            for zone_name, zone_data in unstr_grid.boundary_info.items():
                                grid_dict["zones"][zone_name] = {
                                    "type": "faces",
                                    "bc_type": zone_data.get("type", "unspecified"),
                                    "data": zone_data.get("faces", [])
                                }
                            print(f"  - 边界区域数量: {len(unstr_grid.boundary_info)}")
                        
                        # 创建图形和轴
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # 创建Visualization实例并传入ax
                        visual_obj_with_ax = Visualization(ax=ax)
                        
                        # 尝试使用Visualization类的plot_mesh方法
                        try:
                            visual_obj_with_ax.plot_mesh(grid_dict)
                            print(f"✓ 成功显示 {os.path.basename(file_path)}")
                            
                            # 保存图像到文件
                            img_path = f"test_output_{os.path.basename(file_path).replace('.cas', '.png')}"
                            plt.savefig(img_path)
                            print(f"  - 图像已保存到: {img_path}")
                            plt.close()
                        except Exception as e:
                            print(f"✗ 显示失败: {os.path.basename(file_path)} - {str(e)}")
                            plt.close()
                            all_success = False
                    else:
                        print(f"  - 文件类型不是cas: {result.get('type')}")
                else:
                    print(f"  - 返回类型不是字典: {type(result).__name__}")
            else:
                print(f"✗ 导入失败: {os.path.basename(file_path)}")
                all_success = False
                
        except Exception as e:
            print(f"✗ 处理 {os.path.basename(file_path)} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            all_success = False
    
    return all_success

if __name__ == "__main__":
    success = test_cas_display_core()
    if success:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败！")
        sys.exit(1)
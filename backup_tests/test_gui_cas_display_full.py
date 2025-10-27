#!/usr/bin/env python3
"""
测试GUI中显示.cas文件功能
验证修复后的KeyError: 'nodes'问题
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.file_operations import FileOperations
from gui.mesh_display import MeshDisplayArea
from PyQt5.QtWidgets import QApplication

def test_cas_display_gui():
    """测试GUI中显示.cas文件功能"""
    print("测试GUI中显示.cas文件功能...")
    
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
    
    # 创建QApplication（如果不存在）
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # 创建MeshDisplayArea实例
    mesh_display = MeshDisplayArea()
    
    # 测试每个文件
    all_success = True
    for file_path in found_files:
        print(f"\n测试文件: {os.path.basename(file_path)}")
        try:
            # 使用FileOperations导入网格
            result = file_ops.import_mesh(file_path)
            
            if result:
                print(f"✓ 成功导入 {os.path.basename(file_path)}")
                
                # 尝试显示网格
                if mesh_display.display_mesh(result):
                    print(f"✓ 成功显示 {os.path.basename(file_path)}")
                else:
                    print(f"✗ 显示失败: {os.path.basename(file_path)}")
                    all_success = False
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
    success = test_cas_display_gui()
    if success:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败！")
        sys.exit(1)
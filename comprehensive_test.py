#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyMeshGen综合测试脚本
测试GUI的各种网格生成功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 添加子模块路径
sys.path.append(str(project_root / "fileIO"))
sys.path.append(str(project_root / "data_structure"))
sys.path.append(str(project_root / "meshsize"))
sys.path.append(str(project_root / "visualization"))
sys.path.append(str(project_root / "adfront2"))
sys.path.append(str(project_root / "optimize"))
sys.path.append(str(project_root / "utils"))

def test_triangle_mesh_generation():
    """测试三角形网格生成"""
    print("=== 测试三角形网格生成 ===")
    
    try:
        # 导入GUI模块
        from gui.gui_main import PyMeshGenGUI
        import tkinter as tk
        
        # 创建一个隐藏的根窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        
        # 创建GUI实例
        app = PyMeshGenGUI(root)
        
        # 设置测试参数 - 三角形网格
        app.input_file_var.set("./config/input/anw-hybrid.cas")
        app.output_file_var.set("./out/triangle_mesh.vtk")
        app.debug_level_var.set("1")
        app.mesh_type_var.set("1")  # 三角形网格
        app.viz_enabled_var.set(False)  # 禁用可视化以避免图形界面问题
        
        print("参数设置完成，开始生成三角形网格...")
        
        # 更新参数对象
        app.update_params_from_gui()
        
        # 运行网格生成
        app.run_mesh_generation()
        
        print("三角形网格生成完成！")
        
        # 检查输出文件是否存在
        output_file = Path("./out/triangle_mesh.vtk")
        if output_file.exists():
            print(f"成功生成三角形网格文件: {output_file}")
            return True
        else:
            print("错误：未找到生成的三角形网格文件")
            return False
            
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        return False
    finally:
        # 销毁根窗口
        try:
            root.destroy()
        except:
            pass

def test_mixed_mesh_generation():
    """测试混合网格生成"""
    print("\n=== 测试混合网格生成 ===")
    
    try:
        # 导入GUI模块
        from gui.gui_main import PyMeshGenGUI
        import tkinter as tk
        from data_structure.parameters import Parameters
        
        # 创建一个隐藏的根窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        
        # 创建GUI实例
        app = PyMeshGenGUI(root)
        
        # 为混合网格设置正确的参数对象
        app.params = Parameters("FROM_CASE_JSON", "./config/anw_mixed.json")
        
        # 设置测试参数 - 混合网格
        app.input_file_var.set(app.params.input_file)
        app.output_file_var.set("./out/mixed_mesh.vtk")
        app.debug_level_var.set(str(app.params.debug_level))
        app.mesh_type_var.set(str(app.params.mesh_type))  # 混合网格
        app.viz_enabled_var.set(app.params.viz_enabled)
        
        print("参数设置完成，开始生成混合网格...")
        
        # 更新参数对象
        app.update_params_from_gui()
        
        # 运行网格生成
        app.run_mesh_generation()
        
        print("混合网格生成完成！")
        
        # 检查输出文件是否存在
        output_file = Path("./out/mixed_mesh.vtk")
        if output_file.exists():
            print(f"成功生成混合网格文件: {output_file}")
            return True
        else:
            print("错误：未找到生成的混合网格文件")
            return False
            
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        return False
    finally:
        # 销毁根窗口
        try:
            root.destroy()
        except:
            pass

def main():
    """主测试函数"""
    print("开始PyMeshGen综合测试...")
    
    # 确保输出目录存在
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)
    
    # 运行各项测试
    test_results = []
    
    # 测试三角形网格生成
    result1 = test_triangle_mesh_generation()
    test_results.append(("三角形网格生成", result1))
    
    # 测试混合网格生成
    result2 = test_mixed_mesh_generation()
    test_results.append(("混合网格生成", result2))
    
    # 输出测试结果总结
    print("\n=== 测试结果总结 ===")
    all_passed = True
    for test_name, result in test_results:
        status = "通过" if result else "失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n所有测试通过！GUI网格生成功能正常工作。")
        return 0
    else:
        print("\n部分测试失败！请检查错误信息。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
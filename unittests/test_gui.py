#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试PyMeshGen GUI功能的脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent  # 现在在unittests目录下，需要指向上一级目录
sys.path.insert(0, str(project_root))

# 添加子模块路径
sys.path.append(str(project_root / "fileIO"))
sys.path.append(str(project_root / "data_structure"))
sys.path.append(str(project_root / "meshsize"))
sys.path.append(str(project_root / "visualization"))
sys.path.append(str(project_root / "adfront2"))
sys.path.append(str(project_root / "optimize"))
sys.path.append(str(project_root / "utils"))

def test_gui_mesh_generation():
    """测试GUI网格生成功能"""
    print("开始测试GUI网格生成功能...")
    
    try:
        # 导入GUI模块
        from gui.gui_main import PyMeshGenGUI
        import tkinter as tk
        
        # 创建一个隐藏的根窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        
        # 创建GUI实例
        app = PyMeshGenGUI(root)
        
        # 设置测试参数
        app.input_file_var.set("./config/input/anw-hybrid.cas")
        app.output_file_var.set("./out/test_gui_output.vtk")
        app.debug_level_var.set("1")
        app.mesh_type_var.set("1")  # 三角形网格
        app.viz_enabled_var.set(False)  # 禁用可视化以避免图形界面问题
        
        print("参数设置完成，开始生成网格...")
        
        # 更新参数对象
        app.update_params_from_gui()
        
        # 运行网格生成
        app.run_mesh_generation()
        
        print("网格生成完成！")
        
        # 检查输出文件是否存在
        output_file = Path("./out/test_gui_output.vtk")
        if output_file.exists():
            print(f"成功生成网格文件: {output_file}")
            print("GUI网格生成功能测试通过！")
            return True
        else:
            print("错误：未找到生成的网格文件")
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

if __name__ == "__main__":
    # 确保输出目录存在
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)
    
    # 运行测试
    success = test_gui_mesh_generation()
    
    if success:
        print("\n所有测试通过！")
        sys.exit(0)
    else:
        print("\n测试失败！")
        sys.exit(1)
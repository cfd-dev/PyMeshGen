#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMeshGen GUI启动脚本
用于启动优化后的PyMeshGen图形用户界面
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """主启动函数"""
    try:
        from gui.gui_main import main as gui_main
        
        print("正在启动PyMeshGen图形用户界面...")
        print("注意：属性面板显示功能已修复，现在可以正常显示部件信息。")
        print("=" * 50)
        
        # 启动GUI主程序
        gui_main()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有必要的模块都已安装。")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"运行错误: {e}")
        print("请检查错误信息并尝试解决问题。")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMeshGen GUI启动脚本
用于启动PyMeshGen图形用户界面
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """主启动函数"""
    print("正在启动PyMeshGen图形用户界面...")
    print("=" * 50)

    # Launch the PyQt version
    try:
        print("启动的GUI...")
        from pyqt_gui.gui_main import main as pyqt_main
        print("GUI加载成功，正在启动...")
        pyqt_main()

    except ImportError as e:
        print(f"GUI加载失败: {e}")
        print("请确保所有必要的GUI模块都已安装。")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"运行GUI时出错: {e}")
        print("请检查错误信息并尝试解决问题。")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
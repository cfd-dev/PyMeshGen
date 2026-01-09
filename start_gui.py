#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMeshGen GUI启动脚本
用于启动PyMeshGen图形用户界面
"""

import sys
import os
import traceback
from pathlib import Path

# 添加项目根目录和子目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 添加子目录到Python路径
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils", "gui"]:
    subdir_path = project_root / subdir
    if subdir_path.exists():
        sys.path.insert(0, str(subdir_path))

# 添加 meshio 到 Python 路径
meshio_path = project_root / "3rd_party" / "meshio" / "src"
if meshio_path.exists():
    sys.path.insert(0, str(meshio_path))

def main():
    """主启动函数"""
    print("正在启动PyMeshGen图形用户界面...")
    print("=" * 30)

    # Launch the PyQt version
    try:
        print("启动GUI...")
        from gui.gui_main import main as gui_main
        print("GUI加载成功，正在启动...")
        gui_main()

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
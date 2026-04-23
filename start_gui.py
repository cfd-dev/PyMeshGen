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
from utils.runtime_paths import add_existing_path, find_resource_root

# 添加项目根目录和子目录到Python路径
project_root = Path(__file__).resolve().parent
add_existing_path(project_root, prepend=True)

# 添加子目录到Python路径
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils", "gui"]:
    subdir_path = project_root / subdir
    add_existing_path(subdir_path)

# 添加 meshio 到 Python 路径
resource_root = find_resource_root(
    __file__,
    required_paths=("3rd_party/meshio/src",),
)
meshio_path = resource_root / "3rd_party" / "meshio" / "src"
add_existing_path(meshio_path, prepend=True)

def main():
    """主启动函数"""
    print("正在启动PyMeshGen图形用户界面...")
    print("=" * 30)

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

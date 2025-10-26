#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyMeshGen GUI 启动脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    try:
        # 导入并运行GUI主程序
        from gui.gui_main import main
        main()
    except Exception as e:
        print(f"启动GUI失败: {e}")
        sys.exit(1)
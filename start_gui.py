#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyMeshGen GUI启动脚本
使用重构后的GUI模块
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # 导入并运行GUI
    from gui.gui_main import main
    
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"导入GUI模块失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"运行GUI失败: {e}")
    sys.exit(1)
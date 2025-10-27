#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMeshGen GUI启动脚本
用于启动修复后的PyMeshGen图形用户界面
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from gui.gui_main import main
    
    print("正在启动PyMeshGen图形用户界面...")
    print("注意：属性面板显示功能已修复，现在可以正常显示部件信息。")
    print("=" * 50)
    
    # 启动GUI主程序
    main()
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有必要的模块都已安装。")
    sys.exit(1)
except Exception as e:
    print(f"运行错误: {e}")
    print("请检查错误信息并尝试解决问题。")
    sys.exit(1)
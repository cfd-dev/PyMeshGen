#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试几何模型树功能
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication
from gui.gui_main import SimplifiedPyMeshGenGUI


def test_geometry_tree():
    """测试几何模型树功能"""
    app = QApplication(sys.argv)
    
    gui = SimplifiedPyMeshGenGUI()
    gui.show()
    
    print("几何模型树功能测试:")
    print("1. GUI 已启动")
    print("2. 请导入一个几何文件（STEP/IGES/STL）")
    print("3. 检查左侧的几何模型树是否正确显示")
    print("4. 测试几何元素的显示/隐藏功能")
    print("5. 测试几何元素的选中功能")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_geometry_tree()

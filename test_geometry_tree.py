#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试统一模型树功能（几何和网格）
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication
from gui.gui_main import PyMeshGenGUI


def test_unified_model_tree():
    """测试统一模型树功能（几何和网格）"""
    app = QApplication(sys.argv)
    
    gui = PyMeshGenGUI()
    gui.show()
    
    print("=" * 60)
    print("统一模型树功能测试（几何和网格）")
    print("=" * 60)
    print("\n【几何模型测试】")
    print("1. 请导入一个几何文件（STEP/IGES）")
    print("2. 检查左侧的模型树是否正确显示几何元素（Vertices, Edges, Faces, Bodies）")
    print("3. 测试几何元素的显示/隐藏功能（勾选/取消勾选）")
    print("4. 测试几何元素的选中功能（点击元素查看属性）")
    print("\n【网格模型测试】")
    print("5. 请导入一个网格文件（VTK/VTU/CAS等）")
    print("6. 检查左侧的模型树是否正确显示网格部件（Parts）")
    print("7. 测试网格部件的显示/隐藏功能（勾选/取消勾选）")
    print("8. 测试网格部件的选中功能（点击部件查看属性）")
    print("\n【统一管理测试】")
    print("9. 验证几何和网格可以在同一个模型树中切换显示")
    print("10. 验证模型树可以正确管理不同类型的数据")
    print("=" * 60)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_unified_model_tree()

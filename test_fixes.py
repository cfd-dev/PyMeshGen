#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证修复后的功能
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'fileIO'))

def test_vtk_import():
    """测试VTK文件导入功能"""
    try:
        # 尝试导入read_vtk函数
        from vtk_io import read_vtk
        print("✓ VTK文件读取器导入成功")
        return True
    except ImportError as e:
        print(f"✗ VTK文件读取器导入失败: {e}")
        return False

def test_cas_import():
    """测试CAS文件导入功能"""
    try:
        from fileIO.read_cas import parse_cas_to_unstr_grid
        print("✓ CAS文件读取器导入成功")
        return True
    except Exception as e:
        print(f"✗ CAS文件读取器导入失败: {e}")
        return False

def test_icons():
    """测试图标文件是否存在"""
    icon_dir = os.path.join(os.path.dirname(__file__), "gui", "icons")
    required_icons = [
        "new.png", "open.png", "save.png", 
        "import.png", "export.png", 
        "generate.png", "display.png", "clear.png"
    ]
    
    missing_icons = []
    for icon in required_icons:
        icon_path = os.path.join(icon_dir, icon)
        if os.path.exists(icon_path):
            print(f"✓ {icon} 存在")
        else:
            print(f"✗ {icon} 缺失")
            missing_icons.append(icon)
    
    return len(missing_icons) == 0

def test_gui_import():
    """测试GUI模块导入"""
    try:
        from gui.gui_main import SimplifiedPyMeshGenGUI
        print("✓ GUI模块导入成功")
        return True
    except Exception as e:
        print(f"✗ GUI模块导入失败: {e}")
        return False

def test_file_operations():
    """测试文件操作功能"""
    try:
        from gui.file_operations import FileOperations
        print("✓ 文件操作模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 文件操作模块导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("PyMeshGen 功能测试")
    print("=" * 50)
    
    tests = [
        ("VTK文件导入", test_vtk_import),
        ("CAS文件导入", test_cas_import),
        ("图标文件", test_icons),
        ("GUI模块", test_gui_import),
        ("文件操作", test_file_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n测试 {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "通过" if result else "失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！修复成功！")
        return True
    else:
        print(f"\n⚠️ 有 {len(results) - passed} 个测试失败，需要进一步修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
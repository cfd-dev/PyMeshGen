#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 跳过耗时的网格生成测试
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录和子目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加子目录到Python路径
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils", "gui"]:
    subdir_path = project_root / subdir
    if subdir_path.exists():
        sys.path.insert(0, str(subdir_path))

# 导入所有测试模块
import test_vtk_file_io
import test_cas_file_io
import test_geometry_calculations
import test_mesh_quality
import test_gui_functionality
import test_core_functionality

# 创建测试套件，跳过耗时的网格生成测试
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# 添加快速测试
suite.addTests(loader.loadTestsFromModule(test_vtk_file_io))
suite.addTests(loader.loadTestsFromModule(test_cas_file_io))
suite.addTests(loader.loadTestsFromModule(test_geometry_calculations))
suite.addTests(loader.loadTestsFromModule(test_mesh_quality))
suite.addTests(loader.loadTestsFromModule(test_gui_functionality))
suite.addTests(loader.loadTestsFromModule(test_core_functionality))

# 运行测试
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# 打印总结
print("\n" + "="*70)
print("测试总结")
print("="*70)
print(f"总测试数: {result.testsRun}")
print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
print(f"失败: {len(result.failures)}")
print(f"错误: {len(result.errors)}")
print(f"跳过: {len(result.skipped)}")
print("="*70)

# 返回退出码
sys.exit(0 if result.wasSuccessful() else 1)
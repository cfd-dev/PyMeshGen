#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本 - 运行所有整合后的测试用例
"""

import sys
import os
import unittest
from pathlib import Path

# 添加项目根目录和子目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加子目录到Python路径
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils", "pyqt_gui"]:
    subdir_path = project_root / subdir
    if subdir_path.exists():
        sys.path.insert(0, str(subdir_path))

def run_all_tests():
    """运行所有测试用例"""
    print("=" * 70)
    print("开始运行所有测试用例...")
    print("=" * 70)

    # 获取unittests目录
    unittests_dir = Path(__file__).parent

    # 收集所有测试文件
    test_files = []
    for file in unittests_dir.glob("test_*.py"):
        if file.name != "run_tests.py" and file.name != __file__.split("/")[-1]:
            test_files.append(file)

    print(f"\n找到 {len(test_files)} 个测试文件:")
    for test_file in test_files:
        print(f"  - {test_file.name}")

    print("\n" + "=" * 70)
    print("开始执行测试...")
    print("=" * 70 + "\n")

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 加载所有测试
    for test_file in test_files:
        try:
            module_name = test_file.stem
            spec = __import__(module_name, fromlist=[''])
            tests = loader.loadTestsFromModule(spec)
            suite.addTests(tests)
        except Exception as e:
            print(f"警告: 无法加载测试文件 {test_file.name}: {e}")

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试总数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ 所有测试通过!")
    else:
        print("\n✗ 存在测试失败或错误")
        if result.failures:
            print("\n失败的测试:")
            for test, traceback in result.failures[:5]:
                print(f"  - {test}")
            if len(result.failures) > 5:
                print(f"  ... 还有 {len(result.failures) - 5} 个失败")

        if result.errors:
            print("\n错误的测试:")
            for test, traceback in result.errors[:5]:
                print(f"  - {test}")
            if len(result.errors) > 5:
                print(f"  ... 还有 {len(result.errors) - 5} 个错误")

    print("=" * 70)

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

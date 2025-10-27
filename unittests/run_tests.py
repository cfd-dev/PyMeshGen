#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行所有单元测试的脚本
"""

import unittest
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    """运行所有单元测试"""
    print("开始运行所有单元测试...")
    
    # 获取unittests目录
    unittests_dir = Path(__file__).parent
    
    # 发现并运行所有测试
    loader = unittest.TestLoader()
    suite = loader.discover(str(unittests_dir), pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()

def run_specific_test(test_name):
    """运行特定的测试"""
    print(f"运行特定测试: {test_name}")
    
    # 构建测试模块路径
    test_module = f"test_{test_name}"
    
    try:
        # 导入测试模块
        module = __import__(test_module)
        
        # 创建测试套件
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # 返回测试结果
        return result.wasSuccessful()
    except ImportError as e:
        print(f"无法导入测试模块 {test_module}: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 运行特定测试
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # 运行所有测试
        success = run_all_tests()
    
    # 根据测试结果退出
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
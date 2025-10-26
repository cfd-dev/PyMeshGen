#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试消息输出功能
验证消息能否同时输出到控制台和GUI信息窗口
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入消息模块
from utils.message import info, error, warning, debug, verbose, set_debug_level, DEBUG_LEVEL_DEBUG, DEBUG_LEVEL_VERBOSE

def test_message_output():
    """测试消息输出功能"""
    print("开始测试消息输出功能...")
    
    # 输出不同级别的消息
    info("这是一条信息消息")
    warning("这是一条警告消息")
    error("这是一条错误消息")
    debug("这是一条调试消息")
    verbose("这是一条详细消息")
    
    # 设置调试级别为DEBUG
    set_debug_level(DEBUG_LEVEL_DEBUG)
    print("\n设置调试级别为DEBUG后:")
    debug("这是一条调试消息(应该可见)")
    verbose("这是一条详细消息(应该不可见)")
    
    # 设置调试级别为VERBOSE
    set_debug_level(DEBUG_LEVEL_VERBOSE)
    print("\n设置调试级别为VERBOSE后:")
    debug("这是一条调试消息(应该可见)")
    verbose("这是一条详细消息(应该可见)")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_message_output()
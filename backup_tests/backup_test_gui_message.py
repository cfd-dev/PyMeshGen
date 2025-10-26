#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在GUI环境中测试消息输出功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入消息模块
from utils.message import info, error, warning, debug, verbose, set_debug_level, DEBUG_LEVEL_INFO

def test_gui_message():
    """测试GUI环境中的消息输出"""
    # 输出测试消息
    info("GUI消息测试开始")
    warning("这是一条警告消息")
    error("这是一条错误消息")
    info("GUI消息测试结束")
    
    print("测试消息已发送到GUI信息窗口")

if __name__ == "__main__":
    # 设置调试级别为INFO
    set_debug_level(DEBUG_LEVEL_INFO)
    test_gui_message()
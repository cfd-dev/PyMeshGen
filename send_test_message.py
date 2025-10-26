#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
向正在运行的GUI发送测试消息
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入消息模块
from utils.message import info, error, warning, debug, set_debug_level, DEBUG_LEVEL_DEBUG

def send_test_messages():
    """发送测试消息到GUI"""
    print("开始发送测试消息到GUI...")
    
    # 设置调试级别
    set_debug_level(DEBUG_LEVEL_DEBUG)
    
    # 发送不同类型的消息
    info("这是来自测试脚本的信息消息")
    warning("这是来自测试脚本的警告消息")
    error("这是来自测试脚本的错误消息")
    debug("这是来自测试脚本的调试消息")
    
    print("测试消息发送完成!")

if __name__ == "__main__":
    send_test_messages()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成测试输出信息到GUI
"""

import sys
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent  # 调整路径以适应新的文件位置
sys.path.insert(0, str(project_root))

# 导入消息模块
from utils.message import info, error, warning, debug, set_debug_level, DEBUG_LEVEL_DEBUG

def generate_test_output():
    """生成测试输出信息"""
    print("开始生成测试输出...")
    
    # 设置调试级别
    set_debug_level(DEBUG_LEVEL_DEBUG)
    
    # 生成不同类型的消息
    info("开始执行测试任务...")
    time.sleep(1)
    
    info("正在处理网格数据...")
    time.sleep(1)
    
    warning("检测到部分网格质量较低")
    time.sleep(1)
    
    debug("调试信息: 网格节点数=1000, 单元数=2000")
    time.sleep(1)
    
    info("网格优化完成")
    time.sleep(1)
    
    error("输出文件路径不存在，使用默认路径")
    time.sleep(1)
    
    info("测试任务执行完毕!")
    
    print("测试输出生成完成!")

if __name__ == "__main__":
    generate_test_output()
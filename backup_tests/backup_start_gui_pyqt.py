#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMeshGen GUI启动脚本
用于启动优化后的PyMeshGen图形用户界面（支持PyQt和Tkinter两种GUI框架）
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """主启动函数"""
    print("正在启动PyMeshGen图形用户界面...")
    print("=" * 50)
    
    # 首先尝试使用PyQt版本
    try:
        print("尝试启动PyQt版本的GUI...")
        from pyqt_gui.gui_main import main as pyqt_main
        print("PyQt版本GUI加载成功，正在启动...")
        pyqt_main()
        
    except ImportError as e:
        print(f"PyQt GUI加载失败: {e}")
        print("尝试启动Tkinter版本的GUI作为备份...")
        
        try:
            from gui.gui_main import main as tk_main
            print("Tkinter版本GUI加载成功，正在启动...")
            tk_main()
            
        except ImportError as e2:
            print(f"Tkinter GUI加载也失败: {e2}")
            print("请确保所有必要的GUI模块都已安装。")
            traceback.print_exc()
            sys.exit(1)
        except Exception as e2:
            print(f"运行Tkinter GUI时出错: {e2}")
            print("请检查错误信息并尝试解决问题。")
            traceback.print_exc()
            sys.exit(1)
    
    except Exception as e:
        print(f"运行PyQt GUI时出错: {e}")
        print("尝试启动Tkinter版本的GUI作为备份...")
        
        try:
            from gui.gui_main import main as tk_main
            print("Tkinter版本GUI加载成功，正在启动...")
            tk_main()
            
        except ImportError as e2:
            print(f"Tkinter GUI加载也失败: {e2}")
            print("请确保所有必要的GUI模块都已安装。")
            traceback.print_exc()
            sys.exit(1)
        except Exception as e2:
            print(f"运行Tkinter GUI时出错: {e2}")
            print("请检查错误信息并尝试解决问题。")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
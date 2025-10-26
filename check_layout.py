#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查GUI布局问题
"""

import tkinter as tk
from tkinter import ttk

def check_layout():
    """检查GUI布局"""
    root = tk.Tk()
    root.title("GUI布局检查")
    root.geometry("1200x800")
    
    # 创建主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 创建参数设置区域
    param_frame = ttk.LabelFrame(main_frame, text="参数设置")
    param_frame.pack(fill=tk.X, padx=5, pady=5)
    ttk.Label(param_frame, text="参数设置区域").pack()
    
    # 创建控制按钮区域
    control_frame = ttk.Frame(main_frame)
    control_frame.pack(fill=tk.X, padx=5, pady=5)
    ttk.Label(control_frame, text="控制按钮区域").pack()
    
    # 创建主内容框架
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # 创建网格显示区域
    mesh_frame = ttk.LabelFrame(content_frame, text="网格显示区")
    mesh_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
    ttk.Label(mesh_frame, text="网格显示区域 - 应该占据大部分空间").pack(expand=True)
    
    # 创建信息输出窗口
    info_frame = ttk.LabelFrame(content_frame, text="信息输出")
    info_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=5, pady=5)
    
    # 创建文本框和滚动条
    text_frame = ttk.Frame(info_frame)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    info_text = tk.Text(text_frame, wrap=tk.WORD, height=15)
    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=info_text.yview)
    info_text.configure(yscrollcommand=scrollbar.set)
    
    info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # 添加测试信息
    for i in range(20):
        info_text.insert(tk.END, f"这是第{i+1}行测试信息\n")
    
    # 添加清除按钮
    ttk.Button(info_frame, text="清除信息").pack(pady=5)
    
    # 创建状态栏
    status_var = tk.StringVar(value="就绪")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    root.mainloop()

if __name__ == "__main__":
    check_layout()
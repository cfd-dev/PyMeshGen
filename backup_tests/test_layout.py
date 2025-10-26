#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试GUI布局
"""

import tkinter as tk
from tkinter import ttk

def test_layout():
    """测试GUI布局"""
    root = tk.Tk()
    root.title("GUI布局测试")
    root.geometry("1200x800")
    
    # 创建主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 创建顶部区域
    top_frame = ttk.Frame(main_frame)
    top_frame.pack(fill=tk.X, padx=5, pady=5)
    ttk.Label(top_frame, text="顶部区域").pack()
    
    # 创建主内容框架
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # 创建网格显示区域
    mesh_frame = ttk.LabelFrame(content_frame, text="网格显示区")
    mesh_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
    ttk.Label(mesh_frame, text="网格显示区域").pack(expand=True)
    
    # 创建信息输出窗口
    info_frame = ttk.LabelFrame(content_frame, text="信息输出")
    info_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # 创建文本框和滚动条
    text_frame = ttk.Frame(info_frame)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    info_text = tk.Text(text_frame, wrap=tk.WORD, height=15)
    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=info_text.yview)
    info_text.configure(yscrollcommand=scrollbar.set)
    
    info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # 添加测试信息
    info_text.insert(tk.END, "这是信息输出窗口\n")
    info_text.insert(tk.END, "您应该能看到这个窗口\n")
    info_text.insert(tk.END, "如果看不到，请检查窗口大小和布局设置\n")
    
    # 添加清除按钮
    ttk.Button(info_frame, text="清除信息").pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    test_layout()
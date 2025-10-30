#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI基础组件模块
提供通用的UI元素和基础功能
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
from matplotlib.figure import Figure
from matplotlib.font_manager import FontManager


class BaseFrame:
    """基础框架类，提供通用功能"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
    def pack(self, **kwargs):
        """包装pack方法"""
        self.frame.pack(**kwargs)
        
    def grid(self, **kwargs):
        """包装grid方法"""
        self.frame.grid(**kwargs)


class MenuBar:
    """菜单栏类"""
    
    def __init__(self, root):
        self.root = root
        self.menubar = tk.Menu(root)
        self.root.config(menu=self.menubar)
        
    def create_file_menu(self, commands):
        """创建文件菜单"""
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="文件", menu=file_menu)
        
        for label, command in commands.items():
            if label == "---":
                file_menu.add_separator()
            else:
                file_menu.add_command(label=label, command=command)
                
        return file_menu
    
    def create_view_menu(self, commands):
        """创建视图菜单"""
        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="视图", menu=view_menu)
        
        for label, command in commands.items():
            if label == "---":
                view_menu.add_separator()
            else:
                view_menu.add_command(label=label, command=command)
                
        return view_menu
    
    def create_config_menu(self, commands):
        """创建配置菜单"""
        config_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="配置", menu=config_menu)
        
        for label, command in commands.items():
            if label == "---":
                config_menu.add_separator()
            else:
                config_menu.add_command(label=label, command=command)
            
        return config_menu
    
    def create_mesh_menu(self, commands):
        """创建网格菜单"""
        mesh_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="网格", menu=mesh_menu)
        
        for label, command in commands.items():
            if label == "---":
                mesh_menu.add_separator()
            else:
                mesh_menu.add_command(label=label, command=command)
                
        return mesh_menu
    
    def create_help_menu(self, commands):
        """创建帮助菜单"""
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="帮助", menu=help_menu)
        
        for label, command in commands.items():
            if label == "---":
                help_menu.add_separator()
            else:
                help_menu.add_command(label=label, command=command)
            
        return help_menu


class StatusBar:
    """状态栏类"""
    
    def __init__(self, root):
        self.root = root
        self.status_var = tk.StringVar(value="就绪")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def update_status(self, message):
        """更新状态栏信息"""
        self.status_var.set(message)
        self.root.update_idletasks()


class InfoOutput:
    """信息输出窗口类"""
    
    def __init__(self, parent):
        self.parent = parent
        self.create_info_output_area()
        self.create_info_context_menu()
        
    def create_info_output_area(self):
        """创建信息输出窗口"""
        # 创建信息输出框架
        self.frame = ttk.LabelFrame(self.parent, text="信息输出")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建文本框和滚动条
        text_frame = ttk.Frame(self.frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(text_frame, wrap=tk.WORD, height=8)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定右键事件
        self.info_text.bind("<Button-3>", self.show_info_context_menu)
        
        return self.frame
    
    def append_info_output(self, message):
        """添加信息到输出窗口，线程安全版本"""
        try:
            # 检查info_text是否存在且是有效的tkinter组件
            if hasattr(self, 'info_text') and self.info_text is not None:
                # 在主线程中执行GUI更新
                self.info_text.after(0, lambda: self._insert_message(message))
        except Exception:
            # 在测试环境或没有GUI的情况下忽略错误
            pass
    
    def _insert_message(self, message):
        """实际插入消息的方法（在主线程中调用）"""
        try:
            self.info_text.insert(tk.END, message + "\n")
            self.info_text.see(tk.END)  # 自动滚动到最新信息
        except tk.TclError:
            # 如果GUI组件已被销毁，则忽略错误
            pass
    
    def create_info_context_menu(self):
        """创建信息输出窗口的右键菜单"""
        self.info_context_menu = tk.Menu(self.parent, tearoff=0)
        self.info_context_menu.add_command(label="清除", command=self.clear_info_output, accelerator="Ctrl+D")
        self.info_context_menu.add_separator()
        self.info_context_menu.add_command(label="全选", command=self.select_all_info, accelerator="Ctrl+A")
        self.info_context_menu.add_command(label="复制", command=self.copy_info, accelerator="Ctrl+C")
        
        # 添加键盘快捷键绑定
        self.info_text.bind("<Control-a>", lambda e: self.select_all_info())
        self.info_text.bind("<Control-A>", lambda e: self.select_all_info())
        self.info_text.bind("<Control-c>", lambda e: self.copy_info())
        self.info_text.bind("<Control-C>", lambda e: self.copy_info())
        self.info_text.bind("<Control-d>", lambda e: self.clear_info_output())
        self.info_text.bind("<Control-D>", lambda e: self.clear_info_output())
        
    def show_info_context_menu(self, event):
        """显示信息输出窗口的右键菜单"""
        try:
            self.info_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.info_context_menu.grab_release()
            
    def select_all_info(self):
        """全选信息输出窗口的内容"""
        self.info_text.tag_add(tk.SEL, "1.0", tk.END)
        self.info_text.mark_set(tk.INSERT, "1.0")
        self.info_text.see(tk.INSERT)
        
        # 给出全选成功的反馈
        if hasattr(self.parent, 'status_bar') and hasattr(self.parent.status_bar, 'update_status'):
            self.parent.status_bar.update_status("已全选信息")
            
        return "break"
        
    def copy_info(self):
        """复制选中的信息"""
        try:
            # 尝试获取选中的文本
            selected_text = self.info_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.parent.clipboard_clear()
            self.parent.clipboard_append(selected_text)
        except tk.TclError:
            # 如果没有选中文本，则复制全部内容
            all_text = self.info_text.get("1.0", tk.END)
            self.parent.clipboard_clear()
            self.parent.clipboard_append(all_text)
            
        # 给出复制成功的反馈
        if hasattr(self.parent, 'status_bar') and hasattr(self.parent.status_bar, 'update_status'):
            self.parent.status_bar.update_status("已复制到剪贴板")

    def clear_info_output(self):
        """清除信息输出"""
        self.info_text.delete(1.0, tk.END)
        
        # 给出清除成功的反馈
        if hasattr(self.parent, 'status_bar') and hasattr(self.parent.status_bar, 'update_status'):
            self.parent.status_bar.update_status("已清除信息输出")
        
    def append_info_output(self, message):
        """添加信息到输出窗口，线程安全版本"""
        try:
            # 检查info_text是否存在且是有效的tkinter组件
            if hasattr(self, 'info_text') and self.info_text is not None:
                # 在主线程中执行GUI更新
                self.info_text.after(0, lambda: self._insert_message(message))
        except Exception:
            # 在测试环境或没有GUI的情况下忽略错误
            pass

    def _insert_message(self, message):
        """实际插入消息的方法（在主线程中调用）"""
        try:
            self.info_text.insert(tk.END, message + "\n")
            self.info_text.see(tk.END)  # 自动滚动到最新信息
        except tk.TclError:
            # 如果GUI组件已被销毁，则忽略错误
            pass

    def clear(self):
        """清空信息输出窗口（与clear_info_output一致，供主界面按钮调用）"""
        self.clear_info_output()


class FontManager:
    """字体管理器"""
    
    @staticmethod
    def setup_chinese_font():
        """设置matplotlib中文字体支持"""
        # 尝试设置中文字体
        chinese_fonts = ["Microsoft YaHei", "SimHei", "KaiTi", "SimSun", "FangSong"]
        found_font = None
        
        for font in chinese_fonts:
            try:
                # 检查字体是否可用
                font_manager = FontManager()
                available_fonts = [f.name for f in font_manager.ttflist]
                if font in available_fonts:
                    found_font = font
                    break
            except:
                continue
        
        if found_font:
            matplotlib.rcParams['font.sans-serif'] = [found_font]
            matplotlib.rcParams['axes.unicode_minus'] = False
        else:
            # 如果没有找到中文字体，使用默认字体但关闭unicode minus
            matplotlib.rcParams['axes.unicode_minus'] = False


class DialogBase:
    """对话框基类"""
    
    def __init__(self, parent, title, geometry="400x300"):
        self.parent = parent
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry(geometry)
        
        # 使对话框模态
        self.top.transient(parent)
        self.top.grab_set()
        
        # 创建按钮框架
        self.button_frame = ttk.Frame(self.top)
        self.button_frame.pack(side=tk.BOTTOM, pady=10)
        
        # 添加确定和取消按钮
        self.ok_button = ttk.Button(self.button_frame, text="确定", command=self.ok)
        self.ok_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(self.button_frame, text="取消", command=self.cancel)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # 结果变量
        self.result = None
    
    def ok(self):
        """确定按钮回调，子类应重写此方法"""
        self.top.destroy()
    
    def cancel(self):
        """取消按钮回调"""
        self.top.destroy()


class PanedWindowWrapper:
    """可调整大小的分隔窗口包装器"""
    
    def __init__(self, parent, orient=tk.VERTICAL):
        self.paned_window = ttk.PanedWindow(parent, orient=orient)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def add_frame(self, frame, weight=1):
        """添加框架到分隔窗口"""
        self.paned_window.add(frame, weight=weight)
        
    def pack(self, **kwargs):
        """包装pack方法"""
        self.paned_window.pack(**kwargs)


class LabelFrameWrapper:
    """标签框架包装器"""
    
    def __init__(self, parent, text):
        self.frame = ttk.LabelFrame(parent, text=text)
        
    def pack(self, **kwargs):
        """包装pack方法"""
        self.frame.pack(**kwargs)
        
    def grid(self, **kwargs):
        """包装grid方法"""
        self.frame.grid(**kwargs)
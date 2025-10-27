#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重构后的GUI主模块
整合各个功能模块，提供统一的用户界面
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import vtk

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from gui.gui_base import BaseFrame, MenuBar, StatusBar, InfoOutput, DialogBase
from gui.mesh_display import MeshDisplayArea
from gui.config_manager import ConfigManager, ConfigDialog
from gui.file_operations import FileOperations, ImportDialog, ExportDialog

# 导入项目模块
try:
    from data_structure.parameters import Parameters
    from PyMeshGen import PyMeshGen as MeshGenerator
    from data_structure.basic_elements import NodeElement as Node
except ImportError as e:
    print(f"导入项目模块失败: {e}")
    sys.exit(1)


class SimplifiedPyMeshGenGUI:
    """简化的PyMeshGen GUI主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PyMeshGen - 简化网格生成工具")
        self.root.geometry("1200x800")
        
        # 初始化项目根目录
        self.project_root = project_root
        
        # 初始化各个模块
        self.config_manager = ConfigManager(self.project_root)
        self.file_operations = FileOperations(self.project_root)
        
        # 初始化数据
        self.params = None
        self.mesh_generator = None
        self.current_mesh = None
        
        # 创建UI
        self.create_widgets()
        
        # 初始化状态
        self.update_status("就绪")
    
    def create_widgets(self):
        """创建UI组件"""
        # 创建菜单栏
        self.menu_bar = MenuBar(self.root)
        self.create_menu()
        
        # 创建工具栏（位于菜单栏下方）
        self.create_toolbar()
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧面板
        self.create_left_panel()
        
        # 创建中间面板（网格显示区域）
        self.create_center_panel()
        
        # 创建右侧面板（参数设置和部件管理）
        self.create_right_panel()
        
        # 创建状态栏（在底部创建，确保位于最底部）
        self.status_bar = StatusBar(self.root)
        
        # 创建信息输出窗口（位于状态栏上方）
        self.create_output_panel()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_toolbar(self):
        """创建工具栏"""
        # 工具栏框架
        self.toolbar_frame = ttk.Frame(self.root)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # 获取图标路径
        icon_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
        
        # 创建工具栏按钮
        self.toolbar_buttons = {}
        
        # 文件操作按钮
        try:
            new_icon = tk.PhotoImage(file=os.path.join(icon_dir, "new.png"))
            open_icon = tk.PhotoImage(file=os.path.join(icon_dir, "open.png"))
            save_icon = tk.PhotoImage(file=os.path.join(icon_dir, "save.png"))
            import_icon = tk.PhotoImage(file=os.path.join(icon_dir, "import.png"))
            export_icon = tk.PhotoImage(file=os.path.join(icon_dir, "export.png"))
        except:
            new_icon = open_icon = save_icon = import_icon = export_icon = None
        
        # 添加文件操作按钮
        new_btn = ttk.Button(self.toolbar_frame, text="新建", image=new_icon, compound=tk.TOP, command=self.new_config)
        new_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(new_btn, "创建新的网格配置\n快捷键: Ctrl+N")
        self.toolbar_buttons["new"] = new_btn
        
        open_btn = ttk.Button(self.toolbar_frame, text="打开", image=open_icon, compound=tk.TOP, command=self.open_config)
        open_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(open_btn, "打开已保存的网格配置文件\n支持格式: .json, .cfg\n快捷键: Ctrl+O")
        self.toolbar_buttons["open"] = open_btn
        
        save_btn = ttk.Button(self.toolbar_frame, text="保存", image=save_icon, compound=tk.TOP, command=self.save_config)
        save_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(save_btn, "保存当前网格配置到文件\n格式: .json\n快捷键: Ctrl+S")
        self.toolbar_buttons["save"] = save_btn
        
        # 添加分隔符
        ttk.Separator(self.toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        import_btn = ttk.Button(self.toolbar_frame, text="导入", image=import_icon, compound=tk.TOP, command=self.import_mesh)
        import_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(import_btn, "从外部文件导入网格数据\n支持格式: .vtk, .cas, .msh\n快捷键: Ctrl+I")
        self.toolbar_buttons["import"] = import_btn
        
        export_btn = ttk.Button(self.toolbar_frame, text="导出", image=export_icon, compound=tk.TOP, command=self.export_mesh)
        export_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(export_btn, "将当前网格导出到文件\n支持格式: .vtk, .obj, .stl\n快捷键: Ctrl+E")
        self.toolbar_buttons["export"] = export_btn
        
        # 添加分隔符
        ttk.Separator(self.toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 网格操作按钮
        try:
            generate_icon = tk.PhotoImage(file=os.path.join(icon_dir, "generate.png"))
            display_icon = tk.PhotoImage(file=os.path.join(icon_dir, "display.png"))
            clear_icon = tk.PhotoImage(file=os.path.join(icon_dir, "clear.png"))
        except:
            generate_icon = display_icon = clear_icon = None
        
        generate_btn = ttk.Button(self.toolbar_frame, text="生成", image=generate_icon, compound=tk.TOP, command=self.generate_mesh)
        generate_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(generate_btn, "根据当前配置生成网格\n支持三角形、四边形和混合网格\n快捷键: F5")
        self.toolbar_buttons["generate"] = generate_btn
        
        display_btn = ttk.Button(self.toolbar_frame, text="显示", image=display_icon, compound=tk.TOP, command=self.display_mesh)
        display_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(display_btn, "在显示区域显示当前网格\n支持缩放、旋转和平移操作\n快捷键: F6")
        self.toolbar_buttons["display"] = display_btn
        
        clear_btn = ttk.Button(self.toolbar_frame, text="清空", image=clear_icon, compound=tk.TOP, command=self.clear_mesh)
        clear_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(clear_btn, "清空当前网格数据\n注意: 此操作不可撤销\n快捷键: Delete")
        self.toolbar_buttons["clear"] = clear_btn
        
        # 保存图标引用（防止被垃圾回收）
        self.new_icon = new_icon
        self.open_icon = open_icon
        self.save_icon = save_icon
        self.import_icon = import_icon
        self.export_icon = export_icon
        self.generate_icon = generate_icon
        self.display_icon = display_icon
        self.clear_icon = clear_icon
    
    def create_tooltip(self, widget, text, delay=500):
        """为控件创建提示信息
        
        Args:
            widget: 要添加提示的控件
            text: 提示文本内容
            delay: 鼠标悬停后显示提示的延迟时间(毫秒)，默认500ms
        """
        tooltip = None
        timer_id = None
        
        def show_tooltip(event):
            nonlocal tooltip
            if tooltip is not None:
                return
                
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)  # 无边框窗口
            tooltip.wm_geometry(f"+{event.x_root+15}+{event.y_root+15}")  # 位置偏移15像素
            
            # 创建样式化的标签
            label = tk.Label(
                tooltip, 
                text=text, 
                background="#FFFFE0",  # 浅黄色背景
                foreground="#333333",  # 深灰色文字
                relief=tk.SOLID, 
                borderwidth=1,
                font=("Microsoft YaHei UI", 9),
                padx=8,
                pady=4,
                justify=tk.LEFT
            )
            label.pack()
            
            # 添加阴影效果
            tooltip.attributes("-topmost", True)  # 置于顶层
            
            widget.tooltip = tooltip
        
        def hide_tooltip(event):
            nonlocal tooltip, timer_id
            if timer_id is not None:
                widget.after_cancel(timer_id)
                timer_id = None
                
            if tooltip is not None:
                tooltip.destroy()
                tooltip = None
                if hasattr(widget, 'tooltip'):
                    del widget.tooltip
        
        def schedule_tooltip(event):
            nonlocal timer_id
            if timer_id is not None:
                widget.after_cancel(timer_id)
                
            timer_id = widget.after(delay, lambda: show_tooltip(event))
        
        # 绑定事件
        widget.bind("<Enter>", schedule_tooltip)
        widget.bind("<Leave>", hide_tooltip)
        widget.bind("<Motion>", hide_tooltip)  # 鼠标移动时隐藏，避免遮挡
    
    def create_file_tab(self, icon_dir):
        """创建文件操作选项卡"""
        # 创建选项卡框架
        file_tab = ttk.Frame(self.notebook)
        
        # 加载图标
        try:
            file_icon = tk.PhotoImage(file=os.path.join(icon_dir, "file.png"))
            folder_icon = tk.PhotoImage(file=os.path.join(icon_dir, "folder.png"))
        except:
            file_icon = None
            folder_icon = None
        
        # 添加选项卡
        self.notebook.add(file_tab, text="文件操作", image=file_icon, compound=tk.LEFT)
        
        # 创建按钮框架
        button_frame = ttk.Frame(file_tab)
        button_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建按钮并添加提示信息
        new_config_btn = ttk.Button(button_frame, text="新建配置", command=self.new_config)
        new_config_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(new_config_btn, "创建新的网格配置")
        
        open_config_btn = ttk.Button(button_frame, text="打开配置", command=self.open_config)
        open_config_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(open_config_btn, "打开已保存的网格配置文件")
        
        save_config_btn = ttk.Button(button_frame, text="保存配置", command=self.save_config)
        save_config_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(save_config_btn, "保存当前网格配置到文件")
        
        import_mesh_btn = ttk.Button(button_frame, text="导入网格", command=self.import_mesh)
        import_mesh_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(import_mesh_btn, "从外部文件导入网格数据")
        
        export_mesh_btn = ttk.Button(button_frame, text="导出网格", command=self.export_mesh)
        export_mesh_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(export_mesh_btn, "将当前网格导出到文件")
        
        # 保存图标引用（防止被垃圾回收）
        self.file_icon = file_icon
        self.folder_icon = folder_icon
    
    def create_config_tab(self, icon_dir):
        """创建配置选项卡"""
        # 创建选项卡框架
        config_tab = ttk.Frame(self.notebook)
        
        # 加载图标
        try:
            config_icon = tk.PhotoImage(file=os.path.join(icon_dir, "config.png"))
        except:
            config_icon = None
        
        # 添加选项卡
        self.notebook.add(config_tab, text="参数配置", image=config_icon, compound=tk.LEFT)
        
        # 创建参数设置框架
        params_frame = ttk.Frame(config_tab)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 调试级别
        ttk.Label(params_frame, text="调试级别:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.debug_level_var = tk.StringVar(value="0")
        debug_level_combo = ttk.Combobox(params_frame, textvariable=self.debug_level_var, 
                                        values=["0", "1", "2"], width=10)
        debug_level_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # 输入文件
        ttk.Label(params_frame, text="输入文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_file_var = tk.StringVar()
        input_file_entry = ttk.Entry(params_frame, textvariable=self.input_file_var, width=20)
        input_file_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # 输出文件
        ttk.Label(params_frame, text="输出文件:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.output_file_var = tk.StringVar()
        output_file_entry = ttk.Entry(params_frame, textvariable=self.output_file_var, width=20)
        output_file_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # 网格类型
        ttk.Label(params_frame, text="网格类型:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        mesh_type_frame = ttk.Frame(params_frame)
        mesh_type_frame.grid(row=3, column=1, padx=5, pady=2)
        self.mesh_type_var = tk.StringVar(value="1")
        ttk.Radiobutton(mesh_type_frame, text="三角形", variable=self.mesh_type_var, 
                       value="1").pack(side=tk.LEFT)
        ttk.Radiobutton(mesh_type_frame, text="直角三角形", variable=self.mesh_type_var, 
                       value="2").pack(side=tk.LEFT)
        ttk.Radiobutton(mesh_type_frame, text="混合网格", variable=self.mesh_type_var, 
                       value="3").pack(side=tk.LEFT)
        
        # 可视化开关
        self.viz_enabled_var = tk.BooleanVar(value=True)
        viz_check = ttk.Checkbutton(params_frame, text="启用可视化", variable=self.viz_enabled_var)
        viz_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.create_tooltip(viz_check, "是否启用网格可视化功能")
        
        # 操作按钮
        params_btn = ttk.Button(params_frame, text="参数设置", command=self.edit_params)
        params_btn.grid(row=5, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        self.create_tooltip(params_btn, "打开高级参数设置对话框")
        
        # 保存图标引用（防止被垃圾回收）
        self.config_icon = config_icon
    
    def create_mesh_tab(self, icon_dir):
        """创建网格操作选项卡"""
        # 创建选项卡框架
        mesh_tab = ttk.Frame(self.notebook)
        
        # 加载图标
        try:
            mesh_icon = tk.PhotoImage(file=os.path.join(icon_dir, "mesh.png"))
        except:
            mesh_icon = None
        
        # 添加选项卡
        self.notebook.add(mesh_tab, text="网格操作", image=mesh_icon, compound=tk.LEFT)
        
        # 创建按钮框架
        button_frame = ttk.Frame(mesh_tab)
        button_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建按钮并添加提示信息
        generate_mesh_btn = ttk.Button(button_frame, text="生成网格", command=self.generate_mesh)
        generate_mesh_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(generate_mesh_btn, "根据当前配置生成网格")
        
        display_mesh_btn = ttk.Button(button_frame, text="显示网格", command=self.display_mesh)
        display_mesh_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(display_mesh_btn, "在显示区域显示当前网格")
        
        clear_mesh_btn = ttk.Button(button_frame, text="清空网格", command=self.clear_mesh)
        clear_mesh_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(clear_mesh_btn, "清空当前网格数据")
        
        reset_view_btn = ttk.Button(button_frame, text="重置视图", command=self.reset_view)
        reset_view_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(reset_view_btn, "重置视图到初始状态")
        
        fit_view_btn = ttk.Button(button_frame, text="适应视图", command=self.fit_view)
        fit_view_btn.pack(fill=tk.X, padx=5, pady=2)
        self.create_tooltip(fit_view_btn, "调整视图以适应网格大小")
        
        # 保存图标引用（防止被垃圾回收）
        self.mesh_icon = mesh_icon
    
    def create_part_tab(self, icon_dir):
        """创建部件管理选项卡"""
        # 创建选项卡框架
        part_tab = ttk.Frame(self.notebook)
        
        # 加载图标
        try:
            part_icon = tk.PhotoImage(file=os.path.join(icon_dir, "part.png"))
        except:
            part_icon = None
        
        # 添加选项卡
        self.notebook.add(part_tab, text="部件管理", image=part_icon, compound=tk.LEFT)
        
        # 创建部件列表框架
        list_frame = ttk.Frame(part_tab)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建部件列表
        self.parts_listbox = tk.Listbox(list_frame, height=6)
        self.parts_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.create_tooltip(self.parts_listbox, "显示当前配置中的所有部件")
        
        # 部件操作按钮
        parts_button_frame = ttk.Frame(list_frame)
        parts_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        add_part_btn = ttk.Button(parts_button_frame, text="添加部件", command=self.add_part)
        add_part_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(add_part_btn, "添加新的部件到配置中")
        
        remove_part_btn = ttk.Button(parts_button_frame, text="删除部件", command=self.remove_part)
        remove_part_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(remove_part_btn, "删除选中的部件")
        
        edit_part_btn = ttk.Button(parts_button_frame, text="编辑部件", command=self.edit_part)
        edit_part_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(edit_part_btn, "编辑选中的部件属性")
        
        # 保存图标引用（防止被垃圾回收）
        self.part_icon = part_icon
    
    def create_menu(self):
        """创建菜单"""
        # 文件菜单
        file_commands = {
            "新建配置": self.new_config,
            "打开配置": self.open_config,
            "保存配置": self.save_config,
            "---": None,
            "导入网格": self.import_mesh,
            "导出网格": self.export_mesh,
            "---": None,
            "退出": self.on_closing
        }
        self.menu_bar.create_file_menu(file_commands)
        
        # 配置菜单
        config_commands = {
            "参数设置": self.edit_params,
            "清空网格": self.clear_mesh
        }
        self.menu_bar.create_config_menu(config_commands)
        
        # 网格菜单
        mesh_commands = {
            "生成网格": self.generate_mesh,
            "显示网格": self.display_mesh,
            "---": None,
            "重置视图": self.reset_view,
            "适应视图": self.fit_view
        }
        self.menu_bar.create_mesh_menu(mesh_commands)
        
        # 帮助菜单
        help_commands = {
            "关于": self.show_about
        }
        self.menu_bar.create_help_menu(help_commands)
    
    def create_left_panel(self):
        """创建左侧面板"""
        # 左侧面板框架
        self.left_panel = ttk.Frame(self.main_frame, width=200)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_panel.pack_propagate(False)
        
        # 创建项目信息面板
        info_frame = ttk.LabelFrame(self.left_panel, text="项目信息")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 显示当前配置信息
        ttk.Label(info_frame, text="当前配置: 未加载").pack(anchor=tk.W, padx=5, pady=2)
        self.config_label = ttk.Label(info_frame, text="网格类型: 未设置")
        self.config_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # 显示网格状态
        mesh_frame = ttk.LabelFrame(self.left_panel, text="网格状态")
        mesh_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.mesh_status_label = ttk.Label(mesh_frame, text="状态: 未生成")
        self.mesh_status_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.mesh_info_label = ttk.Label(mesh_frame, text="节点数: 0\n单元数: 0")
        self.mesh_info_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # 创建视图控制面板
        view_frame = ttk.LabelFrame(self.left_panel, text="视图控制")
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        
        reset_view_btn = ttk.Button(view_frame, text="重置视图", command=self.reset_view)
        reset_view_btn.pack(fill=tk.X, padx=5, pady=2)
        
        fit_view_btn = ttk.Button(view_frame, text="适应视图", command=self.fit_view)
        fit_view_btn.pack(fill=tk.X, padx=5, pady=2)
    
    def create_center_panel(self):
        """创建中间面板（网格显示区域）"""
        # 中间面板框架
        self.center_panel = ttk.Frame(self.main_frame)
        self.center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建网格显示区域
        self.mesh_display = MeshDisplayArea(self.center_panel)
        self.mesh_display.pack(fill=tk.BOTH, expand=True)
    
    def create_right_panel(self):
        """创建右侧面板（参数设置和部件管理）"""
        # 右侧面板框架
        self.right_panel = ttk.Frame(self.main_frame, width=250)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.right_panel.pack_propagate(False)
        
        # 创建参数设置面板
        params_frame = ttk.LabelFrame(self.right_panel, text="参数设置")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 调试级别
        ttk.Label(params_frame, text="调试级别:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.debug_level_var = tk.StringVar(value="0")
        debug_level_combo = ttk.Combobox(params_frame, textvariable=self.debug_level_var, 
                                        values=["0", "1", "2"], width=10)
        debug_level_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # 输入文件
        ttk.Label(params_frame, text="输入文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_file_var = tk.StringVar()
        input_file_entry = ttk.Entry(params_frame, textvariable=self.input_file_var, width=15)
        input_file_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # 输出文件
        ttk.Label(params_frame, text="输出文件:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.output_file_var = tk.StringVar()
        output_file_entry = ttk.Entry(params_frame, textvariable=self.output_file_var, width=15)
        output_file_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # 网格类型
        ttk.Label(params_frame, text="网格类型:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        mesh_type_frame = ttk.Frame(params_frame)
        mesh_type_frame.grid(row=3, column=1, padx=5, pady=2)
        self.mesh_type_var = tk.StringVar(value="1")
        ttk.Radiobutton(mesh_type_frame, text="三角形", variable=self.mesh_type_var, 
                       value="1").pack(side=tk.LEFT)
        ttk.Radiobutton(mesh_type_frame, text="直角三角形", variable=self.mesh_type_var, 
                       value="2").pack(side=tk.LEFT)
        ttk.Radiobutton(mesh_type_frame, text="混合", variable=self.mesh_type_var, 
                       value="3").pack(side=tk.LEFT)
        
        # 可视化开关
        self.viz_enabled_var = tk.BooleanVar(value=True)
        viz_check = ttk.Checkbutton(params_frame, text="启用可视化", variable=self.viz_enabled_var)
        viz_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # 参数设置按钮
        params_btn = ttk.Button(params_frame, text="高级参数设置", command=self.edit_params)
        params_btn.grid(row=5, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        # 创建部件管理面板
        parts_frame = ttk.LabelFrame(self.right_panel, text="部件管理")
        parts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建部件列表
        self.parts_listbox = tk.Listbox(parts_frame, height=8)
        self.parts_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 部件操作按钮
        parts_button_frame = ttk.Frame(parts_frame)
        parts_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        add_part_btn = ttk.Button(parts_button_frame, text="添加", command=self.add_part)
        add_part_btn.pack(side=tk.LEFT, padx=2)
        
        remove_part_btn = ttk.Button(parts_button_frame, text="删除", command=self.remove_part)
        remove_part_btn.pack(side=tk.LEFT, padx=2)
        
        edit_part_btn = ttk.Button(parts_button_frame, text="编辑", command=self.edit_part)
        edit_part_btn.pack(side=tk.LEFT, padx=2)
    
    def create_output_panel(self):
        """创建信息输出面板"""
        # 创建底部面板框架
        self.bottom_panel = ttk.Frame(self.root, height=150)
        self.bottom_panel.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.bottom_panel.pack_propagate(False)  # 防止框架被内容压缩
        
        # 创建信息输出区域
        self.info_output = InfoOutput(self.bottom_panel)
        
        # 确保InfoOutput的框架被正确布局
        self.info_output.frame.pack(fill=tk.BOTH, expand=True)
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_bar.update_status(message)
    
    def log_info(self, message):
        """记录信息"""
        self.info_output.append_info_output(f"[INFO] {message}")
    
    def log_warning(self, message):
        """记录警告"""
        self.info_output.append_info_output(f"[WARNING] {message}")
    
    def log_error(self, message):
        """记录错误"""
        self.info_output.append_info_output(f"[ERROR] {message}")
    
    def new_config(self):
        """新建配置"""
        # 创建默认配置字典
        default_config = {
            "debug_level": 0,
            "input_file": "",
            "output_file": "",
            "mesh_type": 1,
            "viz_enabled": True,
            "parts": []
        }
        
        # 创建临时配置文件
        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(default_config, f, indent=2)
            temp_file = f.name
        
        # 使用临时文件创建参数对象
        self.params = Parameters("FROM_CASE_JSON", temp_file)
        
        # 删除临时文件
        import os
        os.unlink(temp_file)
        
        self.update_params_display()
        self.log_info("已创建新配置")
        self.update_status("已创建新配置")
    
    def open_config(self):
        """打开配置"""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                config = self.config_manager.load_config(file_path)
                self.params = self.config_manager.create_params_from_config(config)
                self.update_params_display()
                self.log_info(f"已加载配置文件: {file_path}")
                self.update_status("已加载配置文件")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置文件失败: {str(e)}")
                self.log_error(f"加载配置文件失败: {str(e)}")
    
    def save_config(self):
        """保存配置"""
        if not self.params:
            messagebox.showwarning("警告", "没有可保存的配置")
            return
            
        from tkinter import filedialog
        
        file_path = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                config = self.config_manager.create_config_from_params(self.params)
                self.config_manager.save_config(config, file_path)
                self.log_info(f"已保存配置文件: {file_path}")
                self.update_status("已保存配置文件")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置文件失败: {str(e)}")
                self.log_error(f"保存配置文件失败: {str(e)}")
    
    def edit_params(self):
        """编辑参数"""
        if not self.params:
            messagebox.showwarning("警告", "请先创建或加载配置")
            return
            
        try:
            config = self.config_manager.create_config_from_params(self.params)
            dialog = ConfigDialog(self.root, config)
            self.root.wait_window(dialog.top)
            
            if dialog.result:
                self.params = self.config_manager.create_params_from_config(dialog.result)
                self.update_params_display()
                self.log_info("已更新参数配置")
                self.update_status("已更新参数配置")
        except Exception as e:
            messagebox.showerror("错误", f"编辑参数失败: {str(e)}")
            self.log_error(f"编辑参数失败: {str(e)}")
    
    def generate_mesh(self):
        """生成网格"""
        if not self.params:
            messagebox.showwarning("警告", "请先创建或加载配置")
            return
            
        try:
            self.update_status("正在生成网格...")
            self.log_info("开始生成网格")
            
            # 更新网格状态
            self.mesh_status_label.config(text="状态: 正在生成...")
            
            # 创建网格生成器
            self.mesh_generator = MeshGenerator(self.params)
            
            # 生成网格
            self.current_mesh = self.mesh_generator.generate()
            
            # 更新网格状态
            self.mesh_status_label.config(text="状态: 已生成")
            
            # 获取网格信息
            if hasattr(self.current_mesh, 'nodes') and hasattr(self.current_mesh, 'elements'):
                node_count = len(self.current_mesh.nodes)
                element_count = len(self.current_mesh.elements)
                self.mesh_info_label.config(text=f"节点数: {node_count}\n单元数: {element_count}")
            
            self.log_info("网格生成完成")
            self.update_status("网格生成完成")
            
            # 显示网格
            self.display_mesh()
        except Exception as e:
            self.mesh_status_label.config(text="状态: 生成失败")
            messagebox.showerror("错误", f"生成网格失败: {str(e)}")
            self.log_error(f"生成网格失败: {str(e)}")
            self.update_status("生成网格失败")
    
    def display_mesh(self):
        """显示网格"""
        if not self.current_mesh:
            messagebox.showwarning("警告", "请先生成网格")
            return
            
        try:
            self.update_status("正在显示网格...")
            self.log_info("开始显示网格")
            
            # 更新网格状态
            self.mesh_status_label.config(text="状态: 正在显示...")
            
            # 显示网格
            self.mesh_display.display_mesh(self.current_mesh)
            
            # 更新网格状态
            self.mesh_status_label.config(text="状态: 已显示")
            
            self.log_info("网格显示完成")
            self.update_status("网格显示完成")
        except Exception as e:
            self.mesh_status_label.config(text="状态: 显示失败")
            messagebox.showerror("错误", f"显示网格失败: {str(e)}")
            self.log_error(f"显示网格失败: {str(e)}")
            self.update_status("显示网格失败")
    
    def import_mesh(self):
        """导入网格"""
        dialog = ImportDialog(self.root, self.file_operations)
        self.root.wait_window(dialog.top)
        
        if dialog.result:
            try:
                self.current_mesh = self.file_operations.import_mesh(dialog.result["file_path"])
                
                # 更新网格显示区域的网格数据
                self.mesh_display.mesh_data = self.current_mesh
                
                # 更新网格状态
                self.mesh_status_label.config(text="状态: 已导入")
                
                # 获取网格信息
                if isinstance(self.current_mesh, dict):
                    # 处理字典类型的网格数据
                    node_count = self.current_mesh.get('num_points', 0)
                    element_count = self.current_mesh.get('num_cells', 0)
                    self.mesh_info_label.config(text=f"节点数: {node_count}\n单元数: {element_count}")
                elif hasattr(self.current_mesh, 'num_points') and hasattr(self.current_mesh, 'num_cells'):
                    node_count = self.current_mesh.num_points
                    element_count = self.current_mesh.num_cells
                    self.mesh_info_label.config(text=f"节点数: {node_count}\n单元数: {element_count}")
                elif hasattr(self.current_mesh, 'node_coords') and hasattr(self.current_mesh, 'cells'):
                    node_count = len(self.current_mesh.node_coords)
                    element_count = len(self.current_mesh.cells)
                    self.mesh_info_label.config(text=f"节点数: {node_count}\n单元数: {element_count}")
                elif hasattr(self.current_mesh, 'nodes') and hasattr(self.current_mesh, 'elements'):
                    node_count = len(self.current_mesh.nodes)
                    element_count = len(self.current_mesh.elements)
                    self.mesh_info_label.config(text=f"节点数: {node_count}\n单元数: {element_count}")
                
                if dialog.result["preview"]:
                    self.display_mesh()
                
                self.log_info(f"已导入网格文件: {dialog.result['file_path']}")
                self.update_status("已导入网格文件")
            except Exception as e:
                self.mesh_status_label.config(text="状态: 导入失败")
                messagebox.showerror("错误", f"导入网格失败: {str(e)}")
                self.log_error(f"导入网格失败: {str(e)}")
                self.update_status("导入网格失败")
    
    def export_mesh(self):
        """导出网格"""
        if not self.current_mesh:
            messagebox.showwarning("警告", "没有可导出的网格")
            return
            
        dialog = ExportDialog(self.root, self.file_operations, self.current_mesh)
        self.root.wait_window(dialog.top)
        
        if dialog.result:
            self.log_info(f"已导出网格文件: {dialog.result['file_path']}")
            self.update_status("已导出网格文件")
    
    def clear_mesh(self):
        """清空网格"""
        self.current_mesh = None
        self.mesh_display.clear()
        
        # 更新网格状态
        self.mesh_status_label.config(text="状态: 未生成")
        self.mesh_info_label.config(text="节点数: 0\n单元数: 0")
        
        self.log_info("已清空网格")
        self.update_status("已清空网格")
    
    def reset_view(self):
        """重置视图"""
        self.mesh_display.reset_view()
        self.log_info("已重置视图")
        self.update_status("已重置视图")
    
    def fit_view(self):
        """适应视图"""
        self.mesh_display.fit_view()
        self.log_info("已适应视图")
        self.update_status("已适应视图")
    
    def add_part(self):
        """添加部件"""
        if not self.params:
            messagebox.showwarning("警告", "请先创建或加载配置")
            return
            
        # 这里可以添加一个对话框来设置部件参数
        # 为了简化，我们使用默认值
        part_name = f"部件{len(self.params.part_params) + 1}"
        part = Part(part_name)
        self.params.part_params.append(part)
        
        # 更新部件列表
        self.update_parts_list()
        
        # 切换到部件选项卡
        self.notebook.select(self.part_tab)
        
        self.log_info(f"已添加部件: {part_name}")
        self.update_status("已添加部件")
    
    def remove_part(self):
        """删除部件"""
        if not self.params:
            messagebox.showwarning("警告", "请先创建或加载配置")
            return
            
        selection = self.parts_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的部件")
            return
            
        index = selection[0]
        if 0 <= index < len(self.params.part_params):
            part_name = self.params.part_params[index].part_name
            self.params.part_params.pop(index)
            
            # 更新部件列表
            self.update_parts_list()
            
            self.log_info(f"已删除部件: {part_name}")
            self.update_status("已删除部件")
    
    def edit_part(self):
        """编辑部件"""
        if not self.params:
            messagebox.showwarning("警告", "请先创建或加载配置")
            return
            
        selection = self.parts_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要编辑的部件")
            return
            
        index = selection[0]
        if 0 <= index < len(self.params.part_params):
            # 这里可以添加一个对话框来编辑部件参数
            # 为了简化，我们只显示一个消息
            part_name = self.params.part_params[index].part_name
            messagebox.showinfo("信息", f"编辑部件: {part_name}")
    
    def update_params_display(self):
        """更新参数显示"""
        if not self.params:
            return
            
        # 更新选项卡中的参数显示
        self.debug_level_var.set(str(self.params.debug_level))
        self.input_file_var.set(self.params.input_file or "")
        self.output_file_var.set(self.params.output_file or "")
        self.viz_enabled_var.set(self.params.viz_enabled)
        self.mesh_type_var.set(str(self.params.mesh_type))
        
        # 更新左侧面板的配置信息
        mesh_type_text = "未设置"
        if self.params.mesh_type == 1:
            mesh_type_text = "三角形"
        elif self.params.mesh_type == 2:
            mesh_type_text = "直角三角形"
        elif self.params.mesh_type == 3:
            mesh_type_text = "混合网格"
        
        self.config_label.config(text=f"网格类型: {mesh_type_text}")
        
        # 更新部件列表
        self.update_parts_list()
    
    def update_parts_list(self):
        """更新部件列表"""
        # 更新选项卡中的部件列表
        self.parts_listbox.delete(0, tk.END)
        
        if self.params:
            for part in self.params.part_params:
                self.parts_listbox.insert(tk.END, part.part_name)
    
    def show_about(self):
        """显示关于对话框"""
        messagebox.showinfo("关于", "PyMeshGen - 简化网格生成工具\n\n版本: 1.0\n作者: PyMeshGen团队")
    
    def on_closing(self):
        """窗口关闭事件"""
        if messagebox.askokcancel("退出", "确定要退出吗?"):
            self.root.destroy()


def main():
    """主函数"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建主窗口
    root = tk.Tk()
    app = SimplifiedPyMeshGenGUI(root)
    
    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    main()
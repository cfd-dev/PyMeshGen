#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重构后的GUI主模块
整合各个功能模块，提供统一的用户界面
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox, font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import vtk

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加必要的子目录到路径 (for compatibility with main script approach)
data_structure_dir = os.path.join(project_root, 'data_structure')
fileio_dir = os.path.join(project_root, 'fileIO')
meshsize_dir = os.path.join(project_root, 'meshsize')
visualization_dir = os.path.join(project_root, 'visualization')
adfront2_dir = os.path.join(project_root, 'adfront2')
optimize_dir = os.path.join(project_root, 'optimize')
utils_dir = os.path.join(project_root, 'utils')

for dir_path in [data_structure_dir, fileio_dir, meshsize_dir, visualization_dir, adfront2_dir, optimize_dir, utils_dir]:
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

# 导入自定义模块
from gui.gui_base import BaseFrame, MenuBar, StatusBar, InfoOutput, DialogBase
from gui.mesh_display import MeshDisplayArea
from gui.config_manager import ConfigManager, ConfigDialog
from gui.file_operations import FileOperations, ImportDialog, ExportDialog

# 导入项目模块
try:
    from parameters import Parameters
    from PyMeshGen import PyMeshGen as MeshGenerator
    from basic_elements import NodeElement as Node
except ImportError as e:
    print(f"导入项目模块失败: {e}")
    sys.exit(1)


class SimplifiedPyMeshGenGUI:
    """简化的PyMeshGen GUI主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PyMeshGen - 高性能网格生成工具")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.setup_window_size()
        self.setup_fonts()
        self.setup_styles()
        self.initialize_modules()
        self.initialize_data()
        self.create_widgets()
        self.update_status("就绪")
    
    def setup_window_size(self):
        """设置窗口大小和位置"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 根据屏幕分辨率设置窗口大小
        if screen_width >= 1920 and screen_height >= 1080:
            # 高分辨率屏幕
            window_width = int(screen_width * 0.75)
            window_height = int(screen_height * 0.75)
        elif screen_width >= 1366 and screen_height >= 768:
            # 中等分辨率屏幕
            window_width = int(screen_width * 0.85)
            window_height = int(screen_height * 0.85)
        else:
            # 低分辨率屏幕
            window_width = int(screen_width * 0.95)
            window_height = int(screen_height * 0.95)
        
        # 设置最小窗口大小
        min_width = 1024
        min_height = 768
        
        # 确保窗口大小不小于最小值
        window_width = max(window_width, min_width)
        window_height = max(window_height, min_height)
        
        # 设置窗口大小和位置（居中）
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 设置最小窗口大小
        self.root.minsize(min_width, min_height)
    
    def setup_fonts(self):
        """设置字体"""
        # 设置默认字体大小
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=10)
        text_font = font.nametofont("TkTextFont")
        text_font.configure(size=10)
        fixed_font = font.nametofont("TkFixedFont")
        fixed_font.configure(size=10)
    
    def setup_styles(self):
        """设置样式"""
        style = ttk.Style()
        style.theme_use('clam')  # 使用现代化主题
        
        # 配置自定义样式
        style.configure("Bold.TLabel", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 9))
    
    def initialize_modules(self):
        """初始化模块"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_manager = ConfigManager(self.project_root)
        self.file_operations = FileOperations(self.project_root)
    
    def initialize_data(self):
        """初始化数据"""
        self.params = None
        self.mesh_generator = None
        self.current_mesh = None
        self.cas_parts_info = None  # 初始化cas_parts_info属性
        # 初始化视图控制变量
        self.render_mode_var = tk.StringVar(value="surface")
        self.show_boundary_var = tk.BooleanVar(value=True)
    
    def create_widgets(self):
        """创建UI组件"""
        # 创建菜单栏
        self.menu_bar = MenuBar(self.root)
        self.create_menu()

        # 创建工具栏（位于菜单栏下方）
        self.create_toolbar()

        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建左右两栏布局（3:7比例，支持拖动）
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # 左侧部件信息区域（3/10宽度）
        self.left_panel = ttk.Frame(self.paned_window, width=1)
        self.paned_window.add(self.left_panel, weight=3)
        self.create_left_panel()

        # 右侧网格视图交互区域（7/10宽度）
        right_main_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(right_main_frame, weight=7)
        
        # 在右侧区域中创建垂直分割窗格（网格显示和状态输出）
        self.right_paned = ttk.PanedWindow(right_main_frame, orient=tk.VERTICAL)
        self.right_paned.pack(fill=tk.BOTH, expand=True)
        
        # 上半部分：网格视图交互区域
        self.right_panel = ttk.Frame(self.right_paned)
        self.right_paned.add(self.right_panel, weight=1)
        
        # 下半部分：状态输出面板区域
        bottom_frame = ttk.Frame(self.right_paned)
        self.right_paned.add(bottom_frame, weight=0)
        
        # 创建网格视图
        self.create_right_panel()
        
        # 创建状态输出面板（放在右侧面板的下方部分）
        self.bottom_frame = bottom_frame
        self.create_status_output_panel()
    
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
            "另存为": self.save_config_as,
            "---": None,
            "导入网格": self.import_mesh,
            "导出网格": self.export_mesh,
            "---": None,
            "最近文件": self.show_recent_files,
            "---": None,
            "退出": self.on_closing
        }
        self.menu_bar.create_file_menu(file_commands)
        
        # 视图菜单
        view_commands = {
            "重置视图": self.reset_view,
            "适应视图": self.fit_view,
            "---": None,
            "放大": self.zoom_in,
            "缩小": self.zoom_out,
            "平移": self.pan_view,
            "旋转": self.rotate_view,
            "---": None,
            "显示工具栏": self.toggle_toolbar,
            "显示状态栏": self.toggle_statusbar,
            "显示部件列表": self.toggle_parts_list,
            "显示属性面板": self.toggle_properties_panel,
            "---": None,
            "全屏显示": self.toggle_fullscreen
        }
        self.menu_bar.create_view_menu(view_commands)
        
        # 配置菜单
        config_commands = {
            "参数设置": self.edit_params,
            "网格参数": self.edit_mesh_params,
            "边界条件": self.edit_boundary_conditions,
            "---": None,
            "导入配置": self.import_config,
            "导出配置": self.export_config,
            "---": None,
            "清空网格": self.clear_mesh,
            "重置配置": self.reset_config
        }
        self.menu_bar.create_config_menu(config_commands)
        
        # 网格菜单
        mesh_commands = {
            "生成网格": self.generate_mesh,
            "显示网格": self.display_mesh,
            "网格质量": self.check_mesh_quality,
            "---": None,
            "平滑网格": self.smooth_mesh,
            "优化网格": self.optimize_mesh,
            "---": None,
            "网格统计": self.show_mesh_statistics,
            "导出报告": self.export_mesh_report
        }
        self.menu_bar.create_mesh_menu(mesh_commands)
        
        # 帮助菜单
        help_commands = {
            "用户手册": self.show_user_manual,
            "快速入门": self.show_quick_start,
            "---": None,
            "快捷键": self.show_shortcuts,
            "检查更新": self.check_for_updates,
            "---": None,
            "关于": self.show_about
        }
        self.menu_bar.create_help_menu(help_commands)
    
    def create_left_panel(self):
        """创建左侧部件信息区域（分组更清晰，带滚动）"""
        # 部件列表分组
        parts_frame = ttk.LabelFrame(self.left_panel, text="部件列表")
        parts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))
        # 部件列表带滚动条
        parts_list_frame = ttk.Frame(parts_frame)
        parts_list_frame.pack(fill=tk.BOTH, expand=True)
        self.parts_listbox = tk.Listbox(parts_list_frame, height=10)
        parts_scrollbar = ttk.Scrollbar(parts_list_frame, orient=tk.VERTICAL, command=self.parts_listbox.yview)
        self.parts_listbox.config(yscrollcommand=parts_scrollbar.set)
        self.parts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        parts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.create_tooltip(self.parts_listbox, "显示当前配置中的所有部件")
        # 部件操作按钮
        parts_button_frame = ttk.Frame(parts_frame)
        parts_button_frame.pack(fill=tk.X, padx=5, pady=2)
        add_part_btn = ttk.Button(parts_button_frame, text="添加", command=self.add_part)
        add_part_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(add_part_btn, "添加新的部件到配置中")
        remove_part_btn = ttk.Button(parts_button_frame, text="删除", command=self.remove_part)
        remove_part_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(remove_part_btn, "删除选中的部件")
        edit_part_btn = ttk.Button(parts_button_frame, text="编辑", command=self.edit_part)
        edit_part_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(edit_part_btn, "编辑选中的部件属性")
        # 属性面板分组
        self.props_frame = ttk.LabelFrame(self.left_panel, text="属性面板")
        self.props_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))
        self.props_text = tk.Text(self.props_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        props_scrollbar = ttk.Scrollbar(self.props_frame, orient=tk.VERTICAL, command=self.props_text.yview)
        self.props_text.config(yscrollcommand=props_scrollbar.set)
        self.props_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        props_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.create_tooltip(self.props_text, "显示选中部件的详细属性")
        self.parts_listbox.bind('<<ListboxSelect>>', self.on_part_select)
    
    def create_right_panel(self):
        """创建右侧网格视图交互区域（最优化版，移除所有标签，仅保留纯净的网格显示）"""
        # 移除网格信息框架 - 现在只保留纯净的网格显示区域
        
        # 创建网格显示区域
        self.mesh_display = MeshDisplayArea(self.right_panel)
        self.mesh_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # 增加边距以避免与边缘贴合
        
        # 存储对网格显示区域的引用，以便其他方法使用
        self.main_mesh_display = self.mesh_display
        
        # 为网格显示区域绑定键盘事件
        self.mesh_display.frame.bind("<Key>", self.on_mesh_display_key)
        self.mesh_display.frame.focus_set()  # 确保网格显示区域可以获得键盘焦点
        
        # 为网格显示区域添加右键菜单
        self.create_mesh_display_context_menu()
        
        # 添加键盘快捷键提示
        self.add_view_interaction_hints()
    
    def on_mesh_display_key(self, event):
        """处理网格显示区域的键盘事件"""
        key = event.keysym.lower()
        if key == 'r':  # 重置视图
            self.reset_view()
            self.update_status("已重置视图 (R键)")
        elif key == 'f':  # 适应视图
            self.fit_view()
            self.update_status("已适应视图 (F键)")
        elif key == 'o':  # 切换边界显示
            new_state = not self.show_boundary_var.get()
            self.show_boundary_var.set(new_state)
            self.toggle_boundary_display()
            self.update_status(f"边界显示: {'开启' if new_state else '关闭'} (O键)")
        elif key == '1':  # 切换到表面模式
            self.render_mode_var.set("surface")
            self.change_render_mode()
            self.update_status("渲染模式: 表面 (1键)")
        elif key == '2':  # 切换到线框模式
            self.render_mode_var.set("wireframe")
            self.change_render_mode()
            self.update_status("渲染模式: 线框 (2键)")
        elif key == '3':  # 切换到点云模式
            self.render_mode_var.set("points")
            self.change_render_mode()
            self.update_status("渲染模式: 点云 (3键)")
        elif key == 'plus' or key == 'equal':  # 放大
            self.zoom_in()
            self.update_status("视图已放大 (+键)")
        elif key == 'minus':  # 缩小
            self.zoom_out()
            self.update_status("视图已缩小 (-键)")
    
    def create_mesh_display_context_menu(self):
        """为网格显示区域创建右键上下文菜单"""
        # 创建右键菜单
        self.mesh_context_menu = tk.Menu(self.mesh_display.frame, tearoff=0)
        
        # 视图操作菜单项
        self.mesh_context_menu.add_command(label="重置视图 (R)", command=self.reset_view)
        self.mesh_context_menu.add_command(label="适应视图 (F)", command=self.fit_view)
        self.mesh_context_menu.add_separator()
        
        # 渲染模式菜单项
        self.mesh_context_menu.add_command(label="表面渲染 (1)", command=lambda: self.set_render_mode("surface"))
        self.mesh_context_menu.add_command(label="线框渲染 (2)", command=lambda: self.set_render_mode("wireframe"))
        self.mesh_context_menu.add_command(label="点云渲染 (3)", command=lambda: self.set_render_mode("points"))
        self.mesh_context_menu.add_separator()
        
        # 边界显示菜单项
        show_boundary_menu = tk.Menu(self.mesh_context_menu, tearoff=0)
        show_boundary_menu.add_command(label="开启", command=lambda: self.set_boundary_visibility(True))
        show_boundary_menu.add_command(label="关闭", command=lambda: self.set_boundary_visibility(False))
        self.mesh_context_menu.add_cascade(label="边界显示", menu=show_boundary_menu)
        
        # 绑定右键事件到网格显示区域
        self.mesh_display.frame.bind("<Button-3>", self.show_mesh_context_menu)  # Windows/Linux
        self.mesh_display.frame.bind("<Button-2>", self.show_mesh_context_menu)  # macOS (双击右键)
    
    def show_mesh_context_menu(self, event):
        """显示网格显示区域的右键菜单"""
        try:
            self.mesh_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.mesh_context_menu.grab_release()
    
    def set_render_mode(self, mode):
        """设置渲染模式并更新状态"""
        self.render_mode_var.set(mode)
        self.change_render_mode()
        mode_names = {"surface": "表面", "wireframe": "线框", "points": "点云"}
        self.update_status(f"渲染模式已切换到: {mode_names.get(mode, mode)}")
    
    def set_boundary_visibility(self, visible):
        """设置边界可见性并更新状态"""
        self.show_boundary_var.set(visible)
        self.toggle_boundary_display()
        self.update_status(f"边界显示已{'开启' if visible else '关闭'}")
    
    def add_view_interaction_hints(self):
        """添加视图交互提示信息"""
        # 这个方法可以用来显示当前可用的快捷键提示
        # 可以在状态栏或其他地方显示当前可用的快捷键
        pass
    
    def create_status_output_panel(self):
        """创建状态栏和信息输出面板（左右4:6布局，信息输出区带清空按钮）"""
        # 在底部框架中创建左右分割的状态和输出面板
        self.status_output_paned = ttk.PanedWindow(self.bottom_frame, orient=tk.HORIZONTAL)
        self.status_output_paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧状态显示区域（4/10宽度）
        self.status_panel = ttk.Frame(self.status_output_paned)
        self.status_output_paned.add(self.status_panel, weight=4)
        
        # 右侧信息输出区域（6/10宽度）
        self.output_panel = ttk.Frame(self.status_output_paned)
        self.status_output_paned.add(self.output_panel, weight=6)
        
        # 创建状态面板和输出面板
        self.create_status_panel()
        self.create_output_panel()
    
    def create_status_panel(self):
        """创建状态显示面板"""
        # 创建状态显示框架
        status_frame = ttk.LabelFrame(self.status_panel, text="状态信息")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建状态文本框和滚动条
        text_frame = ttk.Frame(status_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_text = tk.Text(text_frame, wrap=tk.WORD, height=8, state=tk.DISABLED)
        status_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建状态栏（用于显示当前操作状态）
        self.status_bar = StatusBar(status_frame)
    
    def create_output_panel(self):
        """创建信息输出面板（右键菜单可清空）"""
        self.info_output = InfoOutput(self.output_panel)
        self.info_output.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def update_status(self, message):
        """更新状态栏和状态文本框"""
        # 更新状态栏
        self.status_bar.update_status(message)
        
        # 更新状态文本框
        if hasattr(self, 'status_text'):
            self.status_text.config(state=tk.NORMAL)
            # 添加时间戳
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.status_text.see(tk.END)  # 自动滚动到最新信息
            self.status_text.config(state=tk.DISABLED)
    
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
    
    def change_render_mode(self):
        """改变网格渲染方式"""
        if hasattr(self, 'mesh_display') and self.mesh_display:
            render_mode = self.render_mode_var.get()
            
            if render_mode == "surface":
                self.mesh_display.toggle_wireframe(False)
                self.update_status("切换到表面渲染模式")
            elif render_mode == "wireframe":
                self.mesh_display.toggle_wireframe(True)
                self.update_status("切换到线框渲染模式")
            elif render_mode == "points":
                # 点云模式需要特殊处理
                if hasattr(self.mesh_display, 'toggle_points'):
                    self.mesh_display.toggle_points(True)
                    self.update_status("切换到点云渲染模式")
                else:
                    # 如果没有点云模式，则使用线框模式
                    self.mesh_display.toggle_wireframe(True)
                    self.update_status("切换到线框渲染模式（点云模式不可用）")
    
    def toggle_boundary_display(self):
        """切换边界显示"""
        if hasattr(self, 'mesh_display') and self.mesh_display:
            show_boundary = self.show_boundary_var.get()
            if hasattr(self.mesh_display, 'toggle_boundary_display'):
                self.mesh_display.toggle_boundary_display(show_boundary)
                self.update_status(f"边界显示: {'开启' if show_boundary else '关闭'}")
            else:
                self.update_status("边界显示功能不可用")
    
    def toggle_parts_list(self):
        """切换部件列表显示"""
        if hasattr(self, 'left_panel'):
            if self.left_panel.winfo_viewable():
                self.left_panel.pack_forget()
                self.update_status("部件列表已隐藏")
            else:
                self.paned_window.add(self.left_panel, weight=3)
                self.update_status("部件列表已显示")
                
    def toggle_properties_panel(self):
        """切换属性面板显示"""
        if hasattr(self, 'props_frame'):
            if self.props_frame.winfo_viewable():
                self.props_frame.pack_forget()
                self.update_status("属性面板已隐藏")
            else:
                self.props_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                self.update_status("属性面板已显示")
    
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
            
            # 创建网格生成器
            self.mesh_generator = MeshGenerator(self.params)
            
            # 生成网格
            self.current_mesh = self.mesh_generator.generate()
            
            # 获取网格信息 for logging purposes (but don't display it)
            node_count = 0
            element_count = 0
            if hasattr(self.current_mesh, 'nodes') and hasattr(self.current_mesh, 'elements'):
                node_count = len(self.current_mesh.nodes)
                element_count = len(self.current_mesh.elements)
                self.log_info(f"生成网格: 节点数={node_count}, 单元数={element_count}")
            
            self.log_info("网格生成完成")
            self.update_status("网格生成完成")
            
            # 显示网格
            self.display_mesh()
        except Exception as e:
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
            
            # 显示网格
            self.mesh_display.display_mesh(self.current_mesh)
            
            self.log_info("网格显示完成")
            self.update_status("网格显示完成")
        except Exception as e:
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
                
                # 获取网格信息 for logging purposes (but don't display it)
                node_count = 0
                element_count = 0
                if isinstance(self.current_mesh, dict):
                    # 处理字典类型的网格数据
                    node_count = self.current_mesh.get('num_points', 0)
                    element_count = self.current_mesh.get('num_cells', 0)
                    
                    # 如果是cas文件，提取部件信息
                    if self.current_mesh.get('type') == 'cas' and 'parts_info' in self.current_mesh:
                        self.update_parts_list_from_cas(self.current_mesh['parts_info'])
                elif hasattr(self.current_mesh, 'num_points') and hasattr(self.current_mesh, 'num_cells'):
                    node_count = self.current_mesh.num_points
                    element_count = self.current_mesh.num_cells
                elif hasattr(self.current_mesh, 'node_coords') and hasattr(self.current_mesh, 'cells'):
                    node_count = len(self.current_mesh.node_coords)
                    element_count = len(self.current_mesh.cells)
                elif hasattr(self.current_mesh, 'nodes') and hasattr(self.current_mesh, 'elements'):
                    node_count = len(self.current_mesh.nodes)
                    element_count = len(self.current_mesh.elements)
                
                # Log grid info (but don't display it in the view area)
                self.log_info(f"导入网格: 节点数={node_count}, 单元数={element_count}")
                
                if dialog.result["preview"]:
                    self.display_mesh()
                
                self.log_info(f"已导入网格文件: {dialog.result['file_path']}")
                self.update_status("已导入网格文件")
            except Exception as e:
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
    
    def update_parts_list_from_cas(self, parts_info):
        """从cas文件的部件信息更新部件列表"""
        # 更新选项卡中的部件列表
        self.parts_listbox.delete(0, tk.END)
        
        # 存储cas部件信息以便在选择时显示
        self.cas_parts_info = parts_info
        
        # 添加部件到列表
        for part_info in parts_info:
            part_name = part_info.get('part_name', '未知部件')
            self.parts_listbox.insert(tk.END, part_name)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """PyMeshGen v1.0

高性能网格生成工具

© 2023 HighOrderMesh
"""
        messagebox.showinfo("关于", about_text)
    
    def show_user_manual(self):
        """显示用户手册"""
        manual_text = """PyMeshGen 用户手册

1. 文件菜单
   - 新建配置：创建新的网格配置
   - 打开配置：加载已保存的配置文件
   - 保存配置：保存当前配置到文件
   - 导入网格：从外部文件导入网格数据
   - 导出网格：将当前网格导出到文件

2. 视图菜单
   - 重置视图：将视图恢复到初始状态
   - 适应视图：自动调整视图以适应整个网格
   - 放大/缩小：缩放网格显示
   - 显示工具栏：切换工具栏的显示/隐藏
   - 显示状态栏：切换状态栏的显示/隐藏

3. 配置菜单
   - 参数设置：配置网格生成参数
   - 清空网格：清除当前显示的网格

4. 网格菜单
   - 生成网格：根据当前配置生成网格
   - 显示网格：显示/隐藏网格

5. 工具栏
   - 提供常用功能的快速访问按钮

6. 主界面
   - 左侧：部件信息区域，包含部件列表和属性面板
   - 右侧：网格视图交互区域，支持缩放、平移和选择操作
   - 底部：状态栏显示系统状态和信息输出窗口
"""
        messagebox.showinfo("用户手册", manual_text)
    
    def zoom_in(self):
        """放大视图"""
        if hasattr(self, 'mesh_display'):
            self.mesh_display.zoom_in()
            self.update_status("视图已放大")
    
    def zoom_out(self):
        """缩小视图"""
        if hasattr(self, 'mesh_display'):
            self.mesh_display.zoom_out()
            self.update_status("视图已缩小")
    
    def toggle_toolbar(self):
        """切换工具栏显示"""
        if hasattr(self, 'toolbar_frame'):
            if self.toolbar_frame.winfo_viewable():
                self.toolbar_frame.pack_forget()
                self.update_status("工具栏已隐藏")
            else:
                self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
                self.update_status("工具栏已显示")
    
    def toggle_statusbar(self):
        """切换状态栏显示"""
        if hasattr(self, 'bottom_frame'):
            # Get the current position of bottom frame in the paned window
            paned_contents = self.right_paned.panes()
            if self.bottom_frame in paned_contents:
                self.right_paned.forget(self.bottom_frame)  # Hide the bottom frame
                self.log_info("状态栏和信息输出已隐藏")
            else:
                self.right_paned.add(self.bottom_frame, weight=0)  # Show the bottom frame
                self.log_info("状态栏和信息输出已显示")
    
    def on_part_select(self, event):
        """处理部件列表选择事件"""
        # 获取选中的部件索引
        selection = self.parts_listbox.curselection()
        if not selection:
            # 清空属性面板
            self.props_text.config(state=tk.NORMAL)
            self.props_text.delete(1.0, tk.END)
            self.props_text.insert(tk.END, "未选择任何部件\n请从左侧列表中选择一个部件以查看其属性")
            self.props_text.config(state=tk.DISABLED)
            self.update_status("未选择部件")
            return
            
        # 获取选中的部件索引
        index = selection[0]
        
        # 检查是否是cas文件的部件
        if hasattr(self, 'cas_parts_info') and self.cas_parts_info and index < len(self.cas_parts_info):
            # 显示cas部件的属性
            part_info = self.cas_parts_info[index]
            
            # 清空属性文本框
            self.props_text.config(state=tk.NORMAL)
            self.props_text.delete(1.0, tk.END)
            
            # 添加标题
            self.props_text.insert(tk.END, f"=== CAS部件属性 ===\n\n")
            
            # 显示部件属性
            self.props_text.insert(tk.END, f"部件名称: {part_info.get('part_name', '未知')}\n")
            self.props_text.insert(tk.END, f"边界条件类型: {part_info.get('bc_type', '未知')}\n")
            self.props_text.insert(tk.END, f"面数量: {part_info.get('face_count', 0)}\n")
            self.props_text.insert(tk.END, f"节点数量: {len(part_info.get('nodes', []))}\n")
            self.props_text.insert(tk.END, f"单元数量: {len(part_info.get('cells', []))}\n")
            
            # 添加状态信息
            self.props_text.insert(tk.END, f"\n=== 状态信息 ===\n")
            self.props_text.insert(tk.END, f"选择时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.props_text.insert(tk.END, f"部件索引: {index}\n")
            self.props_text.insert(tk.END, f"总部件数: {len(self.cas_parts_info)}\n")
            self.props_text.insert(tk.END, f"数据来源: CAS文件\n")
            
            self.props_text.config(state=tk.DISABLED)
            self.update_status(f"已选中CAS部件: {part_info.get('part_name', f'部件{index}')}")
            return
        
        # 如果有参数对象，显示选中部件的属性
        if hasattr(self, 'params') and self.params:
            try:
                # 修复：直接使用self.params.part_params获取部件列表，而不是调用不存在的get_parts()方法
                parts = self.params.part_params
                if index < len(parts):
                    part = parts[index]
                    
                    # 清空属性文本框
                    self.props_text.config(state=tk.NORMAL)
                    self.props_text.delete(1.0, tk.END)
                    
                    # 添加标题
                    self.props_text.insert(tk.END, f"=== 部件属性 ===\n\n")
                    
                    # 显示部件属性
                    if hasattr(part, 'get_properties'):
                        props = part.get_properties()
                        for key, value in props.items():
                            self.props_text.insert(tk.END, f"{key}: {value}\n")
                    else:
                        # 如果部件没有get_properties方法，显示基本信息
                        self.props_text.insert(tk.END, f"部件类型: {type(part).__name__}\n")
                        if hasattr(part, 'part_name'):
                            self.props_text.insert(tk.END, f"名称: {part.part_name}\n")
                        if hasattr(part, 'name'):
                            self.props_text.insert(tk.END, f"名称: {part.name}\n")
                        if hasattr(part, 'id'):
                            self.props_text.insert(tk.END, f"ID: {part.id}\n")
                        # 显示部件的其他属性
                        if hasattr(part, 'params'):
                            self.props_text.insert(tk.END, f"参数: {part.params}\n")
                        if hasattr(part, 'connectors'):
                            self.props_text.insert(tk.END, f"连接器数量: {len(part.connectors)}\n")
                    
                    # 添加状态信息
                    self.props_text.insert(tk.END, f"\n=== 状态信息 ===\n")
                    self.props_text.insert(tk.END, f"选择时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    self.props_text.insert(tk.END, f"部件索引: {index}\n")
                    self.props_text.insert(tk.END, f"总部件数: {len(parts)}\n")
                    
                    self.props_text.config(state=tk.DISABLED)
                    self.update_status(f"已选中部件: {part.part_name if hasattr(part, 'part_name') else f'部件{index}'}")
            except Exception as e:
                self.log_error(f"显示部件属性时出错: {str(e)}")
                # 在属性面板中显示错误信息
                self.props_text.config(state=tk.NORMAL)
                self.props_text.delete(1.0, tk.END)
                self.props_text.insert(tk.END, f"=== 错误信息 ===\n\n")
                self.props_text.insert(tk.END, f"显示部件属性时出错:\n{str(e)}\n")
                self.props_text.insert(tk.END, f"\n请检查部件数据是否正确\n")
                self.props_text.config(state=tk.DISABLED)
    
    def toggle_statusbar(self):
        """切换状态栏显示"""
        if hasattr(self, 'bottom_frame'):
            # Get the current position of bottom frame in the paned window
            paned_contents = self.right_paned.panes()
            if self.bottom_frame in paned_contents:
                self.right_paned.forget(self.bottom_frame)  # Hide the bottom frame
                self.log_info("状态栏和信息输出已隐藏")
            else:
                self.right_paned.add(self.bottom_frame, weight=0)  # Show the bottom frame
                self.log_info("状态栏和信息输出已显示")
    
    def save_config_as(self):
        """配置另存为"""
        messagebox.showinfo("信息", "配置另存为功能")
        
    def show_recent_files(self):
        """显示最近文件"""
        messagebox.showinfo("信息", "最近文件功能")
        
    def pan_view(self):
        """平移视图"""
        if hasattr(self, 'mesh_display'):
            self.mesh_display.pan_view()
            self.update_status("视图平移模式")
            
    def rotate_view(self):
        """旋转视图"""
        if hasattr(self, 'mesh_display'):
            self.mesh_display.rotate_view()
            self.update_status("视图旋转模式")
            
    def toggle_parts_list(self):
        """切换部件列表显示"""
        if hasattr(self, 'left_panel'):
            if self.left_panel.winfo_viewable():
                self.left_panel.pack_forget()
                self.update_status("部件列表已隐藏")
            else:
                self.paned_window.add(self.left_panel, weight=3)
                self.update_status("部件列表已显示")
                
    def toggle_properties_panel(self):
        """切换属性面板显示"""
        if hasattr(self, 'props_frame'):
            if self.props_frame.winfo_viewable():
                self.props_frame.pack_forget()
                self.update_status("属性面板已隐藏")
            else:
                self.props_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                self.update_status("属性面板已显示")
        
    def toggle_fullscreen(self):
        """切换全屏显示"""
        if self.root.attributes('-fullscreen'):
            self.root.attributes('-fullscreen', False)
            self.update_status("退出全屏模式")
        else:
            self.root.attributes('-fullscreen', True)
            self.update_status("进入全屏模式")
    
    def zoom_in(self):
        """放大视图"""
        if hasattr(self, 'main_mesh_display'):
            self.main_mesh_display.zoom_in()
            self.update_status("视图已放大")
    
    def zoom_out(self):
        """缩小视图"""
        if hasattr(self, 'main_mesh_display'):
            self.main_mesh_display.zoom_out()
            self.update_status("视图已缩小")
    
    def reset_view(self):
        """重置视图"""
        if hasattr(self, 'main_mesh_display'):
            self.main_mesh_display.reset_view()
            self.update_status("视图已重置")
    
    def fit_view(self):
        """适应视图"""
        if hasattr(self, 'main_mesh_display'):
            self.main_mesh_display.fit_view()
            self.update_status("视图已适应")
    
    def pan_view(self):
        """平移视图"""
        if hasattr(self, 'main_mesh_display'):
            self.main_mesh_display.pan_view()
            self.update_status("视图平移模式")
    
    def rotate_view(self):
        """旋转视图"""
        if hasattr(self, 'main_mesh_display'):
            self.main_mesh_display.rotate_view()
            self.update_status("视图旋转模式")
            
    def edit_mesh_params(self):
        """编辑网格参数"""
        messagebox.showinfo("信息", "编辑网格参数功能")
        
    def edit_boundary_conditions(self):
        """编辑边界条件"""
        messagebox.showinfo("信息", "编辑边界条件功能")
        
    def import_config(self):
        """导入配置"""
        messagebox.showinfo("信息", "导入配置功能")
        
    def export_config(self):
        """导出配置"""
        messagebox.showinfo("信息", "导出配置功能")
        
    def reset_config(self):
        """重置配置"""
        messagebox.showinfo("信息", "重置配置功能")
        
    def check_mesh_quality(self):
        """检查网格质量"""
        messagebox.showinfo("信息", "检查网格质量功能")
        
    def smooth_mesh(self):
        """平滑网格"""
        messagebox.showinfo("信息", "平滑网格功能")
        
    def optimize_mesh(self):
        """优化网格"""
        messagebox.showinfo("信息", "优化网格功能")
        
    def show_mesh_statistics(self):
        """显示网格统计"""
        messagebox.showinfo("信息", "显示网格统计功能")
        
    def export_mesh_report(self):
        """导出网格报告"""
        messagebox.showinfo("信息", "导出网格报告功能")
        
    def show_quick_start(self):
        """显示快速入门"""
        messagebox.showinfo("信息", "快速入门功能")
        
    def show_shortcuts(self):
        """显示快捷键"""
        shortcuts_text = """常用快捷键：

Ctrl+N: 新建配置
Ctrl+O: 打开配置
Ctrl+S: 保存配置
Ctrl+I: 导入网格
Ctrl+E: 导出网格
F5: 生成网格
F6: 显示网格
F11: 全屏显示
Esc: 退出全屏
"""
        messagebox.showinfo("快捷键", shortcuts_text)
        
    def check_for_updates(self):
        """检查更新"""
        messagebox.showinfo("信息", "检查更新功能")

    def on_closing(self):
        """窗口关闭事件"""
        if messagebox.askokcancel("退出", "确定要退出吗?"):
            # 清理VTK资源
            if hasattr(self, 'mesh_display') and self.mesh_display:
                try:
                    self.mesh_display.cleanup()
                except:
                    pass
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
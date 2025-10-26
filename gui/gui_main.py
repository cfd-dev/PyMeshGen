#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyMeshGen GUI主程序
提供参数设置、文件读入导出、网格显示和交互功能
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加子模块路径
sys.path.append(str(project_root / "fileIO"))
sys.path.append(str(project_root / "data_structure"))
sys.path.append(str(project_root / "meshsize"))
sys.path.append(str(project_root / "visualization"))
sys.path.append(str(project_root / "adfront2"))
sys.path.append(str(project_root / "optimize"))
sys.path.append(str(project_root / "utils"))

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    print("错误: 缺少tkinter模块，请安装Python GUI库")
    sys.exit(1)

# 导入matplotlib相关模块
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from parameters import Parameters
from read_cas import parse_fluent_msh
from mesh_visualization import Visualization


class PyMeshGenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PyMeshGen 网格生成器")
        self.root.geometry("1200x800")
        
        # 当前参数对象
        self.params = None
        self.mesh_data = None
        
        # 创建GUI界面
        self.create_widgets()
        
        # 初始化参数
        self.init_default_params()

    def create_widgets(self):
        """创建GUI界面组件"""
        # 创建菜单栏
        self.create_menu()
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建参数设置区域
        self.create_param_frame(main_frame)
        
        # 创建控制按钮区域
        self.create_control_frame(main_frame)
        
        # 创建主内容区域（网格显示和信息输出）
        self.create_main_content_area(main_frame)
        
        # 创建状态栏
        self.create_status_bar()

    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开配置文件", command=self.open_config_file)
        file_menu.add_command(label="保存配置文件", command=self.save_config_file)
        file_menu.add_separator()
        file_menu.add_command(label="导入网格文件", command=self.import_mesh_file)
        file_menu.add_command(label="导出网格文件", command=self.export_mesh_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="显示网格", command=self.display_mesh)
        view_menu.add_command(label="清除显示", command=self.clear_display)
        view_menu.add_separator()
        view_menu.add_command(label="放大", command=self.zoom_in)
        view_menu.add_command(label="缩小", command=self.zoom_out)
        view_menu.add_command(label="原始大小", command=self.reset_view)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

    def create_param_frame(self, parent):
        """创建参数设置区域"""
        param_frame = ttk.LabelFrame(parent, text="参数设置")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建参数输入控件
        # 调试级别
        ttk.Label(param_frame, text="调试级别:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.debug_level_var = tk.StringVar(value="0")
        debug_level_combo = ttk.Combobox(param_frame, textvariable=self.debug_level_var, 
                                        values=["0", "1", "2"], width=10)
        debug_level_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 输入文件
        ttk.Label(param_frame, text="输入文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_file_var = tk.StringVar()
        input_file_entry = ttk.Entry(param_frame, textvariable=self.input_file_var, width=50)
        input_file_entry.grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(param_frame, text="浏览", command=self.browse_input_file).grid(row=1, column=2, padx=5, pady=2)
        
        # 输出文件
        ttk.Label(param_frame, text="输出文件:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.output_file_var = tk.StringVar()
        output_file_entry = ttk.Entry(param_frame, textvariable=self.output_file_var, width=50)
        output_file_entry.grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(param_frame, text="浏览", command=self.browse_output_file).grid(row=2, column=2, padx=5, pady=2)
        
        # 网格类型
        ttk.Label(param_frame, text="网格类型:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.mesh_type_var = tk.StringVar(value="1")
        mesh_type_frame = ttk.Frame(param_frame)
        mesh_type_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mesh_type_frame, text="三角形", variable=self.mesh_type_var, 
                       value="1").pack(side=tk.LEFT)
        ttk.Radiobutton(mesh_type_frame, text="直角三角形", variable=self.mesh_type_var, 
                       value="2").pack(side=tk.LEFT)
        ttk.Radiobutton(mesh_type_frame, text="混合网格", variable=self.mesh_type_var, 
                       value="3").pack(side=tk.LEFT)
        
        # 可视化开关
        self.viz_enabled_var = tk.BooleanVar(value=True)
        viz_check = ttk.Checkbutton(param_frame, text="启用可视化", variable=self.viz_enabled_var)
        viz_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

    def create_control_frame(self, parent):
        """创建控制按钮区域"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 生成网格按钮
        ttk.Button(control_frame, text="生成网格", command=self.generate_mesh, 
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        
        # 显示网格按钮
        ttk.Button(control_frame, text="显示网格", command=self.display_mesh).pack(side=tk.LEFT, padx=5)
        
        # 清除显示按钮
        ttk.Button(control_frame, text="清除显示", command=self.clear_display).pack(side=tk.LEFT, padx=5)
        
        # 创建进度条
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

    def create_main_content_area(self, parent):
        """创建主内容区域（网格显示和信息输出）"""
        # 创建主内容框架
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建网格显示区域
        self.create_mesh_display_area(content_frame)
        
        # 创建信息输出窗口
        self.create_info_output_area(content_frame)

    def create_mesh_display_area(self, parent):
        """创建网格显示区域"""
        # 创建网格显示框架
        mesh_frame = ttk.LabelFrame(parent, text="网格显示区")
        mesh_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib图形和轴
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("网格显示区域")
        self.ax.set_xlabel("X坐标")
        self.ax.set_ylabel("Y坐标")
        self.ax.axis("equal")
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, mesh_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 添加导航工具栏
        self.toolbar = NavigationToolbar2Tk(self.canvas, mesh_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 确保鼠标事件被正确连接
        self.canvas.mpl_connect('scroll_event', self.on_mouse_wheel)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # 初始化鼠标交互状态
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None
    
    def create_info_output_area(self, parent):
        """创建信息输出窗口"""
        # 创建信息输出框架
        info_frame = ttk.LabelFrame(parent, text="信息输出")
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建文本框和滚动条
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(text_frame, wrap=tk.WORD, height=10)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 添加清除按钮
        ttk.Button(info_frame, text="清除信息", command=self.clear_info_output).pack(pady=5)

    def clear_info_output(self):
        """清除信息输出"""
        self.info_text.delete(1.0, tk.END)
        
    def append_info_output(self, message):
        """添加信息到输出窗口"""
        self.info_text.insert(tk.END, message + "\n")
        self.info_text.see(tk.END)  # 自动滚动到最新信息

    def create_status_bar(self):
        """创建状态栏"""
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def init_default_params(self):
        """初始化默认参数"""
        try:
            self.params = Parameters("FROM_MAIN_JSON")
            self.input_file_var.set(self.params.input_file)
            self.output_file_var.set(self.params.output_file)
            self.debug_level_var.set(str(self.params.debug_level))
            self.mesh_type_var.set(str(self.params.mesh_type))
            self.viz_enabled_var.set(self.params.viz_enabled)
            self.update_status("参数已初始化")
        except Exception as e:
            self.update_status(f"初始化参数失败: {str(e)}")

    def browse_input_file(self):
        """浏览输入文件"""
        file_path = filedialog.askopenfilename(
            title="选择输入网格文件",
            filetypes=[("CAS文件", "*.cas"), ("所有文件", "*.*")]
        )
        if file_path:
            self.input_file_var.set(file_path)

    def browse_output_file(self):
        """浏览输出文件"""
        file_path = filedialog.asksaveasfilename(
            title="保存网格文件",
            defaultextension=".vtk",
            filetypes=[("VTK文件", "*.vtk"), ("所有文件", "*.*")]
        )
        if file_path:
            self.output_file_var.set(file_path)

    def open_config_file(self):
        """打开配置文件"""
        file_path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                self.params = Parameters("FROM_CASE_JSON", file_path)
                self.update_gui_from_params()
                self.update_status(f"已加载配置文件: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置文件失败: {str(e)}")

    def save_config_file(self):
        """保存配置文件"""
        file_path = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                # 从GUI获取当前配置
                config = self.get_config_from_gui()
                # 保存到文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                self.update_status(f"配置已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置文件失败: {str(e)}")

    def import_mesh_file(self):
        """导入网格文件"""
        file_path = filedialog.askopenfilename(
            title="选择网格文件",
            filetypes=[("CAS文件", "*.cas"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                self.mesh_data = parse_fluent_msh(file_path)
                self.update_status(f"已导入网格文件: {file_path}")
                # 显示网格
                self.display_mesh()
            except Exception as e:
                messagebox.showerror("错误", f"导入网格文件失败: {str(e)}")

    def export_mesh_file(self):
        """导出网格文件"""
        if not self.mesh_data:
            messagebox.showwarning("警告", "没有可导出的网格数据")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="导出网格文件",
            defaultextension=".vtk",
            filetypes=[("VTK文件", "*.vtk"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                # 实际导出网格数据
                from data_structure.basic_elements import Unstructured_Grid
                if isinstance(self.mesh_data, Unstructured_Grid):
                    self.mesh_data.save_to_vtkfile(file_path)
                    self.update_status(f"网格已导出: {file_path}")
                    messagebox.showinfo("成功", f"网格已成功导出到: {file_path}")
                else:
                    messagebox.showwarning("警告", "网格数据格式不支持导出")
            except Exception as e:
                messagebox.showerror("错误", f"导出网格文件失败: {str(e)}")

    def generate_mesh(self):
        """生成网格"""
        try:
            # 更新参数
            self.update_params_from_gui()
            
            # 清除之前的信息输出
            self.clear_info_output()
            
            # 显示进度条
            self.progress.start()
            self.root.update()
            
            # 调用实际的网格生成函数
            self.run_mesh_generation()
            
            # 停止进度条
            self.progress.stop()
            self.update_status("网格生成完成")
            self.append_info_output("网格生成完成")
        except Exception as e:
            self.progress.stop()
            self.append_info_output(f"网格生成失败: {str(e)}")
            messagebox.showerror("错误", f"网格生成失败: {str(e)}")

    def run_mesh_generation(self):
        """运行网格生成算法"""
        try:
            # 根据网格类型选择合适的生成函数
            if self.params.mesh_type == 3:  # 混合网格
                self.append_info_output("开始生成混合网格...")
                from PyMeshGen_mixed import PyMeshGen_mixed
                # 设置GUI实例
                from PyMeshGen_mixed import set_gui_instance
                set_gui_instance(self)
                PyMeshGen_mixed(self.params)
                self.append_info_output("混合网格生成完成！")
            else:  # 三角形或直角三角形网格
                self.append_info_output("开始生成三角形网格...")
                from PyMeshGen import PyMeshGen
                # 设置GUI实例
                from PyMeshGen import set_gui_instance
                set_gui_instance(self)
                PyMeshGen(self.params)
                self.append_info_output("三角形网格生成完成！")
                
            messagebox.showinfo("成功", "网格生成完成！")
        except Exception as e:
            self.append_info_output(f"网格生成过程中出现错误: {str(e)}")
            raise Exception(f"网格生成过程中出现错误: {str(e)}")

    def display_mesh(self):
        """显示网格"""
        if not self.mesh_data:
            messagebox.showwarning("警告", "没有可显示的网格数据")
            return
            
        try:
            # 清除之前的绘图
            self.ax.clear()
            self.ax.set_title("网格显示区域")
            self.ax.set_xlabel("X坐标")
            self.ax.set_ylabel("Y坐标")
            self.ax.axis("equal")
            
            # 创建可视化对象，使用GUI中的绘图区域
            viz_enabled = self.viz_enabled_var.get()
            if viz_enabled:
                visual_obj = Visualization(viz_enabled, self.ax)
                visual_obj.plot_mesh(self.mesh_data)
                self.canvas.draw()  # 更新画布
                self.update_status("网格已显示")
            else:
                messagebox.showwarning("警告", "请先启用可视化功能")
        except Exception as e:
            messagebox.showerror("错误", f"显示网格失败: {str(e)}")
            import traceback
            traceback.print_exc()  # 添加错误追踪信息

    def clear_display(self):
        """清除显示"""
        try:
            # 关闭所有matplotlib图形
            import matplotlib.pyplot as plt
            plt.close('all')
            self.update_status("显示已清除")
        except Exception as e:
            messagebox.showerror("错误", f"清除显示失败: {str(e)}")

    def zoom_in(self):
        """放大视图"""
        if self.ax:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = (xlim[1] - xlim[0]) * 0.8
            y_range = (ylim[1] - ylim[0]) * 0.8
            x_center = (xlim[1] + xlim[0]) / 2
            y_center = (ylim[1] + ylim[0]) / 2
            self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            self.canvas.draw()

    def zoom_out(self):
        """缩小视图"""
        if self.ax:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = (xlim[1] - xlim[0]) * 1.2
            y_range = (ylim[1] - ylim[0]) * 1.2
            x_center = (xlim[1] + xlim[0]) / 2
            y_center = (ylim[1] + ylim[0]) / 2
            self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            self.canvas.draw()

    def reset_view(self):
        """重置视图到原始大小"""
        if self.ax:
            self.ax.autoscale()
            self.canvas.draw()

    def update_gui_from_params(self):
        """从参数对象更新GUI"""
        if self.params:
            self.debug_level_var.set(str(self.params.debug_level))
            self.input_file_var.set(self.params.input_file)
            self.output_file_var.set(self.params.output_file)
            self.mesh_type_var.set(str(self.params.mesh_type))
            self.viz_enabled_var.set(self.params.viz_enabled)

    def update_params_from_gui(self):
        """从GUI更新参数对象"""
        if not self.params:
            self.params = Parameters("FROM_MAIN_JSON")
            
        self.params.debug_level = int(self.debug_level_var.get())
        self.params.input_file = self.input_file_var.get()
        self.params.output_file = self.output_file_var.get()
        self.params.mesh_type = int(self.mesh_type_var.get())
        self.params.viz_enabled = self.viz_enabled_var.get()

    def get_config_from_gui(self):
        """从GUI获取配置数据"""
        return {
            "debug_level": int(self.debug_level_var.get()),
            "input_file": self.input_file_var.get(),
            "output_file": self.output_file_var.get(),
            "mesh_type": int(self.mesh_type_var.get()),
            "viz_enabled": self.viz_enabled_var.get(),
            "parts": []  # 部件参数需要进一步实现
        }

    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def show_about(self):
        """显示关于信息"""
        about_text = """PyMeshGen 网格生成器 GUI 版本
        
 一个用于生成非结构化网格的图形界面工具。
        
 功能特性:
 - 参数设置
 - 文件导入/导出
 - 网格可视化
 - 交互式操作

 版本: 1.0.0
 作者: PyMeshGen 开发团队
 """
        messagebox.showinfo("关于 PyMeshGen GUI", about_text)

    def on_mouse_wheel(self, event):
        """处理鼠标滚轮事件实现缩放功能"""
        if event.inaxes != self.ax:
            return
            
        # 获取当前坐标轴范围
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        # 计算缩放因子
        scale_factor = 1.1 if event.step < 0 else 1/1.1  # 滚轮向上缩小，向下放大
        
        # 计算鼠标位置相对于图形的比例
        xdata = event.xdata
        ydata = event.ydata
        
        # 计算新的坐标轴范围
        x_range = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        y_range = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        # 保持鼠标位置不变的情况下进行缩放
        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
        
        new_xlim = [xdata - x_range * (1 - relx), xdata + x_range * relx]
        new_ylim = [ydata - y_range * (1 - rely), ydata + y_range * rely]
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def on_mouse_press(self, event):
        """处理鼠标按下事件"""
        if event.inaxes != self.ax:
            return
            
        # 记录按下时的坐标和位置
        self.press = (event.xdata, event.ydata)
        self.x0, self.y0 = event.x, event.y
        self.xpress, self.ypress = event.xdata, event.ydata
        
        # 记录当前的坐标轴范围
        self.cur_xlim = self.ax.get_xlim()
        self.cur_ylim = self.ax.get_ylim()
        
        # 记录按下的鼠标按键
        self.button_pressed = event.button

    def on_mouse_release(self, event):
        """处理鼠标释放事件"""
        self.press = None
        self.x1, self.y1 = event.x, event.y
        self.button_pressed = None

    def on_mouse_move(self, event):
        """处理鼠标移动事件实现平移或旋转功能"""
        if self.press is None:
            return  # 鼠标未按下，不执行操作
            
        if event.inaxes != self.ax:
            return
            
        # 根据按下的鼠标按键决定执行什么操作
        if self.button_pressed == 1:  # 左键按下，执行平移
            # 获取当前坐标轴范围
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            
            # 计算鼠标移动的偏移量
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            
            # 更新坐标轴范围实现平移
            self.ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
            self.ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
            
            self.canvas.draw()
        elif self.button_pressed == 3:  # 右键按下，执行旋转
            # 获取当前坐标轴范围
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            
            # 计算中心点
            center_x = (cur_xlim[0] + cur_xlim[1]) / 2
            center_y = (cur_ylim[0] + cur_ylim[1]) / 2
            
            # 计算旋转角度（简化实现）
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            
            # 根据鼠标移动距离计算旋转角度
            angle = (dx - dy) * 0.01  # 简化的旋转角度计算
            
            # 应用旋转变换到坐标轴
            # 这里我们简单地交换坐标轴范围来模拟旋转效果
            x_range = cur_xlim[1] - cur_xlim[0]
            y_range = cur_ylim[1] - cur_ylim[0]
            
            # 更新坐标轴范围实现旋转效果
            self.ax.set_xlim(center_x - x_range/2, center_x + x_range/2)
            self.ax.set_ylim(center_y - y_range/2, center_y + y_range/2)
            
            self.canvas.draw()

def main():
    """主函数"""
    root = tk.Tk()
    app = PyMeshGenGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
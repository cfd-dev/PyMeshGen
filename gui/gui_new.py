#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyMeshGen 简化版 GUI主程序
提供网格显示、信息输出和配置管理功能
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
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
import matplotlib

# 导入消息模块并设置GUI实例
from utils.message import set_gui_instance
from parameters import Parameters
from read_cas import parse_fluent_msh
from mesh_visualization import Visualization


class SimplifiedPyMeshGenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PyMeshGen 网格生成器")
        self.root.geometry("1200x800")
        
        # 设置matplotlib中文字体支持
        self.setup_chinese_font()
        
        # 当前参数对象
        self.params = None
        self.mesh_data = None
        self.config_file_path = None
        
        # 创建GUI界面
        self.create_widgets()
        
        # 初始化参数
        self.init_default_params()
        
        # 设置消息模块的GUI实例引用
        set_gui_instance(self)

    def create_widgets(self):
        """创建GUI界面组件"""
        # 创建菜单栏
        self.create_menu()
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
        file_menu.add_command(label="新建配置", command=self.new_config)
        file_menu.add_command(label="打开配置文件", command=self.open_config_file)
        file_menu.add_command(label="保存配置文件", command=self.save_config_file)
        file_menu.add_command(label="另存为配置文件", command=self.save_as_config_file)
        file_menu.add_separator()
        file_menu.add_command(label="导入网格文件", command=self.import_mesh_file)
        file_menu.add_command(label="导出网格文件", command=self.export_mesh_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 配置菜单
        config_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="配置", menu=config_menu)
        config_menu.add_command(label="编辑配置", command=self.edit_config)
        
        # 网格菜单
        mesh_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="网格", menu=mesh_menu)
        mesh_menu.add_command(label="生成网格", command=self.generate_mesh)
        mesh_menu.add_command(label="显示网格", command=self.display_mesh)
        mesh_menu.add_command(label="清除显示", command=self.clear_display)
        mesh_menu.add_separator()
        mesh_menu.add_command(label="放大", command=self.zoom_in)
        mesh_menu.add_command(label="缩小", command=self.zoom_out)
        mesh_menu.add_command(label="原始大小", command=self.reset_view)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

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
        mesh_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib图形和轴
        # 创建matplotlib图形对象
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("网格显示区域")
        self.ax.set_xlabel("X坐标")
        self.ax.set_ylabel("Y坐标")
        self.ax.axis("equal")
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, mesh_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 设置窗口标题为空字符串，避免显示"Figure 1"
        if hasattr(self.canvas, 'manager') and self.canvas.manager is not None:
            self.canvas.manager.set_window_title('')
        
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
    
    def setup_chinese_font(self):
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

    def create_mesh_display_area(self, parent):
        """创建网格显示区域"""
        # 创建网格显示框架
        mesh_frame = ttk.LabelFrame(parent, text="网格显示区")
        mesh_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib图形和轴
        # 创建matplotlib图形对象
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("网格显示区域")
        self.ax.set_xlabel("X坐标")
        self.ax.set_ylabel("Y坐标")
        # 初始状态不显示坐标轴
        self.ax.set_axis_off()
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, mesh_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 设置窗口标题为空字符串，避免显示"Figure 1"
        if hasattr(self.canvas, 'manager') and self.canvas.manager is not None:
            self.canvas.manager.set_window_title('')
        
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
        info_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建文本框和滚动条
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(text_frame, wrap=tk.WORD, height=15)
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
            self.update_status("参数已初始化")
            
            # 添加欢迎信息到信息输出窗口
            self.append_info_output("欢迎使用PyMeshGen网格生成器!")
            self.append_info_output("请通过'文件'->'打开配置文件'加载配置，然后点击'网格'->'生成网格'开始生成网格。")
            self.append_info_output("-" * 50)
        except Exception as e:
            self.update_status(f"初始化参数失败: {str(e)}")
            self.append_info_output(f"初始化参数失败: {str(e)}")

    def new_config(self):
        """新建配置文件"""
        # 创建默认配置
        default_config = {
            "debug_level": 0,
            "input_file": "",
            "output_file": "",
            "viz_enabled": True,
            "mesh_type": 1,
            "parts": []
        }
        
        # 打开配置编辑对话框
        dialog = ConfigDialog(self.root, default_config)
        config = dialog.result
        
        if config:
            self.config_file_path = None
            # 从配置创建参数对象
            self.create_params_from_config(config)
            self.update_status("新配置已创建")

    def open_config_file(self):
        """打开配置文件"""
        file_path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.config_file_path = file_path
                # 从配置创建参数对象
                self.create_params_from_config(config)
                self.update_status(f"已加载配置文件: {file_path}")
                self.append_info_output(f"已加载配置文件: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置文件失败: {str(e)}")

    def save_config_file(self):
        """保存配置文件"""
        if not self.config_file_path:
            self.save_as_config_file()
            return
            
        try:
            # 从当前参数创建配置
            config = self.create_config_from_params()
            # 保存到文件
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.update_status(f"配置已保存到: {self.config_file_path}")
            self.append_info_output(f"配置已保存到: {self.config_file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置文件失败: {str(e)}")

    def save_as_config_file(self):
        """另存为配置文件"""
        file_path = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                self.config_file_path = file_path
                # 从当前参数创建配置
                config = self.create_config_from_params()
                # 保存到文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                self.update_status(f"配置已保存到: {file_path}")
                self.append_info_output(f"配置已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置文件失败: {str(e)}")

    def create_params_from_config(self, config):
        """从配置创建参数对象"""
        try:
            # 创建临时JSON文件来初始化参数
            temp_config_path = os.path.join(project_root, "temp_config.json")
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            self.params = Parameters("FROM_CASE_JSON", temp_config_path)
            
            # 删除临时文件
            os.remove(temp_config_path)
        except Exception as e:
            messagebox.showerror("错误", f"从配置创建参数失败: {str(e)}")

    def create_config_from_params(self):
        """从参数对象创建配置"""
        if not self.params:
            return {}
            
        # 将Part对象转换为可序列化的字典格式
        parts_data = []
        for part in self.params.part_params:
            # 将Part对象转换为字典
            part_dict = {
                "part_name": part.part_name,
                "part_params": {
                    "part_name": part.part_params.part_name,
                    "max_size": part.part_params.max_size,
                    "PRISM_SWITCH": part.part_params.PRISM_SWITCH,
                    "first_height": part.part_params.first_height,
                    "growth_rate": part.part_params.growth_rate,
                    "growth_method": part.part_params.growth_method,
                    "max_layers": part.part_params.max_layers,
                    "full_layers": part.part_params.full_layers,
                    "multi_direction": part.part_params.multi_direction
                },
                "connectors": []
            }
            
            # 处理connectors
            for connector in part.connectors:
                connector_dict = {
                    "part_name": connector.part_name,
                    "curve_name": connector.curve_name,
                    "param": {
                        "part_name": connector.param.part_name,
                        "max_size": connector.param.max_size,
                        "PRISM_SWITCH": connector.param.PRISM_SWITCH,
                        "first_height": connector.param.first_height,
                        "growth_rate": connector.param.growth_rate,
                        "growth_method": connector.param.growth_method,
                        "max_layers": connector.param.max_layers,
                        "full_layers": connector.param.full_layers,
                        "multi_direction": connector.param.multi_direction
                    }
                }
                part_dict["connectors"].append(connector_dict)
                
            parts_data.append(part_dict)
            
        return {
            "debug_level": self.params.debug_level,
            "input_file": self.params.input_file,
            "output_file": self.params.output_file,
            "viz_enabled": self.params.viz_enabled,
            "mesh_type": self.params.mesh_type,
            "parts": parts_data
        }

    def edit_config(self):
        """编辑配置"""
        if not self.params:
            messagebox.showwarning("警告", "请先加载配置文件")
            return
            
        # 从当前参数创建配置
        config = self.create_config_from_params()
        
        # 打开配置编辑对话框
        dialog = ConfigDialog(self.root, config)
        updated_config = dialog.result
        
        if updated_config:
            # 从更新的配置创建参数对象
            self.create_params_from_config(updated_config)
            self.update_status("配置已更新")
            
    def create_params_from_config(self, config):
        """从配置创建参数对象"""
        # 注意：这个方法需要根据实际需求来实现
        # 目前我们只是更新配置中的基本参数，parts部分需要更复杂的处理
        if not self.params:
            return
            
        # 更新基本参数
        self.params.debug_level = config.get("debug_level", self.params.debug_level)
        self.params.input_file = config.get("input_file", self.params.input_file)
        self.params.output_file = config.get("output_file", self.params.output_file)
        self.params.viz_enabled = config.get("viz_enabled", self.params.viz_enabled)
        self.params.mesh_type = config.get("mesh_type", self.params.mesh_type)
        
        # parts部分的处理比较复杂，需要根据具体需求来实现
        # 这里我们暂时只处理基本参数的更新

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
                self.append_info_output(f"已导入网格文件: {file_path}")
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
                    self.append_info_output(f"网格已导出: {file_path}")
                    messagebox.showinfo("成功", f"网格已成功导出到: {file_path}")
                else:
                    messagebox.showwarning("警告", "网格数据格式不支持导出")
            except Exception as e:
                messagebox.showerror("错误", f"导出网格文件失败: {str(e)}")

    def generate_mesh(self):
        """生成网格"""
        if not self.params:
            messagebox.showwarning("警告", "请先加载配置文件")
            return
            
        try:
            # 检查输入文件是否存在
            input_file = self.params.input_file
            if not input_file or not os.path.exists(input_file):
                messagebox.showerror("错误", f"输入文件不存在: {input_file}\n请检查文件路径或选择有效的输入文件。")
                return
            
            # 清除之前的信息输出
            self.clear_info_output()
            
            # 显示进度信息
            self.append_info_output("开始生成网格...")
            self.update_status("正在生成网格...")
            self.root.update()
            
            # 调用实际的网格生成函数
            self.run_mesh_generation()
            
            self.update_status("网格生成完成")
            self.append_info_output("网格生成完成")
        except Exception as e:
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
            # 尝试从输出文件加载网格数据
            if self.params and self.params.output_file and os.path.exists(self.params.output_file):
                try:
                    from data_structure.basic_elements import Unstructured_Grid
                    self.mesh_data = Unstructured_Grid.from_vtkfile(self.params.output_file)
                    self.append_info_output(f"已从输出文件加载网格数据: {self.params.output_file}")
                except Exception as e:
                    messagebox.showwarning("警告", f"无法从输出文件加载网格数据: {str(e)}")
            
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
            # 确保坐标轴可见
            self.ax.set_axis_on()
            
            # 创建可视化对象，使用GUI中的绘图区域
            if self.params and hasattr(self.params, 'viz_enabled'):
                viz_enabled = self.params.viz_enabled
            else:
                viz_enabled = True
                
            if viz_enabled:
                visual_obj = Visualization(viz_enabled, self.ax)
                visual_obj.plot_mesh(self.mesh_data)
                self.canvas.draw()  # 更新画布
                self.update_status("网格已显示")
                self.append_info_output("网格已显示")
            else:
                messagebox.showwarning("警告", "请在配置中启用可视化功能")
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
            self.append_info_output("显示已清除")
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

    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def show_about(self):
        """显示关于信息"""
        about_text = """PyMeshGen 网格生成器 GUI 版本
        
 一个用于生成非结构化网格的图形界面工具。
        
 功能特性:
 - 简洁的界面设计
 - JSON配置文件管理
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


class ConfigDialog:
    """配置编辑对话框"""
    def __init__(self, parent, config):
        self.top = tk.Toplevel(parent)
        self.top.title("编辑配置")
        self.top.geometry("700x600")
        
        self.result = None
        self.config = config.copy()  # 复制配置以避免修改原始配置
        
        # 创建变量
        self.debug_level_var = tk.StringVar(value=str(config.get("debug_level", 0)))
        self.input_file_var = tk.StringVar(value=config.get("input_file", ""))
        self.output_file_var = tk.StringVar(value=config.get("output_file", ""))
        self.viz_enabled_var = tk.BooleanVar(value=config.get("viz_enabled", True))
        self.mesh_type_var = tk.StringVar(value=str(config.get("mesh_type", 1)))
        
        self.create_widgets(config)
        
        # 使对话框模态
        self.top.transient(parent)
        self.top.grab_set()
        parent.wait_window(self.top)
    
    def create_widgets(self, config):
        """创建对话框组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.top)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 调试级别
        ttk.Label(main_frame, text="调试级别:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        debug_level_combo = ttk.Combobox(main_frame, textvariable=self.debug_level_var, 
                                        values=["0", "1", "2"], width=10)
        debug_level_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 输入文件（只读）
        ttk.Label(main_frame, text="输入文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        input_file_entry = ttk.Entry(main_frame, textvariable=self.input_file_var, width=50, state="readonly")
        input_file_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # 输出文件
        ttk.Label(main_frame, text="输出文件:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        output_file_entry = ttk.Entry(main_frame, textvariable=self.output_file_var, width=50)
        output_file_entry.grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(main_frame, text="浏览", command=self.browse_output_file).grid(row=2, column=2, padx=5, pady=2)
        
        # 网格类型
        ttk.Label(main_frame, text="网格类型:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        mesh_type_frame = ttk.Frame(main_frame)
        mesh_type_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mesh_type_frame, text="三角形", variable=self.mesh_type_var, 
                       value="1").pack(side=tk.LEFT)
        ttk.Radiobutton(mesh_type_frame, text="直角三角形", variable=self.mesh_type_var, 
                       value="2").pack(side=tk.LEFT)
        ttk.Radiobutton(mesh_type_frame, text="混合网格", variable=self.mesh_type_var, 
                       value="3").pack(side=tk.LEFT)
        
        # 可视化开关
        viz_check = ttk.Checkbutton(main_frame, text="启用可视化", variable=self.viz_enabled_var)
        viz_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # 部件配置（使用表格形式显示）
        ttk.Label(main_frame, text="部件配置:").grid(row=5, column=0, sticky=tk.NW, padx=5, pady=2)
        
        # 创建部件配置的表格框架
        parts_frame = ttk.Frame(main_frame)
        parts_frame.grid(row=5, column=1, columnspan=2, padx=5, pady=2, sticky=tk.NSEW)
        
        # 创建Treeview来显示部件配置
        columns = ("部件名称", "最大尺寸", "第一层高度", "增长率", "最大层数")
        self.parts_tree = ttk.Treeview(parts_frame, columns=columns, show="headings", height=8)
        
        # 设置列标题
        for col in columns:
            self.parts_tree.heading(col, text=col)
            self.parts_tree.column(col, width=100)
        
        # 添加滚动条
        parts_scrollbar_y = ttk.Scrollbar(parts_frame, orient=tk.VERTICAL, command=self.parts_tree.yview)
        parts_scrollbar_x = ttk.Scrollbar(parts_frame, orient=tk.HORIZONTAL, command=self.parts_tree.xview)
        self.parts_tree.configure(yscrollcommand=parts_scrollbar_y.set, xscrollcommand=parts_scrollbar_x.set)
        
        # 布局
        self.parts_tree.grid(row=0, column=0, sticky=tk.NSEW)
        parts_scrollbar_y.grid(row=0, column=1, sticky=tk.NS)
        parts_scrollbar_x.grid(row=1, column=0, sticky=tk.EW)
        
        parts_frame.rowconfigure(0, weight=1)
        parts_frame.columnconfigure(0, weight=1)
        
        # 填充部件数据
        self.populate_parts_tree(config.get("parts", []))
        
        # 绑定双击事件以编辑单元格
        self.parts_tree.bind("<Double-1>", self.on_part_double_click)
        
        # 添加部件按钮
        add_part_button = ttk.Button(parts_frame, text="添加部件", command=self.add_part)
        add_part_button.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        # 删除部件按钮
        remove_part_button = ttk.Button(parts_frame, text="删除部件", command=self.remove_part)
        remove_part_button.grid(row=2, column=0, sticky=tk.W, padx=80, pady=5)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)

        # 添加配置文件操作按钮
        ttk.Button(button_frame, text="生成配置文件", command=self.generate_config_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="打开配置文件", command=self.open_config_file_dialog).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="确定", command=self.ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def populate_parts_tree(self, parts):
        """填充部件配置树形视图"""
        # 清空现有项
        for item in self.parts_tree.get_children():
            self.parts_tree.delete(item)
        
        # 添加部件项
        for i, part in enumerate(parts):
            # 获取部件参数
            part_params = part.get("part_params", {})
            part_name = part.get("part_name", f"部件{i+1}")
            max_size = part_params.get("max_size", 1.0)
            first_height = part_params.get("first_height", 0.1)
            growth_rate = part_params.get("growth_rate", 1.2)
            max_layers = part_params.get("max_layers", 3)
            
            # 插入到树形视图
            self.parts_tree.insert("", "end", values=(part_name, max_size, first_height, growth_rate, max_layers), 
                                  tags=(i,))  # 使用tags存储部件索引
    
    def on_part_double_click(self, event):
        """处理部件配置的双击事件以编辑参数"""
        # 获取选中的项
        selected = self.parts_tree.selection()
        if not selected:
            return
            
        # 获取点击的列
        column = self.parts_tree.identify_column(event.x)
        column_index = int(column.replace('#', '')) - 1
        
        # 获取选中项的值和索引
        item = selected[0]
        values = self.parts_tree.item(item, "values")
        part_index = int(self.parts_tree.item(item, "tags")[0])
        
        # 创建编辑窗口
        self.create_edit_window(item, column_index, values, part_index)
    
    def generate_config_file(self):
        """生成配置文件"""
        try:
            # 从当前配置创建配置数据
            config_data = {
                "debug_level": int(self.debug_level_var.get()),
                "input_file": self.input_file_var.get(),
                "output_file": self.output_file_var.get(),
                "viz_enabled": self.viz_enabled_var.get(),
                "mesh_type": int(self.mesh_type_var.get()),
                "parts": self.config.get("parts", [])
            }
            
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                title="保存配置文件",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            
            if file_path:
                # 保存到文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("成功", f"配置文件已保存到: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置文件失败: {str(e)}")
    
    def open_config_file_dialog(self):
        """打开配置文件对话框"""
        try:
            # 选择配置文件
            file_path = filedialog.askopenfilename(
                title="选择配置文件",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            
            if file_path:
                # 读取配置文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 更新界面显示
                self.debug_level_var.set(str(config.get("debug_level", 0)))
                self.input_file_var.set(config.get("input_file", ""))
                self.output_file_var.set(config.get("output_file", ""))
                self.viz_enabled_var.set(config.get("viz_enabled", True))
                self.mesh_type_var.set(str(config.get("mesh_type", 1)))
                
                # 更新配置数据
                self.config = config.copy()
                
                # 更新部件配置显示
                self.populate_parts_tree(config.get("parts", []))
                
                messagebox.showinfo("成功", f"已加载配置文件: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"加载配置文件失败: {str(e)}")
    
    def create_edit_window(self, item, column_index, values, part_index):
        """创建编辑窗口以修改部件参数"""
        # 获取当前值
        current_value = values[column_index]
        
        # 创建顶层窗口
        edit_window = tk.Toplevel(self.top)
        edit_window.title("编辑参数")
        edit_window.geometry("300x100")
        edit_window.transient(self.top)
        edit_window.grab_set()
        
        # 居中显示
        edit_window.geometry("+%d+%d" % (self.top.winfo_rootx()+200, self.top.winfo_rooty()+200))
        
        # 创建输入框
        ttk.Label(edit_window, text="请输入新值:").pack(pady=5)
        entry_var = tk.StringVar(value=str(current_value))
        entry = ttk.Entry(edit_window, textvariable=entry_var, width=30)
        entry.pack(pady=5)
        entry.focus()
        
        # 确定按钮回调
        def save_edit():
            new_value = entry_var.get()
            # 更新显示值
            new_values = list(values)
            new_values[column_index] = new_value
            self.parts_tree.item(item, values=new_values)
            
            # 更新配置数据
            if part_index < len(self.config.get("parts", [])):
                part = self.config["parts"][part_index]
                if "part_params" not in part:
                    part["part_params"] = {}
                    
                # 根据列索引更新对应的参数
                if column_index == 0:  # 部件名称
                    part["part_name"] = new_value
                elif column_index == 1:  # 最大尺寸
                    part["part_params"]["max_size"] = float(new_value) if new_value else 1.0
                elif column_index == 2:  # 第一层高度
                    part["part_params"]["first_height"] = float(new_value) if new_value else 0.1
                elif column_index == 3:  # 增长率
                    part["part_params"]["growth_rate"] = float(new_value) if new_value else 1.2
                elif column_index == 4:  # 最大层数
                    part["part_params"]["max_layers"] = int(new_value) if new_value else 3
            
            edit_window.destroy()
        
        # 创建按钮框架
        button_frame = ttk.Frame(edit_window)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="确定", command=save_edit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=edit_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # 绑定回车键
        entry.bind("<Return>", lambda e: save_edit())
        entry.bind("<Escape>", lambda e: edit_window.destroy())
    
    def add_part(self):
        """添加新部件"""
        # 获取当前部件数量
        parts_count = len(self.config.get("parts", []))
        
        # 创建默认部件配置
        new_part = {
            "part_name": f"部件{parts_count+1}",
            "part_params": {
                "max_size": 1.0,
                "first_height": 0.1,
                "growth_rate": 1.2,
                "max_layers": 3
            }
        }
        
        # 添加到配置中
        if "parts" not in self.config:
            self.config["parts"] = []
        self.config["parts"].append(new_part)
        
        # 更新显示
        self.populate_parts_tree(self.config["parts"])
    
    def remove_part(self):
        """删除选中的部件"""
        selected = self.parts_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要删除的部件")
            return
            
        # 获取选中项的索引
        item = selected[0]
        part_index = int(self.parts_tree.item(item, "tags")[0])
        
        # 从配置中删除
        if "parts" in self.config and 0 <= part_index < len(self.config["parts"]):
            self.config["parts"].pop(part_index)
            
            # 更新显示
            self.populate_parts_tree(self.config["parts"])
    
    def browse_output_file(self):
        """浏览输出文件"""
        file_path = filedialog.asksaveasfilename(
            title="保存网格文件",
            defaultextension=".vtk",
            filetypes=[("VTK文件", "*.vtk"), ("所有文件", "*.*")]
        )
        if file_path:
            self.output_file_var.set(file_path)
    
    def ok(self):
        """确定按钮回调"""
        try:
            self.result = {
                "debug_level": int(self.debug_level_var.get()),
                "input_file": self.input_file_var.get(),
                "output_file": self.output_file_var.get(),
                "viz_enabled": self.viz_enabled_var.get(),
                "mesh_type": int(self.mesh_type_var.get()),
                "parts": self.config.get("parts", [])  # 使用修改后的部件配置
            }
            self.top.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"配置数据格式错误: {str(e)}")
    
    def cancel(self):
        """取消按钮回调"""
        self.top.destroy()


def main():
    """主函数"""
    root = tk.Tk()
    app = SimplifiedPyMeshGenGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
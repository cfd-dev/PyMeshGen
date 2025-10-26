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
                config_data = self.get_config_from_gui()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=False)
                self.update_status(f"配置文件已保存: {file_path}")
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
                # 这里需要实现实际的导出逻辑
                # 暂时显示提示信息
                messagebox.showinfo("提示", f"网格将导出到: {file_path}")
                self.update_status(f"网格已导出: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出网格文件失败: {str(e)}")
                
    def generate_mesh(self):
        """生成网格"""
        try:
            # 更新参数
            self.update_params_from_gui()
            
            # 显示进度条
            self.progress.start()
            self.root.update()
            
            # 这里需要调用实际的网格生成函数
            # 暂时显示提示信息
            messagebox.showinfo("提示", "网格生成功能将在后续版本中实现")
            
            # 停止进度条
            self.progress.stop()
            self.update_status("网格生成完成")
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("错误", f"网格生成失败: {str(e)}")
            
    def display_mesh(self):
        """显示网格"""
        if not self.mesh_data:
            messagebox.showwarning("警告", "没有可显示的网格数据")
            return
            
        try:
            # 创建可视化对象
            viz_enabled = self.viz_enabled_var.get()
            if viz_enabled:
                visual_obj = Visualization(viz_enabled)
                visual_obj.plot_mesh(self.mesh_data)
                self.update_status("网格已显示")
            else:
                messagebox.showwarning("警告", "请先启用可视化功能")
        except Exception as e:
            messagebox.showerror("错误", f"显示网格失败: {str(e)}")
            
    def clear_display(self):
        """清除显示"""
        try:
            # 关闭所有matplotlib图形
            import matplotlib.pyplot as plt
            plt.close('all')
            self.update_status("显示已清除")
        except Exception as e:
            messagebox.showerror("错误", f"清除显示失败: {str(e)}")
            
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


def main():
    """主函数"""
    root = tk.Tk()
    app = PyMeshGenGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
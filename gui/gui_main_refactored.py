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
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧面板
        self.create_left_panel()
        
        # 创建右侧面板
        self.create_right_panel()
        
        # 创建状态栏
        self.status_bar = StatusBar(self.root)
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
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
        self.left_panel = ttk.Frame(self.main_frame, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_panel.pack_propagate(False)
        
        # 参数设置区域
        params_frame = ttk.LabelFrame(self.left_panel, text="参数设置")
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
        
        # 操作按钮区域
        button_frame = ttk.LabelFrame(self.left_panel, text="操作")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="新建配置", command=self.new_config).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="打开配置", command=self.open_config).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="保存配置", command=self.save_config).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="参数设置", command=self.edit_params).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="生成网格", command=self.generate_mesh).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="显示网格", command=self.display_mesh).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="导入网格", command=self.import_mesh).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="导出网格", command=self.export_mesh).pack(fill=tk.X, padx=5, pady=2)
        
        # 部件管理区域
        parts_frame = ttk.LabelFrame(self.left_panel, text="部件管理")
        parts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建部件列表
        self.parts_listbox = tk.Listbox(parts_frame, height=6)
        self.parts_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 部件操作按钮
        parts_button_frame = ttk.Frame(parts_frame)
        parts_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(parts_button_frame, text="添加部件", command=self.add_part).pack(side=tk.LEFT, padx=2)
        ttk.Button(parts_button_frame, text="删除部件", command=self.remove_part).pack(side=tk.LEFT, padx=2)
        ttk.Button(parts_button_frame, text="编辑部件", command=self.edit_part).pack(side=tk.LEFT, padx=2)
    
    def create_right_panel(self):
        """创建右侧面板"""
        # 右侧面板框架
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建网格显示区域
        self.mesh_display = MeshDisplayArea(self.right_panel)
        self.mesh_display.pack(fill=tk.BOTH, expand=True)
        
        # 创建信息输出区域
        self.info_output = InfoOutput(self.right_panel)
    
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
            
            # 创建网格生成器
            self.mesh_generator = MeshGenerator(self.params)
            
            # 生成网格
            self.current_mesh = self.mesh_generator.generate()
            
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
            messagebox.showwarning("警告", "没有可显示的网格")
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
        
        self.update_parts_list()
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
            
        self.debug_level_var.set(str(self.params.debug_level))
        self.input_file_var.set(self.params.input_file or "")
        self.output_file_var.set(self.params.output_file or "")
        self.viz_enabled_var.set(self.params.viz_enabled)
        self.mesh_type_var.set(str(self.params.mesh_type))
        
        self.update_parts_list()
    
    def update_parts_list(self):
        """更新部件列表"""
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
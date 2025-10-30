#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理模块
处理配置文件的加载和保存
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from .gui_base import DialogBase


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, project_root):
        self.project_root = project_root
        
    def load_config(self, file_path):
        """加载配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            raise Exception(f"加载配置文件失败: {str(e)}")
            
    def save_config(self, config, file_path):
        """保存配置文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            raise Exception(f"保存配置文件失败: {str(e)}")
            
    def create_params_from_config(self, config):
        """从配置创建参数对象"""
        try:
            # 导入Parameters类
            try:
                from parameters import Parameters
            except ImportError:
                from data_structure.parameters import Parameters
                
            # 创建临时JSON文件来初始化参数
            temp_config_path = os.path.join(self.project_root, "temp_config.json")
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            params = Parameters("FROM_CASE_JSON", temp_config_path)
            
            # 删除临时文件
            os.remove(temp_config_path)
            return params
        except Exception as e:
            raise Exception(f"从配置创建参数失败: {str(e)}")
            
    def create_config_from_params(self, params):
        """从参数对象创建配置"""
        if not params:
            return {}
            
        try:
            # 将Part对象转换为可序列化的字典格式
            parts_data = []
            for part in params.part_params:
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
                "debug_level": params.debug_level,
                "input_file": params.input_file,
                "output_file": params.output_file,
                "viz_enabled": params.viz_enabled,
                "mesh_type": params.mesh_type,
                "parts": parts_data
            }
        except Exception as e:
            raise Exception(f"从参数创建配置失败: {str(e)}")


class ConfigDialog(DialogBase):
    """配置编辑对话框"""
    
    def __init__(self, parent, config):
        super().__init__(parent, "编辑配置", "700x600")
        self.config = config.copy()  # 复制配置以避免修改原始配置
        
        # 创建变量
        self.debug_level_var = tk.StringVar(value=str(config.get("debug_level", 0)))
        self.input_file_var = tk.StringVar(value=config.get("input_file", ""))
        self.output_file_var = tk.StringVar(value=config.get("output_file", ""))
        self.viz_enabled_var = tk.BooleanVar(value=config.get("viz_enabled", True))
        self.mesh_type_var = tk.StringVar(value=str(config.get("mesh_type", 1)))
        
        self.create_widgets(config)
        
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
        
        # 添加配置文件操作按钮
        ttk.Button(main_frame, text="生成配置文件", command=self.generate_config_file).grid(row=6, column=0, pady=10)
        ttk.Button(main_frame, text="打开配置文件", command=self.open_config_file_dialog).grid(row=6, column=1, pady=10)
    
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
        except ValueError:
            messagebox.showerror("错误", "请确保所有数值字段都输入了有效的数字")
        except Exception as e:
            messagebox.showerror("错误", f"配置数据格式错误: {str(e)}")
    
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
        except ValueError:
            messagebox.showerror("错误", "请确保所有数值字段都输入了有效的数字")
        except Exception as e:
            messagebox.showerror("错误", f"配置数据格式错误: {str(e)}")
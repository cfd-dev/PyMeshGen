#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件操作模块
处理网格文件的导入和导出
"""

import os
import vtk
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from .gui_base import DialogBase


class FileOperations:
    """文件操作类"""
    
    def __init__(self, project_root):
        self.project_root = project_root
        self.supported_import_formats = [".vtk", ".stl", ".obj", ".ply"]
        self.supported_export_formats = [".vtk", ".stl", ".obj", ".ply"]
    
    def import_mesh(self, file_path):
        """导入网格文件"""
        try:
            # 根据文件扩展名选择适当的读取器
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == ".vtk":
                reader = vtk.vtkPolyDataReader()
            elif file_ext == ".stl":
                reader = vtk.vtkSTLReader()
            elif file_ext == ".obj":
                reader = vtk.vtkOBJReader()
            elif file_ext == ".ply":
                reader = vtk.vtkPLYReader()
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            reader.SetFileName(file_path)
            reader.Update()
            
            poly_data = reader.GetOutput()
            if not poly_data or poly_data.GetNumberOfPoints() == 0:
                raise ValueError("导入的网格文件为空或无效")
            
            return poly_data
        except Exception as e:
            raise Exception(f"导入网格文件失败: {str(e)}")
    
    def export_mesh(self, poly_data, file_path):
        """导出网格文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 根据文件扩展名选择适当的写入器
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == ".vtk":
                writer = vtk.vtkPolyDataWriter()
            elif file_ext == ".stl":
                writer = vtk.vtkSTLWriter()
            elif file_ext == ".obj":
                writer = vtk.vtkOBJWriter()
            elif file_ext == ".ply":
                writer = vtk.vtkPLYWriter()
            else:
                raise ValueError(f"不支持的导出格式: {file_ext}")
            
            writer.SetFileName(file_path)
            writer.SetInputData(poly_data)
            writer.Write()
            
            return True
        except Exception as e:
            raise Exception(f"导出网格文件失败: {str(e)}")
    
    def get_mesh_info(self, poly_data):
        """获取网格信息"""
        if not poly_data:
            return None
            
        num_points = poly_data.GetNumberOfPoints()
        num_cells = poly_data.GetNumberOfCells()
        
        # 计算边界框
        bounds = poly_data.GetBounds()
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        
        # 计算尺寸
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        
        return {
            "num_points": num_points,
            "num_cells": num_cells,
            "bounds": bounds,
            "min_point": (min_x, min_y, min_z),
            "max_point": (max_x, max_y, max_z),
            "size": (size_x, size_y, size_z),
            "center": ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
        }


class ImportDialog(DialogBase):
    """导入网格文件对话框"""
    
    def __init__(self, parent, file_operations):
        super().__init__(parent, "导入网格文件", "600x400")
        self.file_operations = file_operations
        self.selected_file = None
        self.mesh_info = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """创建对话框组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.top)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="选择文件")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 文件路径显示
        ttk.Label(file_frame, text="文件路径:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        self.file_path_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # 浏览按钮
        ttk.Button(file_frame, text="浏览", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # 文件格式说明
        format_frame = ttk.LabelFrame(main_frame, text="支持的文件格式")
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        format_text = "支持的文件格式: " + ", ".join(self.file_operations.supported_import_formats)
        ttk.Label(format_frame, text=format_text).pack(padx=5, pady=5)
        
        # 网格信息显示区域
        info_frame = ttk.LabelFrame(main_frame, text="网格信息")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建文本框显示网格信息
        self.info_text = tk.Text(info_frame, height=10, width=70, state=tk.DISABLED)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 预览选项
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preview_frame, text="导入后显示预览", variable=self.preview_var).pack(side=tk.LEFT, padx=5)
    
    def browse_file(self):
        """浏览文件"""
        file_path = filedialog.askopenfilename(
            title="选择网格文件",
            filetypes=[
                ("所有支持的文件", "*.vtk *.stl *.obj *.ply"),
                ("VTK文件", "*.vtk"),
                ("STL文件", "*.stl"),
                ("OBJ文件", "*.obj"),
                ("PLY文件", "*.ply"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.load_mesh_info(file_path)
    
    def load_mesh_info(self, file_path):
        """加载网格信息"""
        try:
            # 导入网格文件
            poly_data = self.file_operations.import_mesh(file_path)
            
            # 获取网格信息
            self.mesh_info = self.file_operations.get_mesh_info(poly_data)
            
            # 更新信息显示
            self.update_info_display()
            
            self.selected_file = file_path
        except Exception as e:
            messagebox.showerror("错误", f"加载网格文件失败: {str(e)}")
            self.selected_file = None
            self.mesh_info = None
            self.update_info_display()
    
    def update_info_display(self):
        """更新信息显示"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        if self.mesh_info:
            info_str = f"文件路径: {self.selected_file}\n\n"
            info_str += f"顶点数量: {self.mesh_info['num_points']}\n"
            info_str += f"单元数量: {self.mesh_info['num_cells']}\n\n"
            info_str += f"边界框:\n"
            info_str += f"  X: {self.mesh_info['bounds'][0]:.4f} ~ {self.mesh_info['bounds'][1]:.4f}\n"
            info_str += f"  Y: {self.mesh_info['bounds'][2]:.4f} ~ {self.mesh_info['bounds'][3]:.4f}\n"
            info_str += f"  Z: {self.mesh_info['bounds'][4]:.4f} ~ {self.mesh_info['bounds'][5]:.4f}\n\n"
            info_str += f"尺寸:\n"
            info_str += f"  X: {self.mesh_info['size'][0]:.4f}\n"
            info_str += f"  Y: {self.mesh_info['size'][1]:.4f}\n"
            info_str += f"  Z: {self.mesh_info['size'][2]:.4f}\n\n"
            info_str += f"中心点: ({self.mesh_info['center'][0]:.4f}, {self.mesh_info['center'][1]:.4f}, {self.mesh_info['center'][2]:.4f})"
            
            self.info_text.insert(tk.END, info_str)
        else:
            self.info_text.insert(tk.END, "请选择有效的网格文件")
        
        self.info_text.config(state=tk.DISABLED)
    
    def ok(self):
        """确定按钮回调"""
        if not self.selected_file:
            messagebox.showwarning("警告", "请先选择一个有效的网格文件")
            return
            
        self.result = {
            "file_path": self.selected_file,
            "mesh_info": self.mesh_info,
            "preview": self.preview_var.get()
        }
        self.top.destroy()


class ExportDialog(DialogBase):
    """导出网格文件对话框"""
    
    def __init__(self, parent, file_operations, poly_data):
        super().__init__(parent, "导出网格文件", "600x400")
        self.file_operations = file_operations
        self.poly_data = poly_data
        
        # 获取网格信息
        self.mesh_info = file_operations.get_mesh_info(poly_data) if poly_data else None
        
        self.create_widgets()
    
    def create_widgets(self):
        """创建对话框组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.top)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="输出文件")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 文件路径显示
        ttk.Label(file_frame, text="保存路径:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        self.file_path_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # 浏览按钮
        ttk.Button(file_frame, text="浏览", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # 文件格式选择
        ttk.Label(file_frame, text="文件格式:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.format_var = tk.StringVar(value=".vtk")
        format_combo = ttk.Combobox(file_frame, textvariable=self.format_var, 
                                   values=self.file_operations.supported_export_formats, 
                                   width=10, state="readonly")
        format_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        format_combo.bind("<<ComboboxSelected>>", self.on_format_change)
        
        # 网格信息显示区域
        info_frame = ttk.LabelFrame(main_frame, text="网格信息")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建文本框显示网格信息
        self.info_text = tk.Text(info_frame, height=10, width=70, state=tk.DISABLED)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 更新信息显示
        self.update_info_display()
    
    def browse_file(self):
        """浏览文件"""
        # 根据选择的格式设置默认扩展名
        file_ext = self.format_var.get()
        default_ext = file_ext if file_ext.startswith(".") else f".{file_ext}"
        
        file_path = filedialog.asksaveasfilename(
            title="保存网格文件",
            defaultextension=default_ext,
            filetypes=[
                ("所有支持的文件", "*.vtk *.stl *.obj *.ply"),
                ("VTK文件", "*.vtk"),
                ("STL文件", "*.stl"),
                ("OBJ文件", "*.obj"),
                ("PLY文件", "*.ply"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
    
    def on_format_change(self, event):
        """格式改变事件处理"""
        # 更新文件扩展名
        current_path = self.file_path_var.get()
        if current_path:
            # 移除旧扩展名
            path_without_ext = os.path.splitext(current_path)[0]
            # 添加新扩展名
            new_ext = self.format_var.get()
            new_path = f"{path_without_ext}{new_ext}"
            self.file_path_var.set(new_path)
    
    def update_info_display(self):
        """更新信息显示"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        if self.mesh_info:
            info_str = f"顶点数量: {self.mesh_info['num_points']}\n"
            info_str += f"单元数量: {self.mesh_info['num_cells']}\n\n"
            info_str += f"边界框:\n"
            info_str += f"  X: {self.mesh_info['bounds'][0]:.4f} ~ {self.mesh_info['bounds'][1]:.4f}\n"
            info_str += f"  Y: {self.mesh_info['bounds'][2]:.4f} ~ {self.mesh_info['bounds'][3]:.4f}\n"
            info_str += f"  Z: {self.mesh_info['bounds'][4]:.4f} ~ {self.mesh_info['bounds'][5]:.4f}\n\n"
            info_str += f"尺寸:\n"
            info_str += f"  X: {self.mesh_info['size'][0]:.4f}\n"
            info_str += f"  Y: {self.mesh_info['size'][1]:.4f}\n"
            info_str += f"  Z: {self.mesh_info['size'][2]:.4f}\n\n"
            info_str += f"中心点: ({self.mesh_info['center'][0]:.4f}, {self.mesh_info['center'][1]:.4f}, {self.mesh_info['center'][2]:.4f})"
            
            self.info_text.insert(tk.END, info_str)
        else:
            self.info_text.insert(tk.END, "没有有效的网格数据")
        
        self.info_text.config(state=tk.DISABLED)
    
    def ok(self):
        """确定按钮回调"""
        if not self.poly_data:
            messagebox.showwarning("警告", "没有可导出的网格数据")
            return
            
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showwarning("警告", "请指定输出文件路径")
            return
            
        try:
            # 导出网格文件
            self.file_operations.export_mesh(self.poly_data, file_path)
            self.result = {
                "file_path": file_path,
                "success": True
            }
            self.top.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"导出网格文件失败: {str(e)}")
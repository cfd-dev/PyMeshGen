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
        self.supported_import_formats = [".vtk", ".stl", ".obj", ".ply", ".cas"]
        self.supported_export_formats = [".vtk", ".stl", ".obj", ".ply"]
    
    def extract_parts_from_cas(self, mesh_data):
        """从cas文件导入的网格数据中提取部件信息"""
        parts_info = []
        
        # 检查是否是cas文件导入的数据
        if isinstance(mesh_data, dict) and mesh_data.get('type') == 'cas' and 'unstr_grid' in mesh_data:
            unstr_grid = mesh_data['unstr_grid']
            
            # 检查是否有边界信息
            if hasattr(unstr_grid, 'boundary_info') and unstr_grid.boundary_info:
                # 从边界信息中提取部件
                for part_name, part_data in unstr_grid.boundary_info.items():
                    # 创建部件信息字典
                    part_info = {
                        'part_name': part_name,
                        'bc_type': part_data.get('type', 'unknown'),
                        'face_count': len(part_data.get('faces', [])),
                        'nodes': set(),
                        'cells': set()
                    }
                    
                    # 收集部件中的节点和单元
                    for face in part_data.get('faces', []):
                        # 添加节点
                        for node_idx in face.get('nodes', []):
                            part_info['nodes'].add(node_idx - 1)  # 转换为0基索引
                        
                        # 添加单元
                        left_cell = face.get('left_cell', 0)
                        right_cell = face.get('right_cell', 0)
                        if left_cell > 0:
                            part_info['cells'].add(left_cell - 1)  # 转换为0基索引
                        if right_cell > 0:
                            part_info['cells'].add(right_cell - 1)  # 转换为0基索引
                    
                    # 转换set为list以便序列化
                    part_info['nodes'] = list(part_info['nodes'])
                    part_info['cells'] = list(part_info['cells'])
                    
                    parts_info.append(part_info)
        
        return parts_info
    
    def import_mesh(self, file_path):
        """导入网格文件"""
        try:
            # 根据文件扩展名选择适当的读取器
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == ".vtk":
                # 尝试使用vtkUnstructuredGridReader读取非结构化网格
                reader = vtk.vtkUnstructuredGridReader()
                reader.SetFileName(file_path)
                reader.Update()
                unstructured_grid = reader.GetOutput()
                
                # 如果读取失败，尝试使用vtkPolyDataReader
                if not unstructured_grid or unstructured_grid.GetNumberOfPoints() == 0:
                    reader = vtk.vtkPolyDataReader()
                    reader.SetFileName(file_path)
                    reader.Update()
                    poly_data = reader.GetOutput()
                else:
                    # 将非结构化网格转换为多边形数据
                    geometry_filter = vtk.vtkGeometryFilter()
                    geometry_filter.SetInputData(unstructured_grid)
                    geometry_filter.Update()
                    poly_data = geometry_filter.GetOutput()
                
                # 将vtkPolyData转换为统一的数据结构
                if poly_data and poly_data.GetNumberOfPoints() > 0:
                    # 获取节点坐标
                    points = poly_data.GetPoints()
                    num_points = points.GetNumberOfPoints()
                    node_coords = []
                    for i in range(num_points):
                        x, y, z = points.GetPoint(i)
                        node_coords.append([x, y, z])  # 保存x,y,z坐标
                    
                    # 获取单元信息
                    num_cells = poly_data.GetNumberOfCells()
                    cells = []
                    for i in range(num_cells):
                        cell = poly_data.GetCell(i)
                        if cell.GetCellType() == vtk.VTK_TRIANGLE:
                            # 三角形单元
                            point_ids = []
                            for j in range(cell.GetNumberOfPoints()):
                                point_ids.append(cell.GetPointId(j))
                            cells.append(point_ids)
                        elif cell.GetCellType() == vtk.VTK_QUAD:
                            # 四边形单元
                            point_ids = []
                            for j in range(cell.GetNumberOfPoints()):
                                point_ids.append(cell.GetPointId(j))
                            cells.append(point_ids)
                    
                    # 创建统一的数据结构
                    mesh_data = {
                        'type': 'vtk',
                        'node_coords': node_coords,
                        'cells': cells,
                        'num_points': num_points,
                        'num_cells': len(cells),
                        'vtk_poly_data': poly_data  # 保留原始VTK数据以备后用
                    }
                    return mesh_data
                else:
                    raise ValueError("导入的VTK文件为空或无效")
                    
            elif file_ext == ".stl":
                reader = vtk.vtkSTLReader()
                reader.SetFileName(file_path)
                reader.Update()
                poly_data = reader.GetOutput()
                
                # 将vtkPolyData转换为统一的数据结构
                if poly_data and poly_data.GetNumberOfPoints() > 0:
                    # 获取节点坐标
                    points = poly_data.GetPoints()
                    num_points = points.GetNumberOfPoints()
                    node_coords = []
                    for i in range(num_points):
                        x, y, z = points.GetPoint(i)
                        node_coords.append([x, y, z])  # 保存x,y,z坐标
                    
                    # 获取单元信息
                    num_cells = poly_data.GetNumberOfCells()
                    cells = []
                    for i in range(num_cells):
                        cell = poly_data.GetCell(i)
                        if cell.GetCellType() == vtk.VTK_TRIANGLE:
                            # 三角形单元
                            point_ids = []
                            for j in range(cell.GetNumberOfPoints()):
                                point_ids.append(cell.GetPointId(j))
                            cells.append(point_ids)
                    
                    # 创建统一的数据结构
                    mesh_data = {
                        'type': 'stl',
                        'node_coords': node_coords,
                        'cells': cells,
                        'num_points': num_points,
                        'num_cells': len(cells),
                        'vtk_poly_data': poly_data  # 保留原始VTK数据以备后用
                    }
                    return mesh_data
                else:
                    raise ValueError("导入的STL文件为空或无效")
                    
            elif file_ext == ".obj":
                reader = vtk.vtkOBJReader()
                reader.SetFileName(file_path)
                reader.Update()
                poly_data = reader.GetOutput()
                
                # 将vtkPolyData转换为统一的数据结构
                if poly_data and poly_data.GetNumberOfPoints() > 0:
                    # 获取节点坐标
                    points = poly_data.GetPoints()
                    num_points = points.GetNumberOfPoints()
                    node_coords = []
                    for i in range(num_points):
                        x, y, z = points.GetPoint(i)
                        node_coords.append([x, y, z])  # 保存x,y,z坐标
                    
                    # 获取单元信息
                    num_cells = poly_data.GetNumberOfCells()
                    cells = []
                    for i in range(num_cells):
                        cell = poly_data.GetCell(i)
                        if cell.GetCellType() == vtk.VTK_TRIANGLE:
                            # 三角形单元
                            point_ids = []
                            for j in range(cell.GetNumberOfPoints()):
                                point_ids.append(cell.GetPointId(j))
                            cells.append(point_ids)
                        elif cell.GetCellType() == vtk.VTK_QUAD:
                            # 四边形单元
                            point_ids = []
                            for j in range(cell.GetNumberOfPoints()):
                                point_ids.append(cell.GetPointId(j))
                            cells.append(point_ids)
                    
                    # 创建统一的数据结构
                    mesh_data = {
                        'type': 'obj',
                        'node_coords': node_coords,
                        'cells': cells,
                        'num_points': num_points,
                        'num_cells': len(cells),
                        'vtk_poly_data': poly_data  # 保留原始VTK数据以备后用
                    }
                    return mesh_data
                else:
                    raise ValueError("导入的OBJ文件为空或无效")
                    
            elif file_ext == ".ply":
                reader = vtk.vtkPLYReader()
                reader.SetFileName(file_path)
                reader.Update()
                poly_data = reader.GetOutput()
                
                # 将vtkPolyData转换为统一的数据结构
                if poly_data and poly_data.GetNumberOfPoints() > 0:
                    # 获取节点坐标
                    points = poly_data.GetPoints()
                    num_points = points.GetNumberOfPoints()
                    node_coords = []
                    for i in range(num_points):
                        x, y, z = points.GetPoint(i)
                        node_coords.append([x, y, z])  # 保存x,y,z坐标
                    
                    # 获取单元信息
                    num_cells = poly_data.GetNumberOfCells()
                    cells = []
                    for i in range(num_cells):
                        cell = poly_data.GetCell(i)
                        if cell.GetCellType() == vtk.VTK_TRIANGLE:
                            # 三角形单元
                            point_ids = []
                            for j in range(cell.GetNumberOfPoints()):
                                point_ids.append(cell.GetPointId(j))
                            cells.append(point_ids)
                        elif cell.GetCellType() == vtk.VTK_QUAD:
                            # 四边形单元
                            point_ids = []
                            for j in range(cell.GetNumberOfPoints()):
                                point_ids.append(cell.GetPointId(j))
                            cells.append(point_ids)
                    
                    # 创建统一的数据结构
                    mesh_data = {
                        'type': 'ply',
                        'node_coords': node_coords,
                        'cells': cells,
                        'num_points': num_points,
                        'num_cells': len(cells),
                        'vtk_poly_data': poly_data  # 保留原始VTK数据以备后用
                    }
                    return mesh_data
                else:
                    raise ValueError("导入的PLY文件为空或无效")
                    
            elif file_ext == ".cas":
                # 对于cas文件，使用专门的解析器
                import sys
                sys.path.append(os.path.join(self.project_root, 'fileIO'))
                from read_cas import parse_cas_to_unstr_grid
                
                # 解析cas文件并转换为Unstructured_Grid对象
                unstr_grid = parse_cas_to_unstr_grid(file_path)
                
                # 将Unstructured_Grid转换为统一的数据结构
                if unstr_grid and hasattr(unstr_grid, 'node_coords') and hasattr(unstr_grid, 'cell_container'):
                    # 获取节点坐标
                    node_coords = unstr_grid.node_coords
                    
                    # 获取单元信息
                    cells = []
                    for cell in unstr_grid.cell_container:
                        if hasattr(cell, 'nodes'):
                            point_ids = []
                            for node in cell.nodes:
                                point_ids.append(node.idx)
                            cells.append(point_ids)
                    
                    # 创建统一的数据结构
                    mesh_data = {
                        'type': 'cas',
                        'node_coords': node_coords,
                        'cells': cells,
                        'num_points': len(node_coords),
                        'num_cells': len(cells),
                        'unstr_grid': unstr_grid  # 保留原始Unstructured_Grid对象以备后用
                    }
                    
                    # 提取部件信息
                    parts_info = self.extract_parts_from_cas(mesh_data)
                    if parts_info:
                        mesh_data['parts_info'] = parts_info
                    
                    return mesh_data
                else:
                    raise ValueError("导入的CAS文件为空或无效")
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
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
    
    def get_mesh_info(self, mesh_data):
        """获取网格信息"""
        if not mesh_data:
            return None
        
        # 检查是否是字典类型（VTK/STL/OBJ/PLY文件导入的结果）
        if isinstance(mesh_data, dict) and 'node_coords' in mesh_data and 'cells' in mesh_data:
            # 处理字典类型的网格数据
            node_coords = mesh_data['node_coords']
            cells = mesh_data['cells']
            
            num_points = len(node_coords)
            num_cells = len(cells)
            
            # 计算边界框
            if num_points > 0:
                coords = np.array(node_coords)
                
                # 检查坐标维度
                if coords.shape[1] >= 3:
                    # 三维坐标
                    min_x, min_y, min_z = np.min(coords, axis=0)
                    max_x, max_y, max_z = np.max(coords, axis=0)
                elif coords.shape[1] == 2:
                    # 二维坐标，添加z=0
                    min_x, min_y = np.min(coords, axis=0)
                    max_x, max_y = np.max(coords, axis=0)
                    min_z = max_z = 0.0
                else:
                    # 其他情况，使用默认值
                    min_x = min_y = min_z = 0.0
                    max_x = max_y = max_z = 1.0
                
                # 计算尺寸
                size_x = max_x - min_x
                size_y = max_y - min_y
                size_z = max_z - min_z
                
                return {
                    "num_points": num_points,
                    "num_cells": num_cells,
                    "bounds": (min_x, max_x, min_y, max_y, min_z, max_z),
                    "min_point": (min_x, min_y, min_z),
                    "max_point": (max_x, max_y, max_z),
                    "size": (size_x, size_y, size_z),
                    "center": ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
                }
            else:
                return {
                    "num_points": 0,
                    "num_cells": 0,
                    "bounds": (0, 0, 0, 0, 0, 0),
                    "min_point": (0, 0, 0),
                    "max_point": (0, 0, 0),
                    "size": (0, 0, 0),
                    "center": (0, 0, 0)
                }
        # 检查是否是Unstructured_Grid对象（cas文件导入的结果）
        elif hasattr(mesh_data, 'node_coords') and hasattr(mesh_data, 'cell_container'):
            # 处理Unstructured_Grid对象
            num_points = len(mesh_data.node_coords)
            num_cells = len(mesh_data.cell_container)
            
            # 计算边界框
            if num_points > 0:
                coords = np.array(mesh_data.node_coords)
                min_x, min_y, min_z = np.min(coords, axis=0)
                max_x, max_y, max_z = np.max(coords, axis=0)
                
                # 计算尺寸
                size_x = max_x - min_x
                size_y = max_y - min_y
                size_z = max_z - min_z
                
                return {
                    "num_points": num_points,
                    "num_cells": num_cells,
                    "bounds": (min_x, max_x, min_y, max_y, min_z, max_z),
                    "min_point": (min_x, min_y, min_z),
                    "max_point": (max_x, max_y, max_z),
                    "size": (size_x, size_y, size_z),
                    "center": ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
                }
            else:
                return {
                    "num_points": 0,
                    "num_cells": 0,
                    "bounds": (0, 0, 0, 0, 0, 0),
                    "min_point": (0, 0, 0),
                    "max_point": (0, 0, 0),
                    "size": (0, 0, 0),
                    "center": (0, 0, 0)
                }
        else:
            # 处理VTK PolyData对象
            num_points = mesh_data.GetNumberOfPoints()
            num_cells = mesh_data.GetNumberOfCells()
            
            # 计算边界框
            bounds = mesh_data.GetBounds()
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
                ("所有支持的文件", "*.vtk *.stl *.obj *.ply *.cas"),
                ("VTK文件", "*.vtk"),
                ("STL文件", "*.stl"),
                ("OBJ文件", "*.obj"),
                ("PLY文件", "*.ply"),
                ("CAS文件", "*.cas"),
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
            mesh_data = self.file_operations.import_mesh(file_path)
            
            # 检查返回的数据类型
            if isinstance(mesh_data, dict):
                # 如果是字典，直接使用其中的信息
                if 'node_coords' in mesh_data and 'cells' in mesh_data:
                    # 从字典中提取网格信息
                    node_coords = mesh_data['node_coords']
                    cells = mesh_data['cells']
                    
                    num_points = len(node_coords)
                    num_cells = len(cells)
                    
                    # 计算边界框
                    if num_points > 0:
                        coords = np.array(node_coords)
                        min_x, min_y, min_z = np.min(coords, axis=0)
                        max_x, max_y, max_z = np.max(coords, axis=0)
                        
                        # 计算尺寸
                        size_x = max_x - min_x
                        size_y = max_y - min_y
                        size_z = max_z - min_z
                        
                        self.mesh_info = {
                            "num_points": num_points,
                            "num_cells": num_cells,
                            "bounds": (min_x, max_x, min_y, max_y, min_z, max_z),
                            "min_point": (min_x, min_y, min_z),
                            "max_point": (max_x, max_y, max_z),
                            "size": (size_x, size_y, size_z),
                            "center": ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
                        }
                    else:
                        self.mesh_info = {
                            "num_points": 0,
                            "num_cells": 0,
                            "bounds": (0, 0, 0, 0, 0, 0),
                            "min_point": (0, 0, 0),
                            "max_point": (0, 0, 0),
                            "size": (0, 0, 0),
                            "center": (0, 0, 0)
                        }
                else:
                    self.mesh_info = None
            else:
                # 如果是VTK对象或Unstructured_Grid对象，使用get_mesh_info方法
                self.mesh_info = self.file_operations.get_mesh_info(mesh_data)
            
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
    
    def __init__(self, parent, file_operations, mesh_data):
        super().__init__(parent, "导出网格文件", "600x400")
        self.file_operations = file_operations
        self.mesh_data = mesh_data
        
        # 获取网格信息
        self.mesh_info = file_operations.get_mesh_info(mesh_data) if mesh_data else None
        
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
        if not self.mesh_data:
            messagebox.showwarning("警告", "没有可导出的网格数据")
            return
            
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showwarning("警告", "请指定输出文件路径")
            return
            
        try:
            # 检查是否是Unstructured_Grid对象（cas文件导入的结果）
            if hasattr(self.mesh_data, 'node_coords') and hasattr(self.mesh_data, 'cell_container'):
                # 对于Unstructured_Grid对象，使用save_to_vtkfile方法
                self.mesh_data.save_to_vtkfile(file_path)
            else:
                # 对于VTK PolyData对象，使用file_operations的export_mesh方法
                self.file_operations.export_mesh(self.mesh_data, file_path)
                
            self.result = {
                "file_path": file_path,
                "success": True
            }
            self.top.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"导出网格文件失败: {str(e)}")
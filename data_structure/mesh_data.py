#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用网格数据结构
用于统一处理不同格式的网格数据
"""

class MeshData:
    """通用网格数据类"""
    
    def __init__(self, file_path=None, mesh_type=None):
        """
        初始化网格数据对象
        
        Args:
            file_path: 网格文件路径
            mesh_type: 网格类型（如vtk、stl、obj、cas、msh、ply）
        """
        self.file_path = file_path
        self.mesh_type = mesh_type
        
        # 基本网格数据
        self.node_coords = []  # 节点坐标列表，每个元素为[x, y, z]
        self.cells = []  # 单元列表，每个元素为节点索引列表
        self.num_points = 0  # 节点数量
        self.num_cells = 0  # 单元数量
        self.dimension = 2  # 网格维度 (2D/3D)
        
        # 扩展数据
        self.vtk_poly_data = None  # VTK多边形数据对象
        self.unstr_grid = None  # 非结构化网格对象
        
        # 部件和边界信息
        self.parts_info = {}  # 部件信息，格式：{part_name: part_data}
        self.boundary_info = {}  # 边界信息
        
        # 网格质量数据
        self.quality_data = {}  # 网格质量数据
    
    def update_counts(self):
        """更新节点和单元数量"""
        self.num_points = len(self.node_coords)
        self.num_cells = len(self.cells)
    
    def from_dict(self, mesh_dict):
        """
        从字典创建网格数据对象
        
        Args:
            mesh_dict: 包含网格数据的字典
        """
        # 基本属性
        self.mesh_type = mesh_dict.get('type', None)
        self.dimension = mesh_dict.get('dimension', 2)
        self.file_path = mesh_dict.get('file_path', None)
        
        # 网格数据
        self.node_coords = mesh_dict.get('node_coords', [])
        self.cells = mesh_dict.get('cells', [])
        
        # 扩展数据
        self.vtk_poly_data = mesh_dict.get('vtk_poly_data', None)
        self.unstr_grid = mesh_dict.get('unstr_grid', None)
        
        # 部件和边界信息
        self.parts_info = mesh_dict.get('parts_info', {})
        self.boundary_info = mesh_dict.get('boundary_info', {})
        
        # 更新计数
        self.update_counts()
    
    def to_dict(self):
        """
        将网格数据对象转换为字典
        
        Returns:
            包含网格数据的字典
        """
        return {
            'type': self.mesh_type,
            'file_path': self.file_path,
            'node_coords': self.node_coords,
            'cells': self.cells,
            'num_points': self.num_points,
            'num_cells': self.num_cells,
            'dimension': self.dimension,
            'vtk_poly_data': self.vtk_poly_data,
            'unstr_grid': self.unstr_grid,
            'parts_info': self.parts_info,
            'boundary_info': self.boundary_info,
            'quality_data': self.quality_data
        }
    
    def __repr__(self):
        """返回网格数据的字符串表示"""
        return f"MeshData(type={self.mesh_type}, points={self.num_points}, cells={self.num_cells})"

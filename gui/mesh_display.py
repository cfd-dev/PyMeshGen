#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网格显示模块
处理网格可视化和交互功能
使用VTK进行网格渲染和显示
"""

import os
import tkinter as tk
from tkinter import ttk
import vtk
from vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
from .gui_base import BaseFrame, LabelFrameWrapper


class MeshDisplayArea(BaseFrame):
    """网格显示区域类，使用VTK进行渲染"""
    
    def __init__(self, parent, figsize=(16, 9), dpi=100):
        super().__init__(parent)
        self.figsize = figsize
        self.dpi = dpi
        self.mesh_data = None
        self.params = None
        
        # VTK相关组件
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.mesh_actor = None
        self.boundary_actors = []
        
        # 创建网格显示区域
        self.create_mesh_display_area()
        
    def create_mesh_display_area(self):
        """创建VTK网格显示区域"""
        # 创建主框架
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建VTK渲染窗口和交互器
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        
        # 创建VTK交互器
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        # 创建一个Tkinter框架来容纳VTK窗口
        self.vtk_frame = ttk.Frame(main_frame)
        self.vtk_frame.pack(fill=tk.BOTH, expand=True)
        
        # 使用GetRenderWindowId来获取窗口ID，然后在Tkinter中嵌入
        self.render_window.SetSize(800, 600)
        self.render_window.SetOffScreenRendering(0)  # 确保不是离屏渲染
        
        # 初始化交互器
        self.interactor.Initialize()
        self.interactor.CreateRepeatingTimer(100)  # 100ms定时器，用于更新
        
        # 设置背景色
        self.renderer.SetBackground(0.9, 0.9, 0.9)
        self.renderer.ResetCamera()
        
        # 初始化网格演员列表
        self.mesh_actor = None
        self.boundary_actors = []
        
        # 创建工具栏
        self.create_toolbar(main_frame)
        
    def create_toolbar(self, parent):
        """创建工具栏"""
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # 重置视图按钮
        reset_btn = ttk.Button(toolbar_frame, text="重置视图", command=self.reset_view)
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # 适应视图按钮
        fit_btn = ttk.Button(toolbar_frame, text="适应视图", command=self.fit_view)
        fit_btn.pack(side=tk.LEFT, padx=5)
        
        # 显示边界按钮
        self.show_boundary_var = tk.BooleanVar(value=True)
        boundary_btn = ttk.Checkbutton(toolbar_frame, text="显示边界", 
                                      variable=self.show_boundary_var,
                                      command=self.toggle_boundary_display)
        boundary_btn.pack(side=tk.LEFT, padx=5)
        
        # 线框/实体模式切换
        self.wireframe_var = tk.BooleanVar(value=False)
        wireframe_btn = ttk.Checkbutton(toolbar_frame, text="线框模式", 
                                       variable=self.wireframe_var,
                                       command=self.toggle_wireframe)
        wireframe_btn.pack(side=tk.LEFT, padx=5)
        
    def set_mesh_data(self, mesh_data):
        """设置网格数据"""
        self.mesh_data = mesh_data
        
    def set_params(self, params):
        """设置参数对象"""
        self.params = params
        
    def display_mesh(self, mesh_data=None):
        """显示网格"""
        # 如果提供了mesh_data参数，则更新self.mesh_data
        if mesh_data is not None:
            self.mesh_data = mesh_data
            
        if not self.mesh_data:
            # 尝试从输出文件加载网格数据
            if self.params and self.params.output_file and os.path.exists(self.params.output_file):
                try:
                    from fileIO.vtk_io import parse_vtk_msh
                    self.mesh_data = parse_vtk_msh(self.params.output_file)
                except Exception as e:
                    print(f"无法从输出文件加载网格数据: {str(e)}")
            
            if not self.mesh_data:
                return False
                
        try:
            # 清除之前的网格
            self.clear_mesh_actors()
            
            # 检查可视化是否启用
            if self.params and hasattr(self.params, 'viz_enabled'):
                viz_enabled = self.params.viz_enabled
            else:
                viz_enabled = True
                
            if viz_enabled:
                # 根据网格数据类型创建VTK网格
                vtk_mesh = self.create_vtk_mesh()
                
                if vtk_mesh:
                    # 创建映射器和演员
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(vtk_mesh)
                    
                    # 创建网格演员
                    self.mesh_actor = vtk.vtkActor()
                    self.mesh_actor.SetMapper(mapper)
                    
                    # 设置属性
                    self.mesh_actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # 浅灰色
                    self.mesh_actor.GetProperty().SetLineWidth(1.0)
                    
                    # 根据wireframe_var设置显示模式
                    if self.wireframe_var.get():
                        self.mesh_actor.GetProperty().SetRepresentationToWireframe()
                    else:
                        self.mesh_actor.GetProperty().SetRepresentationToSurface()
                    
                    # 添加到渲染器
                    self.renderer.AddActor(self.mesh_actor)
                    
                    # 如果需要，显示边界
                    if self.show_boundary_var.get():
                        self.display_boundary()
                    
                    # 重置相机以显示整个网格
                    self.renderer.ResetCamera()
                    
                    # 更新渲染窗口
                    self.render_window.Render()
                    
                    # 如果交互器没有启动，则启动它
                    if not self.interactor.GetInitialized():
                        self.interactor.Initialize()
                    
                    # 确保VTK窗口嵌入到Tkinter中
                    if hasattr(self.render_window, 'GetGenericWindowId'):
                        # 获取VTK窗口ID
                        vtk_window_id = self.render_window.GetGenericWindowId()
                        
                        # 在Tkinter中创建一个Frame来容纳VTK窗口
                        if not hasattr(self, 'vtk_embed_frame'):
                            import tkinter as tk
                            self.vtk_embed_frame = tk.Frame(self.vtk_frame, width=800, height=600)
                            self.vtk_embed_frame.pack(fill=tk.BOTH, expand=True)
                        
                        # 设置窗口父级
                        if hasattr(tk, '_default_root'):
                            tk_root = tk._default_root
                            if tk_root:
                                self.render_window.SetWindowInfo(str(tk_root.winfo_id()))
                        
                        # 显示窗口
                        self.render_window.Start()
                    
                    return True
                else:
                    print("无法创建VTK网格")
                    return False
            else:
                return False
        except Exception as e:
            print(f"显示网格失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def create_vtk_mesh(self):
        """根据网格数据创建VTK网格对象"""
        try:
            # 创建VTK点和单元
            points = vtk.vtkPoints()
            polys = vtk.vtkCellArray()
            
            # 处理不同类型的网格数据
            if isinstance(self.mesh_data, dict) and 'type' in self.mesh_data:
                if self.mesh_data['type'] in ['vtk', 'stl', 'obj', 'ply']:
                    # 使用统一的数据结构
                    node_coords = self.mesh_data['node_coords']
                    cells = self.mesh_data['cells']
                    
                    # 添加点
                    for i, coord in enumerate(node_coords):
                        if len(coord) >= 2:
                            if len(coord) == 2:
                                points.InsertNextPoint(coord[0], coord[1], 0.0)
                            else:
                                points.InsertNextPoint(coord[0], coord[1], coord[2])
                    
                    # 添加单元
                    for cell in cells:
                        if len(cell) == 3:  # 三角形
                            triangle = vtk.vtkTriangle()
                            triangle.GetPointIds().SetId(0, cell[0])
                            triangle.GetPointIds().SetId(1, cell[1])
                            triangle.GetPointIds().SetId(2, cell[2])
                            polys.InsertNextCell(triangle)
                        elif len(cell) == 4:  # 四边形
                            quad = vtk.vtkQuad()
                            quad.GetPointIds().SetId(0, cell[0])
                            quad.GetPointIds().SetId(1, cell[1])
                            quad.GetPointIds().SetId(2, cell[2])
                            quad.GetPointIds().SetId(3, cell[3])
                            polys.InsertNextCell(quad)
                
                elif self.mesh_data['type'] == 'cas':
                    # 对于.cas文件，使用Unstructured_Grid对象
                    if 'unstr_grid' in self.mesh_data:
                        return self.create_vtk_mesh_from_unstr_grid(self.mesh_data['unstr_grid'])
            
            elif hasattr(self.mesh_data, 'node_coords') and hasattr(self.mesh_data, 'cells'):
                # 处理具有node_coords和cells属性的对象
                node_coords = self.mesh_data.node_coords
                cells = self.mesh_data.cells
                
                # 添加点
                for i, coord in enumerate(node_coords):
                    if len(coord) >= 2:
                        if len(coord) == 2:
                            points.InsertNextPoint(coord[0], coord[1], 0.0)
                        else:
                            points.InsertNextPoint(coord[0], coord[1], coord[2])
                
                # 添加单元
                for cell in cells:
                    if len(cell) == 3:  # 三角形
                        triangle = vtk.vtkTriangle()
                        triangle.GetPointIds().SetId(0, cell[0])
                        triangle.GetPointIds().SetId(1, cell[1])
                        triangle.GetPointIds().SetId(2, cell[2])
                        polys.InsertNextCell(triangle)
                    elif len(cell) == 4:  # 四边形
                        quad = vtk.vtkQuad()
                        quad.GetPointIds().SetId(0, cell[0])
                        quad.GetPointIds().SetId(1, cell[1])
                        quad.GetPointIds().SetId(2, cell[2])
                        quad.GetPointIds().SetId(3, cell[3])
                        polys.InsertNextCell(quad)
            
            elif hasattr(self.mesh_data, 'node_coords') and hasattr(self.mesh_data, 'cell_container'):
                # 如果是Unstructured_Grid对象
                return self.create_vtk_mesh_from_unstr_grid(self.mesh_data)
            
            # 创建VTK多边形数据
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)
            
            return polydata
            
        except Exception as e:
            print(f"创建VTK网格失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def create_vtk_mesh_from_unstr_grid(self, unstr_grid):
        """从Unstructured_Grid对象创建VTK网格"""
        try:
            # 创建VTK点和单元
            points = vtk.vtkPoints()
            polys = vtk.vtkCellArray()
            
            # 添加点
            for i, coord in enumerate(unstr_grid.node_coords):
                if len(coord) >= 2:
                    if len(coord) == 2:
                        points.InsertNextPoint(coord[0], coord[1], 0.0)
                    else:
                        points.InsertNextPoint(coord[0], coord[1], coord[2])
            
            # 添加单元
            for cell in unstr_grid.cell_container:
                if cell is None:
                    continue
                    
                node_ids = cell.node_ids
                if len(node_ids) == 3:  # 三角形
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, node_ids[0])
                    triangle.GetPointIds().SetId(1, node_ids[1])
                    triangle.GetPointIds().SetId(2, node_ids[2])
                    polys.InsertNextCell(triangle)
                elif len(node_ids) == 4:  # 四边形
                    quad = vtk.vtkQuad()
                    quad.GetPointIds().SetId(0, node_ids[0])
                    quad.GetPointIds().SetId(1, node_ids[1])
                    quad.GetPointIds().SetId(2, node_ids[2])
                    quad.GetPointIds().SetId(3, node_ids[3])
                    polys.InsertNextCell(quad)
            
            # 创建VTK多边形数据
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)
            
            return polydata
            
        except Exception as e:
            print(f"从Unstructured_Grid创建VTK网格失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def display_boundary(self):
        """显示边界"""
        try:
            # 清除之前的边界演员
            for actor in self.boundary_actors:
                self.renderer.RemoveActor(actor)
            self.boundary_actors.clear()
            
            # 获取边界信息
            boundary_info = None
            
            if isinstance(self.mesh_data, dict) and 'unstr_grid' in self.mesh_data:
                unstr_grid = self.mesh_data['unstr_grid']
                if hasattr(unstr_grid, 'boundary_info'):
                    boundary_info = unstr_grid.boundary_info
            elif hasattr(self.mesh_data, 'boundary_info'):
                boundary_info = self.mesh_data.boundary_info
            
            if not boundary_info:
                return
                
            # 定义边界类型颜色映射
            bc_colors = {
                "wall": (1.0, 0.0, 0.0),  # 红色
                "pressure-inlet": (0.0, 1.0, 0.0),  # 绿色
                "pressure-outlet": (0.0, 0.0, 1.0),  # 蓝色
                "symmetry": (1.0, 1.0, 0.0),  # 黄色
                "pressure-far-field": (0.0, 1.0, 1.0),  # 青色
                "unspecified": (0.5, 0.5, 0.5),  # 灰色
            }
            
            # 为每个边界区域创建演员
            for zone_name, zone_data in boundary_info.items():
                bc_type = zone_data.get("bc_type", "unspecified")
                faces = zone_data.get("faces", [])
                
                if not faces:
                    continue
                    
                # 创建边界多边形数据
                boundary_points = vtk.vtkPoints()
                boundary_polys = vtk.vtkCellArray()
                point_map = {}  # 用于映射原始点到新点索引
                
                # 添加面
                for face in faces:
                    # 确保面是有效的
                    if len(face) < 2:
                        continue
                        
                    # 创建多边形
                    polygon = vtk.vtkPolygon()
                    polygon.GetPointIds().SetNumberOfIds(len(face))
                    
                    # 添加点并设置多边形顶点
                    for i, node_id in enumerate(face):
                        if node_id not in point_map:
                            # 添加新点
                            coord = self.mesh_data.node_coords[node_id] if hasattr(self.mesh_data, 'node_coords') else self.mesh_data['node_coords'][node_id]
                            if len(coord) == 2:
                                point_id = boundary_points.InsertNextPoint(coord[0], coord[1], 0.0)
                            else:
                                point_id = boundary_points.InsertNextPoint(coord[0], coord[1], coord[2])
                            point_map[node_id] = point_id
                        else:
                            point_id = point_map[node_id]
                            
                        polygon.GetPointIds().SetId(i, point_id)
                    
                    boundary_polys.InsertNextCell(polygon)
                
                # 创建边界多边形数据
                boundary_polydata = vtk.vtkPolyData()
                boundary_polydata.SetPoints(boundary_points)
                boundary_polydata.SetPolys(boundary_polys)
                
                # 创建边界映射器和演员
                boundary_mapper = vtk.vtkPolyDataMapper()
                boundary_mapper.SetInputData(boundary_polydata)
                
                boundary_actor = vtk.vtkActor()
                boundary_actor.SetMapper(boundary_mapper)
                
                # 设置边界颜色
                color = bc_colors.get(bc_type, (0.5, 0.5, 0.5))
                boundary_actor.GetProperty().SetColor(color)
                boundary_actor.GetProperty().SetLineWidth(2.0)
                
                # 添加到渲染器和边界演员列表
                self.renderer.AddActor(boundary_actor)
                self.boundary_actors.append(boundary_actor)
                
        except Exception as e:
            print(f"显示边界失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def clear_mesh_actors(self):
        """清除所有网格相关的演员"""
        # 清除主网格演员
        if self.mesh_actor:
            self.renderer.RemoveActor(self.mesh_actor)
            self.mesh_actor = None
            
        # 清除边界演员
        for actor in self.boundary_actors:
            self.renderer.RemoveActor(actor)
        self.boundary_actors.clear()
        
    def clear_display(self):
        """清除显示"""
        try:
            # 清除所有演员
            self.clear_mesh_actors()
            
            # 更新渲染窗口
            if self.render_window:
                self.render_window.Render()
        except Exception as e:
            print(f"清除显示失败: {str(e)}")
            
    def clear(self):
        """清除显示（别名方法，与clear_display功能相同）"""
        self.clear_display()
            
    def zoom_in(self):
        """放大视图"""
        if self.renderer:
            camera = self.renderer.GetActiveCamera()
            camera.Zoom(1.2)
            self.render_window.Render()

    def zoom_out(self):
        """缩小视图"""
        if self.renderer:
            camera = self.renderer.GetActiveCamera()
            camera.Zoom(0.8)
            self.render_window.Render()

    def reset_view(self):
        """重置视图到原始大小"""
        if self.renderer:
            self.renderer.ResetCamera()
            self.render_window.Render()
            
    def fit_view(self):
        """适应视图以显示所有内容"""
        if self.renderer:
            self.renderer.ResetCamera()
            self.render_window.Render()
            
    def toggle_boundary_display(self):
        """切换边界显示"""
        if self.mesh_actor:
            if self.show_boundary_var.get():
                self.display_boundary()
            else:
                # 清除边界演员
                for actor in self.boundary_actors:
                    self.renderer.RemoveActor(actor)
                self.boundary_actors.clear()
            
            self.render_window.Render()
            
    def toggle_wireframe(self):
        """切换线框/实体模式"""
        if self.mesh_actor:
            if self.wireframe_var.get():
                self.mesh_actor.GetProperty().SetRepresentationToWireframe()
            else:
                self.mesh_actor.GetProperty().SetRepresentationToSurface()
            
            self.render_window.Render()
            
    def pan_view(self):
        """平移视图"""
        # VTK的交互器默认支持鼠标左键旋转，右键平移，滚轮缩放
        # 这里我们只更新状态提示，实际平移由鼠标交互完成
        pass
            
    def rotate_view(self):
        """旋转视图"""
        # VTK的交互器默认支持鼠标左键旋转，右键平移，滚轮缩放
        # 这里我们只更新状态提示，实际旋转由鼠标交互完成
        pass
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
        
        # 显示控制变量
        self.show_boundary_var = tk.BooleanVar(value=True)
        self.wireframe_var = tk.BooleanVar(value=False)
        
        # 创建网格显示区域
        self.create_mesh_display_area()
        
    def create_mesh_display_area(self):
        """创建VTK网格显示区域"""
        # 创建主框架
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建VTK渲染器
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.9, 0.9, 0.9)
        
        # 创建VTK渲染窗口
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)
        
        # 创建一个Tkinter框架来容纳VTK窗口
        self.vtk_frame = ttk.Frame(main_frame)
        self.vtk_frame.pack(fill=tk.BOTH, expand=True)
        
        # 初始化交互器
        self.interactor = None
        self.embedded = False
        
        # 延迟嵌入VTK窗口，确保框架已经完全创建
        self.vtk_frame.after(100, self.embed_vtk_window)
        
        # 重置相机
        self.renderer.ResetCamera()
        
        # 初始化网格演员列表
        self.mesh_actor = None
        self.boundary_actors = []
    
    def embed_vtk_window(self):
        """将VTK窗口嵌入到Tkinter框架中"""
        try:
            # 创建VTK交互器
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.render_window)
            
            # 检测操作系统
            import platform
            system = platform.system()
            
            # Windows系统使用特殊处理
            if system == "Windows":
                try:
                    # 获取窗口句柄
                    window_id = self.vtk_frame.winfo_id()
                    
                    # 设置窗口信息
                    self.render_window.SetWindowInfo(str(window_id))
                    
                    # 初始化交互器
                    self.interactor.Initialize()
                    
                    # 设置窗口大小
                    self.vtk_frame.update_idletasks()
                    width = self.vtk_frame.winfo_width()
                    height = self.vtk_frame.winfo_height()
                    if width <= 1 or height <= 1:
                        width, height = 800, 600
                    self.render_window.SetSize(width, height)
                    
                    # 不要启动交互器，只初始化它
                    # self.interactor.Start()  # 这行会导致GIL问题
                    
                    # 强制更新
                    self.vtk_frame.update_idletasks()
                    
                    print("使用Windows窗口ID方法成功嵌入VTK窗口")
                    self.embedded = True
                    return True
                    
                except Exception as e:
                    print(f"Windows窗口ID方法失败: {str(e)}")
                    
                    # 尝试备用方法
                    try:
                        # 直接创建一个简单的渲染窗口
                        self.render_window.SetOffScreenRendering(0)
                        self.interactor.Initialize()
                        
                        # 设置窗口大小
                        self.vtk_frame.update_idletasks()
                        width = self.vtk_frame.winfo_width()
                        height = self.vtk_frame.winfo_height()
                        if width <= 1 or height <= 1:
                            width, height = 800, 600
                        self.render_window.SetSize(width, height)
                        
                        # 不要启动交互器，只初始化它
                        # self.interactor.Start()  # 这行会导致GIL问题
                        
                        # 强制更新
                        self.vtk_frame.update_idletasks()
                        
                        print("使用Windows备用方法成功嵌入VTK窗口")
                        self.embedded = True
                        return True
                        
                    except Exception as e2:
                        print(f"Windows备用方法也失败: {str(e2)}")
            
            # 尝试使用vtkTkRenderWindowInteractor
            try:
                # 导入vtkTkRenderWindowInteractor
                from vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
                
                # 创建Tkinter渲染窗口交互器
                self.render_window_interactor = vtkTkRenderWindowInteractor(
                    self.vtk_frame, rw=self.render_window
                )
                self.render_window_interactor.pack(fill=tk.BOTH, expand=True)
                
                # 设置交互器
                self.interactor.SetRenderWindow(self.render_window)
                
                # 初始化交互器
                self.interactor.Initialize()
                
                # 强制更新
                self.vtk_frame.update_idletasks()
                
                print("使用vtkTkRenderWindowInteractor成功嵌入VTK窗口")
                self.embedded = True
                return True
                
            except ImportError:
                print("vtkTkRenderWindowInteractor不可用，尝试其他方法")
            except Exception as e:
                print(f"vtkTkRenderWindowInteractor方法失败: {str(e)}")
            
            # 备用方法：直接使用GetRenderWindow()
            try:
                # 获取窗口ID
                window_id = self.vtk_frame.winfo_id()
                
                # 设置窗口信息
                self.render_window.SetWindowInfo(str(window_id))
                
                # 初始化交互器
                self.interactor.Initialize()
                
                # 强制更新
                self.vtk_frame.update_idletasks()
                
                print("使用GetRenderWindow()方法成功嵌入VTK窗口")
                self.embedded = True
                return True
                
            except Exception as e:
                print(f"GetRenderWindow()方法失败: {str(e)}")
            
            # 所有方法都失败
            print("无法嵌入VTK窗口")
            self.embedded = False
            return False
            
        except Exception as e:
            print(f"嵌入VTK窗口时发生错误: {str(e)}")
            self.embedded = False
            return False
        
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
            # 确保VTK窗口已嵌入
            if not self.embedded or not self.interactor:
                print("VTK窗口未正确嵌入，尝试重新嵌入...")
                self.embed_vtk_window()
                # 等待嵌入完成
                import time
                time.sleep(0.5)
                
            # 如果仍然未嵌入，则返回失败
            if not self.embedded or not self.interactor:
                print("VTK窗口嵌入失败，无法显示网格")
                return False
                
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
                    
                    # 检测操作系统
                    import platform
                    system = platform.system()
                    
                    # 如果交互器没有启动，则启动它
                    if not self.interactor.GetInitialized():
                        self.interactor.Initialize()
                    
                    # 对于macOS，可能需要特殊的渲染处理
                    if system == "Darwin":
                        # 强制更新窗口
                        self.vtk_frame.update_idletasks()
                        self.render_window.Render()
                        
                        # 如果窗口仍然没有显示，尝试重新初始化
                        if hasattr(self, 'render_window_interactor'):
                            try:
                                self.render_window_interactor.Render()
                            except:
                                pass
                    
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
            
    def clear_boundary_actors(self):
        """清除所有边界演员"""
        for actor in self.boundary_actors:
            self.renderer.RemoveActor(actor)
        self.boundary_actors = []
    
    def toggle_boundary_display(self, show_boundary=None):
        """切换边界显示"""
        if show_boundary is not None:
            self.show_boundary_var.set(show_boundary)
        
        if self.show_boundary_var.get():
            self.display_boundary()
        else:
            self.clear_boundary_actors()
        
        # 更新渲染窗口
        if self.render_window:
            self.render_window.Render()
    
    def toggle_points(self, show_points=True):
        """切换点云显示模式"""
        if self.mesh_actor:
            if show_points:
                self.mesh_actor.GetProperty().SetRepresentationToPoints()
                self.mesh_actor.GetProperty().SetPointSize(3.0)
            else:
                # 切换回表面模式
                self.mesh_actor.GetProperty().SetRepresentationToSurface()
            
            # 更新渲染窗口
            if self.render_window:
                self.render_window.Render()
    
    def reset_view(self):
        """重置视图到原始大小"""
        if self.renderer:
            self.renderer.ResetCamera()
            if self.render_window:
                self.render_window.Render()
    
    def reset_camera(self):
        """重置相机以适应整个网格"""
        if self.renderer:
            self.renderer.ResetCamera()
            if self.render_window:
                self.render_window.Render()
    
    def zoom_in(self):
        """放大视图"""
        if self.renderer:
            camera = self.renderer.GetActiveCamera()
            camera.Zoom(1.2)
            if self.render_window:
                self.render_window.Render()
    
    def zoom_out(self):
        """缩小视图"""
        if self.renderer:
            camera = self.renderer.GetActiveCamera()
            camera.Zoom(0.8)
            if self.render_window:
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
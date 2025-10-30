#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网格显示模块
处理网格可视化和交互功能
使用VTK进行网格渲染和显示
"""

import os
import time
import tkinter as tk
from tkinter import ttk
import vtk
from vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
from .gui_base import BaseFrame, LabelFrameWrapper


class MeshDisplayArea(BaseFrame):
    """网格显示区域类，使用VTK进行渲染"""
    
    def __init__(self, parent, figsize=(16, 9), dpi=100, offscreen=False):
        super().__init__(parent)
        self.figsize = figsize
        self.dpi = dpi
        self.mesh_data = None
        self.params = None
        self.offscreen = offscreen  # 离屏渲染模式标志
        
        # VTK相关组件
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.mesh_actor = None
        self.boundary_actors = []
        self.axes_actor = None  # 添加坐标轴演员引用
        
        # 显示控制变量
        self.show_boundary_var = tk.BooleanVar(value=True)
        self.wireframe_var = tk.BooleanVar(value=False)
        self.render_mode = tk.StringVar(value="surface")  # 新增：渲染模式 (surface/wireframe/points)
        
        # 渲染状态标志
        self._render_in_progress = False
        self.embedded = False
        
        # 创建网格显示区域
        self.create_mesh_display_area()
        
        # 设置网格显示区域可以接收键盘焦点
        if hasattr(self.frame, 'focus_set'):
            self.frame.focus_set()
        if hasattr(self.frame, 'config'):
            self.frame.config(takefocus=True)
    
    def __del__(self):
        """清理资源"""
        self.cleanup()
    
    def cleanup(self):
        """清理VTK资源以避免内存泄漏"""
        # 清除边界演员
        if self.renderer and self.boundary_actors:
            for actor in self.boundary_actors:
                try:
                    self.renderer.RemoveActor(actor)
                except:
                    pass
            self.boundary_actors.clear()
        
        # 清除网格演员
        if self.renderer and self.mesh_actor:
            try:
                self.renderer.RemoveActor(self.mesh_actor)
            except:
                pass
            self.mesh_actor = None
        
        # 清除坐标轴演员
        if self.renderer and self.axes_actor:
            try:
                self.renderer.RemoveActor(self.axes_actor)
            except:
                pass
            self.axes_actor = None
        
        # 清除渲染器中的所有演员
        if self.renderer:
            try:
                self.renderer.RemoveAllViewProps()
            except:
                pass
        
        # 停止交互器
        if self.interactor:
            try:
                self.interactor.TerminateApp()
            except:
                pass
    
    def create_mesh_display_area(self):
        """创建VTK网格显示区域"""
        # 创建主框架（仅在非离屏模式下创建）
        if not self.offscreen:
            main_frame = ttk.Frame(self.frame)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # 设置主框架可以接收键盘焦点
            main_frame.bind("<Button-1>", lambda e: main_frame.focus_set())
            main_frame.config(takefocus=True)
        
        # 创建VTK渲染器
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)  # 设置为白色背景
        
        # 创建VTK渲染窗口
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)
        
        # 离屏渲染模式下启用离屏渲染
        if self.offscreen:
            self.render_window.SetOffScreenRendering(1)
            self.embedded = True  # 离屏模式下不需要嵌入
        else:
            # 创建一个Tkinter框架来容纳VTK窗口
            self.vtk_frame = ttk.Frame(main_frame)
            self.vtk_frame.pack(fill=tk.BOTH, expand=True)
            
            # 设置VTK框架可以接收键盘焦点
            self.vtk_frame.bind("<Button-1>", lambda e: self.vtk_frame.focus_set())
            self.vtk_frame.config(takefocus=True)
            
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
        # 离屏渲染模式下不需要嵌入
        if self.offscreen:
            return True
            
        try:
            # 创建VTK交互器
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.render_window)
            
            # 检测操作系统
            system = self._get_system()
            
            # Windows系统使用特殊处理
            if system == "Windows":
                try:
                    # 获取窗口句柄
                    window_id = self.vtk_frame.winfo_id()
                    
                    # 设置窗口信息
                    self.render_window.SetWindowInfo(str(window_id))
                    
                    # 初始化交互器
                    self.interactor.Initialize()
                    
                    # 设置交互器样式，支持鼠标操作
                    style = vtk.vtkInteractorStyleTrackballCamera()
                    self.interactor.SetInteractorStyle(style)
                    
                    # 设置窗口大小
                    self._update_window_size()
                    
                    # 添加渲染窗口回调，确保渲染持续进行
                    self.render_window.AddObserver('EndEvent', self.on_render_end)
                    
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
                        
                        # 设置交互器样式，支持鼠标操作
                        style = vtk.vtkInteractorStyleTrackballCamera()
                        self.interactor.SetInteractorStyle(style)
                        
                        # 设置窗口大小
                        self._update_window_size()
                        
                        # 添加渲染窗口回调，确保渲染持续进行
                        self.render_window.AddObserver('EndEvent', self.on_render_end)
                        
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
                self.interactor = self.render_window_interactor.GetRenderWindow().GetInteractor()
                
                # 初始化交互器
                self.interactor.Initialize()
                
                # 设置交互器样式，支持鼠标操作
                style = vtk.vtkInteractorStyleTrackballCamera()
                self.interactor.SetInteractorStyle(style)
                
                # 添加渲染窗口回调，确保渲染持续进行
                self.render_window.AddObserver('EndEvent', self.on_render_end)
                
                # 设置可以接收键盘焦点
                self.render_window_interactor.bind("<Button-1>", lambda e: self.render_window_interactor.focus_set())
                self.render_window_interactor.config(takefocus=True)
                
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
                
                # 设置交互器样式，支持鼠标操作
                style = vtk.vtkInteractorStyleTrackballCamera()
                self.interactor.SetInteractorStyle(style)
                
                # 添加渲染窗口回调，确保渲染持续进行
                self.render_window.AddObserver('EndEvent', self.on_render_end)
                
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
    
    def _get_system(self):
        """获取操作系统名称"""
        import platform
        return platform.system()
    
    def _update_window_size(self):
        """更新窗口大小"""
        self.vtk_frame.update_idletasks()
        width = self.vtk_frame.winfo_width()
        height = self.vtk_frame.winfo_height()
        if width <= 1 or height <= 1:
            width, height = 800, 600
        self.render_window.SetSize(width, height)
    
    def on_render_end(self, obj, event):
        """渲染结束事件回调，确保网格不会消失"""
        try:
            # 如果渲染窗口存在，强制更新
            if self.render_window:
                # 防止递归调用，使用标志检查
                if not hasattr(self, '_render_in_progress'):
                    self._render_in_progress = True
                    self.render_window.Render()
                    self._render_in_progress = False
        except Exception as e:
            print(f"渲染回调错误: {str(e)}")
            # 确保在出错时重置标志
            if hasattr(self, '_render_in_progress'):
                self._render_in_progress = False
    
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
            # 确保VTK窗口已嵌入（离屏模式下不需要交互器）
            if not self.offscreen and (not self.embedded or not self.interactor):
                print("VTK窗口未正确嵌入，尝试重新嵌入...")
                self.embed_vtk_window()
                # 等待嵌入完成
                time.sleep(0.5)
                
            # 如果仍然未嵌入，则返回失败（离屏模式下不需要交互器）
            if not self.offscreen and (not self.embedded or not self.interactor):
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
                    
                    # 设置网格属性
                    self._apply_default_mesh_properties()
                    
                    # 设置显示模式
                    self._apply_render_mode()
                    
                    # 添加到渲染器
                    self.renderer.AddActor(self.mesh_actor)
                    
                    # 添加坐标轴
                    self.add_axes()
                    
                    # 如果需要，显示边界
                    if self.show_boundary_var.get():
                        self.display_boundary()
                    
                    # 重置相机以显示整个网格
                    self.renderer.ResetCamera()
                    
                    # 更新渲染窗口
                    self.render_window.Render()
                    
                    # 检测操作系统
                    system = self._get_system()
                    
                    # 如果交互器没有启动，则启动它（离屏模式下跳过）
                    if not self.offscreen and not self.interactor.GetInitialized():
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
            return False
    
    def _apply_default_mesh_properties(self):
        """应用默认网格属性"""
        if self.mesh_actor:
            self.mesh_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # 黑色
            self.mesh_actor.GetProperty().SetLineWidth(2.0)  # 增加线宽使线框更明显
            self.mesh_actor.GetProperty().SetPointSize(2.0)   # 设置点云大小
    
    def _apply_render_mode(self):
        """应用当前渲染模式"""
        if not self.mesh_actor:
            return
            
        mode = self.render_mode.get()
        if mode == "wireframe":
            self.mesh_actor.GetProperty().SetRepresentationToWireframe()
            self.mesh_actor.GetProperty().EdgeVisibilityOn()
            self.mesh_actor.GetProperty().SetLineWidth(2.0)
        elif mode == "points":
            self.mesh_actor.GetProperty().SetRepresentationToPoints()
            self.mesh_actor.GetProperty().SetPointSize(4.0)
        else:  # surface
            self.mesh_actor.GetProperty().SetRepresentationToSurface()
            self.mesh_actor.GetProperty().EdgeVisibilityOff()
    
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
            return None
            
    def display_boundary(self):
        """显示边界"""
        try:
            # 清除之前的边界演员
            for actor in self.boundary_actors:
                self.renderer.RemoveActor(actor)
            self.boundary_actors.clear()
            
            # 获取边界信息
            boundary_info = self._get_boundary_info()
            
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
                bc_type = zone_data.get("type", zone_data.get("bc_type", "unspecified"))
                faces = zone_data.get("faces", [])
                
                if not faces:
                    continue
                
                # 创建边界多边形数据
                boundary_polydata = self._create_boundary_polydata(faces)
                
                if boundary_polydata:
                    # 创建边界映射器和演员
                    boundary_mapper = vtk.vtkPolyDataMapper()
                    boundary_mapper.SetInputData(boundary_polydata)
                    
                    boundary_actor = vtk.vtkActor()
                    boundary_actor.SetMapper(boundary_mapper)
                    
                    # 设置边界颜色
                    color = bc_colors.get(bc_type, (0.5, 0.5, 0.5))
                    boundary_actor.GetProperty().SetColor(color)
                    boundary_actor.GetProperty().SetLineWidth(2.5)  # 增加线宽
                    
                    # 添加到渲染器和边界演员列表
                    self.renderer.AddActor(boundary_actor)
                    self.boundary_actors.append(boundary_actor)
                
        except Exception as e:
            print(f"显示边界失败: {str(e)}")
    
    def _get_boundary_info(self):
        """获取边界信息"""
        boundary_info = None
        
        if isinstance(self.mesh_data, dict) and 'unstr_grid' in self.mesh_data:
            unstr_grid = self.mesh_data['unstr_grid']
            if hasattr(unstr_grid, 'boundary_info'):
                boundary_info = unstr_grid.boundary_info
        elif hasattr(self.mesh_data, 'boundary_info'):
            boundary_info = self.mesh_data.boundary_info
        
        return boundary_info
    
    def _create_boundary_polydata(self, faces):
        """创建边界多边形数据"""
        try:
            boundary_points = vtk.vtkPoints()
            boundary_polys = vtk.vtkCellArray()
            point_map = {}  # 用于映射原始点到新点索引
            
            # 添加面
            for face in faces:
                # 确保面是有效的
                if not isinstance(face, dict) or "nodes" not in face:
                    continue
                    
                # 获取节点列表
                nodes = face["nodes"]
                if not nodes or len(nodes) < 2:
                    continue
                    
                # 创建多边形
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(len(nodes))
                
                # 添加点并设置多边形顶点
                for i, node_id in enumerate(nodes):
                    if node_id not in point_map:
                        # 添加新点
                        coord = None
                        
                        # 确保node_id是整数
                        try:
                            node_id_int = int(node_id)
                        except (ValueError, TypeError):
                            print(f"无效的节点ID: {node_id}")
                            continue
                        
                        # 转换为0基索引（从1基索引转换为0基索引）
                        node_id_0 = node_id_int - 1
                        
                        # 尝试从不同结构中获取节点坐标
                        coord = self._get_node_coord(node_id_0)
                        
                        if coord is None:
                            print(f"无法获取节点坐标: {node_id}")
                            continue
                            
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
            
            return boundary_polydata
        except Exception as e:
            print(f"创建边界多边形数据失败: {str(e)}")
            return None
    
    def _get_node_coord(self, node_id_0):
        """获取节点坐标"""
        coord = None
        
        # 尝试从不同结构中获取节点坐标
        if isinstance(self.mesh_data, dict):
            if 'unstr_grid' in self.mesh_data:
                unstr_grid = self.mesh_data['unstr_grid']
                if hasattr(unstr_grid, 'node_coords') and isinstance(unstr_grid.node_coords, list):
                    # 确保node_id_0是有效的索引
                    if 0 <= node_id_0 < len(unstr_grid.node_coords):
                        coord = unstr_grid.node_coords[node_id_0]
            elif 'node_coords' in self.mesh_data and isinstance(self.mesh_data['node_coords'], list):
                if 0 <= node_id_0 < len(self.mesh_data['node_coords']):
                    coord = self.mesh_data['node_coords'][node_id_0]
        elif hasattr(self.mesh_data, 'node_coords') and isinstance(self.mesh_data.node_coords, list):
            if 0 <= node_id_0 < len(self.mesh_data.node_coords):
                coord = self.mesh_data.node_coords[node_id_0]
        elif hasattr(self.mesh_data, 'unstr_grid'):
            unstr_grid = self.mesh_data.unstr_grid
            if hasattr(unstr_grid, 'node_coords') and isinstance(unstr_grid.node_coords, list):
                if 0 <= node_id_0 < len(unstr_grid.node_coords):
                    coord = unstr_grid.node_coords[node_id_0]
        
        return coord
    
    def add_axes(self):
        """添加坐标轴到渲染器"""
        try:
            # 创建坐标轴
            axes = vtk.vtkAxesActor()
            
            # 设置坐标轴大小
            axes.SetTotalLength(1.0, 1.0, 1.0)
            
            # 设置坐标轴标签字号
            axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
            axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
            axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
            
            # 创建坐标轴变换，将坐标轴放置在左下角
            transform = self._create_axes_transform()
            axes.SetUserTransform(transform)
            
            # 添加到渲染器
            self.renderer.AddActor(axes)
            
            # 保存坐标轴引用以便后续操作
            self.axes_actor = axes
            
        except Exception as e:
            print(f"添加坐标轴失败: {str(e)}")
    
    def _create_axes_transform(self):
        """创建坐标轴变换"""
        transform = vtk.vtkTransform()
        
        # 获取渲染器的边界
        bounds = self.renderer.ComputeVisiblePropBounds()
        if bounds and all(bound != float('inf') and bound != float('-inf') for bound in bounds):
            # 计算网格中心点和尺寸
            center_x = (bounds[0] + bounds[1]) / 2
            center_y = (bounds[2] + bounds[3]) / 2
            center_z = (bounds[4] + bounds[5]) / 2
            
            # 计算网格尺寸
            size_x = bounds[1] - bounds[0]
            size_y = bounds[3] - bounds[2]
            size_z = bounds[5] - bounds[4]
            
            # 计算最大尺寸
            max_size = max(size_x, size_y, size_z) if size_x > 0 and size_y > 0 and size_z > 0 else 1.0
            
            # 计算左下角位置（稍微偏移，避免与网格重叠）
            offset_x = center_x - size_x * 0.6
            offset_y = center_y - size_y * 0.6
            offset_z = center_z - size_z * 0.6
            
            # 设置变换，将坐标轴移动到左下角
            transform.Translate(offset_x, offset_y, offset_z if size_z > 0 else 0.0)
            
            # 根据网格大小调整坐标轴比例
            scale = max_size * 0.15  # 坐标轴大小为网格最大尺寸的15%
            transform.Scale(scale, scale, scale)
        else:
            # 如果没有边界信息，使用默认设置
            transform.Translate(-1, -1, -1)
            transform.Scale(0.5, 0.5, 0.5)
        
        return transform
    
    def clear_mesh_actors(self):
        """清除所有网格相关的演员"""
        # 清除主网格演员
        if self.mesh_actor:
            try:
                self.renderer.RemoveActor(self.mesh_actor)
            except:
                pass  # 忽略移除演员时的错误
            self.mesh_actor = None
            
        # 清除坐标轴演员
        if hasattr(self, 'axes_actor') and self.axes_actor:
            try:
                self.renderer.RemoveActor(self.axes_actor)
            except:
                pass  # 忽略移除演员时的错误
            self.axes_actor = None
            
        # 清除边界演员
        for actor in self.boundary_actors:
            try:
                self.renderer.RemoveActor(actor)
            except:
                pass  # 忽略移除演员时的错误
        self.boundary_actors.clear()
        
    def clear_display(self):
        """清除显示"""
        try:
            # 清除所有演员
            self.clear_mesh_actors()
            
            # 更新渲染窗口
            if self.render_window:
                # 防止递归调用，使用标志检查
                if not hasattr(self, '_render_in_progress'):
                    self._render_in_progress = True
                    self.render_window.Render()
                    self._render_in_progress = False
        except Exception as e:
            print(f"清除显示失败: {str(e)}")
            # 确保在出错时重置标志
            if hasattr(self, '_render_in_progress'):
                self._render_in_progress = False
    
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
            # 防止递归调用，使用标志检查
            if not hasattr(self, '_render_in_progress'):
                self._render_in_progress = True
                self.render_window.Render()
                self._render_in_progress = False
    
    def set_render_mode(self, mode):
        """设置渲染模式"""
        if mode in ["surface", "wireframe", "points"]:
            self.render_mode.set(mode)
            self._apply_render_mode()
            
            # 更新渲染窗口
            if self.render_window:
                # 防止递归调用，使用标志检查
                if not hasattr(self, '_render_in_progress'):
                    self._render_in_progress = True
                    self.render_window.Render()
                    self._render_in_progress = False
    
    def reset_view(self):
        """重置视图到原始大小"""
        if self.renderer:
            try:
                self.renderer.ResetCamera()
            except Exception as e:
                print(f"重置视图失败: {str(e)}")
                return
            
            if self.render_window:
                # 防止递归调用，使用标志检查
                if not hasattr(self, '_render_in_progress'):
                    self._render_in_progress = True
                    self.render_window.Render()
                    self._render_in_progress = False
    
    def reset_camera(self):
        """重置相机以适应整个网格"""
        if self.renderer:
            try:
                self.renderer.ResetCamera()
            except Exception as e:
                print(f"重置相机失败: {str(e)}")
                return
                
            if self.render_window:
                # 防止递归调用，使用标志检查
                if not hasattr(self, '_render_in_progress'):
                    self._render_in_progress = True
                    self.render_window.Render()
                    self._render_in_progress = False
    
    def zoom_in(self):
        """放大视图"""
        if self.renderer:
            try:
                camera = self.renderer.GetActiveCamera()
                camera.Zoom(1.2)
            except Exception as e:
                print(f"放大视图失败: {str(e)}")
                return
                
            if self.render_window:
                # 防止递归调用，使用标志检查
                if not hasattr(self, '_render_in_progress'):
                    self._render_in_progress = True
                    self.render_window.Render()
                    self._render_in_progress = False
    
    def zoom_out(self):
        """缩小视图"""
        if self.renderer:
            try:
                camera = self.renderer.GetActiveCamera()
                camera.Zoom(0.8)
            except Exception as e:
                print(f"缩小视图失败: {str(e)}")
                return
                
            if self.render_window:
                # 防止递归调用，使用标志检查
                if not hasattr(self, '_render_in_progress'):
                    self._render_in_progress = True
                    self.render_window.Render()
                    self._render_in_progress = False
            
    def fit_view(self):
        """适应视图以显示所有内容"""
        if self.renderer:
            try:
                self.renderer.ResetCamera()
            except Exception as e:
                print(f"适应视图失败: {str(e)}")
                return
                
            # 防止递归调用，使用标志检查
            if self.render_window and not hasattr(self, '_render_in_progress'):
                self._render_in_progress = True
                self.render_window.Render()
                self._render_in_progress = False
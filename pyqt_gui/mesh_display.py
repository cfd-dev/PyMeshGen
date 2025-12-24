#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt网格显示模块
处理网格可视化和交互功能
使用VTK进行网格渲染和显示
"""

import os
import time
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class MeshDisplayArea:
    """网格显示区域类，使用VTK进行渲染"""
    
    def __init__(self, parent=None, figsize=(16, 9), dpi=100, offscreen=False):
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
        self.show_boundary = True
        self.wireframe = False
        self.render_mode = "wireframe"

        # 渲染状态标志
        self._render_in_progress = False

        # 创建网格显示区域
        self.create_mesh_display_area(parent)

        # 启用交互功能
        self.enable_interaction()

    def __del__(self):
        """清理资源"""
        self.cleanup()
    
    def cleanup(self):
        """清理VTK资源以避免内存泄漏"""
        if self.renderer and self.boundary_actors:
            for actor in self.boundary_actors:
                try:
                    self.renderer.RemoveActor(actor)
                except:
                    pass
            self.boundary_actors.clear()

        if self.renderer and self.mesh_actor:
            try:
                self.renderer.RemoveActor(self.mesh_actor)
            except:
                pass
            self.mesh_actor = None

        if self.renderer and self.axes_actor:
            try:
                self.renderer.RemoveActor(self.axes_actor)
            except:
                pass
            self.axes_actor = None

        if self.renderer:
            try:
                self.renderer.RemoveAllViewProps()
            except:
                pass

        if self.interactor:
            try:
                self.interactor.TerminateApp()
            except:
                pass
    
    def create_mesh_display_area(self, parent):
        """创建VTK网格显示区域"""
        self.frame = QVTKRenderWindowInteractor(parent)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)

        self.render_window = self.frame.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.frame.SetInteractorStyle(style)

        self.frame.SetRenderWindow(self.render_window)

        self.frame.Initialize()
        self.frame.Start()

        self.renderer.ResetCamera()

        self.mesh_actor = None
        self.boundary_actors = []

        return self.frame

    def enable_interaction(self):
        """启用交互功能"""
        if self.frame:
            self.frame.Initialize()
            self.frame.Start()

    def disable_interaction(self):
        """禁用交互功能"""
        if self.frame:
            self.frame.TerminateApp()

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
                print("没有有效的网格数据，无法显示网格")
                return False
                
        try:
            self.clear_mesh_actors()

            if self.params and hasattr(self.params, 'viz_enabled'):
                viz_enabled = self.params.viz_enabled
            else:
                viz_enabled = True

            if viz_enabled:
                vtk_mesh = self.create_vtk_mesh()

                if vtk_mesh:
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(vtk_mesh)

                    self.mesh_actor = vtk.vtkActor()
                    self.mesh_actor.SetMapper(mapper)

                    self._apply_default_mesh_properties()
                    self._apply_render_mode()

                    self.renderer.AddActor(self.mesh_actor)
                    self.add_axes()

                    if self.show_boundary:
                        self.display_boundary()

                    self.renderer.ResetCamera()
                    self.render_window.Render()
                    self.enable_interaction()

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
            self.mesh_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
            self.mesh_actor.GetProperty().SetLineWidth(2.0)
            self.mesh_actor.GetProperty().SetPointSize(2.0)

    def _apply_render_mode(self):
        """应用当前渲染模式"""
        if not self.mesh_actor:
            return

        mode = self.render_mode
        if mode == "wireframe":
            self.mesh_actor.GetProperty().SetRepresentationToWireframe()
            self.mesh_actor.GetProperty().EdgeVisibilityOn()
            self.mesh_actor.GetProperty().SetLineWidth(2.0)
        elif mode == "points":
            self.mesh_actor.GetProperty().SetRepresentationToPoints()
            self.mesh_actor.GetProperty().SetPointSize(4.0)
        elif mode == "surface-wireframe" or mode == "mixed":
            self.mesh_actor.GetProperty().SetRepresentationToSurface()
            self.mesh_actor.GetProperty().EdgeVisibilityOn()
            self.mesh_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
            self.mesh_actor.GetProperty().SetLineWidth(1.5)
        else:
            self.mesh_actor.GetProperty().SetRepresentationToSurface()
            self.mesh_actor.GetProperty().EdgeVisibilityOff()

    def create_vtk_mesh(self):
        """根据网格数据创建VTK网格对象"""
        try:
            if not self.mesh_data:
                print("mesh_data为空，无法创建VTK网格")
                return None

            points = vtk.vtkPoints()
            polys = vtk.vtkCellArray()

            def add_point(coord):
                """添加点到VTK点集"""
                if len(coord) >= 2:
                    if len(coord) == 2:
                        points.InsertNextPoint(coord[0], coord[1], 0.0)
                    else:
                        points.InsertNextPoint(coord[0], coord[1], coord[2])

            def add_cell(cell):
                """添加单元到VTK单元集"""
                if len(cell) == 3:
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, cell[0])
                    triangle.GetPointIds().SetId(1, cell[1])
                    triangle.GetPointIds().SetId(2, cell[2])
                    polys.InsertNextCell(triangle)
                elif len(cell) == 4:
                    quad = vtk.vtkQuad()
                    quad.GetPointIds().SetId(0, cell[0])
                    quad.GetPointIds().SetId(1, cell[1])
                    quad.GetPointIds().SetId(2, cell[2])
                    quad.GetPointIds().SetId(3, cell[3])
                    polys.InsertNextCell(quad)

            if isinstance(self.mesh_data, dict):
                if 'type' in self.mesh_data:
                    if self.mesh_data['type'] in ['vtk', 'stl', 'obj', 'ply']:
                        if 'node_coords' not in self.mesh_data or 'cells' not in self.mesh_data:
                            print("mesh_data缺少必要的键")
                            return None

                        node_coords = self.mesh_data['node_coords']
                        cells = self.mesh_data['cells']

                        for coord in node_coords:
                            add_point(coord)

                        for cell in cells:
                            add_cell(cell)

                    elif self.mesh_data['type'] == 'cas':
                        if 'unstr_grid' in self.mesh_data:
                            return self.create_vtk_mesh_from_unstr_grid(self.mesh_data['unstr_grid'])
                else:
                    if 'node_coords' in self.mesh_data and 'cells' in self.mesh_data:
                        node_coords = self.mesh_data['node_coords']
                        cells = self.mesh_data['cells']

                        for coord in node_coords:
                            add_point(coord)

                        for cell in cells:
                            add_cell(cell)

            elif hasattr(self.mesh_data, 'node_coords') and hasattr(self.mesh_data, 'cells'):
                node_coords = self.mesh_data.node_coords
                cells = self.mesh_data.cells

                for coord in node_coords:
                    add_point(coord)

                for cell in cells:
                    add_cell(cell)

            elif hasattr(self.mesh_data, 'node_coords') and hasattr(self.mesh_data, 'cell_container'):
                return self.create_vtk_mesh_from_unstr_grid(self.mesh_data)

            if points.GetNumberOfPoints() == 0:
                print("没有有效的点数据，无法创建VTK网格")
                return None

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
            points = vtk.vtkPoints()
            polys = vtk.vtkCellArray()

            for coord in unstr_grid.node_coords:
                if len(coord) >= 2:
                    if len(coord) == 2:
                        points.InsertNextPoint(coord[0], coord[1], 0.0)
                    else:
                        points.InsertNextPoint(coord[0], coord[1], coord[2])

            for cell in unstr_grid.cell_container:
                if cell is None:
                    continue

                node_ids = cell.node_ids
                if len(node_ids) == 3:
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, node_ids[0])
                    triangle.GetPointIds().SetId(1, node_ids[1])
                    triangle.GetPointIds().SetId(2, node_ids[2])
                    polys.InsertNextCell(triangle)
                elif len(node_ids) == 4:
                    quad = vtk.vtkQuad()
                    quad.GetPointIds().SetId(0, node_ids[0])
                    quad.GetPointIds().SetId(1, node_ids[1])
                    quad.GetPointIds().SetId(2, node_ids[2])
                    quad.GetPointIds().SetId(3, node_ids[3])
                    polys.InsertNextCell(quad)

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
            for actor in self.boundary_actors:
                self.renderer.RemoveActor(actor)
            self.boundary_actors.clear()

            boundary_info = self._get_boundary_info()

            if not boundary_info:
                return

            bc_colors = {
                "wall": (1.0, 0.0, 0.0),
                "pressure-inlet": (0.0, 1.0, 0.0),
                "pressure-outlet": (0.0, 0.0, 1.0),
                "symmetry": (1.0, 1.0, 0.0),
                "pressure-far-field": (0.0, 1.0, 1.0),
                "unspecified": (0.5, 0.5, 0.5),
            }

            for zone_name, zone_data in boundary_info.items():
                bc_type = zone_data.get("type", zone_data.get("bc_type", "unspecified"))
                faces = zone_data.get("faces", [])

                if not faces:
                    continue

                boundary_polydata = self._create_boundary_polydata(faces)

                if boundary_polydata:
                    boundary_mapper = vtk.vtkPolyDataMapper()
                    boundary_mapper.SetInputData(boundary_polydata)

                    boundary_actor = vtk.vtkActor()
                    boundary_actor.SetMapper(boundary_mapper)

                    color = bc_colors.get(bc_type, (0.5, 0.5, 0.5))
                    boundary_actor.GetProperty().SetColor(color)
                    boundary_actor.GetProperty().SetLineWidth(2.5)

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
            point_map = {}

            for face in faces:
                if not isinstance(face, dict) or "nodes" not in face:
                    continue

                nodes = face["nodes"]
                if not nodes or len(nodes) < 2:
                    continue

                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(len(nodes))

                for i, node_id in enumerate(nodes):
                    if node_id not in point_map:
                        try:
                            node_id_int = int(node_id)
                        except (ValueError, TypeError):
                            print(f"无效的节点ID: {node_id}")
                            continue

                        node_id_0 = node_id_int - 1
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

        if self.mesh_data:
            if isinstance(self.mesh_data, dict):
                if 'unstr_grid' in self.mesh_data:
                    unstr_grid = self.mesh_data['unstr_grid']
                    if hasattr(unstr_grid, 'node_coords') and isinstance(unstr_grid.node_coords, list):
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
            # Create a 2D coordinate system that stays fixed in the corner of the viewport
            # This will use a special actor that is not affected by camera zoom

            # Create the axes actor
            axes = vtk.vtkAxesActor()

            # Set axis lengths
            axes.SetTotalLength(1.0, 1.0, 1.0)

            # Set better colors for visibility
            axes.GetXAxisShaftProperty().SetColor(1, 0, 0)  # Red for X
            axes.GetYAxisShaftProperty().SetColor(0, 1, 0)  # Green for Y
            axes.GetZAxisShaftProperty().SetColor(0, 0, 1)  # Blue for Z

            axes.GetXAxisTipProperty().SetColor(1, 0, 0)   # Red for X tip
            axes.GetYAxisTipProperty().SetColor(0, 1, 0)   # Green for Y tip
            axes.GetZAxisTipProperty().SetColor(0, 0, 1)   # Blue for Z tip

            # Set label properties
            axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(14)
            axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(14)
            axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(14)

            # Use a special approach to keep the axes in the corner regardless of zoom
            # Create a widget to position the axes in the viewport
            self.axes_widget = vtk.vtkOrientationMarkerWidget()
            self.axes_widget.SetOrientationMarker(axes)
            self.axes_widget.SetInteractor(self.frame.GetRenderWindow().GetInteractor())
            self.axes_widget.SetEnabled(True)
            self.axes_widget.InteractiveOff()  # Disable interaction with the axes

            # Position the axes in the lower-left corner of the viewport
            self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)  # Lower-left corner

            # Save reference to axes widget for later operations
            self.axes_actor = axes
            self.axes_widget_ref = self.axes_widget

        except Exception as e:
            print(f"添加坐标轴失败: {str(e)}")
            # Fallback to original method if the widget approach fails
            try:
                # 创建坐标轴
                axes = vtk.vtkAxesActor()
                axes.SetTotalLength(1.0, 1.0, 1.0)

                # Set better colors for visibility
                axes.GetXAxisShaftProperty().SetColor(1, 0, 0)  # Red for X
                axes.GetYAxisShaftProperty().SetColor(0, 1, 0)  # Green for Y
                axes.GetZAxisShaftProperty().SetColor(0, 0, 1)  # Blue for Z

                axes.GetXAxisTipProperty().SetColor(1, 0, 0)   # Red for X tip
                axes.GetYAxisTipProperty().SetColor(0, 1, 0)   # Green for Y tip
                axes.GetZAxisTipProperty().SetColor(0, 0, 1)   # Blue for Z tip

                # Set label properties
                axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(14)
                axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(14)
                axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(14)

                # Add to renderer with a fixed transform
                transform = vtk.vtkTransform()
                transform.Translate(-0.8, -0.8, 0)  # Fixed position
                transform.Scale(0.1, 0.1, 0.1)      # Fixed scale
                axes.SetUserTransform(transform)

                self.renderer.AddActor(axes)
                self.axes_actor = axes
            except Exception as fallback_e:
                print(f"备用坐标轴方法也失败: {str(fallback_e)}")
    
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
            self.clear_mesh_actors()
            self.render_window.Render()
        except Exception as e:
            print(f"清除显示失败: {str(e)}")

    def clear(self):
        """清除显示的网格"""
        self.clear_display()

    def clear_boundary_actors(self):
        """清除所有边界演员"""
        for actor in self.boundary_actors:
            self.renderer.RemoveActor(actor)
        self.boundary_actors = []
    
    def toggle_boundary_display(self, show_boundary=None):
        """切换边界显示"""
        if show_boundary is not None:
            self.show_boundary = show_boundary
        
        if self.show_boundary:
            self.display_boundary()
        else:
            self.clear_boundary_actors()
        
        # 更新渲染窗口
        self.render_window.Render()
    
    def set_render_mode(self, mode):
        """设置渲染模式"""
        if mode in ["surface", "wireframe", "points"]:
            self.render_mode = mode
            self._apply_render_mode()
            
            # 更新渲染窗口
            self.render_window.Render()
    
    def reset_view(self):
        """重置视图到原始大小"""
        if self.renderer:
            try:
                self.renderer.ResetCamera()
            except Exception as e:
                print(f"重置视图失败: {str(e)}")
                return

            self.render_window.Render()

    def zoom_in(self):
        """放大视图"""
        if self.renderer:
            try:
                camera = self.renderer.GetActiveCamera()
                camera.Zoom(1.2)
            except Exception as e:
                print(f"放大视图失败: {str(e)}")
                return

            self.render_window.Render()

    def zoom_out(self):
        """缩小视图"""
        if self.renderer:
            try:
                camera = self.renderer.GetActiveCamera()
                camera.Zoom(0.8)
            except Exception as e:
                print(f"缩小视图失败: {str(e)}")
                return

            self.render_window.Render()

    def fit_view(self):
        """适应视图以显示所有内容"""
        if self.renderer:
            try:
                self.renderer.ResetCamera()
            except Exception as e:
                print(f"适应视图失败: {str(e)}")
                return

            self.render_window.Render()
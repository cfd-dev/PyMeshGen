#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt网格显示模块
处理网格可视化和交互功能
使用VTK进行网格渲染和显示
"""

import os
import time
import numpy as np
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
        # 设置蓝色到白色的渐变背景
        self.renderer.SetBackground(0.9, 0.9, 1.0)  # 浅蓝色背景
        self.renderer.SetBackground2(1.0, 1.0, 1.0)  # 白色背景
        self.renderer.SetGradientBackground(True)  # 启用渐变背景

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
        
        # 在创建显示区域后立即添加坐标轴
        self.add_axes()

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
        
    def display_mesh(self, mesh_data=None, render_immediately=True):
        """显示网格
        
        Args:
            mesh_data: 网格数据对象
            render_immediately: 是否立即渲染，默认为True。设为False可批量渲染提高性能
        """
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
                    pass
            
            if not self.mesh_data:
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
                        self.display_boundary(render_immediately=False)

                    self.renderer.ResetCamera()
                    
                    # 根据参数决定是否立即渲染
                    if render_immediately:
                        self.render_window.Render()
                    
                    self.enable_interaction()

                    return True
                else:
                    return False
            else:
                return False
        except Exception as e:
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
            self.mesh_actor.GetProperty().EdgeVisibilityOff()
            self.mesh_actor.GetProperty().SetLineWidth(2.0)
        elif mode == "surface-wireframe":
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
                # 对于CAS类型的MeshData对象，直接使用node_coords和cells属性
                node_coords = self.mesh_data.node_coords
                cells = self.mesh_data.cells

                for coord in node_coords:
                    add_point(coord)

                for cell in cells:
                    add_cell(cell)

            elif hasattr(self.mesh_data, 'node_coords') and hasattr(self.mesh_data, 'cell_container'):
                # 对于Unstructured_Grid对象，直接使用node_coords和cell_container属性
                node_coords = np.array(self.mesh_data.node_coords, dtype=np.float64)
                cell_container = self.mesh_data.cell_container

                # 批量插入点数据
                for i in range(len(node_coords)):
                    coord = node_coords[i]
                    add_point(coord)

                # 预导入单元类型以避免循环中重复导入
                from data_structure.basic_elements import (
                    Quadrilateral, Tetrahedron, Pyramid, Prism, Hexahedron
                )

                for cell in cell_container:
                    if cell is None:
                        continue
                    node_ids = cell.node_ids
                    num_nodes = len(node_ids)
                    
                    if num_nodes == 3:
                        triangle = vtk.vtkTriangle()
                        triangle.GetPointIds().SetId(0, node_ids[0])
                        triangle.GetPointIds().SetId(1, node_ids[1])
                        triangle.GetPointIds().SetId(2, node_ids[2])
                        polys.InsertNextCell(triangle)
                    elif num_nodes == 4:
                        # 检查单元类型：四边形(2D)或四面体(3D)
                        if isinstance(cell, Tetrahedron):
                            tetra = vtk.vtkTetra()
                            tetra.GetPointIds().SetId(0, node_ids[0])
                            tetra.GetPointIds().SetId(1, node_ids[1])
                            tetra.GetPointIds().SetId(2, node_ids[2])
                            tetra.GetPointIds().SetId(3, node_ids[3])
                            polys.InsertNextCell(tetra)
                        elif isinstance(cell, Quadrilateral):
                            quad = vtk.vtkQuad()
                            quad.GetPointIds().SetId(0, node_ids[0])
                            quad.GetPointIds().SetId(1, node_ids[1])
                            quad.GetPointIds().SetId(2, node_ids[2])
                            quad.GetPointIds().SetId(3, node_ids[3])
                            polys.InsertNextCell(quad)
                    elif num_nodes == 5:
                        # 金字塔单元
                        if isinstance(cell, Pyramid):
                            pyramid = vtk.vtkPyramid()
                            pyramid.GetPointIds().SetId(0, node_ids[0])
                            pyramid.GetPointIds().SetId(1, node_ids[1])
                            pyramid.GetPointIds().SetId(2, node_ids[2])
                            pyramid.GetPointIds().SetId(3, node_ids[3])
                            pyramid.GetPointIds().SetId(4, node_ids[4])
                            polys.InsertNextCell(pyramid)
                    elif num_nodes == 6:
                        # 三棱柱单元
                        if isinstance(cell, Prism):
                            wedge = vtk.vtkWedge()
                            wedge.GetPointIds().SetId(0, node_ids[0])
                            wedge.GetPointIds().SetId(1, node_ids[1])
                            wedge.GetPointIds().SetId(2, node_ids[2])
                            wedge.GetPointIds().SetId(3, node_ids[3])
                            wedge.GetPointIds().SetId(4, node_ids[4])
                            wedge.GetPointIds().SetId(5, node_ids[5])
                            polys.InsertNextCell(wedge)
                    elif num_nodes == 8:
                        # 六面体单元
                        if isinstance(cell, Hexahedron):
                            hexahedron = vtk.vtkHexahedron()
                            hexahedron.GetPointIds().SetId(0, node_ids[0])
                            hexahedron.GetPointIds().SetId(1, node_ids[1])
                            hexahedron.GetPointIds().SetId(2, node_ids[2])
                            hexahedron.GetPointIds().SetId(3, node_ids[3])
                            hexahedron.GetPointIds().SetId(4, node_ids[4])
                            hexahedron.GetPointIds().SetId(5, node_ids[5])
                            hexahedron.GetPointIds().SetId(6, node_ids[6])
                            hexahedron.GetPointIds().SetId(7, node_ids[7])
                            polys.InsertNextCell(hexahedron)

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

            # 使用numpy数组批量处理节点坐标以提高性能
            node_coords = np.array(unstr_grid.node_coords, dtype=np.float64)
            num_points = len(node_coords)
            
            # 批量插入点数据
            for i in range(num_points):
                coord = node_coords[i]
                if len(coord) >= 2:
                    if len(coord) == 2:
                        points.InsertNextPoint(coord[0], coord[1], 0.0)
                    else:
                        points.InsertNextPoint(coord[0], coord[1], coord[2])

            # 预导入单元类型以避免循环中重复导入
            from data_structure.basic_elements import (
                Quadrilateral, Tetrahedron, Pyramid, Prism, Hexahedron
            )

            for cell in unstr_grid.cell_container:
                if cell is None:
                    continue

                node_ids = cell.node_ids
                num_nodes = len(node_ids)
                
                if num_nodes == 3:
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, node_ids[0])
                    triangle.GetPointIds().SetId(1, node_ids[1])
                    triangle.GetPointIds().SetId(2, node_ids[2])
                    polys.InsertNextCell(triangle)
                elif num_nodes == 4:
                    # 检查单元类型：四边形(2D)或四面体(3D)
                    if isinstance(cell, Tetrahedron):
                        tetra = vtk.vtkTetra()
                        tetra.GetPointIds().SetId(0, node_ids[0])
                        tetra.GetPointIds().SetId(1, node_ids[1])
                        tetra.GetPointIds().SetId(2, node_ids[2])
                        tetra.GetPointIds().SetId(3, node_ids[3])
                        polys.InsertNextCell(tetra)
                    elif isinstance(cell, Quadrilateral):
                        quad = vtk.vtkQuad()
                        quad.GetPointIds().SetId(0, node_ids[0])
                        quad.GetPointIds().SetId(1, node_ids[1])
                        quad.GetPointIds().SetId(2, node_ids[2])
                        quad.GetPointIds().SetId(3, node_ids[3])
                        polys.InsertNextCell(quad)
                elif num_nodes == 5:
                    # 金字塔单元
                    if isinstance(cell, Pyramid):
                        pyramid = vtk.vtkPyramid()
                        pyramid.GetPointIds().SetId(0, node_ids[0])
                        pyramid.GetPointIds().SetId(1, node_ids[1])
                        pyramid.GetPointIds().SetId(2, node_ids[2])
                        pyramid.GetPointIds().SetId(3, node_ids[3])
                        pyramid.GetPointIds().SetId(4, node_ids[4])
                        polys.InsertNextCell(pyramid)
                elif num_nodes == 6:
                    # 三棱柱单元
                    if isinstance(cell, Prism):
                        wedge = vtk.vtkWedge()
                        wedge.GetPointIds().SetId(0, node_ids[0])
                        wedge.GetPointIds().SetId(1, node_ids[1])
                        wedge.GetPointIds().SetId(2, node_ids[2])
                        wedge.GetPointIds().SetId(3, node_ids[3])
                        wedge.GetPointIds().SetId(4, node_ids[4])
                        wedge.GetPointIds().SetId(5, node_ids[5])
                        polys.InsertNextCell(wedge)
                elif num_nodes == 8:
                    # 六面体单元
                    if isinstance(cell, Hexahedron):
                        hexahedron = vtk.vtkHexahedron()
                        hexahedron.GetPointIds().SetId(0, node_ids[0])
                        hexahedron.GetPointIds().SetId(1, node_ids[1])
                        hexahedron.GetPointIds().SetId(2, node_ids[2])
                        hexahedron.GetPointIds().SetId(3, node_ids[3])
                        hexahedron.GetPointIds().SetId(4, node_ids[4])
                        hexahedron.GetPointIds().SetId(5, node_ids[5])
                        hexahedron.GetPointIds().SetId(6, node_ids[6])
                        hexahedron.GetPointIds().SetId(7, node_ids[7])
                        polys.InsertNextCell(hexahedron)

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)

            return polydata

        except Exception as e:
            print(f"从Unstructured_Grid创建VTK网格失败: {str(e)}")
            return None
            
    def display_boundary(self, render_immediately=True):
        """显示边界
        
        Args:
            render_immediately: 是否立即渲染，默认为True。设为False可批量渲染提高性能
        """
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
                "velocity-inlet": (1.0, 0.0, 1.0),
                "interface": (1.0, 0.5, 0.0),
                "outflow": (0.0, 1.0, 1.0),
                "unspecified": (0.5, 0.5, 0.5),
            }

            for zone_name, zone_data in boundary_info.items():
                bc_type = zone_data.get("bc_type", "unspecified")
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

            # 根据参数决定是否立即渲染
            if render_immediately:
                self.render_window.Render()

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
        """创建边界多边形数据 - 支持2D线段和3D面"""
        try:
            boundary_points = vtk.vtkPoints()
            boundary_polys = vtk.vtkCellArray()  # For polygons (3+ nodes)
            boundary_lines = vtk.vtkCellArray()  # For line segments (2 nodes)
            point_map = {}

            for face in faces:
                if not isinstance(face, dict) or "nodes" not in face:
                    continue

                nodes = face["nodes"]
                if not nodes or len(nodes) < 2:
                    continue

                num_nodes = len(nodes)

                # Handle 2-node cases as line segments (for 2D boundaries)
                if num_nodes == 2:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetNumberOfIds(2)

                    for i, node_id in enumerate(nodes):
                        if node_id not in point_map:
                            try:
                                node_id_int = int(node_id)
                            except (ValueError, TypeError):
                                print(f"无效的节点ID: {node_id}")
                                continue

                            coord = self._get_node_coord(node_id_int)

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

                        line.GetPointIds().SetId(i, point_id)

                    boundary_lines.InsertNextCell(line)

                # Handle 3+ node cases as polygons (for 3D boundaries)
                else:
                    polygon = vtk.vtkPolygon()
                    polygon.GetPointIds().SetNumberOfIds(num_nodes)

                    for i, node_id in enumerate(nodes):
                        if node_id not in point_map:
                            try:
                                node_id_int = int(node_id)
                            except (ValueError, TypeError):
                                print(f"无效的节点ID: {node_id}")
                                continue

                            coord = self._get_node_coord(node_id_int)

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

            # Add both polygons and lines to the polydata
            if boundary_polys.GetNumberOfCells() > 0:
                boundary_polydata.SetPolys(boundary_polys)
            if boundary_lines.GetNumberOfCells() > 0:
                boundary_polydata.SetLines(boundary_lines)

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
            self.display_boundary(render_immediately=False)
        else:
            self.clear_boundary_actors()
        
        # 更新渲染窗口
        self.render_window.Render()
    
    def highlight_part(self, part_name, highlight=True, parts_info=None):
        """高亮显示指定部件

        Args:
            part_name (str): 要高亮的部件名称
            highlight (bool): 是否高亮
            parts_info (dict): 可选，部件信息字典，用于直接传递部件数据
        """
        try:
            # 首先清除所有高亮状态
            self.clear_highlights()

            # 如果不进行高亮或部件名称为空，直接返回
            if not highlight or not part_name:
                return

            # 检查渲染器和渲染窗口是否可用
            if not self.renderer or not self.render_window:
                return

            # 检查网格数据是否可用
            if not self.mesh_data:
                return

            # 获取边界信息
            boundary_info = self._get_boundary_info()

            # 直接从mesh_data中获取部件信息，支持更多数据格式
            part_faces = []

            # 优先使用传递的parts_info
            if parts_info:
                if isinstance(parts_info, dict) and part_name in parts_info:
                    part_faces = parts_info[part_name].get('faces', [])
            # 然后检查mesh_data中是否有parts_info
            elif isinstance(self.mesh_data, dict) and 'parts_info' in self.mesh_data:
                parts_info = self.mesh_data['parts_info']
                if isinstance(parts_info, dict) and part_name in parts_info:
                    part_faces = parts_info[part_name].get('faces', [])
            # 检查是否有边界信息
            elif boundary_info and part_name in boundary_info:
                part_data = boundary_info[part_name]
                part_faces = part_data.get("faces", [])
            # 检查是否有直接的部件数据
            elif isinstance(self.mesh_data, dict) and part_name in self.mesh_data:
                part_faces = self.mesh_data[part_name].get('faces', [])

            # 检查是否有面数据
            if not part_faces:
                # 如果没有面数据，尝试高亮整个网格
                if self.mesh_actor:
                    # 保存原始颜色
                    if not hasattr(self, 'original_mesh_color'):
                        self.original_mesh_color = list(self.mesh_actor.GetProperty().GetColor())
                        self.original_mesh_line_width = self.mesh_actor.GetProperty().GetLineWidth()

                    # 设置高亮颜色
                    self.mesh_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
                    self.mesh_actor.GetProperty().SetLineWidth(4.0)
                    self.render_window.Render()

                    # 保存高亮演员引用
                    if not hasattr(self, 'highlight_actors'):
                        self.highlight_actors = []
                    self.highlight_actors.append(self.mesh_actor)

                    return
                else:
                    return

            # 检查是否有节点坐标数据
            has_node_coords = False
            if isinstance(self.mesh_data, dict):
                if 'unstr_grid' in self.mesh_data and hasattr(self.mesh_data['unstr_grid'], 'node_coords'):
                    has_node_coords = True
                elif 'node_coords' in self.mesh_data:
                    has_node_coords = True
            elif hasattr(self.mesh_data, 'node_coords'):
                has_node_coords = True

            if not has_node_coords:
                # 尝试高亮整个网格作为备选方案
                if self.mesh_actor:
                    # 保存原始颜色
                    if not hasattr(self, 'original_mesh_color'):
                        self.original_mesh_color = list(self.mesh_actor.GetProperty().GetColor())
                        self.original_mesh_line_width = self.mesh_actor.GetProperty().GetLineWidth()

                    # 设置高亮颜色
                    self.mesh_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
                    self.mesh_actor.GetProperty().SetLineWidth(4.0)
                    self.render_window.Render()

                    # 保存高亮演员引用
                    if not hasattr(self, 'highlight_actors'):
                        self.highlight_actors = []
                    self.highlight_actors.append(self.mesh_actor)

                return

            # 创建高亮多边形数据
            highlight_polydata = self._create_boundary_polydata(part_faces)

            if highlight_polydata:
                highlight_mapper = vtk.vtkPolyDataMapper()
                highlight_mapper.SetInputData(highlight_polydata)

                highlight_actor = vtk.vtkActor()
                highlight_actor.SetMapper(highlight_mapper)

                # 设置高亮颜色（黄色）
                highlight_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
                highlight_actor.GetProperty().SetLineWidth(4.0)
                highlight_actor.GetProperty().SetOpacity(0.8)

                # 添加高亮演员
                self.renderer.AddActor(highlight_actor)

                # 保存高亮演员引用
                if not hasattr(self, 'highlight_actors'):
                    self.highlight_actors = []
                self.highlight_actors.append(highlight_actor)

                # 更新渲染窗口
                self.render_window.Render()

        except Exception as e:
            pass
    
    def clear_highlights(self):
        """清除所有高亮状态"""
        try:
            # 恢复原始网格属性（如果有）
            if self.mesh_actor is not None and hasattr(self, 'original_mesh_color'):
                self.mesh_actor.GetProperty().SetColor(self.original_mesh_color)
                self.mesh_actor.GetProperty().SetLineWidth(self.original_mesh_line_width)
                
                # 移除原始网格颜色和线宽的引用
                if hasattr(self, 'original_mesh_color'):
                    delattr(self, 'original_mesh_color')
                if hasattr(self, 'original_mesh_line_width'):
                    delattr(self, 'original_mesh_line_width')
            
            # 移除高亮演员
            if hasattr(self, 'highlight_actors'):
                for actor in self.highlight_actors:
                    # 只移除我们创建的高亮演员，不移除原始网格演员
                    if actor != self.mesh_actor:
                        try:
                            self.renderer.RemoveActor(actor)
                        except:
                            pass
                self.highlight_actors.clear()
                
                # 更新渲染窗口
                if self.render_window:
                    self.render_window.Render()
        except Exception as e:
            pass
    
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

        # 清除高亮演员
        self.clear_highlights()

        # 清除额外演员（由display_part方法添加的）
        if hasattr(self, 'additional_actors'):
            for actor in self.additional_actors:
                try:
                    self.renderer.RemoveActor(actor)
                except:
                    pass  # 忽略移除演员时的错误
            self.additional_actors.clear()
    
    def set_render_mode(self, mode):
        """设置渲染模式

        Args:
            mode: 渲染模式，可选值:
                - "surface": 实体模式
                - "wireframe": 线框模式
                - "surface-wireframe": 实体+线框模式
        """
        if mode in ["surface", "wireframe", "surface-wireframe"]:
            self.render_mode = mode
            self._apply_render_mode()

            # Apply render mode to all additional actors (individual parts)
            if hasattr(self, 'additional_actors'):
                for actor in self.additional_actors:
                    if mode == "wireframe":
                        actor.GetProperty().SetRepresentationToWireframe()
                        actor.GetProperty().EdgeVisibilityOff()
                        actor.GetProperty().SetLineWidth(2.0)
                    elif mode == "surface-wireframe":
                        actor.GetProperty().SetRepresentationToSurface()
                        actor.GetProperty().EdgeVisibilityOn()
                        actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
                        actor.GetProperty().SetLineWidth(1.5)
                    else:  # surface mode
                        actor.GetProperty().SetRepresentationToSurface()
                        actor.GetProperty().EdgeVisibilityOff()

            # 更新渲染窗口
            self.render_window.Render()
    
    def set_background_gradient(self, color1, color2):
        """设置渐变背景色
        
        Args:
            color1 (tuple): 第一个颜色 (R, G, B)，范围0-1
            color2 (tuple): 第二个颜色 (R, G, B)，范围0-1
        """
        if self.renderer:
            self.renderer.SetBackground(color1[0], color1[1], color1[2])
            self.renderer.SetBackground2(color2[0], color2[1], color2[2])
            self.renderer.SetGradientBackground(True)
            self.render_window.Render()
    
    def set_background_color(self, color):
        """设置单一背景色
        
        Args:
            color (tuple): 颜色 (R, G, B)，范围0-1
        """
        if self.renderer:
            self.renderer.SetBackground(color[0], color[1], color[2])
            self.renderer.SetGradientBackground(False)
            self.render_window.Render()

    def reset_view(self):
        """重置视图到xy平面"""
        if self.renderer:
            try:
                camera = self.renderer.GetActiveCamera()
                camera.SetPosition(0, 0, 1)
                camera.SetFocalPoint(0, 0, 0)
                camera.SetViewUp(0, 1, 0)
                # Use perspective projection instead of parallel for better zoom flexibility
                camera.SetParallelProjection(False)
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
                # Instead of using Zoom(1.2), we'll decrease the camera's distance to focal point
                # Get current position and focal point
                pos = camera.GetPosition()
                focal = camera.GetFocalPoint()

                # Calculate direction vector from focal point to camera position
                direction = [pos[i] - focal[i] for i in range(3)]

                # Move camera closer to focal point (zoom in)
                new_pos = [focal[i] + 0.8 * direction[i] for i in range(3)]

                camera.SetPosition(new_pos)
            except Exception as e:
                print(f"放大视图失败: {str(e)}")
                return

            self.render_window.Render()

    def zoom_out(self):
        """缩小视图"""
        if self.renderer:
            try:
                camera = self.renderer.GetActiveCamera()
                # Instead of using Zoom(0.8), we'll increase the camera's distance to focal point
                # Get current position and focal point
                pos = camera.GetPosition()
                focal = camera.GetFocalPoint()

                # Calculate direction vector from focal point to camera position
                direction = [pos[i] - focal[i] for i in range(3)]

                # Move camera further from focal point (zoom out)
                new_pos = [focal[i] + 1.25 * direction[i] for i in range(3)]

                camera.SetPosition(new_pos)
            except Exception as e:
                print(f"缩小视图失败: {str(e)}")
                return

            self.render_window.Render()

    def fit_view(self):
        """适应视图以显示所有内容"""
        if self.renderer:
            try:
                camera = self.renderer.GetActiveCamera()
                # Ensure perspective projection for proper fitting
                camera.SetParallelProjection(False)
                self.renderer.ResetCamera()
            except Exception as e:
                print(f"适应视图失败: {str(e)}")
                return

            self.render_window.Render()
    
    def display_part(self, part_name, parts_info=None, color=None, render_immediately=True):
        """显示指定的部件，保留其他已显示的内容

        Args:
            part_name (str): 要显示的部件名称
            parts_info (dict): 可选，部件信息字典，用于直接传递部件数据
            color (tuple): 可选，指定颜色 (R, G, B)，范围0-1
            render_immediately (bool): 是否立即渲染，默认为True。设为False可批量渲染提高性能
        """
        try:
            # 检查渲染器和渲染窗口是否可用
            if not self.renderer or not self.render_window:
                return False

            # 检查网格数据是否可用
            if not self.mesh_data:
                return False

            # Extract the actual part name from formatted text (e.g., "部件1 - Max Size: 1.0, Prism: True" -> "部件1")
            actual_part_name = part_name.split(' - ')[0] if ' - ' in part_name else part_name

            # 获取边界信息
            boundary_info = self._get_boundary_info()

            # 直接从mesh_data中获取部件信息，支持更多数据格式
            part_faces = []
            part_data = None

            # 优先使用传递的parts_info
            if parts_info:
                if isinstance(parts_info, dict):
                    # Try exact match first
                    if actual_part_name in parts_info:
                        part_data = parts_info[actual_part_name]
                        # 检查是否有 mesh_elements 或 geometry_elements（用于 DefaultPart）
                        if 'mesh_elements' in part_data:
                            mesh_elements = part_data['mesh_elements']
                            if 'faces' in mesh_elements:
                                # mesh_elements['faces'] 包含的是面索引列表
                                # 需要转换为面数据格式
                                if isinstance(self.mesh_data, dict):
                                    cells = self.mesh_data.get('cells', [])
                                elif hasattr(self.mesh_data, 'cells'):
                                    cells = self.mesh_data.cells
                                else:
                                    cells = []
                                part_faces = [{'nodes': cells[i] if i < len(cells) else []} for i in mesh_elements['faces']]
                        elif 'geometry_elements' in part_data:
                            # 几何元素暂时不显示
                            part_faces = []
                        else:
                            part_faces = part_data.get('faces', [])
                    # If not found, try to find by partial match or key that starts with the part name
                    else:
                        for key in parts_info:
                            if key == actual_part_name or key.startswith(actual_part_name):
                                part_data = parts_info[key]
                                if 'mesh_elements' in part_data:
                                    mesh_elements = part_data['mesh_elements']
                                    if 'faces' in mesh_elements:
                                        if isinstance(self.mesh_data, dict):
                                            cells = self.mesh_data.get('cells', [])
                                        elif hasattr(self.mesh_data, 'cells'):
                                            cells = self.mesh_data.cells
                                        else:
                                            cells = []
                                        part_faces = [{'nodes': cells[i] if i < len(cells) else []} for i in mesh_elements['faces']]
                                elif 'geometry_elements' in part_data:
                                    part_faces = []
                                else:
                                    part_faces = parts_info[key].get('faces', [])
                                break
            # 然后检查mesh_data中是否有parts_info
            elif isinstance(self.mesh_data, dict) and 'parts_info' in self.mesh_data:
                parts_info = self.mesh_data['parts_info']
                if isinstance(parts_info, dict):
                    # Try exact match first
                    if actual_part_name in parts_info:
                        part_data = parts_info[actual_part_name]
                        if 'mesh_elements' in part_data:
                            mesh_elements = part_data['mesh_elements']
                            if 'faces' in mesh_elements:
                                cells = self.mesh_data.get('cells', [])
                                part_faces = [{'nodes': cells[i] if i < len(cells) else []} for i in mesh_elements['faces']]
                        elif 'geometry_elements' in part_data:
                            part_faces = []
                        else:
                            part_faces = parts_info[actual_part_name].get('faces', [])
                    # If not found, try to find by partial match or key that starts with the part name
                    else:
                        for key in parts_info:
                            if key == actual_part_name or key.startswith(actual_part_name):
                                part_data = parts_info[key]
                                if 'mesh_elements' in part_data:
                                    mesh_elements = part_data['mesh_elements']
                                    if 'faces' in mesh_elements:
                                        cells = self.mesh_data.get('cells', [])
                                        part_faces = [{'nodes': cells[i] if i < len(cells) else []} for i in mesh_elements['faces']]
                                elif 'geometry_elements' in part_data:
                                    part_faces = []
                                else:
                                    part_faces = parts_info[key].get('faces', [])
                                break
            # 检查是否有边界信息
            elif boundary_info and isinstance(boundary_info, dict):
                # Try exact match first
                if actual_part_name in boundary_info:
                    part_data = boundary_info[actual_part_name]
                    part_faces = part_data.get("faces", [])
                # If not found, try to find by partial match or key that starts with the part name
                else:
                    for key in boundary_info:
                        if key == actual_part_name or key.startswith(actual_part_name):
                            part_data = boundary_info[key]
                            part_faces = part_data.get("faces", [])
                            break
            # 检查是否有直接的部件数据
            elif isinstance(self.mesh_data, dict):
                # Try exact match first
                if actual_part_name in self.mesh_data:
                    part_data = self.mesh_data[actual_part_name]
                    part_faces = part_data.get('faces', []) if isinstance(part_data, dict) else []
                # If not found, try to find by partial match or key that starts with the part name
                else:
                    for key in self.mesh_data:
                        if key == actual_part_name or key.startswith(actual_part_name):
                            part_data = self.mesh_data[key]
                            part_faces = part_data.get('faces', []) if isinstance(part_data, dict) else []
                            break

            # 如果已有部件数据但无面数据，避免回退显示整个网格
            if not part_faces and part_data:
                if render_immediately:
                    self.render_window.Render()
                return False

            # 检查是否有面数据
            if part_faces:
                # 创建只包含该部件的VTK网格
                part_polydata = self._create_boundary_polydata(part_faces)

                if part_polydata:
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(part_polydata)

                    part_actor = vtk.vtkActor()
                    part_actor.SetMapper(mapper)

                    # 设置部件颜色（如果未指定颜色，则使用不同颜色区分不同部件）
                    if color:
                        part_actor.GetProperty().SetColor(color)
                    else:
                        # Generate a unique color for each part based on its name
                        color_index = hash(actual_part_name) % 10  # Use hash to get consistent color per part
                        colors = [
                            (1.0, 0.0, 0.0),  # Red
                            (0.0, 1.0, 0.0),  # Green
                            (0.0, 0.0, 1.0),  # Blue
                            (1.0, 1.0, 0.0),  # Yellow
                            (1.0, 0.0, 1.0),  # Magenta
                            (0.0, 1.0, 1.0),  # Cyan
                            (1.0, 0.5, 0.0),  # Orange
                            (0.5, 0.0, 1.0),  # Purple
                            (0.0, 0.5, 1.0),  # Light Blue
                            (1.0, 0.5, 0.5),  # Pink
                        ]
                        part_actor.GetProperty().SetColor(colors[color_index])

                    # Apply current render mode to the part actor
                    if self.render_mode == "wireframe":
                        part_actor.GetProperty().SetRepresentationToWireframe()
                        part_actor.GetProperty().EdgeVisibilityOff()
                        part_actor.GetProperty().SetLineWidth(2.0)
                    elif self.render_mode == "surface-wireframe":
                        part_actor.GetProperty().SetRepresentationToSurface()
                        part_actor.GetProperty().EdgeVisibilityOn()
                        part_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
                        part_actor.GetProperty().SetLineWidth(1.5)
                    else:  # surface mode
                        part_actor.GetProperty().SetRepresentationToSurface()
                        part_actor.GetProperty().EdgeVisibilityOff()

                    part_actor.GetProperty().SetOpacity(1.0)

                    # 添加部件演员
                    self.renderer.AddActor(part_actor)

                    # Note: We don't set self.mesh_actor here to preserve existing actors
                    # 保存额外演员引用 for potential cleanup later
                    if not hasattr(self, 'additional_actors'):
                        self.additional_actors = []
                    self.additional_actors.append(part_actor)
            else:
                # 如果没有找到特定部件的数据，尝试显示整个网格
                vtk_mesh = self.create_vtk_mesh()
                if vtk_mesh:
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(vtk_mesh)

                    additional_actor = vtk.vtkActor()
                    additional_actor.SetMapper(mapper)

                    self._apply_default_mesh_properties()
                    if color:
                        additional_actor.GetProperty().SetColor(color)
                    else:
                        additional_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # Blue

                    # Apply current render mode to the additional actor
                    if self.render_mode == "wireframe":
                        additional_actor.GetProperty().SetRepresentationToWireframe()
                        additional_actor.GetProperty().EdgeVisibilityOff()
                        additional_actor.GetProperty().SetLineWidth(2.0)
                    elif self.render_mode == "surface-wireframe":
                        additional_actor.GetProperty().SetRepresentationToSurface()
                        additional_actor.GetProperty().EdgeVisibilityOn()
                        additional_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
                        additional_actor.GetProperty().SetLineWidth(1.5)
                    else:  # surface mode
                        additional_actor.GetProperty().SetRepresentationToSurface()
                        additional_actor.GetProperty().EdgeVisibilityOff()

                    self.renderer.AddActor(additional_actor)

                    # 保存额外演员引用 for potential cleanup later
                    if not hasattr(self, 'additional_actors'):
                        self.additional_actors = []
                    self.additional_actors.append(additional_actor)

            # 重渲染 - 仅当render_immediately为True时才渲染
            if render_immediately:
                self.render_window.Render()

            return True

        except Exception as e:
            print(f"显示部件时出错: {str(e)}")
            return False

    def _is_volume_part(self, part_data):
        if not isinstance(part_data, dict):
            return False
        mesh_elements = part_data.get('mesh_elements') or {}
        if mesh_elements.get('bodies'):
            return True
        if part_data.get('cell_dimension') == 3:
            return True
        if part_data.get('bc_type') == 'volume':
            return True
        return False

    def show_only_selected_part(self, part_name, parts_info=None):
        """只显示选中的部件，隐藏其他所有部件

        Args:
            part_name (str): 要显示的部件名称
            parts_info (dict): 可选，部件信息字典，用于直接传递部件数据
        """
        try:
            # 清空所有演员
            self.clear_mesh_actors()

            # 检查渲染器和渲染窗口是否可用
            if not self.renderer or not self.render_window:
                return False

            # 检查网格数据是否可用
            if not self.mesh_data:
                return False

            # 获取边界信息
            boundary_info = self._get_boundary_info()

            # 直接从mesh_data中获取部件信息，支持更多数据格式
            part_faces = []
            part_data = None

            # 优先使用传递的parts_info
            if parts_info:
                if isinstance(parts_info, dict) and part_name in parts_info:
                    part_data = parts_info[part_name]
                    part_faces = part_data.get('faces', [])
            # 然后检查mesh_data中是否有parts_info
            elif isinstance(self.mesh_data, dict) and 'parts_info' in self.mesh_data:
                parts_info = self.mesh_data['parts_info']
                if isinstance(parts_info, dict) and part_name in parts_info:
                    part_data = parts_info[part_name]
                    part_faces = part_data.get('faces', [])
            # 检查是否有边界信息
            elif boundary_info and part_name in boundary_info:
                part_data = boundary_info[part_name]
                part_faces = part_data.get("faces", [])
            # 检查是否有直接的部件数据
            elif isinstance(self.mesh_data, dict) and part_name in self.mesh_data:
                part_data = self.mesh_data[part_name]
                part_faces = part_data.get('faces', [])

            # 如果已有部件数据但无面数据，避免回退显示整个网格
            if not part_faces and part_data:
                return False

            # 检查是否有面数据
            if part_faces:
                # 创建只包含该部件的VTK网格
                part_polydata = self._create_boundary_polydata(part_faces)

                if part_polydata:
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(part_polydata)

                    part_actor = vtk.vtkActor()
                    part_actor.SetMapper(mapper)

                    # 设置部件颜色（黄色）
                    part_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
                    part_actor.GetProperty().SetLineWidth(4.0)
                    part_actor.GetProperty().SetOpacity(1.0)

                    # 添加部件演员
                    self.renderer.AddActor(part_actor)

                    # 保存部件演员引用
                    self.mesh_actor = part_actor
            else:
                # 如果没有找到特定部件的数据，尝试显示整个网格
                vtk_mesh = self.create_vtk_mesh()
                if vtk_mesh:
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(vtk_mesh)

                    self.mesh_actor = vtk.vtkActor()
                    self.mesh_actor.SetMapper(mapper)

                    self._apply_default_mesh_properties()
                    self._apply_render_mode()

                    self.renderer.AddActor(self.mesh_actor)

            # 添加坐标轴
            self.add_axes()

            # 重置相机并渲染
            self.renderer.ResetCamera()
            self.render_window.Render()
            self.enable_interaction()

            return True

        except Exception as e:
            return False

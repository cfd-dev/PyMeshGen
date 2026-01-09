#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt文件操作模块
处理网格文件的导入和导出功能
"""

import os
import sys
import vtk
from vtk.util import numpy_support
import numpy as np

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加 meshio 到 sys.path
meshio_path = os.path.join(project_root, "3rd_party", "meshio", "src")
if meshio_path not in sys.path:
    sys.path.insert(0, meshio_path)

# 导入通用网格数据类
from data_structure.mesh_data import MeshData


class FileOperations:
    """文件操作类"""

    def __init__(self, project_root):
        self.project_root = project_root
        self.mesh_dir = os.path.join(project_root, "meshes")
        
        # 确保网格目录存在
        if not os.path.exists(self.mesh_dir):
            os.makedirs(self.mesh_dir, exist_ok=True)

    def import_mesh_original(self, file_path):
        """导入网格文件（原始方法备份）"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.vtk':
                return self._import_vtk(file_path)
            elif file_ext == '.stl':
                return self._import_stl(file_path)
            elif file_ext == '.obj':
                return self._import_obj(file_path)
            elif file_ext == '.cas':
                return self._import_cas(file_path)
            elif file_ext == '.msh':
                return self._import_msh(file_path)
            elif file_ext == '.ply':
                return self._import_ply(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
        except Exception as e:
            print(f"导入网格失败: {str(e)}")
            return None

    def import_mesh(self, file_path):
        """导入网格文件（使用 meshio 实现）"""
        try:
            # 尝试导入 meshio
            try:
                import meshio
            except ImportError as e:
                print(f"无法导入 meshio: {e}")
                print("尝试使用原始方法...")
                return self.import_mesh_original(file_path)
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 对于 CAS 文件，使用原始方法（因为 meshio 不支持 CAS 格式）
            if file_ext == '.cas':
                return self.import_mesh_original(file_path)
            
            # 使用 meshio 读取网格文件
            mesh = meshio.read(file_path)
            
            # 创建 MeshData 对象
            mesh_data = MeshData(file_path=file_path, mesh_type=file_ext[1:])
            
            # 提取节点坐标
            mesh_data.node_coords = mesh.points.tolist() if hasattr(mesh.points, 'tolist') else list(mesh.points)
            
            # 提取单元数据
            mesh_data.cells = []
            for cell_block in mesh.cells:
                cell_type = cell_block.type
                cell_data = cell_block.data
                for cell in cell_data:
                    mesh_data.cells.append(cell.tolist() if hasattr(cell, 'tolist') else list(cell))
            
            # 提取点数据（如果有）
            if mesh.point_data:
                mesh_data.point_data = dict(mesh.point_data)
            
            # 提取单元数据（如果有）
            if mesh.cell_data:
                mesh_data.cell_data_dict = dict(mesh.cell_data)
            
            # 创建 VTK PolyData 对象用于可视化
            try:
                import vtk
                from vtk.util import numpy_support
                
                # 创建 VTK 点
                vtk_points = vtk.vtkPoints()
                for point in mesh_data.node_coords:
                    vtk_points.InsertNextPoint(point)
                
                # 创建 VTK 单元
                vtk_cells = vtk.vtkCellArray()
                for cell in mesh_data.cells:
                    vtk_cell = vtk.vtkPolygon()
                    vtk_cell.GetPointIds().SetNumberOfIds(len(cell))
                    for i, node_id in enumerate(cell):
                        vtk_cell.GetPointIds().SetId(i, node_id)
                    vtk_cells.InsertNextCell(vtk_cell)
                
                # 创建 PolyData
                poly_data = vtk.vtkPolyData()
                poly_data.SetPoints(vtk_points)
                poly_data.SetPolys(vtk_cells)
                
                mesh_data.vtk_poly_data = poly_data
            except Exception as e:
                print(f"创建 VTK PolyData 失败: {str(e)}")
            
            # 更新计数
            mesh_data.update_counts()
            
            return mesh_data
                
        except Exception as e:
            print(f"使用 meshio 导入网格失败: {str(e)}")
            # 如果 meshio 失败，回退到原始方法
            print("尝试使用原始方法...")
            return self.import_mesh_original(file_path)

    def _import_vtk(self, file_path):
        """导入VTK文件"""
        try:
            # 检查文件扩展名，使用合适的读取器
            file_ext = os.path.splitext(file_path)[1].lower()
            poly_data = None
            
            if file_ext == '.vtk':
                # 先读取文件头，确定文件类型
                file_type = None
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('DATASET'):
                            file_type = line.split()[1]
                            break
                
                # 根据文件类型选择合适的读取器
                if file_type == 'POLYDATA':
                    reader = vtk.vtkPolyDataReader()
                    reader.SetFileName(file_path)
                    reader.Update()
                    poly_data = reader.GetOutput()
                elif file_type == 'UNSTRUCTURED_GRID':
                    reader = vtk.vtkUnstructuredGridReader()
                    reader.SetFileName(file_path)
                    reader.ReadAllScalarsOn()
                    reader.ReadAllVectorsOn()
                    reader.Update()
                    
                    # 将UnstructuredGrid转换为PolyData
                    geometry_filter = vtk.vtkGeometryFilter()
                    geometry_filter.SetInputData(reader.GetOutput())
                    geometry_filter.Update()
                    poly_data = geometry_filter.GetOutput()
                else:
                    # 对于其他类型或无法确定类型，尝试使用UnstructuredGridReader
                    reader = vtk.vtkUnstructuredGridReader()
                    reader.SetFileName(file_path)
                    reader.ReadAllScalarsOn()
                    reader.ReadAllVectorsOn()
                    reader.Update()
                    
                    # 将UnstructuredGrid转换为PolyData
                    geometry_filter = vtk.vtkGeometryFilter()
                    geometry_filter.SetInputData(reader.GetOutput())
                    geometry_filter.Update()
                    poly_data = geometry_filter.GetOutput()
            
            if not poly_data or poly_data.GetNumberOfPoints() == 0:
                print(f"无法读取VTK文件: {file_path}")
                return None
            
            # 提取节点坐标
            points = poly_data.GetPoints()
            node_coords = []
            if points:
                num_points = points.GetNumberOfPoints()
                for i in range(num_points):
                    point = points.GetPoint(i)
                    node_coords.append(list(point))
            
            # 提取单元数据
            cells = poly_data.GetPolys()
            cell_data = []
            if cells:
                cell_array = cells.GetData()
                num_cells = cells.GetNumberOfCells()
                offset = 0
                
                for i in range(num_cells):
                    num_ids = cell_array.GetValue(offset)
                    # 只处理三角形和四边形单元
                    if num_ids == 3 or num_ids == 4:
                        cell_ids = []
                        for j in range(1, num_ids + 1):
                            cell_ids.append(cell_array.GetValue(offset + j))
                        cell_data.append(cell_ids)
                    offset += num_ids + 1
            
            # 创建MeshData对象
            mesh_data = MeshData(file_path=file_path, mesh_type='vtk')
            mesh_data.node_coords = node_coords
            mesh_data.cells = cell_data
            mesh_data.vtk_poly_data = poly_data
            mesh_data.update_counts()
            
            return mesh_data
            
        except Exception as e:
            print(f"导入VTK文件失败: {str(e)}")
            return None

    def _import_stl(self, file_path):
        """导入STL文件"""
        try:
            # 创建STL读取器
            reader = vtk.vtkSTLReader()
            reader.SetFileName(file_path)
            reader.Update()
            
            # 获取多边形数据
            poly_data = reader.GetOutput()
            
            # 提取节点坐标
            points = poly_data.GetPoints()
            if points:
                num_points = points.GetNumberOfPoints()
                node_coords = []
                for i in range(num_points):
                    point = points.GetPoint(i)
                    node_coords.append(list(point))
            else:
                node_coords = []
            
            # 提取面数据
            polys = poly_data.GetPolys()
            if polys:
                cell_array = polys.GetData()
                num_cells = polys.GetNumberOfCells()
                cell_data = []
                
                # 解析面数据
                offset = 0
                for i in range(num_cells):
                    num_ids = cell_array.GetValue(offset)
                    cell_ids = []
                    for j in range(1, num_ids + 1):
                        cell_ids.append(cell_array.GetValue(offset + j))
                    cell_data.append(cell_ids)
                    offset += num_ids + 1
            else:
                cell_data = []
            
            # 创建MeshData对象
            mesh_data = MeshData(file_path=file_path, mesh_type='stl')
            mesh_data.node_coords = node_coords
            mesh_data.cells = cell_data
            mesh_data.vtk_poly_data = poly_data
            mesh_data.update_counts()
            
            return mesh_data
            
        except Exception as e:
            print(f"导入STL文件失败: {str(e)}")
            return None

    def _import_obj(self, file_path):
        """导入OBJ文件"""
        try:
            # 创建OBJ读取器
            reader = vtk.vtkOBJReader()
            reader.SetFileName(file_path)
            reader.Update()
            
            # 获取多边形数据
            poly_data = reader.GetOutput()
            
            # 提取节点坐标
            points = poly_data.GetPoints()
            if points:
                num_points = points.GetNumberOfPoints()
                node_coords = []
                for i in range(num_points):
                    point = points.GetPoint(i)
                    node_coords.append(list(point))
            else:
                node_coords = []
            
            # 提取面数据
            polys = poly_data.GetPolys()
            if polys:
                cell_array = polys.GetData()
                num_cells = polys.GetNumberOfCells()
                cell_data = []
                
                # 解析面数据
                offset = 0
                for i in range(num_cells):
                    num_ids = cell_array.GetValue(offset)
                    cell_ids = []
                    for j in range(1, num_ids + 1):
                        cell_ids.append(cell_array.GetValue(offset + j))
                    cell_data.append(cell_ids)
                    offset += num_ids + 1
            else:
                cell_data = []
            
            # 创建MeshData对象
            mesh_data = MeshData(file_path=file_path, mesh_type='obj')
            mesh_data.node_coords = node_coords
            mesh_data.cells = cell_data
            mesh_data.vtk_poly_data = poly_data
            mesh_data.update_counts()
            
            return mesh_data
            
        except Exception as e:
            print(f"导入OBJ文件失败: {str(e)}")
            return None

    def _import_cas(self, file_path):
        """导入CAS文件 - 用于CFD网格"""
        try:
            # 创建MeshData对象
            mesh_data = MeshData(file_path=file_path, mesh_type='cas')

            # 首先尝试使用专门的CAS解析函数
            try:
                from fileIO.read_cas import parse_cas_to_unstr_grid
                unstr_grid = parse_cas_to_unstr_grid(file_path)

                if unstr_grid:
                    # 提取节点坐标
                    node_coords = []
                    if hasattr(unstr_grid, 'node_coords'):
                        for coord in unstr_grid.node_coords:
                            if len(coord) == 2:
                                node_coords.append([coord[0], coord[1], 0.0])  # 补充z坐标
                            else:
                                node_coords.append(list(coord))
                    elif hasattr(unstr_grid, 'nodes'):
                        for node in unstr_grid.nodes:
                            node_coords.append(list(node))

                    # 提取单元数据
                    cells = []
                    if hasattr(unstr_grid, 'cell_container'):
                        for cell in unstr_grid.cell_container:
                            if cell is not None:  # 确保cell不为None
                                if hasattr(cell, 'node_ids'):
                                    cell_ids = [nid if isinstance(nid, int) and nid > 0 else nid
                                               for nid in cell.node_ids]
                                    cells.append(cell_ids)
                                elif hasattr(cell, 'nodes'):
                                    # 如果cell有nodes属性
                                    cell_ids = []
                                    for node in cell.nodes:
                                        if hasattr(node, 'id'):
                                            cell_ids.append(node.id) 
                                    if cell_ids:
                                        cells.append(cell_ids)

                    # 更新MeshData对象
                    mesh_data.node_coords = node_coords
                    mesh_data.cells = cells
                    mesh_data.unstr_grid = unstr_grid  # 保留原始对象

                    # Extract parts information from boundary_info
                    boundary_info = getattr(unstr_grid, 'boundary_info', {})
                    if boundary_info:
                        parts_info = {}
                        for part_name, part_data in boundary_info.items():
                            # Create part info with essential properties
                            parts_info[part_name] = {
                                'bc_type': part_data.get('bc_type', 'unspecified'),
                                'faces': part_data.get('faces', []),
                                'face_count': len(part_data.get('faces', [])),
                                'part_name': part_name
                            }
                        mesh_data.parts_info = parts_info
                    else:
                        mesh_data.parts_info = {}

                    mesh_data.boundary_info = boundary_info  # 边界信息
                    mesh_data.update_counts()

                    return mesh_data
            except ImportError:
                print("未找到CAS解析模块，使用备用方法")
            except Exception as e:
                print(f"使用专门的CAS解析函数失败: {str(e)}")

            # 备用方法：使用VTK_IO模块
            try:
                from fileIO.vtk_io import parse_vtk_msh
                vtk_mesh_data = parse_vtk_msh(file_path)

                if vtk_mesh_data:
                    if isinstance(vtk_mesh_data, dict):
                        # 从字典更新MeshData对象
                        mesh_data.node_coords = vtk_mesh_data.get('node_coords', [])
                        mesh_data.cells = vtk_mesh_data.get('cells', [])

                        # Extract parts information from the imported data
                        parts_info = vtk_mesh_data.get('parts_info', {})
                        if not parts_info and 'boundary_info' in vtk_mesh_data:
                            # Extract from boundary_info if parts_info is not directly available
                            boundary_info = vtk_mesh_data.get('boundary_info', {})
                            parts_info = {}
                            for part_name, part_data in boundary_info.items():
                                parts_info[part_name] = {
                                    'bc_type': part_data.get('bc_type', 'unspecified'),
                                    'faces': part_data.get('faces', []),
                                    'face_count': len(part_data.get('faces', [])),
                                    'part_name': part_name
                                }

                        mesh_data.parts_info = parts_info
                        mesh_data.boundary_info = vtk_mesh_data.get('boundary_info', {})
                    else:
                        # 如果返回的是Unstructured_Grid对象
                        mesh_data.unstr_grid = vtk_mesh_data
                        mesh_data.node_coords = getattr(vtk_mesh_data, 'node_coords', [])

                        # Extract parts information from boundary_info
                        boundary_info = getattr(vtk_mesh_data, 'boundary_info', {})
                        if boundary_info:
                            parts_info = {}
                            for part_name, part_data in boundary_info.items():
                                parts_info[part_name] = {
                                    'bc_type': part_data.get('bc_type', 'unspecified'),
                                    'faces': part_data.get('faces', []),
                                    'face_count': len(part_data.get('faces', [])),
                                    'part_name': part_name
                                }
                            mesh_data.parts_info = parts_info
                        else:
                            mesh_data.parts_info = {}

                        mesh_data.boundary_info = boundary_info

                    mesh_data.update_counts()
                    return mesh_data
            except ImportError:
                pass  # 继续下面的实现
            except Exception as e:
                print(f"使用VTK_IO模块解析CAS文件失败: {str(e)}")

            # 简单的文本解析（根据实际需求调整）
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024)  # 读取前1024个字符

            # 基本的CAS文件检测
            if 'FLUENT' in content.upper() or 'CASE' in content.upper():
                # 这是一个CAS文件，但需要更复杂的解析逻辑
                # 这里返回一个基本结构
                mesh_data.update_counts()
                return mesh_data
            else:
                print("文件不包含FLUENT或CASE标识，可能不是有效的CAS文件")
                # 即使不是标准CAS文件，也返回一个基本结构而不是None
                mesh_data.update_counts()
                return mesh_data

        except Exception as e:
            print(f"导入CAS文件失败: {str(e)}")
            # 返回一个基本结构而不是None，以避免调用代码出现'NoneType'错误
            mesh_data = MeshData(file_path=file_path, mesh_type='cas')
            mesh_data.update_counts()
            return mesh_data

    def _import_msh(self, file_path):
        """导入Gmsh MSH文件"""
        try:
            # 创建MeshData对象
            mesh_data = MeshData(file_path=file_path, mesh_type='msh')
            
            # MSH文件格式解析（简化版）
            # 通常需要使用Gmsh Python API，这里提供基本解析
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            nodes = []
            elements = []
            in_nodes = False
            in_elements = False
            
            for line in lines:
                line = line.strip()
                
                if line == '$Nodes':
                    in_nodes = True
                    continue
                elif line == '$EndNodes':
                    in_nodes = False
                    continue
                elif line == '$Elements':
                    in_elements = True
                    continue
                elif line == '$EndElements':
                    in_elements = False
                    continue
                
                if in_nodes and line.isdigit():
                    # 跳过节点数量行
                    continue
                elif in_nodes and line:
                    # 解析节点数据: node_number x y z
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            nodes.append([x, y, z])
                        except (ValueError, IndexError):
                            continue
                
                if in_elements and line.isdigit():
                    # 跳过元素数量行
                    continue
                elif in_elements and line:
                    # 解析元素数据
                    parts = line.split()
                    if len(parts) >= 5:  # 至少包含类型、标签、节点数和节点ID
                        try:
                            # 通常格式: element_number type tag_count node_ids...
                            node_ids = [int(parts[i]) - 1 for i in range(3, len(parts))]  # 转换为0基索引
                            elements.append(node_ids)
                        except (ValueError, IndexError):
                            continue
            
            # 更新MeshData对象
            mesh_data.node_coords = nodes
            mesh_data.cells = elements
            mesh_data.update_counts()
            
            return mesh_data
            
        except Exception as e:
            print(f"导入MSH文件失败: {str(e)}")
            return None

    def _import_ply(self, file_path):
        """导入PLY文件"""
        try:
            # 创建MeshData对象
            mesh_data = MeshData(file_path=file_path, mesh_type='ply')
            
            # 创建PLY读取器
            reader = vtk.vtkPLYReader()
            reader.SetFileName(file_path)
            reader.Update()
            
            # 获取多边形数据
            poly_data = reader.GetOutput()
            
            # 提取节点坐标
            points = poly_data.GetPoints()
            if points:
                num_points = points.GetNumberOfPoints()
                node_coords = []
                for i in range(num_points):
                    point = points.GetPoint(i)
                    node_coords.append(list(point))
            else:
                node_coords = []
            
            # 提取面数据
            polys = poly_data.GetPolys()
            if polys:
                cell_array = polys.GetData()
                num_cells = polys.GetNumberOfCells()
                cell_data = []
                
                # 解析面数据
                offset = 0
                for i in range(num_cells):
                    num_ids = cell_array.GetValue(offset)
                    cell_ids = []
                    for j in range(1, num_ids + 1):
                        cell_ids.append(cell_array.GetValue(offset + j))
                    cell_data.append(cell_ids)
                    offset += num_ids + 1
            else:
                cell_data = []
            
            # 更新MeshData对象
            mesh_data.node_coords = node_coords
            mesh_data.cells = cell_data
            mesh_data.vtk_poly_data = poly_data
            mesh_data.update_counts()
            
            return mesh_data
            
        except Exception as e:
            print(f"导入PLY文件失败: {str(e)}")
            return None

    def export_mesh(self, mesh_data, file_path):
        """导出网格文件"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.vtk':
                return self._export_vtk(mesh_data, file_path)
            elif file_ext == '.stl':
                return self._export_stl(mesh_data, file_path)
            elif file_ext == '.obj':
                return self._export_obj(mesh_data, file_path)
            elif file_ext == '.ply':
                return self._export_ply(mesh_data, file_path)
            else:
                raise ValueError(f"不支持的导出格式: {file_ext}")
                
        except Exception as e:
            print(f"导出网格失败: {str(e)}")
            return False

    def _export_vtk(self, mesh_data, file_path):
        """导出VTK文件"""
        try:
            # 检查mesh_data是否包含vtk_poly_data
            if isinstance(mesh_data, dict) and 'vtk_poly_data' in mesh_data:
                vtk_poly_data = mesh_data['vtk_poly_data']
            elif hasattr(mesh_data, 'GetPoints') and hasattr(mesh_data, 'GetCells'):
                # mesh_data本身就是一个VTK对象
                vtk_poly_data = mesh_data
            else:
                # 从字典数据创建VTK对象
                vtk_poly_data = self._create_vtk_from_dict(mesh_data)
            
            if vtk_poly_data:
                # 创建VTK写入器
                writer = vtk.vtkUnstructuredGridWriter()
                writer.SetFileName(file_path)
                writer.SetInputData(vtk_poly_data)
                writer.Write()
                return True
            else:
                print("无法创建VTK数据对象")
                return False
                
        except Exception as e:
            print(f"导出VTK文件失败: {str(e)}")
            return False

    def _export_stl(self, mesh_data, file_path):
        """导出STL文件"""
        try:
            # 从字典数据创建VTK对象
            vtk_poly_data = self._create_vtk_from_dict(mesh_data)
            
            if vtk_poly_data:
                # 创建STL写入器
                writer = vtk.vtkSTLWriter()
                writer.SetFileName(file_path)
                writer.SetInputData(vtk_poly_data)
                writer.Write()
                return True
            else:
                print("无法创建VTK数据对象")
                return False
                
        except Exception as e:
            print(f"导出STL文件失败: {str(e)}")
            return False

    def _export_obj(self, mesh_data, file_path):
        """导出OBJ文件"""
        try:
            # 从字典数据创建VTK对象
            vtk_poly_data = self._create_vtk_from_dict(mesh_data)
            
            if vtk_poly_data:
                # 创建OBJ写入器
                writer = vtk.vtkOBJWriter()
                writer.SetFileName(file_path)
                writer.SetInputData(vtk_poly_data)
                writer.Write()
                return True
            else:
                print("无法创建VTK数据对象")
                return False
                
        except Exception as e:
            print(f"导出OBJ文件失败: {str(e)}")
            return False

    def _export_ply(self, mesh_data, file_path):
        """导出PLY文件"""
        try:
            # 从字典数据创建VTK对象
            vtk_poly_data = self._create_vtk_from_dict(mesh_data)
            
            if vtk_poly_data:
                # 创建PLY写入器
                writer = vtk.vtkPLYWriter()
                writer.SetFileName(file_path)
                writer.SetInputData(vtk_poly_data)
                writer.Write()
                return True
            else:
                print("无法创建VTK数据对象")
                return False
                
        except Exception as e:
            print(f"导出PLY文件失败: {str(e)}")
            return False

    def _create_vtk_from_dict(self, mesh_data):
        """从字典数据创建VTK对象"""
        try:
            # 创建VTK点
            vtk_points = vtk.vtkPoints()
            
            # 添加节点坐标
            if isinstance(mesh_data, dict):
                node_coords = mesh_data.get('node_coords', [])
            elif hasattr(mesh_data, 'node_coords'):
                node_coords = mesh_data.node_coords
            else:
                node_coords = []
            
            for coord in node_coords:
                if len(coord) == 2:
                    vtk_points.InsertNextPoint(coord[0], coord[1], 0.0)
                elif len(coord) >= 3:
                    vtk_points.InsertNextPoint(coord[0], coord[1], coord[2])
            
            # 创建VTK单元
            vtk_cells = vtk.vtkCellArray()
            
            # 添加单元数据
            if isinstance(mesh_data, dict):
                cells = mesh_data.get('cells', [])
            elif hasattr(mesh_data, 'cells'):
                cells = mesh_data.cells
            else:
                cells = []
            
            for cell in cells:
                if len(cell) == 3:  # 三角形
                    triangle = vtk.vtkTriangle()
                    for i, node_id in enumerate(cell):
                        triangle.GetPointIds().SetId(i, node_id)
                    vtk_cells.InsertNextCell(triangle)
                elif len(cell) == 4:  # 四边形
                    quad = vtk.vtkQuad()
                    for i, node_id in enumerate(cell):
                        quad.GetPointIds().SetId(i, node_id)
                    vtk_cells.InsertNextCell(quad)
                elif len(cell) > 4:  # 多边形
                    polygon = vtk.vtkPolygon()
                    polygon.GetPointIds().SetNumberOfIds(len(cell))
                    for i, node_id in enumerate(cell):
                        polygon.GetPointIds().SetId(i, node_id)
                    vtk_cells.InsertNextCell(polygon)
            
            # 创建VTK多边形数据
            vtk_poly_data = vtk.vtkPolyData()
            vtk_poly_data.SetPoints(vtk_points)
            vtk_poly_data.SetPolys(vtk_cells)
            
            return vtk_poly_data
            
        except Exception as e:
            print(f"从字典数据创建VTK对象失败: {str(e)}")
            return None

    def get_supported_import_formats(self):
        """获取支持的导入格式"""
        return [
            "VTK文件 (*.vtk)",
            "STL文件 (*.stl)", 
            "OBJ文件 (*.obj)",
            "CAS文件 (*.cas)",
            "Gmsh文件 (*.msh)",
            "PLY文件 (*.ply)"
        ]

    def get_supported_export_formats(self):
        """获取支持的导出格式"""
        return [
            "VTK文件 (*.vtk)",
            "STL文件 (*.stl)",
            "OBJ文件 (*.obj)",
            "PLY文件 (*.ply)"
        ]
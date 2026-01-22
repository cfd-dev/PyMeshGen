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

# 导入统一网格数据类
from data_structure.unstructured_grid import Unstructured_Grid

# 导入通用CGNS读取器
try:
    from fileIO.universal_cgns_reader import UniversalCGNSReader
    CGNS_READER_AVAILABLE = True
except ImportError:
    CGNS_READER_AVAILABLE = False
    print("警告: UniversalCGNSReader 未找到，CGNS 文件将使用 meshio 读取")


class FileOperations:
    """文件操作类"""

    def __init__(self, project_root, log_callback=None):
        self.project_root = project_root
        self.mesh_dir = os.path.join(project_root, "meshes")
        self.log_callback = log_callback
        
        # 确保网格目录存在
        if not os.path.exists(self.mesh_dir):
            os.makedirs(self.mesh_dir, exist_ok=True)

    def _infer_dimension_from_vtk_cells(self, vtk_grid):
        """根据VTK单元类型推断网格维度（2D/3D）"""
        if vtk_grid is None or not hasattr(vtk_grid, 'GetCellTypes'):
            return None
        vtk_cell_types = vtk.vtkCellTypes()
        vtk_grid.GetCellTypes(vtk_cell_types)
        if vtk_cell_types.GetNumberOfTypes() == 0:
            return None
        from data_structure.vtk_types import VTKCellType
        from utils.geom_toolkit import detect_mesh_dimension_by_cell_type
        cell_types = []
        for i in range(vtk_cell_types.GetNumberOfTypes()):
            cell_type = vtk_cell_types.GetCellType(i)
            try:
                cell_types.append(VTKCellType(cell_type))
            except ValueError:
                continue
        if not cell_types:
            return None
        return detect_mesh_dimension_by_cell_type(cell_types)

    def _log(self, message, level="info"):
        """记录日志到GUI信息窗口"""
        if self.log_callback:
            if level == "info":
                self.log_callback(message)
            elif level == "error":
                self.log_callback(message)
            elif level == "warning":
                self.log_callback(message)
        else:
            print(message)

    def _build_unstructured_grid(
        self,
        node_coords,
        cells,
        dimension,
        file_path=None,
        mesh_type=None,
        vtk_poly_data=None,
        boundary_nodes_idx=None,
        cell_dimension=None,
    ):
        grid = Unstructured_Grid.from_cells(
            node_coords=node_coords,
            cells=cells,
            boundary_nodes_idx=boundary_nodes_idx or [],
            grid_dimension=dimension,
            cell_dimension=cell_dimension,
        )
        grid.file_path = file_path
        grid.mesh_type = mesh_type
        if vtk_poly_data is not None:
            grid.vtk_poly_data = vtk_poly_data
        grid.update_counts()
        return grid

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
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 对于 CGNS 文件，使用 UniversalCGNSReader
            if file_ext == '.cgns' and CGNS_READER_AVAILABLE:
                return self._import_cgns(file_path)
            
            # 对于 CAS 文件，使用原始方法（因为 meshio 不支持 CAS 格式）
            if file_ext == '.cas':
                return self.import_mesh_original(file_path)
            
            # 尝试导入 meshio
            try:
                import meshio
            except ImportError as e:
                print(f"无法导入 meshio: {e}")
                print("尝试使用原始方法...")
                return self.import_mesh_original(file_path)
            
            # 使用 meshio 读取网格文件
            mesh = meshio.read(file_path)
            
            node_coords = mesh.points.tolist() if hasattr(mesh.points, 'tolist') else list(mesh.points)

            # 提取单元数据
            cells = []
            for cell_block in mesh.cells:
                cell_type = cell_block.type
                cell_data = cell_block.data
                for cell in cell_data:
                    cells.append(cell.tolist() if hasattr(cell, 'tolist') else list(cell))

            # 设置网格维度（优先使用单元拓扑维度）
            cell_dims = [getattr(cell_block, 'dim', None) for cell_block in mesh.cells]
            cell_dims = [dim for dim in cell_dims if dim in (2, 3)]
            if cell_dims:
                mesh_dimension = max(cell_dims)
            else:
                mesh_dimension = 2

            mesh_data = self._build_unstructured_grid(
                node_coords=node_coords,
                cells=cells,
                dimension=mesh_dimension,
                file_path=file_path,
                mesh_type=file_ext[1:],
                cell_dimension=mesh_dimension,
            )

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
                    if len(point) >= 3:
                        vtk_points.InsertNextPoint(point[0], point[1], point[2])
                    elif len(point) == 2:
                        vtk_points.InsertNextPoint(point[0], point[1], 0.0)
                
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
            unstructured_grid = None
            
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
                    unstructured_grid = reader.GetOutput()
                    
                    # 将UnstructuredGrid转换为PolyData
                    geometry_filter = vtk.vtkGeometryFilter()
                    geometry_filter.SetInputData(unstructured_grid)
                    geometry_filter.Update()
                    poly_data = geometry_filter.GetOutput()
                else:
                    # 对于其他类型或无法确定类型，尝试使用UnstructuredGridReader
                    reader = vtk.vtkUnstructuredGridReader()
                    reader.SetFileName(file_path)
                    reader.ReadAllScalarsOn()
                    reader.ReadAllVectorsOn()
                    reader.Update()
                    unstructured_grid = reader.GetOutput()
                    
                    # 将UnstructuredGrid转换为PolyData
                    geometry_filter = vtk.vtkGeometryFilter()
                    geometry_filter.SetInputData(unstructured_grid)
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
            
            mesh_data = self._build_unstructured_grid(
                node_coords=node_coords,
                cells=cell_data,
                dimension=2,
                file_path=file_path,
                mesh_type='vtk',
                vtk_poly_data=poly_data,
                cell_dimension=2,
            )
            if unstructured_grid:
                mesh_dim = self._infer_dimension_from_vtk_cells(unstructured_grid)
            else:
                mesh_dim = self._infer_dimension_from_vtk_cells(poly_data)
            if mesh_dim in (2, 3):
                mesh_data.dimension = mesh_dim
            else:
                mesh_data.dimension = 2
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
            
            mesh_data = self._build_unstructured_grid(
                node_coords=node_coords,
                cells=cell_data,
                dimension=2,
                file_path=file_path,
                mesh_type='stl',
                vtk_poly_data=poly_data,
                cell_dimension=2,
            )
            
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
            
            mesh_data = self._build_unstructured_grid(
                node_coords=node_coords,
                cells=cell_data,
                dimension=2,
                file_path=file_path,
                mesh_type='obj',
                vtk_poly_data=poly_data,
                cell_dimension=2,
            )
            
            return mesh_data
            
        except Exception as e:
            print(f"导入OBJ文件失败: {str(e)}")
            return None

    def _import_cgns(self, file_path):
        """导入CGNS文件 - 使用 UniversalCGNSReader"""
        try:
            self._log(f"使用 UniversalCGNSReader 读取 CGNS 文件: {file_path}")
            
            # 创建 CGNS 读取器
            reader = UniversalCGNSReader(file_path)
            
            # 读取 CGNS 文件
            if not reader.read():
                self._log(f"读取 CGNS 文件失败", level="error")
                return None
            
            node_coords = reader.points.tolist() if hasattr(reader.points, 'tolist') else list(reader.points)

            # 提取单元数据（区分表面与体单元）
            surface_cells = []
            volume_cells = []
            line_cells = []
            cell_groups = []
            volume_parts_info = {}
            section_entries = []

            cell_info_list = reader.cell_info if reader.cell_info else []
            for idx, cell in enumerate(reader.cells):
                cell_data = cell['data']
                num_cells = cell.get('num_cells', cell_data.shape[0])
                nodes_per_cell = cell.get('num_nodes', cell_data.shape[1] if cell_data.ndim > 1 else 0)
                cell_type = cell.get('type', '')
                cell_info = cell_info_list[idx] if idx < len(cell_info_list) else {}
                section_name = cell_info.get('name', f"Section_{idx}")
                element_range = cell_info.get('element_range')

                dimension = self._infer_cgns_cell_dimension(cell_type, nodes_per_cell)

                section_entries.append({
                    'name': section_name,
                    'type': cell_type,
                    'dimension': dimension,
                    'num_cells': num_cells,
                    'nodes_per_cell': nodes_per_cell,
                    'element_range': element_range,
                    'cell_data': cell_data
                })

            body_dim = 0
            for entry in section_entries:
                if entry['dimension'] in (2, 3):
                    body_dim = max(body_dim, entry['dimension'])
            mesh_dimension = body_dim if body_dim in (2, 3) else 2

            boundary_names = set()
            if reader.boundary_info:
                for _, part_data in reader.boundary_info.items():
                    boundary_names.add(part_data.get('bc_name') or part_data.get('family_name'))

            volume_part_name = None
            if body_dim == 2:
                volume_part_name = self._get_cgns_volume_part_name(section_entries, boundary_names)

            for entry in section_entries:
                section_name = entry['name']
                cell_type = entry['type']
                dimension = entry['dimension']
                num_cells = entry['num_cells']
                nodes_per_cell = entry['nodes_per_cell']
                element_range = entry['element_range']
                cell_data = entry['cell_data']
                indices = []

                if body_dim == 2:
                    if dimension == 2:
                        start_index = len(surface_cells)
                        for cell_nodes in cell_data:
                            surface_cells.append(cell_nodes.tolist() if hasattr(cell_nodes, 'tolist') else list(cell_nodes))
                        indices = list(range(start_index, start_index + num_cells))
                        part_key = volume_part_name or section_name
                        part_entry = volume_parts_info.get(part_key, {'part_name': part_key})
                        mesh_elements = part_entry.get('mesh_elements', {})
                        mesh_elements.setdefault('faces', []).extend(indices)
                        part_entry['mesh_elements'] = mesh_elements
                        part_entry.setdefault('bc_type', 'volume')
                        part_entry.setdefault('cell_type', cell_type)
                        part_entry.setdefault('cell_dimension', dimension)
                        volume_parts_info[section_name] = part_entry
                    elif dimension == 1:
                        start_index = len(line_cells)
                        for cell_nodes in cell_data:
                            line_cells.append(cell_nodes.tolist() if hasattr(cell_nodes, 'tolist') else list(cell_nodes))
                        indices = list(range(start_index, start_index + num_cells))
                else:
                    if dimension == 3:
                        start_index = len(volume_cells)
                        for cell_nodes in cell_data:
                            volume_cells.append(cell_nodes.tolist() if hasattr(cell_nodes, 'tolist') else list(cell_nodes))
                        indices = list(range(start_index, start_index + num_cells))
                        part_entry = volume_parts_info.get(section_name, {'part_name': section_name})
                        mesh_elements = part_entry.get('mesh_elements', {})
                        mesh_elements.setdefault('bodies', []).extend(indices)
                        part_entry['mesh_elements'] = mesh_elements
                        part_entry.setdefault('bc_type', 'volume')
                        part_entry.setdefault('cell_type', cell_type)
                        part_entry.setdefault('cell_dimension', dimension)
                        volume_parts_info[section_name] = part_entry
                    elif dimension == 2:
                        start_index = len(surface_cells)
                        for cell_nodes in cell_data:
                            surface_cells.append(cell_nodes.tolist() if hasattr(cell_nodes, 'tolist') else list(cell_nodes))
                        indices = list(range(start_index, start_index + num_cells))
                    elif dimension == 1:
                        start_index = len(line_cells)
                        for cell_nodes in cell_data:
                            line_cells.append(cell_nodes.tolist() if hasattr(cell_nodes, 'tolist') else list(cell_nodes))
                        indices = list(range(start_index, start_index + num_cells))

                cell_groups.append({
                    'name': section_name,
                    'type': cell_type,
                    'dimension': dimension,
                    'num_cells': num_cells,
                    'nodes_per_cell': nodes_per_cell,
                    'indices': indices,
                    'element_range': element_range
                })

            mesh_data = self._build_unstructured_grid(
                node_coords=node_coords,
                cells=surface_cells,
                dimension=mesh_dimension,
                file_path=file_path,
                mesh_type='cgns',
                cell_dimension=2 if surface_cells else mesh_dimension,
            )
            mesh_data.volume_cells = volume_cells
            mesh_data.cell_groups = cell_groups
            if line_cells:
                mesh_data.line_cells = line_cells
            
            # 提取元数据
            if reader.metadata:
                mesh_data.metadata = reader.metadata
            
            # 提取边界信息并转换为parts_info
            parts_info = {}
            boundary_info_with_faces = {}
            if reader.boundary_info:
                # 将boundary_info转换为parts_info格式，并提取faces数据
                for part_name, part_data in reader.boundary_info.items():
                    family_name = part_data.get('family_name', part_name)
                    bc_name = part_data.get('bc_name') or part_name
                    grid_location = (part_data.get('grid_location') or '').lower()
                    element_indices = []

                    element_list = part_data.get('element_list')
                    element_range = part_data.get('element_range')
                    point_list = part_data.get('point_list')
                    point_range = part_data.get('point_range')

                    if element_list:
                        element_indices = [int(v) for v in element_list]
                    elif element_range and len(element_range) >= 2:
                        element_indices = list(range(int(element_range[0]), int(element_range[1]) + 1))
                    elif grid_location not in ('vertex', 'vertices'):
                        if point_list:
                            element_indices = [int(v) for v in point_list]
                        elif point_range and len(point_range) >= 2:
                            element_indices = list(range(int(point_range[0]), int(point_range[1]) + 1))

                    faces = self._extract_faces_from_element_indices(reader.cells, reader.cell_info, element_indices)

                    boundary_part_name = bc_name
                    if boundary_part_name in volume_parts_info:
                        boundary_part_name = f"{boundary_part_name}_boundary"

                    parts_info[boundary_part_name] = {
                        'bc_type': part_data.get('bc_type', 'boundary'),
                        'point_range': point_range,
                        'element_range': element_range,
                        'family_name': family_name,
                        'grid_location': part_data.get('grid_location'),
                        'part_name': boundary_part_name,
                        'faces': faces
                    }

                    boundary_info_with_faces[boundary_part_name] = {
                        'bc_type': part_data.get('bc_type', 'boundary'),
                        'point_range': point_range,
                        'element_range': element_range,
                        'family_name': family_name,
                        'faces': faces
                    }

            # 添加体单元部件信息
            if volume_parts_info:
                for part_name, section_data in volume_parts_info.items():
                    target_name = part_name
                    if target_name in parts_info:
                        target_name = f"{target_name}_volume"
                    updated_section = dict(section_data)
                    updated_section['part_name'] = target_name
                    parts_info[target_name] = updated_section

            mesh_data.parts_info = parts_info
            mesh_data.boundary_info = boundary_info_with_faces
            
            # 创建 VTK PolyData 对象用于可视化
            try:
                import vtk
                from vtk.util import numpy_support
                
                # 创建 VTK 点
                vtk_points = vtk.vtkPoints()
                for point in mesh_data.node_coords:
                    if len(point) >= 3:
                        vtk_points.InsertNextPoint(point[0], point[1], point[2])
                    elif len(point) == 2:
                        vtk_points.InsertNextPoint(point[0], point[1], 0.0)
                
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
                self._log(f"创建 VTK PolyData 失败: {str(e)}", level="warning")
            mesh_data.update_counts()

            self._log(self._format_cgns_summary(section_entries, body_dim))

            self._log(f"✅ 成功读取 CGNS 文件")
            self._log(f"   节点数: {len(mesh_data.node_coords)}")
            self._log(f"   面单元数: {len(mesh_data.cells)}")
            if mesh_data.volume_cells:
                self._log(f"   体单元数: {len(mesh_data.volume_cells)}")
            
            # 记录边界信息
            if mesh_data.parts_info:
                self._log(f"   部件数: {len(mesh_data.parts_info)}")
                for part_name, part_data in mesh_data.parts_info.items():
                    range_info = part_data.get('element_range') or part_data.get('point_range')
                    mesh_elements = part_data.get('mesh_elements', {})
                    if range_info and len(range_info) >= 2:
                        self._log(f"   - {part_name}: 单元范围 [{range_info[0]}, {range_info[1]}]")
                    elif mesh_elements.get('bodies'):
                        self._log(f"   - {part_name}: 体单元 {len(mesh_elements.get('bodies', []))} 个")
                    elif mesh_elements.get('faces'):
                        self._log(f"   - {part_name}: 面 {len(mesh_elements.get('faces', []))} 个")
                    else:
                        self._log(f"   - {part_name}")
            
            return mesh_data
            
        except Exception as e:
            print(f"导入CGNS文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_faces_from_point_range(self, cells, point_range):
        """从point_range提取faces数据
        
        Args:
            cells: 单元数据列表
            point_range: 单元索引范围 [start, end]
            
        Returns:
            list: faces数据，每个元素是包含节点索引的字典
        """
        faces = []
        if not point_range or len(point_range) != 2:
            return faces
        
        start_idx = point_range[0] - 1  # 转换为0基索引
        end_idx = point_range[1] - 1    # 转换为0基索引
        
        # 遍历所有单元组，找到属于该范围的单元
        current_global_idx = 1  # CGNS使用1基索引
        
        for cell_group in cells:
            cell_data = cell_group['data']
            num_cells = cell_data.shape[0]
            
            # 计算这个单元组的全局索引范围
            group_start = current_global_idx
            group_end = current_global_idx + num_cells - 1
            
            # 检查是否有重叠
            overlap_start = max(start_idx, group_start - 1)
            overlap_end = min(end_idx, group_end - 1)
            
            if overlap_start <= overlap_end:
                # 有重叠，提取这些单元
                local_start = overlap_start - (group_start - 1)
                local_end = overlap_end - (group_start - 1) + 1
                
                for i in range(local_start, local_end):
                    cell_nodes = cell_data[i]
                    faces.append({
                        'nodes': cell_nodes.tolist() if hasattr(cell_nodes, 'tolist') else list(cell_nodes)
                    })
            
            current_global_idx += num_cells
        
        return faces

    def _extract_faces_from_element_indices(self, cells, cell_info, element_indices):
        """从单元索引列表提取边界faces（2D面/1D线）"""
        faces = []
        if not element_indices:
            return faces

        current_global_idx = 1  # CGNS使用1基索引
        for idx, cell_group in enumerate(cells):
            cell_data = cell_group['data']
            num_cells = cell_data.shape[0]
            nodes_per_cell = cell_group.get('num_nodes', cell_data.shape[1] if cell_data.ndim > 1 else 0)
            cell_type = cell_group.get('type', '')

            dimension = self._infer_cgns_cell_dimension(cell_type, nodes_per_cell)
            if dimension not in (1, 2):
                current_global_idx += num_cells
                continue

            info = cell_info[idx] if cell_info and idx < len(cell_info) else {}
            element_range = info.get('element_range')
            if element_range and len(element_range) >= 2:
                group_start = int(element_range[0])
                group_end = int(element_range[1])
            else:
                group_start = current_global_idx
                group_end = current_global_idx + num_cells - 1

            for elem_id in element_indices:
                if group_start <= elem_id <= group_end:
                    local_index = elem_id - group_start
                    if 0 <= local_index < num_cells:
                        cell_nodes = cell_data[local_index]
                        faces.append({
                            'nodes': cell_nodes.tolist() if hasattr(cell_nodes, 'tolist') else list(cell_nodes)
                        })

            current_global_idx += num_cells

        return faces

    def _infer_cgns_cell_dimension(self, cell_type, nodes_per_cell):
        """根据单元类型与节点数判断单元维度"""
        cell_type = (cell_type or '').lower()
        if 'bar' in cell_type or 'line' in cell_type:
            return 1
        if 'tri' in cell_type:
            return 2
        if 'quad' in cell_type:
            return 2
        if 'tet' in cell_type or 'hexa' in cell_type or 'hex' in cell_type or 'wedge' in cell_type or 'prism' in cell_type or 'penta' in cell_type or 'pyra' in cell_type:
            return 3
        if nodes_per_cell <= 2:
            return 1
        if nodes_per_cell in (3, 4):
            return 2
        return 3

    def _get_cgns_volume_part_name(self, section_entries, boundary_names):
        candidates = []
        for entry in section_entries:
            if entry['dimension'] != 2:
                continue
            section_name = entry['name']
            if section_name in boundary_names:
                continue
            candidates.append(section_name)
        if candidates:
            unique = []
            for name in candidates:
                if name not in unique:
                    unique.append(name)
            for name in unique:
                if name.lower() == 'fluid':
                    return name
            if len(unique) == 1:
                return unique[0]
        return "Fluid"

    def _format_cgns_summary(self, section_entries, body_dim):
        line_count = 0
        tri_count = 0
        quad_count = 0
        volume_count = 0

        for entry in section_entries:
            dimension = entry['dimension']
            cell_type = (entry.get('type') or '').lower()
            count = entry.get('num_cells', 0)
            if dimension == 1:
                line_count += count
            elif dimension == 2:
                if 'tri' in cell_type:
                    tri_count += count
                elif 'quad' in cell_type:
                    quad_count += count
                else:
                    tri_count += count
            elif dimension == 3:
                volume_count += count

        summary = "   CGNS统计: "
        parts = []
        if line_count:
            parts.append(f"线单元 {line_count}")
        if quad_count:
            parts.append(f"四边形 {quad_count}")
        if tri_count:
            parts.append(f"三角形 {tri_count}")
        if volume_count:
            parts.append(f"体单元 {volume_count}")
        if not parts:
            parts.append("未识别单元")
        return summary + ", ".join(parts)

    def _import_cas(self, file_path):
        """导入CAS文件 - 用于CFD网格"""
        try:
            mesh_data = self._build_unstructured_grid(
                node_coords=[],
                cells=[],
                dimension=2,
                file_path=file_path,
                mesh_type='cas',
            )

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
                    volume_cells = []
                    if hasattr(unstr_grid, 'cell_container'):
                        for cell in unstr_grid.cell_container:
                            if cell is not None:  # 确保cell不为None
                                if hasattr(cell, 'node_ids'):
                                    cell_ids = [nid if isinstance(nid, int) and nid >= 0 else nid
                                                for nid in cell.node_ids]
                                    if hasattr(unstr_grid, 'dimension') and unstr_grid.dimension == 3:
                                        cell_name = cell.__class__.__name__
                                        if cell_name in ('Triangle', 'Quadrilateral'):
                                            cells.append(cell_ids)
                                        elif cell_name in ('Tetrahedron', 'Pyramid', 'Prism', 'Hexahedron'):
                                            volume_cells.append(cell_ids)
                                    else:
                                        cells.append(cell_ids)
                                elif hasattr(cell, 'nodes'):
                                    # 如果cell有nodes属性
                                    cell_ids = []
                                    for node in cell.nodes:
                                        if hasattr(node, 'id'):
                                            cell_ids.append(node.id) 
                                    if cell_ids:
                                        cells.append(cell_ids)

                    mesh_data.node_coords = node_coords
                    if hasattr(unstr_grid, 'dimension') and unstr_grid.dimension == 3:
                        mesh_data.set_cells(cells, grid_dimension=2)
                        mesh_data.volume_cells = volume_cells
                    else:
                        mesh_data.set_cells(cells)
                    if hasattr(unstr_grid, 'dimension') and unstr_grid.dimension in (2, 3):
                        mesh_data.dimension = int(unstr_grid.dimension)
                    else:
                        mesh_data.dimension = 2

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
                    if hasattr(unstr_grid, 'dimension') and unstr_grid.dimension == 3:
                        if not cells and boundary_info:
                            surface_cells = []
                            for part_data in boundary_info.values():
                                for face in part_data.get('faces', []):
                                    nodes = face.get('nodes', [])
                                    if nodes:
                                        surface_cells.append(list(nodes))
                            if surface_cells:
                                mesh_data.set_cells(surface_cells, grid_dimension=2)
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
                        mesh_data.node_coords = vtk_mesh_data.get('node_coords', [])
                        mesh_data.set_cells(vtk_mesh_data.get('cells', []))
                        from utils.geom_toolkit import detect_mesh_dimension_by_metadata
                        mesh_data.dimension = detect_mesh_dimension_by_metadata(vtk_mesh_data, default_dim=mesh_data.dimension)

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
                        mesh_data = vtk_mesh_data
                        if not hasattr(mesh_data, 'node_coords'):
                            mesh_data.node_coords = []
                        if hasattr(vtk_mesh_data, 'dimension') and vtk_mesh_data.dimension in (2, 3):
                            mesh_data.dimension = int(vtk_mesh_data.dimension)
                        else:
                            mesh_data.dimension = 2

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
                mesh_data.dimension = 2
                mesh_data.update_counts()
                return mesh_data
            else:
                print("文件不包含FLUENT或CASE标识，可能不是有效的CAS文件")
                # 即使不是标准CAS文件，也返回一个基本结构而不是None
                mesh_data.dimension = 2
                mesh_data.update_counts()
                return mesh_data

        except Exception as e:
            print(f"导入CAS文件失败: {str(e)}")
            # 返回一个基本结构而不是None，以避免调用代码出现'NoneType'错误
            mesh_data = self._build_unstructured_grid(
                node_coords=[],
                cells=[],
                dimension=2,
                file_path=file_path,
                mesh_type='cas',
            )
            return mesh_data

    def _import_msh(self, file_path):
        """导入Gmsh MSH文件"""
        try:
            mesh_data = None
            
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
            
            mesh_data = self._build_unstructured_grid(
                node_coords=nodes,
                cells=elements,
                dimension=2,
                file_path=file_path,
                mesh_type='msh',
                cell_dimension=2,
            )
            
            return mesh_data
            
        except Exception as e:
            print(f"导入MSH文件失败: {str(e)}")
            return None

    def _import_ply(self, file_path):
        """导入PLY文件"""
        try:
            # PLY读取后构建统一网格对象
            
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
            
            mesh_data = self._build_unstructured_grid(
                node_coords=node_coords,
                cells=cell_data,
                dimension=2,
                file_path=file_path,
                mesh_type='ply',
                vtk_poly_data=poly_data,
                cell_dimension=2,
            )
            
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
            if hasattr(mesh_data, 'save_to_vtkfile') and callable(getattr(mesh_data, 'save_to_vtkfile')):
                mesh_data.save_to_vtkfile(file_path)
                return True

            # 检查mesh_data是否包含vtk_poly_data
            if isinstance(mesh_data, dict) and 'vtk_poly_data' in mesh_data:
                vtk_data = mesh_data['vtk_poly_data']
            elif hasattr(mesh_data, 'GetPoints') and hasattr(mesh_data, 'GetCells'):
                # mesh_data本身就是一个VTK对象
                vtk_data = mesh_data
            else:
                # 从字典数据创建VTK对象
                vtk_data = self._create_vtk_from_dict(mesh_data)
            
            if vtk_data:
                output_dir = os.path.dirname(file_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

                if isinstance(vtk_data, vtk.vtkUnstructuredGrid):
                    writer = vtk.vtkUnstructuredGridWriter()
                elif isinstance(vtk_data, vtk.vtkPolyData):
                    writer = vtk.vtkPolyDataWriter()
                else:
                    writer = vtk.vtkDataSetWriter()
                writer.SetFileName(file_path)
                writer.SetInputData(vtk_data)
                return bool(writer.Write())
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
            elif hasattr(mesh_data, 'cell_container'):
                # 从cell_container提取单元节点ID
                cells = [
                    cell.node_ids
                    for cell in mesh_data.cell_container
                    if cell is not None and hasattr(cell, 'node_ids')
                ]
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

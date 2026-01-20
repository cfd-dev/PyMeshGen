#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
异步导入线程模块
用于在后台线程中执行耗时的文件导入操作，避免GUI卡顿
"""

import os
from PyQt5.QtCore import QThread, pyqtSignal
from fileIO.read_cas import parse_fluent_msh, reconstruct_mesh_from_cas


class GeometryImportThread(QThread):
    """几何导入线程类"""

    progress_updated = pyqtSignal(str, int)  # 消息, 进度百分比
    import_finished = pyqtSignal(object)  # 导入的几何形状
    import_failed = pyqtSignal(str)  # 错误信息

    def __init__(self, file_path, create_vtk_actors=None):
        super().__init__()
        self.file_path = file_path
        self._is_running = True

        # 如果create_vtk_actors参数未指定，则根据文件扩展名决定
        if create_vtk_actors is None:
            file_ext = os.path.splitext(file_path)[1].lower()
            # 对于STL文件，创建VTK actors；对于其他文件，延迟到模型树加载后显示
            self.create_vtk_actors = (file_ext == '.stl')
        else:
            self.create_vtk_actors = create_vtk_actors

    def run(self):
        """执行几何导入操作"""
        try:
            from fileIO.geometry_io import import_geometry_file, get_shape_statistics

            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            self.progress_updated.emit(f"开始导入几何文件: {os.path.basename(self.file_path)}", 5)
            
            if not self._is_running:
                return

            self.progress_updated.emit(f"读取{file_ext.upper()}文件...", 20)

            if not self._is_running:
                return

            shape = import_geometry_file(self.file_path)

            if not self._is_running:
                return

            self.progress_updated.emit("提取几何统计信息...", 60)

            if not self._is_running:
                return

            stats = get_shape_statistics(shape)

            if not self._is_running:
                return

            result = {
                'shape': shape,
                'stats': stats,
                'file_path': self.file_path
            }

            if self.create_vtk_actors:
                self.progress_updated.emit("创建几何显示...", 75)

                if not self._is_running:
                    return

                try:
                    from fileIO.occ_to_vtk import create_shape_actor, create_geometry_edges_actor

                    main_actor = create_shape_actor(
                        shape,
                        mesh_quality=2.0,
                        display_mode='surface',
                        color=(0.8, 0.8, 0.9),
                        opacity=0.8
                    )

                    if not self._is_running:
                        return

                    self.progress_updated.emit("创建几何边缘显示...", 85)

                    edges_actor = create_geometry_edges_actor(
                        shape,
                        color=(0.0, 0.0, 0.0),
                        line_width=1.5,
                        sample_rate=0.1,
                        max_points_per_edge=50
                    )

                    result['main_actor'] = main_actor
                    result['edges_actor'] = edges_actor

                except Exception as e:
                    self.progress_updated.emit(f"创建显示失败: {str(e)}", 90)
                    result['main_actor'] = None
                    result['edges_actor'] = None

            self.progress_updated.emit("完成几何文件导入", 100)
            
            self.import_finished.emit(result)

        except Exception as e:
            self.import_failed.emit(f"几何导入失败: {str(e)}")

    def stop(self):
        """停止导入操作"""
        self._is_running = False
        self.wait()


class MeshImportThread(QThread):
    """网格导入线程类"""

    progress_updated = pyqtSignal(str, int)  # 消息, 进度百分比
    import_finished = pyqtSignal(object)  # 导入的网格数据
    import_failed = pyqtSignal(str)  # 错误信息

    def __init__(self, file_path, file_operations):
        super().__init__()
        self.file_path = file_path
        self.file_operations = file_operations
        self._is_running = True

    def run(self):
        """执行导入操作"""
        try:
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            self.progress_updated.emit(f"开始导入文件: {os.path.basename(self.file_path)}", 5)
            
            if file_ext == '.cas':
                self._import_cas()
            else:
                self.progress_updated.emit(f"使用通用导入方法: {file_ext}", 10)
                mesh_data = self.file_operations.import_mesh(self.file_path)
                if mesh_data:
                    self.progress_updated.emit("文件导入完成", 100)
                    self.import_finished.emit(mesh_data)
                else:
                    self.import_failed.emit("导入失败：无法读取文件")
                    
        except Exception as e:
            self.import_failed.emit(f"导入失败: {str(e)}")

    def _import_cas(self):
        """导入CAS文件"""
        try:
            self.progress_updated.emit("开始解析CAS文件结构...", 5)

            if not self._is_running:
                return

            # Parse the CAS file
            raw_cas_data = parse_fluent_msh(self.file_path)

            if not self._is_running:
                return

            self.progress_updated.emit("解析网格节点信息...", 20)

            if not self._is_running:
                return

            # Reconstruct the mesh from CAS data
            unstr_grid = reconstruct_mesh_from_cas(raw_cas_data)

            if not self._is_running:
                return

            self.progress_updated.emit("重建网格拓扑结构...", 40)

            if not self._is_running:
                return

            self.progress_updated.emit("处理网格单元...", 60)

            if not self._is_running:
                return

            from data_structure.mesh_data import MeshData
            mesh_data = MeshData(file_path=self.file_path, mesh_type='cas')

            if hasattr(unstr_grid, 'node_coords'):
                mesh_data.node_coords = [list(coord) for coord in unstr_grid.node_coords]
            elif hasattr(unstr_grid, 'nodes'):
                mesh_data.node_coords = [list(node.coords) for node in unstr_grid.nodes]

            if hasattr(unstr_grid, 'cell_container'):
                mesh_data.cells = []
                total_cells = len(unstr_grid.cell_container)
                for i, cell in enumerate(unstr_grid.cell_container):
                    if not self._is_running:
                        return
                    if cell is not None and hasattr(cell, 'node_ids'):
                        mesh_data.cells.append(cell.node_ids)

                    # Update progress periodically during cell processing
                    if i % max(1, total_cells // 10) == 0:  # Update every 10% of cells
                        progress = 60 + int(20 * i / total_cells)
                        self.progress_updated.emit(f"处理网格单元... ({i}/{total_cells})", progress)

            if hasattr(unstr_grid, 'dimension') and unstr_grid.dimension in (2, 3):
                mesh_data.dimension = int(unstr_grid.dimension)

            self.progress_updated.emit("提取边界信息...", 80)

            if not self._is_running:
                return

            if hasattr(unstr_grid, 'boundary_info'):
                mesh_data.boundary_info = unstr_grid.boundary_info
                parts_info = {}
                total_parts = len(unstr_grid.boundary_info)
                for i, (part_name, part_data) in enumerate(unstr_grid.boundary_info.items()):
                    if not self._is_running:
                        return
                    parts_info[part_name] = {
                        'bc_type': part_data.get('bc_type', 'unspecified'),
                        'faces': part_data.get('faces', []),
                        'face_count': len(part_data.get('faces', [])),
                        'part_name': part_name
                    }

                    # Update progress periodically during part processing
                    if total_parts > 0:
                        progress = 80 + int(15 * i / total_parts)
                        self.progress_updated.emit(f"提取部件信息... ({i+1}/{total_parts})", progress)

                mesh_data.parts_info = parts_info

            mesh_data.unstr_grid = unstr_grid
            mesh_data.update_counts()

            if not self._is_running:
                return

            self.progress_updated.emit("完成CAS文件导入", 100)
            self.import_finished.emit(mesh_data)

        except Exception as e:
            self.import_failed.emit(f"CAS文件导入失败: {str(e)}")

    def stop(self):
        """停止导入操作"""
        self._is_running = False
        self.wait()

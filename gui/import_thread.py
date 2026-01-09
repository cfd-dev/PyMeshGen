#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
异步导入线程模块
用于在后台线程中执行耗时的文件导入操作，避免GUI卡顿
"""

import os
from PyQt5.QtCore import QThread, pyqtSignal
from fileIO.read_cas import parse_fluent_msh, reconstruct_mesh_from_cas


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
            self.progress_updated.emit("解析CAS文件结构...", 10)
            
            if not self._is_running:
                return

            raw_cas_data = parse_fluent_msh(self.file_path)
            
            if not self._is_running:
                return

            self.progress_updated.emit("重建网格数据...", 60)
            
            if not self._is_running:
                return

            unstr_grid = reconstruct_mesh_from_cas(raw_cas_data)
            
            if not self._is_running:
                return

            self.progress_updated.emit("提取网格信息...", 80)
            
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
                for cell in unstr_grid.cell_container:
                    if cell is not None and hasattr(cell, 'node_ids'):
                        mesh_data.cells.append(cell.node_ids)
            
            if hasattr(unstr_grid, 'boundary_info'):
                mesh_data.boundary_info = unstr_grid.boundary_info
                parts_info = {}
                for part_name, part_data in unstr_grid.boundary_info.items():
                    parts_info[part_name] = {
                        'bc_type': part_data.get('bc_type', 'unspecified'),
                        'faces': part_data.get('faces', []),
                        'face_count': len(part_data.get('faces', [])),
                        'part_name': part_name
                    }
                mesh_data.parts_info = parts_info
            
            mesh_data.unstr_grid = unstr_grid
            mesh_data.update_counts()
            
            if not self._is_running:
                return

            self.progress_updated.emit("CAS文件导入完成", 100)
            self.import_finished.emit(mesh_data)
            
        except Exception as e:
            self.import_failed.emit(f"CAS文件导入失败: {str(e)}")

    def stop(self):
        """停止导入操作"""
        self._is_running = False
        self.wait()

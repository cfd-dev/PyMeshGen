import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class GeometryOperations:
    """几何操作器 - 负责几何文件的导入、导出和分析"""

    def __init__(self, gui):
        self.gui = gui

    def import_geometry(self):
        """导入几何文件（使用异步线程，避免GUI卡顿）"""
        from gui.import_thread import GeometryImportThread

        file_path, _ = QFileDialog.getOpenFileName(
            self.gui,
            "导入几何文件",
            os.path.join(self.gui.project_root, "geometries"),
            "几何文件 (*.step *.stp *.iges *.igs *.stl);;STEP文件 (*.step *.stp);;IGES文件 (*.iges *.igs);;STL文件 (*.stl);;所有文件 (*.*)"
        )

        if file_path:
            try:
                self.gui.geometry_import_thread = GeometryImportThread(file_path)

                self.gui.geometry_import_thread.progress_updated.connect(self.gui.on_geometry_import_progress)
                self.gui.geometry_import_thread.import_finished.connect(self.gui.on_geometry_import_finished)
                self.gui.geometry_import_thread.import_failed.connect(self.gui.on_geometry_import_failed)

                if hasattr(self.gui, 'ribbon') and hasattr(self.gui.ribbon, 'buttons'):
                    import_btn = self.gui.ribbon.buttons.get('file', {}).get('import_geometry')
                    if import_btn:
                        import_btn.setEnabled(False)

                self.gui.geometry_import_thread.start()
                self.gui.log_info(f"开始导入几何: {file_path}")
                self.gui.update_status("正在导入几何...")

            except Exception as e:
                QMessageBox.critical(self.gui, "错误", f"导入几何失败: {str(e)}")
                self.gui.log_error(f"导入几何失败: {str(e)}")

    def on_geometry_import_progress(self, message, progress):
        """几何导入进度更新回调"""
        self.gui.status_bar.show_progress(message, progress)
        self.gui.log_info(f"{message} ({progress}%)")

    def on_geometry_import_finished(self, result):
        """几何导入完成回调"""
        try:
            from fileIO.occ_to_vtk import create_shape_actor

            shape = result['shape']
            stats = result['stats']
            file_path = result['file_path']

            self.gui.log_info(f"几何统计信息:")
            self.gui.log_info(f"  - 顶点数: {stats['num_vertices']}")
            self.gui.log_info(f"  - 边数: {stats['num_edges']}")
            self.gui.log_info(f"  - 面数: {stats['num_faces']}")
            self.gui.log_info(f"  - 实体数: {stats['num_solids']}")

            bbox_min, bbox_max = stats['bounding_box']
            self.gui.log_info(f"  - 边界框: ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}) 到 ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f})")

            self.gui.current_geometry = shape
            self.gui.geometry_actors = {}
            self.gui.geometry_actor = None
            self.gui.geometry_edges_actor = None

            self.gui.update_status("正在创建几何显示...")

            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                from fileIO.occ_to_vtk import create_geometry_edges_actor

                self.gui.geometry_actor = create_shape_actor(
                    shape,
                    mesh_quality=2.0,
                    display_mode='surface',
                    color=(0.8, 0.8, 0.9),
                    opacity=0.8
                )
                self.gui.mesh_display.renderer.AddActor(self.gui.geometry_actor)
                self.gui.geometry_actors['main'] = [self.gui.geometry_actor]

                self.gui.geometry_edges_actor = create_geometry_edges_actor(
                    shape,
                    color=(0.0, 0.0, 0.0),
                    line_width=1.5,
                    sample_rate=0.1,
                    max_points_per_edge=50
                )
                self.gui.mesh_display.renderer.AddActor(self.gui.geometry_edges_actor)
                self.gui.geometry_actors['edges'] = [self.gui.geometry_edges_actor]

                if self.gui.render_mode == "wireframe":
                    self.gui.geometry_actor.SetVisibility(False)
                    self.gui.geometry_edges_actor.SetVisibility(True)
                elif self.gui.render_mode == "surface-wireframe":
                    self.gui.geometry_actor.SetVisibility(True)
                    self.gui.geometry_edges_actor.SetVisibility(True)
                else:
                    self.gui.geometry_actor.SetVisibility(True)
                    self.gui.geometry_edges_actor.SetVisibility(False)

                self.gui.mesh_display.renderer.ResetCamera()
                self.gui.mesh_display.render_window.Render()

            if hasattr(self.gui, 'model_tree_widget'):
                self.gui.model_tree_widget.load_geometry(shape, geometry_name="几何")

            self.gui.log_info(f"几何导入成功: {file_path}")
            self.gui.update_status("几何导入成功")

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"处理几何数据失败：{str(e)}")
            self.gui.log_info(f"处理几何数据失败：{str(e)}")
            self.gui.update_status("几何处理失败")

    def on_geometry_import_failed(self, error_message):
        """几何导入失败回调"""
        QMessageBox.critical(self.gui, "错误", f"导入几何失败：{error_message}")
        self.gui.log_error(f"导入几何失败：{error_message}")
        self.gui.update_status("几何导入失败")

        if hasattr(self.gui, 'ribbon') and hasattr(self.gui.ribbon, 'buttons'):
            import_btn = self.gui.ribbon.buttons.get('file', {}).get('import_geometry')
            if import_btn:
                import_btn.setEnabled(True)

    def export_geometry(self):
        """导出几何文件"""
        if not hasattr(self.gui, 'current_geometry') or self.gui.current_geometry is None:
            QMessageBox.warning(self.gui, "警告", "请先导入几何文件")
            self.gui.log_info("未导入几何文件，无法导出")
            self.gui.update_status("未导入几何文件")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.gui,
            "导出几何文件",
            os.path.join(self.gui.project_root, "geometries"),
            "几何文件 (*.step *.stp *.iges *.igs *.stl);;STEP文件 (*.step *.stp);;IGES文件 (*.iges *.igs);;STL文件 (*.stl);;所有文件 (*.*)"
        )

        if file_path:
            try:
                from fileIO.geometry_io import export_geometry_file

                self.gui.log_info(f"开始导出几何: {file_path}")
                self.gui.update_status("正在导出几何...")

                success = export_geometry_file(self.gui.current_geometry, file_path)

                if success:
                    self.gui.log_info(f"已导出几何: {file_path}")
                    self.gui.update_status("已导出几何")
                else:
                    QMessageBox.warning(self.gui, "警告", "导出几何失败")
                    self.gui.log_info("导出几何失败")
                    self.gui.update_status("导出几何失败")

            except Exception as e:
                QMessageBox.critical(self.gui, "错误", f"导出几何失败: {str(e)}")
                self.gui.log_error(f"导出几何失败: {str(e)}")
                self.gui.update_status("导出几何失败")

    def export_mesh(self):
        """导出网格"""
        from gui.file_operations import FileOperations

        if not self.gui.current_mesh:
            QMessageBox.warning(self.gui, "警告", "没有可导出的网格")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.gui,
            "导出网格文件",
            os.path.join(self.gui.project_root, "meshes", "mesh.vtk"),
            "网格文件 (*.vtk *.stl *.obj *.msh *.ply)"
        )

        if file_path:
            try:
                file_ops = FileOperations(self.gui.project_root)
                vtk_poly_data = None
                if isinstance(self.gui.current_mesh, dict):
                    vtk_poly_data = self.gui.current_mesh.get('vtk_poly_data')

                if vtk_poly_data:
                    file_ops.export_mesh(vtk_poly_data, file_path)
                else:
                    if hasattr(self.gui.current_mesh, 'node_coords') and hasattr(self.gui.current_mesh, 'cell_container'):
                        if hasattr(self.gui.current_mesh, 'save_to_vtkfile'):
                            self.gui.current_mesh.save_to_vtkfile(file_path)
                        else:
                            QMessageBox.warning(self.gui, "警告", "当前网格格式不支持直接保存，请使用VTK格式")
                            return
                    else:
                        QMessageBox.warning(self.gui, "警告", "无法获取有效的VTK数据进行导出")
                        return

                self.gui.log_info(f"已导出网格文件: {file_path}")
                self.gui.update_status("已导出网格文件")
            except Exception as e:
                QMessageBox.critical(self.gui, "错误", f"导出网格失败: {str(e)}")
                self.gui.log_error(f"导出网格失败: {str(e)}")

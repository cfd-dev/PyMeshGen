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

                if hasattr(self.gui, '_reset_progress_cache'):
                    self.gui._reset_progress_cache("geometry")
                if hasattr(self.gui, '_set_ribbon_button_enabled'):
                    self.gui._set_ribbon_button_enabled('file', 'import_geometry', False)

                self.gui.geometry_import_thread.start()
                self.gui.log_info(f"开始导入几何: {file_path}")
                self.gui.update_status("正在导入几何...")

            except Exception as e:
                QMessageBox.critical(self.gui, "错误", f"导入几何失败: {str(e)}")
                self.gui.log_error(f"导入几何失败: {str(e)}")

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

    def create_geometry_from_points(self, points, mode="point"):
        """创建点/线/曲线几何"""
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.GC import GC_MakeSegment
            from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
            from OCC.Core.TColgp import TColgp_Array1OfPnt
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return

        if not points:
            QMessageBox.warning(self.gui, "警告", "未提供点数据")
            return

        shape = None
        if mode == "point":
            p = gp_Pnt(*points[0])
            shape = BRepBuilderAPI_MakeVertex(p).Vertex()
        elif mode == "line":
            if len(points) < 2:
                QMessageBox.warning(self.gui, "警告", "直线需要两个点")
                return
            p1 = gp_Pnt(*points[0])
            p2 = gp_Pnt(*points[1])
            shape = BRepBuilderAPI_MakeEdge(GC_MakeSegment(p1, p2).Value()).Edge()
        elif mode == "curve":
            if len(points) < 2:
                QMessageBox.warning(self.gui, "警告", "曲线至少需要两个点")
                return
            arr = TColgp_Array1OfPnt(1, len(points))
            for i, pt in enumerate(points, start=1):
                arr.SetValue(i, gp_Pnt(*pt))
            spline = GeomAPI_PointsToBSpline(arr).Curve()
            shape = BRepBuilderAPI_MakeEdge(spline).Edge()
        else:
            return

        self._merge_geometry(shape, mode_label=mode)

    def create_geometry_circle(self, center, radius, start_angle=0.0, end_angle=360.0):
        """创建圆/圆弧几何"""
        try:
            from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Circ
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
            from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeCircle
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return

        center_pnt = gp_Pnt(*center)
        axis = gp_Ax2(center_pnt, gp_Dir(0, 0, 1))
        if abs(end_angle - start_angle) >= 360.0:
            circle = gp_Circ(axis, radius)
            shape = BRepBuilderAPI_MakeEdge(GC_MakeCircle(circle).Value()).Edge()
        else:
            start_rad = start_angle * 3.141592653589793 / 180.0
            end_rad = end_angle * 3.141592653589793 / 180.0
            arc = GC_MakeArcOfCircle(axis, radius, start_rad, end_rad, True).Value()
            shape = BRepBuilderAPI_MakeEdge(arc).Edge()

        self._merge_geometry(shape, mode_label="circle")

    def _merge_geometry(self, new_shape, mode_label=""):
        if new_shape is None:
            return
        try:
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.TopoDS import TopoDS_Compound
            from fileIO.geometry_io import get_shape_statistics
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何合并失败: {str(e)}")
            return

        if not hasattr(self.gui, 'current_geometry') or self.gui.current_geometry is None:
            compound = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(compound)
            builder.Add(compound, new_shape)
            self.gui.current_geometry = compound
        else:
            compound = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(compound)
            builder.Add(compound, self.gui.current_geometry)
            builder.Add(compound, new_shape)
            self.gui.current_geometry = compound

        stats = get_shape_statistics(self.gui.current_geometry)
        self.gui.current_geometry_stats = stats
        self._refresh_geometry_display(stats)

        if hasattr(self.gui, 'model_tree_widget'):
            self.gui.model_tree_widget.load_geometry(self.gui.current_geometry, "几何")

        if hasattr(self.gui, '_create_default_part_for_geometry'):
            self.gui._create_default_part_for_geometry(self.gui.current_geometry, stats)

        self.gui.log_info(f"已创建几何: {mode_label}")
        self.gui.update_status("几何已更新")

    def _refresh_geometry_display(self, stats):
        try:
            from fileIO.occ_to_vtk import create_shape_actor, create_geometry_edges_actor
        except Exception:
            return

        if not hasattr(self.gui, 'mesh_display') or not hasattr(self.gui.mesh_display, 'renderer'):
            return

        if hasattr(self.gui, 'part_manager') and hasattr(self.gui.part_manager, 'cleanup_geometry_actors'):
            self.gui.part_manager.cleanup_geometry_actors()

        self.gui.geometry_actors = {}
        self.gui.geometry_actor = None
        self.gui.geometry_edges_actor = None
        self.gui.geometry_points_actor = None

        has_surface = (stats.get('num_faces', 0) + stats.get('num_solids', 0)) > 0
        has_edges = stats.get('num_edges', 0) > 0
        has_vertices = stats.get('num_vertices', 0) > 0

        if has_surface:
            self.gui.geometry_actor = create_shape_actor(
                self.gui.current_geometry,
                mesh_quality=8.0,
                display_mode='surface',
                color=(0.8, 0.8, 0.9),
                opacity=0.8
            )
            self.gui.mesh_display.renderer.AddActor(self.gui.geometry_actor)
            self.gui.geometry_actors['main'] = [self.gui.geometry_actor]

        if has_edges:
            self.gui.geometry_edges_actor = create_geometry_edges_actor(
                self.gui.current_geometry,
                color=(0.0, 0.0, 0.0),
                line_width=1.5,
                sample_rate=0.5,
                max_points_per_edge=20
            )
            self.gui.mesh_display.renderer.AddActor(self.gui.geometry_edges_actor)
            self.gui.geometry_actors['edges'] = [self.gui.geometry_edges_actor]

        if has_vertices:
            self.gui.geometry_points_actor = create_shape_actor(
                self.gui.current_geometry,
                mesh_quality=8.0,
                display_mode='points',
                color=(1.0, 0.0, 0.0),
                opacity=1.0
            )
            self.gui.mesh_display.renderer.AddActor(self.gui.geometry_points_actor)
            self.gui.geometry_actors['points'] = [self.gui.geometry_points_actor]

        if hasattr(self.gui, 'view_controller'):
            self.gui.view_controller._apply_render_mode_to_geometry(self.gui.render_mode)

        if hasattr(self.gui.mesh_display, 'render_window'):
            self.gui.mesh_display.render_window.Render()


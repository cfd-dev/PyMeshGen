import os
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QComboBox,
    QPushButton,
)


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
                unit_dialog = _GeometryUnitDialog(self.gui)
                if unit_dialog.exec_() != QDialog.Accepted:
                    self.gui.log_info("导入几何已取消（未设置单位）")
                    self.gui.update_status("导入几何已取消")
                    return
                unit = unit_dialog.get_unit()

                self.gui.geometry_import_thread = GeometryImportThread(file_path, unit=unit)

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
            return False

        if not points:
            QMessageBox.warning(self.gui, "警告", "未提供点数据")
            return False

        shape = None
        if mode == "point":
            p = gp_Pnt(*points[0])
            shape = BRepBuilderAPI_MakeVertex(p).Vertex()
        elif mode == "line":
            if len(points) < 2:
                QMessageBox.warning(self.gui, "警告", "直线需要两个点")
                return False
            p1 = gp_Pnt(*points[0])
            p2 = gp_Pnt(*points[1])
            shape = BRepBuilderAPI_MakeEdge(GC_MakeSegment(p1, p2).Value()).Edge()
        elif mode == "curve":
            if len(points) < 2:
                QMessageBox.warning(self.gui, "警告", "曲线至少需要两个点")
                return False
            arr = TColgp_Array1OfPnt(1, len(points))
            for i, pt in enumerate(points, start=1):
                arr.SetValue(i, gp_Pnt(*pt))
            spline = GeomAPI_PointsToBSpline(arr).Curve()
            shape = BRepBuilderAPI_MakeEdge(spline).Edge()
        else:
            return False

        self._merge_geometry(shape, mode_label=mode)
        return True

    def create_geometry_circle(self, center, radius, start_angle=0.0, end_angle=360.0):
        """创建圆/圆弧几何"""
        try:
            from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Circ
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
            from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeCircle
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return False

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
        return True

    def create_geometry_polyline(self, points):
        """创建多段线/折线"""
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
            from OCC.Core.gp import gp_Pnt
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return False
        if len(points) < 2:
            QMessageBox.warning(self.gui, "警告", "多段线至少需要两个点")
            return False
        wire_maker = BRepBuilderAPI_MakeWire()
        for idx in range(len(points) - 1):
            p1 = gp_Pnt(*points[idx])
            p2 = gp_Pnt(*points[idx + 1])
            wire_maker.Add(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        if not wire_maker.IsDone():
            QMessageBox.warning(self.gui, "警告", "多段线创建失败")
            return False
        self._merge_geometry(wire_maker.Wire(), mode_label="polyline")
        return True

    def create_geometry_polygon(self, points):
        """创建多边形（闭合线框）"""
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
            from OCC.Core.gp import gp_Pnt
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return False
        if len(points) < 3:
            QMessageBox.warning(self.gui, "警告", "多边形至少需要三个点")
            return False
        wire_maker = BRepBuilderAPI_MakeWire()
        for idx in range(len(points) - 1):
            p1 = gp_Pnt(*points[idx])
            p2 = gp_Pnt(*points[idx + 1])
            wire_maker.Add(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        if points[0] != points[-1]:
            wire_maker.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(*points[-1]), gp_Pnt(*points[0])).Edge())
        if not wire_maker.IsDone():
            QMessageBox.warning(self.gui, "警告", "多边形创建失败")
            return False
        self._merge_geometry(wire_maker.Wire(), mode_label="polygon")
        return True

    def create_geometry_rectangle(self, p1, p2):
        """创建矩形（基于两对角点，假定XY平面）"""
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
            from OCC.Core.gp import gp_Pnt
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return False
        if len(p1) < 3 or len(p2) < 3:
            QMessageBox.warning(self.gui, "警告", "矩形需要两个点")
            return False
        if abs(p1[2] - p2[2]) > 1e-6:
            QMessageBox.warning(self.gui, "警告", "矩形点Z坐标不一致，已使用第一个点的Z")
        z = p1[2]
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        corners = [
            (x1, y1, z),
            (x2, y1, z),
            (x2, y2, z),
            (x1, y2, z),
        ]
        wire_maker = BRepBuilderAPI_MakeWire()
        for idx in range(len(corners)):
            p_start = gp_Pnt(*corners[idx])
            p_end = gp_Pnt(*corners[(idx + 1) % len(corners)])
            wire_maker.Add(BRepBuilderAPI_MakeEdge(p_start, p_end).Edge())
        if not wire_maker.IsDone():
            QMessageBox.warning(self.gui, "警告", "矩形创建失败")
            return False
        self._merge_geometry(wire_maker.Wire(), mode_label="rectangle")
        return True

    def create_geometry_ellipse(self, center, major_radius, minor_radius, start_angle=0.0, end_angle=360.0):
        """创建椭圆/椭圆弧（XY平面）"""
        try:
            from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Elips
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
            from OCC.Core.GC import GC_MakeEllipse, GC_MakeArcOfEllipse
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return False
        if major_radius <= 0 or minor_radius <= 0:
            QMessageBox.warning(self.gui, "警告", "椭圆半径必须为正数")
            return False
        center_pnt = gp_Pnt(*center)
        axis = gp_Ax2(center_pnt, gp_Dir(0, 0, 1))
        ellipse = gp_Elips(axis, major_radius, minor_radius)
        if abs(end_angle - start_angle) >= 360.0:
            shape = BRepBuilderAPI_MakeEdge(GC_MakeEllipse(ellipse).Value()).Edge()
        else:
            start_rad = start_angle * 3.141592653589793 / 180.0
            end_rad = end_angle * 3.141592653589793 / 180.0
            arc = GC_MakeArcOfEllipse(ellipse, start_rad, end_rad, True).Value()
            shape = BRepBuilderAPI_MakeEdge(arc).Edge()
        self._merge_geometry(shape, mode_label="ellipse")
        return True

    def create_geometry_box(self, corner1, corner2):
        """创建长方体（基于两对角点）"""
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
            from OCC.Core.gp import gp_Pnt
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return False
        p1 = gp_Pnt(*corner1)
        p2 = gp_Pnt(*corner2)
        shape = BRepPrimAPI_MakeBox(p1, p2).Shape()
        self._merge_geometry(shape, mode_label="box")
        return True

    def create_geometry_sphere(self, center, radius):
        """创建圆球"""
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
            from OCC.Core.gp import gp_Pnt
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return False
        if radius <= 0:
            QMessageBox.warning(self.gui, "警告", "半径必须为正数")
            return False
        shape = BRepPrimAPI_MakeSphere(gp_Pnt(*center), radius).Shape()
        self._merge_geometry(shape, mode_label="sphere")
        return True

    def create_geometry_cylinder(self, base_center, radius, height):
        """创建圆柱（沿Z轴）"""
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
            from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"几何创建失败: {str(e)}")
            return False
        if radius <= 0 or height <= 0:
            QMessageBox.warning(self.gui, "警告", "半径和高度必须为正数")
            return False
        axis = gp_Ax2(gp_Pnt(*base_center), gp_Dir(0, 0, 1))
        shape = BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()
        self._merge_geometry(shape, mode_label="cylinder")
        return True

    def delete_geometry(self, element_map):
        """删除选中的几何元素"""
        if not element_map:
            return False
        if not hasattr(self.gui, 'current_geometry') or self.gui.current_geometry is None:
            QMessageBox.warning(self.gui, "警告", "请先导入几何文件")
            self.gui.log_info("未导入几何文件，无法删除几何元素")
            self.gui.update_status("未导入几何文件")
            return False
        if getattr(self.gui, 'geometry_display_source', None) == 'stl':
            QMessageBox.warning(self.gui, "警告", "STL几何不支持元素级删除")
            self.gui.log_info("STL几何不支持元素级删除")
            self.gui.update_status("STL几何不支持元素级删除")
            return False

        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
            from OCC.Core.BRepTools import BRepTools_ReShape
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"删除几何失败: {str(e)}")
            self.gui.log_error(f"删除几何失败: {str(e)}")
            return False

        element_types = [
            ("vertices", TopAbs_VERTEX),
            ("edges", TopAbs_EDGE),
            ("faces", TopAbs_FACE),
            ("bodies", TopAbs_SOLID),
        ]

        remove_map = {}
        for key, occ_type in element_types:
            selected = element_map.get(key, []) or []
            if not selected:
                continue
            explorer = TopExp_Explorer(self.gui.current_geometry, occ_type)
            while explorer.More():
                shape = explorer.Current()
                for candidate in selected:
                    try:
                        if hasattr(shape, "IsEqual") and shape.IsEqual(candidate):
                            remove_map[shape] = True
                            break
                        if hasattr(shape, "IsSame") and shape.IsSame(candidate):
                            remove_map[shape] = True
                            break
                    except Exception:
                        continue
                explorer.Next()

        if not remove_map:
            QMessageBox.warning(self.gui, "警告", "未找到可删除的几何元素")
            return False

        reshaper = BRepTools_ReShape()
        for shape in remove_map.keys():
            reshaper.Remove(shape)

        try:
            new_shape = reshaper.Apply(self.gui.current_geometry, True)
        except Exception:
            new_shape = reshaper.Apply(self.gui.current_geometry)

        self.gui.log_info(f"删除后 new_shape 类型: {type(new_shape)}, IsNull: {new_shape.IsNull() if hasattr(new_shape, 'IsNull') else 'N/A'}")

        try:
            from fileIO.geometry_io import get_shape_statistics
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"删除几何失败: {str(e)}")
            self.gui.log_error(f"删除几何失败: {str(e)}")
            return False

        stats = get_shape_statistics(new_shape)
        previous_shape = self.gui.current_geometry
        self.gui.current_geometry = new_shape
        self.gui.current_geometry_stats = stats

        if hasattr(self.gui, 'part_manager') and hasattr(self.gui.part_manager, 'cleanup_geometry_actors'):
            self.gui.part_manager.cleanup_geometry_actors()

        if stats.get('num_vertices', 0) == 0 and stats.get('num_edges', 0) == 0 and stats.get('num_faces', 0) == 0 and stats.get('num_solids', 0) == 0:
            if hasattr(self.gui, 'part_manager') and hasattr(self.gui.part_manager, 'cleanup_geometry_actors'):
                self.gui.part_manager.cleanup_geometry_actors()
            if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor and hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                try:
                    self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_actor)
                except Exception:
                    pass
                self.gui.geometry_actor = None
            if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor and hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                try:
                    self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_edges_actor)
                except Exception:
                    pass
                self.gui.geometry_edges_actor = None
            if hasattr(self.gui, 'geometry_points_actor') and self.gui.geometry_points_actor and hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                try:
                    self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_points_actor)
                except Exception:
                    pass
                self.gui.geometry_points_actor = None
            if hasattr(self.gui, 'geometry_actors'):
                self.gui.geometry_actors = {}
            if hasattr(self.gui, 'geometry_actors_cache'):
                self.gui.geometry_actors_cache = {}
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
                self.gui.mesh_display.render_window.Render()
        else:
            self._refresh_geometry_display(stats)

        if hasattr(self.gui, 'model_tree_widget'):
            self.gui.model_tree_widget.load_geometry(new_shape, "几何")
            if hasattr(self.gui.model_tree_widget, 'set_pending_label_restore'):
                self.gui.model_tree_widget.set_pending_label_restore(previous_shape, new_shape)

        if hasattr(self.gui, '_rebuild_parts_for_geometry'):
            self.gui._rebuild_parts_for_geometry(previous_shape, new_shape, remove_map)

        if hasattr(self.gui, 'model_tree_widget') and stats.get('num_vertices', 0) == 0 and stats.get('num_edges', 0) == 0 and stats.get('num_faces', 0) == 0 and stats.get('num_solids', 0) == 0:
            self.gui.model_tree_widget.load_geometry(None, "几何")
            if hasattr(self.gui, 'cas_parts_info') and self.gui.cas_parts_info and 'DefaultPart' in self.gui.cas_parts_info:
                del self.gui.cas_parts_info['DefaultPart']
                if hasattr(self.gui, 'model_tree_widget'):
                    self.gui.model_tree_widget.load_parts({'parts_info': self.gui.cas_parts_info})
                self.gui.log_info("已清除 DefaultPart")

        self.gui.log_info("已删除选中几何元素")
        self.gui.update_status("几何已更新")
        return True

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

        if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
            try:
                self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_actor)
            except Exception:
                pass
        if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
            try:
                self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_edges_actor)
            except Exception:
                pass
        if hasattr(self.gui, 'geometry_points_actor') and self.gui.geometry_points_actor:
            try:
                self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_points_actor)
            except Exception:
                pass

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
                sample_rate=0.0025,
                max_points_per_edge=2000
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

        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
            self.gui.mesh_display.render_window.Render()


class _GeometryUnitDialog(QDialog):
    """几何单位设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("几何单位设置")
        self.setMinimumWidth(360)
        self._unit_options = ["mm", "cm", "m", "inch", "ft", "um", "km", "mi"]
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        source_group = QGroupBox("单位来源")
        source_layout = QVBoxLayout(source_group)
        self.auto_radio = QRadioButton("从数模文件中获取单位")
        self.convert_radio = QRadioButton("将单位转换为")
        self.auto_radio.setChecked(True)
        source_layout.addWidget(self.auto_radio)
        source_layout.addWidget(self.convert_radio)
        main_layout.addWidget(source_group)

        convert_group = QGroupBox("转换单位")
        convert_layout = QHBoxLayout(convert_group)
        convert_layout.addWidget(QLabel("目标单位:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(self._unit_options)
        self.unit_combo.setEnabled(False)
        convert_layout.addWidget(self.unit_combo)
        convert_layout.addStretch()
        main_layout.addWidget(convert_group)

        self.convert_radio.toggled.connect(self.unit_combo.setEnabled)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("确定")
        cancel_button = QPushButton("取消")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

    def get_unit(self):
        if self.convert_radio.isChecked():
            return self.unit_combo.currentText()
        return "auto"



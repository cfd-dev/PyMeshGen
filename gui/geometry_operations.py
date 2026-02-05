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
                # 导入前重置为整体显示模式
                self.gui.display_mode = "full"
                if hasattr(self.gui, 'view_controller') and hasattr(self.gui.view_controller, 'set_display_mode'):
                    self.gui.view_controller.set_display_mode("full")

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
        circle = gp_Circ(axis, radius)
        if abs(end_angle - start_angle) >= 360.0:
            shape = BRepBuilderAPI_MakeEdge(GC_MakeCircle(circle).Value()).Edge()
        else:
            start_rad = start_angle * 3.141592653589793 / 180.0
            end_rad = end_angle * 3.141592653589793 / 180.0
            arc = GC_MakeArcOfCircle(circle, start_rad, end_rad, True).Value()
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
        if corner1 is None or corner2 is None:
            QMessageBox.warning(self.gui, "警告", "长方体需要两个不同的角点")
            return False
        dx = abs(corner1[0] - corner2[0])
        dy = abs(corner1[1] - corner2[1])
        dz = abs(corner1[2] - corner2[2])
        if dx <= 0 or dy <= 0 or dz <= 0:
            QMessageBox.warning(self.gui, "警告", "长方体三方向长度必须为正数")
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
            parts_info = self.gui._get_parts_info() if hasattr(self.gui, '_get_parts_info') else None
            if parts_info and 'DefaultPart' in parts_info:
                del parts_info['DefaultPart']
                if hasattr(self.gui, 'model_tree_widget'):
                    self.gui.model_tree_widget.load_parts({'parts_info': parts_info})
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


class LineMeshGenerationDialog(QDialog):
    """线网格生成参数设置对话框"""
    
    UNIFORM = "uniform"
    GEOMETRIC = "geometric"
    TANH = "tanh"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("线网格生成参数设置")
        self.setMinimumWidth(450)
        self.selected_edges = []
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QRadioButton, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit, QGridLayout
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 几何线选择组
        line_group = QGroupBox("几何线选择")
        line_layout = QVBoxLayout()
        line_layout.setSpacing(5)
        
        info_layout = QHBoxLayout()
        self.lbl_selected_lines = QLabel("未选择几何线")
        self.lbl_selected_lines.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self.lbl_selected_lines)
        line_layout.addLayout(info_layout)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.btn_pick_lines = QPushButton("拾取几何线")
        self.btn_pick_lines.setCheckable(True)
        self.btn_pick_lines.clicked.connect(self.on_pick_lines_clicked)
        button_layout.addWidget(self.btn_pick_lines)
        
        self.btn_clear_lines = QPushButton("清除选择")
        self.btn_clear_lines.clicked.connect(self.on_clear_lines_clicked)
        button_layout.addWidget(self.btn_clear_lines)
        
        button_layout.addStretch()
        line_layout.addLayout(button_layout)
        
        line_group.setLayout(line_layout)
        layout.addWidget(line_group)
        
        # 离散化方法组
        method_group = QGroupBox("离散化方法")
        method_layout = QVBoxLayout()
        method_layout.setSpacing(8)
        
        self.radio_uniform = QRadioButton("均匀分布")
        self.radio_uniform.setChecked(True)
        self.radio_uniform.toggled.connect(self.on_method_changed)
        method_layout.addWidget(self.radio_uniform)
        
        self.radio_geometric = QRadioButton("几何级数分布")
        self.radio_geometric.toggled.connect(self.on_method_changed)
        method_layout.addWidget(self.radio_geometric)
        
        self.radio_tanh = QRadioButton("Tanh函数分布")
        self.radio_tanh.toggled.connect(self.on_method_changed)
        method_layout.addWidget(self.radio_tanh)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # 离散化参数组
        params_group = QGroupBox("离散化参数")
        params_layout = QGridLayout()
        params_layout.setSpacing(10)
        
        params_layout.addWidget(QLabel("网格数量:"), 0, 0)
        self.spin_num_elements = QSpinBox()
        self.spin_num_elements.setRange(2, 1000)
        self.spin_num_elements.setValue(10)
        self.spin_num_elements.valueChanged.connect(self.on_params_changed)
        params_layout.addWidget(self.spin_num_elements, 0, 1)
        
        params_layout.addWidget(QLabel("起始尺寸:"), 1, 0)
        self.spin_start_size = QDoubleSpinBox()
        self.spin_start_size.setRange(1e-6, 1e6)
        self.spin_start_size.setValue(0.1)
        self.spin_start_size.setDecimals(6)
        self.spin_start_size.setSuffix(" (可选)")
        params_layout.addWidget(self.spin_start_size, 1, 1)
        
        params_layout.addWidget(QLabel("结束尺寸:"), 2, 0)
        self.spin_end_size = QDoubleSpinBox()
        self.spin_end_size.setRange(1e-6, 1e6)
        self.spin_end_size.setValue(0.2)
        self.spin_end_size.setDecimals(6)
        self.spin_end_size.setSuffix(" (可选)")
        params_layout.addWidget(self.spin_end_size, 2, 1)
        
        params_layout.addWidget(QLabel("增长比率:"), 3, 0)
        self.spin_growth_rate = QDoubleSpinBox()
        self.spin_growth_rate.setRange(0.1, 10.0)
        self.spin_growth_rate.setValue(1.2)
        self.spin_growth_rate.setDecimals(2)
        self.spin_growth_rate.setSingleStep(0.1)
        params_layout.addWidget(self.spin_growth_rate, 3, 1)
        
        params_layout.addWidget(QLabel("Tanh拉伸系数:"), 4, 0)
        self.spin_tanh_factor = QDoubleSpinBox()
        self.spin_tanh_factor.setRange(0.1, 10.0)
        self.spin_tanh_factor.setValue(2.0)
        self.spin_tanh_factor.setDecimals(2)
        self.spin_tanh_factor.setSingleStep(0.1)
        params_layout.addWidget(self.spin_tanh_factor, 4, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        self.update_params_group_state()
        
        # 边界条件组
        bc_group = QGroupBox("边界条件")
        bc_layout = QGridLayout()
        bc_layout.setSpacing(10)
        
        bc_layout.addWidget(QLabel("边界类型:"), 0, 0)
        self.combo_bc_type = QComboBox()
        from data_structure.cgns_types import CGNSBCTypeName
        self.combo_bc_type.addItems(list(CGNSBCTypeName.TYPE_NAMES.values()))
        self.combo_bc_type.setCurrentText("BCWall")
        bc_layout.addWidget(self.combo_bc_type, 0, 1)
        
        bc_layout.addWidget(QLabel("部件名称:"), 1, 0)
        self.edit_part_name = QLineEdit("default_line")
        bc_layout.addWidget(self.edit_part_name, 1, 1)
        
        bc_group.setLayout(bc_layout)
        layout.addWidget(bc_group)
        
        # 按钮组
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        layout.addStretch()
        
        self.btn_preview = QPushButton("预览")
        self.btn_preview.clicked.connect(self.on_preview_clicked)
        button_layout.addWidget(self.btn_preview)
        
        self.btn_generate = QPushButton("生成线网格")
        self.btn_generate.clicked.connect(self.on_generate_clicked)
        button_layout.addWidget(self.btn_generate)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(button_layout)
        
    def get_discretization_method(self):
        """获取当前选择的离散化方法"""
        if self.radio_uniform.isChecked():
            return self.UNIFORM
        elif self.radio_geometric.isChecked():
            return self.GEOMETRIC
        elif self.radio_tanh.isChecked():
            return self.TANH
        return self.UNIFORM
        
    def get_parameters(self):
        """获取所有参数"""
        return {
            'method': self.get_discretization_method(),
            'num_elements': self.spin_num_elements.value(),
            'start_size': self.spin_start_size.value(),
            'end_size': self.spin_end_size.value(),
            'growth_rate': self.spin_growth_rate.value(),
            'tanh_factor': self.spin_tanh_factor.value(),
            'bc_type': self.combo_bc_type.currentText(),
            'part_name': self.edit_part_name.text()
        }
        
    def set_selected_lines_count(self, count):
        """设置选中的几何线数量"""
        if count > 0:
            self.lbl_selected_lines.setText(f"已选择 {count} 条几何线")
            self.lbl_selected_lines.setStyleSheet("color: #006400; font-weight: bold;")
        else:
            self.lbl_selected_lines.setText("未选择几何线")
            self.lbl_selected_lines.setStyleSheet("color: #666; font-style: italic;")
            
    def on_method_changed(self):
        """离散化方法改变时的处理"""
        self.update_params_group_state()
        
    def update_params_group_state(self):
        """更新参数组的状态"""
        method = self.get_discretization_method()
        
        if method == self.UNIFORM:
            self.spin_start_size.setEnabled(False)
            self.spin_end_size.setEnabled(False)
            self.spin_growth_rate.setEnabled(False)
            self.spin_tanh_factor.setEnabled(False)
        elif method == self.GEOMETRIC:
            self.spin_start_size.setEnabled(True)
            self.spin_end_size.setEnabled(True)
            self.spin_growth_rate.setEnabled(True)
            self.spin_tanh_factor.setEnabled(False)
        elif method == self.TANH:
            self.spin_start_size.setEnabled(True)
            self.spin_end_size.setEnabled(True)
            self.spin_growth_rate.setEnabled(False)
            self.spin_tanh_factor.setEnabled(True)
            
    def on_params_changed(self):
        """参数改变时的处理"""
        pass
        
    def on_pick_lines_clicked(self):
        """点击拾取几何线按钮"""
        if self.btn_pick_lines.isChecked():
            self.btn_pick_lines.setText("取消拾取")
        else:
            self.btn_pick_lines.setText("拾取几何线")
            
    def on_clear_lines_clicked(self):
        """清除选择的线"""
        self.set_selected_lines_count(0)
        self.btn_pick_lines.setChecked(False)
        self.btn_pick_lines.setText("拾取几何线")
        
    def on_preview_clicked(self):
        """预览按钮点击"""
        params = self.get_parameters()
        self.accept()
        
    def on_generate_clicked(self):
        """生成按钮点击"""
        params = self.get_parameters()
        self.accept()



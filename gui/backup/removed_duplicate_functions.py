#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backup of removed duplicate GUI functions.
These functions were removed during refactoring to avoid duplication.
"""

from PyQt5.QtWidgets import QMessageBox


class PartManagerBackup:
    """Backup: duplicate DefaultPart creation helpers."""

    def __init__(self, gui):
        self.gui = gui

    def create_default_part_for_mesh(self, mesh_data):
        """为导入的网格创建Default部件"""
        try:
            if hasattr(self.gui, '_create_default_part_for_mesh'):
                self.gui._create_default_part_for_mesh(mesh_data)
                return

            if not hasattr(self.gui, 'cas_parts_info') or self.gui.cas_parts_info is None:
                self.gui.cas_parts_info = {}

            mesh_elements = {
                "vertices": [],
                "edges": [],
                "faces": [],
                "bodies": []
            }

            if hasattr(mesh_data, 'node_coords'):
                num_vertices = len(mesh_data.node_coords)
                mesh_elements["vertices"] = list(range(num_vertices))

            if hasattr(mesh_data, 'cells'):
                num_faces = len(mesh_data.cells)
                mesh_elements["faces"] = list(range(num_faces))

            part_name = "DefaultPart"
            part_info = {
                'part_name': part_name,
                'bc_type': '',
                'mesh_elements': mesh_elements,
                'num_vertices': len(mesh_elements["vertices"]),
                'num_edges': len(mesh_elements["edges"]),
                'num_faces': len(mesh_elements["faces"]),
                'num_bodies': len(mesh_elements["bodies"])
            }

            self.gui.cas_parts_info[part_name] = part_info

            if hasattr(self.gui, 'model_tree_widget'):
                self.gui.model_tree_widget.load_parts({'parts_info': self.gui.cas_parts_info})

            self.gui.log_info("已自动创建Default部件，包含所有网格元素")

        except Exception as e:
            self.gui.log_error(f"创建Default部件失败: {str(e)}")

    def create_default_part_for_geometry(self, shape, stats):
        """为导入的几何创建Default部件"""
        try:
            if hasattr(self.gui, '_create_default_part_for_geometry'):
                self.gui._create_default_part_for_geometry(shape, stats)
                return

            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID

            if not hasattr(self.gui, 'cas_parts_info') or self.gui.cas_parts_info is None:
                self.gui.cas_parts_info = {}

            geometry_elements = {
                "vertices": [],
                "edges": [],
                "faces": [],
                "bodies": []
            }

            vertex_index = 0
            explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
            while explorer.More():
                geometry_elements["vertices"].append(vertex_index)
                vertex_index += 1
                explorer.Next()

            edge_index = 0
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while explorer.More():
                geometry_elements["edges"].append(edge_index)
                edge_index += 1
                explorer.Next()

            face_index = 0
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                geometry_elements["faces"].append(face_index)
                face_index += 1
                explorer.Next()

            body_index = 0
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while explorer.More():
                geometry_elements["bodies"].append(body_index)
                body_index += 1
                explorer.Next()

            self.gui.log_info("Default部件元素收集:")
            self.gui.log_info(f"  - 顶点: {len(geometry_elements['vertices'])} 个")
            self.gui.log_info(f"  - 边: {len(geometry_elements['edges'])} 个")
            self.gui.log_info(f"  - 面: {len(geometry_elements['faces'])} 个")
            self.gui.log_info(f"  - 体: {len(geometry_elements['bodies'])} 个")

            part_name = "DefaultPart"
            part_info = {
                'part_name': part_name,
                'bc_type': '',
                'geometry_elements': geometry_elements,
                'num_vertices': len(geometry_elements["vertices"]),
                'num_edges': len(geometry_elements["edges"]),
                'num_faces': len(geometry_elements["faces"]),
                'num_solids': len(geometry_elements["bodies"])
            }

            self.gui.cas_parts_info[part_name] = part_info

            if hasattr(self.gui, 'model_tree_widget'):
                self.gui.model_tree_widget.load_parts({'parts_info': self.gui.cas_parts_info})

            self.gui.log_info("已自动创建Default部件，包含所有几何元素")

        except Exception as e:
            self.gui.log_error(f"创建Default部件失败: {str(e)}")


class GeometryOperationsBackup:
    """Backup: duplicate geometry import callbacks."""

    def __init__(self, gui):
        self.gui = gui

    def on_geometry_import_progress(self, message, progress):
        """几何导入进度更新回调"""
        if hasattr(self.gui, '_update_progress'):
            self.gui._update_progress(message, progress, "geometry")
        else:
            self.gui.status_bar.show_progress(message, progress)
            if progress % 20 == 0 or progress == 100:
                self.gui.log_info(f"{message} ({progress}%)")
        if progress == 100:
            self.gui.status_bar.hide_progress()

    def on_geometry_import_finished(self, result):
        """几何导入完成回调"""
        try:
            shape = result['shape']
            stats = result['stats']
            file_path = result['file_path']

            self.gui.log_info("几何统计信息:")
            self.gui.log_info(f"  - 顶点数: {stats['num_vertices']}")
            self.gui.log_info(f"  - 边数: {stats['num_edges']}")
            self.gui.log_info(f"  - 面数: {stats['num_faces']}")
            self.gui.log_info(f"  - 实体数: {stats['num_solids']}")

            bbox_min, bbox_max = stats['bounding_box']
            self.gui.log_info(
                f"  - 边界框: ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}) "
                f"到 ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f})"
            )

            self.gui.current_geometry = shape
            self.gui.geometry_actors = {}
            self.gui.geometry_actor = None
            self.gui.geometry_edges_actor = None

            self.gui.update_status("正在添加几何到显示...")

            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                main_actor = result.get('main_actor')
                edges_actor = result.get('edges_actor')

                if main_actor is not None:
                    self.gui.geometry_actor = main_actor
                    self.gui.mesh_display.renderer.AddActor(self.gui.geometry_actor)
                    self.gui.geometry_actors['main'] = [self.gui.geometry_actor]

                if edges_actor is not None:
                    self.gui.geometry_edges_actor = edges_actor
                    self.gui.mesh_display.renderer.AddActor(self.gui.geometry_edges_actor)
                    self.gui.geometry_actors['edges'] = [self.gui.geometry_edges_actor]

                if self.gui.geometry_actor is not None:
                    if self.gui.render_mode == "wireframe":
                        self.gui.geometry_actor.SetVisibility(False)
                        if self.gui.geometry_edges_actor:
                            self.gui.geometry_edges_actor.SetVisibility(True)
                    elif self.gui.render_mode == "surface-wireframe":
                        self.gui.geometry_actor.SetVisibility(True)
                        if self.gui.geometry_edges_actor:
                            self.gui.geometry_edges_actor.SetVisibility(True)
                    else:
                        self.gui.geometry_actor.SetVisibility(True)
                        if self.gui.geometry_edges_actor:
                            self.gui.geometry_edges_actor.SetVisibility(False)

                    self.gui.mesh_display.renderer.ResetCamera()
                    self.gui.mesh_display.render_window.Render()

            if hasattr(self.gui, 'model_tree_widget'):
                self.gui.update_status("正在加载模型树...")
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

        if hasattr(self.gui, '_set_ribbon_button_enabled'):
            self.gui._set_ribbon_button_enabled('file', 'import_geometry', True)

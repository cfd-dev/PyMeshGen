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


class GuiMainPartManagerWrappersBackup:
    """Backup: gui_main wrappers that delegated to PartManager."""

    def __init__(self, gui):
        self.gui = gui

    def add_part(self):
        """添加部件"""
        self.gui.part_manager.add_part()

    def remove_part(self):
        """删除部件"""
        self.gui.part_manager.remove_part()

    def edit_part(self):
        """编辑部件参数（从右键菜单调用）"""
        self.gui.part_manager.edit_mesh_params()

    def show_selected_part(self):
        """显示选中部件（从右键菜单调用）"""
        self.gui.part_manager.show_selected_part()

    def show_only_selected_part(self):
        """只显示选中部件，隐藏其他所有部件（从右键菜单调用）"""
        self.gui.part_manager.show_only_selected_part()

    def show_all_parts(self):
        """显示所有部件（从右键菜单调用）"""
        self.gui.part_manager.show_all_parts()

    def update_parts_list(self):
        """更新部件列表"""
        self.gui.part_manager.update_parts_list(update_status=False)

    def update_parts_list_from_cas(self, parts_info):
        """从cas文件的部件信息更新部件列表"""
        self.gui.part_manager.update_parts_list_from_cas(parts_info=parts_info, update_status=False)

    def handle_part_visibility_change(self, part_name, is_visible):
        """处理部件可见性变化"""
        self.gui.part_manager.handle_part_visibility_change(part_name, is_visible)

    def on_part_created(self, part_info):
        """处理新建部件的回调"""
        self.gui.part_manager.on_part_created(part_info)

    def refresh_display_all_parts(self):
        """刷新显示所有可见部件"""
        self.gui.part_manager.refresh_display_all_parts()

    def switch_display_mode(self, mode):
        """切换显示模式"""
        self.gui.part_manager.switch_display_mode(mode)

    def edit_mesh_params(self):
        """编辑部件参数"""
        self.gui.part_manager.edit_mesh_params()

    def on_geometry_visibility_changed(self, element_type, visible):
        """几何元素类别可见性改变时的回调"""
        self.gui.part_manager.on_geometry_visibility_changed(element_type, visible)

    def on_geometry_element_visibility_changed(self, element_type, element_index, visible):
        """单个几何元素可见性改变时的回调"""
        self.gui.part_manager.on_geometry_element_visibility_changed(element_type, element_index, visible)

    def on_mesh_part_visibility_changed(self, visible):
        """网格部件类别可见性改变时的回调"""
        self.gui.part_manager.on_mesh_part_visibility_changed(visible)

    def on_mesh_part_element_visibility_changed(self, part_index, visible):
        """单个网格部件可见性改变时的回调"""
        self.gui.part_manager.on_mesh_part_element_visibility_changed(part_index, visible)

    def on_mesh_part_selected(self, part_data, part_index):
        """网格部件被选中时的回调"""
        self.gui.part_manager.on_mesh_part_selected(part_data, part_index)

    def _update_geometry_element_display(self):
        """更新几何元素的显示"""
        self.gui.part_manager._update_geometry_element_display()

    def _update_mesh_part_display(self):
        """更新网格部件的显示"""
        self.gui.part_manager._update_mesh_part_display()

    def _update_geometry_display_for_parts(self, visible_parts):
        """根据可见部件更新几何元素的显示"""
        self.gui.part_manager._update_geometry_display_for_parts(visible_parts)

    def on_geometry_element_selected(self, element_type, element_data, element_index):
        """几何元素被选中时的回调"""
        self.gui.part_manager.on_geometry_element_selected(element_type, element_data, element_index)

    def on_model_tree_visibility_changed(self, *args):
        """模型树可见性改变的回调"""
        self.gui.part_manager.on_model_tree_visibility_changed(*args)

    def on_model_tree_selected(self, category, element_type, element_index, element_obj):
        """模型树元素被选中的回调"""
        self.gui.part_manager.on_model_tree_selected(category, element_type, element_index, element_obj)

    def on_mesh_element_selected(self, element_type, element_data, element_index):
        """网格元素被选中时的回调"""
        self.gui.part_manager.on_mesh_element_selected(element_type, element_data, element_index)

    def on_part_selected(self, element_type, element_data, element_index):
        """部件被选中时的回调"""
        self.gui.part_manager.on_part_selected(element_type, element_data, element_index)

    def _get_edge_length(self, edge):
        """获取边的长度"""
        return self.gui.part_manager._get_edge_length(edge)

    def _get_face_area(self, face):
        """获取面的面积"""
        return self.gui.part_manager._get_face_area(face)

    def _get_solid_volume(self, solid):
        """获取体的体积"""
        return self.gui.part_manager._get_solid_volume(solid)


class UIHelpersBackup:
    """Backup: toggle helpers removed from ui_helpers."""

    def __init__(self, gui):
        self.gui = gui

    def toggle_toolbar(self):
        """切换功能区显示"""
        if hasattr(self.gui, 'ribbon') and self.gui.ribbon:
            if self.gui.ribbon.isVisible():
                self.gui.ribbon.hide()
                self.gui.update_status("功能区已隐藏")
            else:
                self.gui.ribbon.show()
                self.gui.update_status("功能区已显示")

    def toggle_statusbar(self):
        """切换状态栏显示"""
        if hasattr(self.gui, 'status_bar') and hasattr(self.gui.status_bar, 'status_bar'):
            if self.gui.status_bar.status_bar.isVisible():
                self.gui.status_bar.status_bar.hide()
                self.gui.log_info("状态栏已隐藏")
            else:
                self.gui.status_bar.status_bar.show()
                self.gui.log_info("状态栏已显示")


class GuiMainViewWrappersBackup:
    """Backup: gui_main view-controller wrappers."""

    def __init__(self, gui):
        self.gui = gui

    def set_render_mode(self, mode):
        """设置渲染模式"""
        self.gui.view_controller.set_render_mode(mode)

    def _toggle_boundary_display(self):
        """切换边界显示"""
        self.gui.view_controller.toggle_boundary_display()

    def reset_view(self):
        """重置视图"""
        self.gui.view_controller.reset_view()

    def fit_view(self):
        """适应视图"""
        self.gui.view_controller.fit_view()

    def set_view_x_positive(self):
        """设置X轴正向视图"""
        self.gui.view_controller.set_view_x_positive()

    def set_view_x_negative(self):
        """设置X轴负向视图"""
        self.gui.view_controller.set_view_x_negative()

    def set_view_y_positive(self):
        """设置Y轴正向视图"""
        self.gui.view_controller.set_view_y_positive()

    def set_view_y_negative(self):
        """设置Y轴负向视图"""
        self.gui.view_controller.set_view_y_negative()

    def set_view_z_positive(self):
        """设置Z轴正向视图"""
        self.gui.view_controller.set_view_z_positive()

    def set_view_z_negative(self):
        """设置Z轴负向视图"""
        self.gui.view_controller.set_view_z_negative()

    def set_view_isometric(self):
        """设置等轴测视图"""
        self.gui.view_controller.set_view_isometric()

    def zoom_in(self):
        """放大视图"""
        self.gui.view_controller.zoom_in()

    def zoom_out(self):
        """缩小视图"""
        self.gui.view_controller.zoom_out()

    def toggle_toolbar(self):
        """切换功能区显示"""
        self.gui.view_controller.toggle_toolbar()

    def toggle_statusbar(self):
        """切换状态栏显示"""
        self.gui.view_controller.toggle_statusbar()

    def set_background_color(self):
        """设置视图区背景色"""
        self.gui.view_controller.set_background_color()

    def toggle_fullscreen(self):
        """切换全屏显示"""
        self.gui.view_controller.toggle_fullscreen()


class GuiMainConfigWrappersBackup:
    """Backup: gui_main config helpers."""

    def __init__(self, gui):
        self.gui = gui

    def edit_params(self):
        """编辑全局参数"""
        self.gui.ui_helpers.edit_params()

    def edit_boundary_conditions(self):
        """编辑边界条件"""
        self.gui.mesh_operations.edit_boundary_conditions()


class GuiMainMeshWrappersBackup:
    """Backup: gui_main mesh-operation wrappers."""

    def __init__(self, gui):
        self.gui = gui

    def generate_mesh(self):
        """生成网格"""
        self.gui.mesh_operations.generate_mesh()

    def check_mesh_quality(self):
        """检查网格质量"""
        self.gui.mesh_operations.check_mesh_quality()

    def smooth_mesh(self):
        """平滑网格"""
        self.gui.mesh_operations.smooth_mesh()

    def optimize_mesh(self):
        """优化网格"""
        self.gui.mesh_operations.optimize_mesh()

    def show_mesh_statistics(self):
        """显示网格统计信息"""
        self.gui.mesh_operations.show_mesh_statistics()

    def extract_boundary_mesh_info(self):
        """提取边界网格及部件信息"""
        self.gui.mesh_operations.extract_boundary_mesh_info()

    def export_mesh_report(self):
        """导出网格报告"""
        self.gui.mesh_operations.export_mesh_report()


class GuiMainHelpWrappersBackup:
    """Backup: gui_main help-module wrappers."""

    def __init__(self, gui):
        self.gui = gui

    def show_about(self):
        """显示关于对话框"""
        self.gui.help_module.show_about()

    def show_user_manual(self):
        """显示用户手册"""
        self.gui.help_module.show_user_manual()

    def show_quick_start(self):
        """显示快速入门"""
        self.gui.help_module.show_quick_start()

    def check_for_updates(self):
        """检查更新"""
        self.gui.help_module.check_for_updates()

    def show_shortcuts(self):
        """显示快捷键"""
        self.gui.help_module.show_shortcuts()

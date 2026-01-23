#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视图控制模块
处理3D视图的缩放、旋转、重置等操作
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ViewController:
    """视图控制类"""

    def __init__(self, gui_instance):
        self.gui = gui_instance
        self._picking_helper = None
        self._picking_enabled = False

    def reset_view(self):
        """重置视图"""
        if hasattr(self.gui, 'mesh_display'):
            self.gui.mesh_display.reset_view()
        self.gui.log_info("已重置视图")
        self.gui.update_status("已重置视图")

    def fit_view(self):
        """适应视图"""
        if hasattr(self.gui, 'mesh_display'):
            self.gui.mesh_display.fit_view()
        self.gui.log_info("已适应视图")
        self.gui.update_status("已适应视图")

    def set_view_x_positive(self):
        """设置X轴正向视图"""
        if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display.renderer:
            camera = self.gui.mesh_display.renderer.GetActiveCamera()
            camera.SetPosition(1, 0, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            self.gui.mesh_display.renderer.ResetCamera()
            self.gui.mesh_display.render_window.Render()
            self.gui.log_info("已切换到X轴正向视图")
            self.gui.update_status("已切换到X轴正向视图")

    def set_view_x_negative(self):
        """设置X轴负向视图"""
        if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display.renderer:
            camera = self.gui.mesh_display.renderer.GetActiveCamera()
            camera.SetPosition(-1, 0, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            self.gui.mesh_display.renderer.ResetCamera()
            self.gui.mesh_display.render_window.Render()
            self.gui.log_info("已切换到X轴负向视图")
            self.gui.update_status("已切换到X轴负向视图")

    def set_view_y_positive(self):
        """设置Y轴正向视图"""
        if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display.renderer:
            camera = self.gui.mesh_display.renderer.GetActiveCamera()
            camera.SetPosition(0, 1, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            self.gui.mesh_display.renderer.ResetCamera()
            self.gui.mesh_display.render_window.Render()
            self.gui.log_info("已切换到Y轴正向视图")
            self.gui.update_status("已切换到Y轴正向视图")

    def set_view_y_negative(self):
        """设置Y轴负向视图"""
        if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display.renderer:
            camera = self.gui.mesh_display.renderer.GetActiveCamera()
            camera.SetPosition(0, -1, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            self.gui.mesh_display.renderer.ResetCamera()
            self.gui.mesh_display.render_window.Render()
            self.gui.log_info("已切换到Y轴负向视图")
            self.gui.update_status("已切换到Y轴负向视图")

    def set_view_z_positive(self):
        """设置Z轴正向视图"""
        if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display.renderer:
            camera = self.gui.mesh_display.renderer.GetActiveCamera()
            camera.SetPosition(0, 0, 1)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)
            self.gui.mesh_display.renderer.ResetCamera()
            self.gui.mesh_display.render_window.Render()
            self.gui.log_info("已切换到Z轴正向视图")
            self.gui.update_status("已切换到Z轴正向视图")

    def set_view_z_negative(self):
        """设置Z轴负向视图"""
        if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display.renderer:
            camera = self.gui.mesh_display.renderer.GetActiveCamera()
            camera.SetPosition(0, 0, -1)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)
            self.gui.mesh_display.renderer.ResetCamera()
            self.gui.mesh_display.render_window.Render()
            self.gui.log_info("已切换到Z轴负向视图")
            self.gui.update_status("已切换到Z轴负向视图")

    def set_view_isometric(self):
        """设置等轴测视图"""
        if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display.renderer:
            camera = self.gui.mesh_display.renderer.GetActiveCamera()
            camera.SetPosition(1, 1, 1)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            self.gui.mesh_display.renderer.ResetCamera()
            self.gui.mesh_display.render_window.Render()
            self.gui.log_info("已切换到等轴测视图")
            self.gui.update_status("已切换到等轴测视图")

    def zoom_in(self):
        """放大视图"""
        if hasattr(self.gui, 'mesh_display'):
            self.gui.mesh_display.zoom_in()
        self.gui.update_status("视图已放大")

    def zoom_out(self):
        """缩小视图"""
        if hasattr(self.gui, 'mesh_display'):
            self.gui.mesh_display.zoom_out()
        self.gui.update_status("视图已缩小")

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

    def toggle_picking_mode(self):
        """切换拾取模式开关"""
        self.set_picking_mode(not self._picking_enabled)

    def set_picking_mode(self, enabled):
        """设置拾取模式"""
        if enabled:
            if self._picking_helper is None:
                if not hasattr(self.gui, 'mesh_display') or self.gui.mesh_display is None:
                    self.gui.log_warning("拾取模式开启失败: 未找到视图区")
                    self.gui.update_status("拾取模式开启失败")
                    return
                from .geometry_picking import GeometryPickingHelper
                self._picking_helper = GeometryPickingHelper(self.gui.mesh_display, gui=self.gui)
            self._picking_helper.enable()
            self._picking_enabled = True
            self.gui.log_info("拾取模式已开启")
            self.gui.update_status("拾取模式: 开启")
        else:
            if self._picking_helper is not None:
                self._picking_helper.cleanup()
                self._picking_helper = None
            self._picking_enabled = False
            self.gui.log_info("拾取模式已关闭")
            self.gui.update_status("拾取模式: 关闭")

    def _apply_render_mode_to_geometry(self, mode):
        display_mode = getattr(self.gui, 'display_mode', 'full')
        use_element_display = display_mode == "elements"

        state = None
        if hasattr(self.gui, 'part_manager') and hasattr(self.gui.part_manager, '_get_geometry_visibility_state'):
            state = self.gui.part_manager._get_geometry_visibility_state(render_mode=mode)

        if state:
            show_faces = state['show_faces']
            show_edges = state['show_edges']
            show_vertices = state['show_vertices']
        else:
            type_states = {'vertices': True, 'edges': True, 'faces': True, 'bodies': True}
            if hasattr(self.gui, 'model_tree_widget') and hasattr(self.gui.model_tree_widget, 'get_geometry_type_states'):
                type_states = self.gui.model_tree_widget.get_geometry_type_states()

            stats = getattr(self.gui, 'current_geometry_stats', None)
            face_count = stats.get('num_faces', 0) if stats else 0
            solid_count = stats.get('num_solids', 0) if stats else 0
            has_surface = (face_count > 0) or (solid_count > 0)

            faces_checked = type_states.get('faces', True)
            bodies_checked = type_states.get('bodies', False)
            edges_checked = type_states.get('edges', False)
            vertices_checked = type_states.get('vertices', False)

            if face_count > 0:
                surface_checked = faces_checked
            elif solid_count > 0:
                surface_checked = bodies_checked
            else:
                surface_checked = False

            if mode == "wireframe":
                show_faces = False
                show_edges = edges_checked
                show_vertices = vertices_checked
            elif mode == "surface-wireframe":
                show_faces = surface_checked
                show_edges = edges_checked
                show_vertices = vertices_checked
            else:
                show_faces = surface_checked
                if not has_surface:
                    show_edges = edges_checked
                    show_vertices = vertices_checked
                else:
                    show_edges = False
                    show_vertices = False

        if not use_element_display:
            if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
                self.gui.geometry_actor.SetVisibility(show_faces)
                if show_faces:
                    self.gui.geometry_actor.GetProperty().SetRepresentationToSurface()
                    self.gui.geometry_actor.GetProperty().EdgeVisibilityOff()

            if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
                self.gui.geometry_edges_actor.SetVisibility(show_edges)

            if hasattr(self.gui, 'geometry_points_actor') and self.gui.geometry_points_actor:
                self.gui.geometry_points_actor.SetVisibility(show_vertices)
        else:
            if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
                self.gui.geometry_actor.SetVisibility(False)
            if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
                self.gui.geometry_edges_actor.SetVisibility(False)
            if hasattr(self.gui, 'geometry_points_actor') and self.gui.geometry_points_actor:
                self.gui.geometry_points_actor.SetVisibility(False)

        if hasattr(self.gui, 'geometry_actors'):
            for elem_type, actors in self.gui.geometry_actors.items():
                for actor in actors:
                    if elem_type == 'faces' or elem_type == 'bodies':
                        actor.SetVisibility(show_faces)
                        if show_faces:
                            actor.GetProperty().SetRepresentationToSurface()
                            actor.GetProperty().EdgeVisibilityOff()
                    elif elem_type == 'edges':
                        actor.SetVisibility(show_edges)
                    elif elem_type == 'vertices':
                        actor.SetVisibility(show_vertices)

    def set_render_mode(self, mode):
        """设置渲染模式"""
        self.gui.render_mode = mode
        if hasattr(self.gui, 'mesh_display'):
            self.gui.mesh_display.set_render_mode(mode)

        self._apply_render_mode_to_geometry(mode)

        mode_messages = {
            "surface": "渲染模式: 实体模式 (1键)",
            "wireframe": "渲染模式: 线框模式 (2键)",
            "surface-wireframe": "渲染模式: 实体+线框模式 (3键)",
        }
        self.gui.update_status(mode_messages.get(mode, f"渲染模式: {mode}"))

        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
            self.gui.mesh_display.render_window.Render()

    def toggle_boundary_display(self):
        """切换边界显示"""
        new_state = not self.gui.show_boundary
        self.gui.show_boundary = new_state
        self.gui.mesh_display.toggle_boundary_display(new_state)
        self.gui.update_status(f"边界显示: {'开启' if new_state else '关闭'} (O键)")

    def set_background_color(self):
        """设置背景颜色"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QColorDialog, QLabel, QComboBox, QGroupBox
        from PyQt5.QtGui import QColor

        dialog = QDialog(self.gui)
        dialog.setWindowTitle("设置背景色")
        dialog.setModal(True)
        dialog.resize(400, 300)

        layout = QVBoxLayout(dialog)

        mode_group = QGroupBox("背景模式")
        mode_layout = QHBoxLayout(mode_group)

        mode_label = QLabel("背景模式:")
        mode_combo = QComboBox()
        mode_combo.addItems(["渐变背景", "单一颜色"])
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(mode_combo)

        color_group = QGroupBox("颜色设置")
        color_layout = QVBoxLayout(color_group)

        color1_layout = QHBoxLayout()
        color1_label = QLabel("起始颜色:")
        color1_button = QPushButton()
        color1_button.setFixedSize(50, 30)
        color1_button.setStyleSheet("background-color: rgb(230, 230, 255)")
        color1_layout.addWidget(color1_label)
        color1_layout.addWidget(color1_button)
        color1_layout.addStretch()

        color2_layout = QHBoxLayout()
        color2_label = QLabel("结束颜色:")
        color2_button = QPushButton()
        color2_button.setFixedSize(50, 30)
        color2_button.setStyleSheet("background-color: rgb(255, 255, 255)")
        color2_layout.addWidget(color2_label)
        color2_layout.addWidget(color2_button)
        color2_layout.addStretch()

        color_layout.addLayout(color1_layout)
        color_layout.addLayout(color2_layout)

        preset_group = QGroupBox("预设方案")
        preset_layout = QHBoxLayout(preset_group)

        def set_preset_blue_white():
            nonlocal color1, color2
            mode_combo.setCurrentIndex(0)
            color1 = (0.9, 0.9, 1.0)
            color1_button.setStyleSheet("background-color: rgb(230, 230, 255)")
            color2 = (1.0, 1.0, 1.0)
            color2_button.setStyleSheet("background-color: rgb(255, 255, 255)")

        def set_preset_black_white():
            nonlocal color1, color2
            mode_combo.setCurrentIndex(0)
            color1 = (0.0, 0.0, 0.0)
            color1_button.setStyleSheet("background-color: rgb(0, 0, 0)")
            color2 = (1.0, 1.0, 1.0)
            color2_button.setStyleSheet("background-color: rgb(255, 255, 255)")

        def set_preset_white():
            nonlocal color1, color2
            mode_combo.setCurrentIndex(1)
            color1 = (1.0, 1.0, 1.0)
            color1_button.setStyleSheet("background-color: rgb(255, 255, 255)")
            color2 = (1.0, 1.0, 1.0)
            color2_button.setStyleSheet("background-color: rgb(255, 255, 255)")

        blue_white_btn = QPushButton("蓝-白渐变")
        blue_white_btn.clicked.connect(set_preset_blue_white)

        black_white_btn = QPushButton("黑-白渐变")
        black_white_btn.clicked.connect(set_preset_black_white)

        white_btn = QPushButton("纯白背景")
        white_btn.clicked.connect(set_preset_white)

        preset_layout.addWidget(blue_white_btn)
        preset_layout.addWidget(black_white_btn)
        preset_layout.addWidget(white_btn)

        button_layout = QHBoxLayout()
        apply_button = QPushButton("应用")
        cancel_button = QPushButton("取消")

        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)

        layout.addWidget(mode_group)
        layout.addWidget(color_group)
        layout.addWidget(preset_group)
        layout.addStretch()
        layout.addLayout(button_layout)

        color1 = (0.9, 0.9, 1.0)
        color2 = (1.0, 1.0, 1.0)

        def choose_color1():
            nonlocal color1
            current_color = color1_button.palette().color(color1_button.backgroundRole())
            color = QColorDialog.getColor(current_color, dialog, "选择起始颜色")
            if color.isValid():
                color1 = (color.redF(), color.greenF(), color.blueF())
                color1_button.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})")

        def choose_color2():
            nonlocal color2
            current_color = color2_button.palette().color(color2_button.backgroundRole())
            color = QColorDialog.getColor(current_color, dialog, "选择结束颜色")
            if color.isValid():
                color2 = (color.redF(), color.greenF(), color.blueF())
                color2_button.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})")

        color1_button.clicked.connect(choose_color1)
        color2_button.clicked.connect(choose_color2)

        def on_mode_changed(index):
            if index == 0:
                color2_label.setEnabled(True)
                color2_button.setEnabled(True)
            else:
                color2_label.setEnabled(False)
                color2_button.setEnabled(False)

        mode_combo.currentIndexChanged.connect(on_mode_changed)

        def apply_background():
            if hasattr(self.gui, 'mesh_display'):
                if mode_combo.currentIndex() == 0:
                    self.gui.mesh_display.set_background_gradient(color1, color2)
                    self.gui.log_info(f"已设置渐变背景色: 起始色{color1}, 结束色{color2}")
                else:
                    self.gui.mesh_display.set_background_color(color1)
                    self.gui.log_info(f"已设置单一背景色: {color1}")
                self.gui.update_status("背景色已更新")
            dialog.accept()

        apply_button.clicked.connect(apply_background)
        cancel_button.clicked.connect(dialog.reject)

        dialog.exec_()

    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.gui.isFullScreen():
            self.gui.showNormal()
            self.gui.update_status("退出全屏模式")
        else:
            self.gui.showFullScreen()
            self.gui.update_status("进入全屏模式")

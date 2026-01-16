#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backup of duplicated gui_main helper implementations that were centralized.
These blocks now delegate to ViewController or UIHelpers.
"""

GUI_MAIN_SET_RENDER_MODE_BACKUP = """
def set_render_mode(self, mode):
    self.render_mode = mode
    if hasattr(self, 'mesh_display'):
        self.mesh_display.set_render_mode(mode)

    if hasattr(self, 'geometry_actor') and self.geometry_actor:
        if mode == "wireframe":
            self.geometry_actor.SetVisibility(False)
        elif mode == "surface-wireframe":
            self.geometry_actor.SetVisibility(True)
            self.geometry_actor.GetProperty().SetRepresentationToSurface()
            self.geometry_actor.GetProperty().EdgeVisibilityOff()
        else:
            self.geometry_actor.SetVisibility(True)
            self.geometry_actor.GetProperty().SetRepresentationToSurface()
            self.geometry_actor.GetProperty().EdgeVisibilityOff()

    if hasattr(self, 'geometry_edges_actor') and self.geometry_edges_actor:
        if mode == "wireframe":
            self.geometry_edges_actor.SetVisibility(True)
        elif mode == "surface-wireframe":
            self.geometry_edges_actor.SetVisibility(False)
        else:
            self.geometry_edges_actor.SetVisibility(False)

    if hasattr(self, 'geometry_actors'):
        for elem_type, actors in self.geometry_actors.items():
            for actor in actors:
                if mode == "wireframe":
                    if elem_type == 'faces' or elem_type == 'bodies':
                        actor.SetVisibility(False)
                    elif elem_type == 'edges' or elem_type == 'vertices':
                        actor.SetVisibility(True)
                elif mode == "surface-wireframe":
                    if elem_type == 'faces' or elem_type == 'bodies':
                        actor.SetVisibility(True)
                        actor.GetProperty().SetRepresentationToSurface()
                        actor.GetProperty().EdgeVisibilityOff()
                    elif elem_type == 'edges' or elem_type == 'vertices':
                        actor.SetVisibility(True)
                else:
                    if elem_type == 'faces' or elem_type == 'bodies':
                        actor.SetVisibility(True)
                        actor.GetProperty().SetRepresentationToSurface()
                        actor.GetProperty().EdgeVisibilityOff()
                    elif elem_type == 'edges' or elem_type == 'vertices':
                        actor.SetVisibility(False)

    mode_messages = {
        "surface": "渲染模式: 实体模式 (1键)",
        "wireframe": "渲染模式: 线框模式 (2键)",
        "surface-wireframe": "渲染模式: 实体+线框模式 (3键)",
    }
    self.update_status(mode_messages.get(mode, f"渲染模式: {mode}"))

    if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'render_window'):
        self.mesh_display.render_window.Render()
"""

GUI_MAIN_TOGGLE_BOUNDARY_DISPLAY_BACKUP = """
def _toggle_boundary_display(self):
    new_state = not self.show_boundary
    self.show_boundary = new_state
    self.mesh_display.toggle_boundary_display(new_state)
    self.update_status(f"边界显示: {'开启' if new_state else '关闭'} (O键)")
"""

GUI_MAIN_LOGGING_BACKUP = """
def log_info(self, message):
    if hasattr(self, 'info_output'):
        self.info_output.log_info(message)

def log_error(self, message):
    if hasattr(self, 'info_output'):
        self.info_output.log_error(message)

def log_warning(self, message):
    if hasattr(self, 'info_output'):
        self.info_output.log_warning(message)

def update_status(self, message):
    if hasattr(self, 'status_bar'):
        self.status_bar.update_status(message)
"""

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

GUI_MAIN_EDIT_PART_BACKUP = """
def edit_part(self):
    QMessageBox.information(self, "提示", "编辑部件参数功能暂未实现，请使用模型树右键菜单")

    # 为每个当前部件创建参数，优先使用已保存的参数
    for part_name in current_part_names:
        if part_name in saved_params_map:
            # 使用已保存的参数
            parts_params.append(saved_params_map[part_name])
        else:
            # 使用默认参数
            parts_params.append({
                "part_name": part_name,
                "max_size": 1e6,
                "PRISM_SWITCH": "off",
                "first_height": 0.01,
                "growth_rate": 1.2,
                "max_layers": 5,
                "full_layers": 5,
                "multi_direction": False
            })

    self.log_info(f"已准备 {len(parts_params)} 个部件的参数")

    # 确保current_row在有效范围内，防止索引超出范围
    if current_row >= len(parts_params):
        current_row = 0  # 如果超出范围，选择第一个部件
    elif current_row < 0:
        current_row = 0  # 如果没有选中任何部件，选择第一个部件

    # 创建并显示对话框，默认选中当前部件
    dialog = PartParamsDialog(self, parts=parts_params, current_part=current_row)
    if dialog.exec_() == QDialog.Accepted:
        # 获取设置后的参数
        self.parts_params = dialog.get_parts_params()
        self.log_info(f"已更新部件参数，共 {len(self.parts_params)} 个部件")
        self.update_status("部件参数已更新")
    else:
        self.log_info("取消设置部件参数")
        self.update_status("已取消部件参数设置")
"""

GUI_MAIN_UPDATE_PARTS_LIST_BACKUP = """
def update_parts_list(self):
    if hasattr(self, 'model_tree_widget') and hasattr(self, 'cas_parts_info') and self.cas_parts_info:
        self.model_tree_widget.load_parts(self.cas_parts_info)
        self.log_info("部件列表已更新")
    else:
        self.log_info("没有部件信息需要更新")
"""

GUI_MAIN_UPDATE_PARTS_LIST_FROM_CAS_BACKUP = """
def update_parts_list_from_cas(self, parts_info):
    self.cas_parts_info = parts_info

    if hasattr(self, 'model_tree_widget'):
        self.model_tree_widget.load_parts(parts_info)
        self.log_info(f"已从CAS更新部件列表，共 {len(parts_info)} 个部件")
    else:
        self.log_info("模型树组件未初始化，无法更新部件列表")
"""

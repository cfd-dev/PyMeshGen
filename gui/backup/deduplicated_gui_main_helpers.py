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

GUI_MAIN_ON_PART_CREATED_BACKUP = """
def on_part_created(self, part_info):
    # 处理新建部件的回调
    if not part_info:
        return

    part_name = part_info.get('part_name', '新部件')
    geometry_elements = part_info.get('geometry_elements', {})
    mesh_elements = part_info.get('mesh_elements', {})

    # 初始化cas_parts_info（如果不存在）
    if not hasattr(self, 'cas_parts_info') or self.cas_parts_info is None:
        self.cas_parts_info = {}

    # 从DefaultPart中移除已分配的元素
    if 'DefaultPart' in self.cas_parts_info:
        default_part = self.cas_parts_info['DefaultPart']

        # 从网格元素中移除
        if 'mesh_elements' in default_part:
            default_mesh_elements = default_part['mesh_elements']
            for elem_type, elem_list in mesh_elements.items():
                if elem_type in default_mesh_elements:
                    for elem_idx in elem_list:
                        if elem_idx in default_mesh_elements[elem_type]:
                            default_mesh_elements[elem_type].remove(elem_idx)

            # 更新计数
            default_part['num_vertices'] = len(default_mesh_elements.get('vertices', []))
            default_part['num_edges'] = len(default_mesh_elements.get('edges', []))
            default_part['num_faces'] = len(default_mesh_elements.get('faces', []))
            default_part['num_bodies'] = len(default_mesh_elements.get('bodies', []))

        # 从几何元素中移除
        if 'geometry_elements' in default_part:
            default_geo_elements = default_part['geometry_elements']
            for elem_type, elem_list in geometry_elements.items():
                if elem_type in default_geo_elements:
                    for elem_idx in elem_list:
                        if elem_idx in default_geo_elements[elem_type]:
                            default_geo_elements[elem_type].remove(elem_idx)

            # 更新计数
            default_part['num_vertices'] = len(default_geo_elements.get('vertices', []))
            default_part['num_edges'] = len(default_geo_elements.get('edges', []))
            default_part['num_faces'] = len(default_geo_elements.get('faces', []))
            default_part['num_solids'] = len(default_geo_elements.get('bodies', []))

        self.log_info(f"已从DefaultPart中移除分配给 {part_name} 的元素")

    # 将新建部件的数据转换为与cas部件兼容的格式
    converted_part_info = {
        'type': 'user_created',
        'bc_type': 'wall',
        'faces': [],
        'nodes': [],
        'cells': [],
        'geometry_elements': geometry_elements,
        'mesh_elements': mesh_elements
    }

    # 如果有网格数据，尝试提取面数据
    if hasattr(self, 'current_mesh') and self.current_mesh:
        if hasattr(self.current_mesh, 'cells'):
            # 获取选中的网格面
            selected_faces = mesh_elements.get('faces', [])
            if selected_faces:
                for face_idx in selected_faces:
                    if face_idx < len(self.current_mesh.cells):
                        cell = self.current_mesh.cells[face_idx]
                        converted_part_info['faces'].append({
                            'nodes': list(cell) if hasattr(cell, '__iter__') else [cell]
                        })

    # 将部件添加到cas_parts_info中
    self.cas_parts_info[part_name] = converted_part_info

    # 更新部件列表显示
    if hasattr(self, 'model_tree_widget'):
        self.model_tree_widget.load_parts(self.cas_parts_info)

    self.log_info(f"已创建部件: {part_name}")
    self.update_status(f"部件已创建: {part_name}")

    # 刷新显示
    self.refresh_display_all_parts()
"""

GUI_MAIN_SWITCH_DISPLAY_MODE_BACKUP = """
def switch_display_mode(self, mode):
    if hasattr(self, 'mesh_display') and self.mesh_display:
        self.mesh_display.set_render_mode(mode)
        self.update_status(f"显示模式已切换为: {mode}")

        # Refresh the display to apply the new mode to all visible parts
        self.refresh_display_all_parts()
"""

GUI_MAIN_VISIBILITY_HANDLERS_BACKUP = """
def on_geometry_visibility_changed(self, element_type, visible):
    self._update_geometry_element_display()

def on_geometry_element_visibility_changed(self, element_type, element_index, visible):
    self._update_geometry_element_display()

def on_mesh_part_visibility_changed(self, visible):
    self._update_mesh_part_display()

def on_mesh_part_element_visibility_changed(self, part_index, visible):
    self._update_mesh_part_display()

def on_mesh_part_selected(self, part_data, part_index):
    part_name = part_data.get('part_name', f'Part_{part_index}') if isinstance(part_data, dict) else f'Part_{part_index}'
    self.log_info(f"选中网格部件: {part_name}")

    if hasattr(self, 'props_text'):
        info_text = f"选中部件: {part_name}\\n"
        info_text += f"索引: {part_index}\\n"

        if isinstance(part_data, dict):
            bc_type = part_data.get('bc_type', '未知')
            info_text += f"边界条件: {bc_type}\\n"

            if 'faces' in part_data:
                info_text += f"面数量: {len(part_data['faces'])}\\n"

            if 'nodes' in part_data:
                info_text += f"节点数量: {len(part_data['nodes'])}\\n"

        self.props_text.setPlainText(info_text)
        self.update_status(f"已选中网格部件: {part_name}")
"""

GUI_MAIN_UPDATE_GEOMETRY_ELEMENT_DISPLAY_BACKUP = """
def _update_geometry_element_display(self):
    if not hasattr(self, 'model_tree_widget') or not hasattr(self, 'current_geometry'):
        return

    visible_elements = self.model_tree_widget.get_visible_elements(category='geometry')

    if hasattr(self, 'geometry_actors'):
        for elem_type, actors in self.geometry_actors.items():
            for actor in actors:
                if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                    self.mesh_display.renderer.RemoveActor(actor)

    if hasattr(self, 'geometry_actor') and self.geometry_actor:
        if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
            self.mesh_display.renderer.RemoveActor(self.geometry_actor)
        self.geometry_actor = None

    if hasattr(self, 'geometry_edges_actor') and self.geometry_edges_actor:
        if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
            self.mesh_display.renderer.RemoveActor(self.geometry_edges_actor)
        self.geometry_edges_actor = None

    self.geometry_actors = {}

    from fileIO.occ_to_vtk import create_vertex_actor, create_edge_actor, create_face_actor, create_solid_actor, create_shape_actor, create_geometry_edges_actor

    if 'geometry' in visible_elements and 'vertices' in visible_elements['geometry'] and visible_elements['geometry']['vertices']:
        self.geometry_actors['vertices'] = []
        for elem_index, elem_data in visible_elements['geometry']['vertices']:
            actor = create_vertex_actor(elem_data, color=(1.0, 0.0, 0.0), point_size=8.0)
            self.geometry_actors['vertices'].append(actor)
            if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                self.mesh_display.renderer.AddActor(actor)

    if 'geometry' in visible_elements and 'edges' in visible_elements['geometry'] and visible_elements['geometry']['edges']:
        self.geometry_actors['edges'] = []
        for elem_index, elem_data in visible_elements['geometry']['edges']:
            actor = create_edge_actor(elem_data, color=(0.0, 0.0, 1.0), line_width=2.0)
            self.geometry_actors['edges'].append(actor)
            if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                self.mesh_display.renderer.AddActor(actor)

    if 'geometry' in visible_elements and 'faces' in visible_elements['geometry'] and visible_elements['geometry']['faces']:
        self.geometry_actors['faces'] = []
        for elem_index, elem_data in visible_elements['geometry']['faces']:
            actor = create_face_actor(elem_data, color=(0.0, 1.0, 0.0), opacity=0.6)
            self.geometry_actors['faces'].append(actor)
            if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                self.mesh_display.renderer.AddActor(actor)

    if 'geometry' in visible_elements and 'bodies' in visible_elements['geometry'] and visible_elements['geometry']['bodies']:
        self.geometry_actors['bodies'] = []
        for elem_index, elem_data in visible_elements['geometry']['bodies']:
            actor = create_solid_actor(elem_data, color=(0.8, 0.8, 0.9), opacity=0.5)
            self.geometry_actors['bodies'].append(actor)
            if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                self.mesh_display.renderer.AddActor(actor)

    if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'render_window'):
        self.mesh_display.render_window.Render()
"""

GUI_MAIN_UPDATE_MESH_PART_DISPLAY_BACKUP = """
def _update_mesh_part_display(self):
    if not hasattr(self, 'model_tree_widget') or not hasattr(self, 'current_mesh'):
        return

    # 使用 refresh_display_all_parts 来正确更新部件显示
    self.refresh_display_all_parts()
"""

GUI_MAIN_UPDATE_GEOMETRY_DISPLAY_FOR_PARTS_BACKUP = """
def _update_geometry_display_for_parts(self, visible_parts):
    if not hasattr(self, 'model_tree_widget') or not hasattr(self, 'current_geometry'):
        return

    # 获取所有部件的几何元素索引
    part_geometry_elements = {}
    if hasattr(self, 'cas_parts_info') and self.cas_parts_info:
        for part_name, part_data in self.cas_parts_info.items():
            if 'geometry_elements' in part_data:
                part_geometry_elements[part_name] = part_data['geometry_elements']

    # 如果没有部件包含几何元素，则显示所有几何元素
    if not part_geometry_elements:
        return

    # 收集所有可见部件的几何元素索引
    visible_geometry_indices = {'vertices': set(), 'edges': set(), 'faces': set(), 'bodies': set()}
    for part_name in visible_parts:
        if part_name in part_geometry_elements:
            for elem_type, indices in part_geometry_elements[part_name].items():
                if elem_type in visible_geometry_indices:
                    visible_geometry_indices[elem_type].update(indices)

    # 获取当前几何元素的可见性
    visible_elements = self.model_tree_widget.get_visible_elements(category='geometry')

    # 移除所有几何元素actor
    if hasattr(self, 'geometry_actors'):
        for elem_type, actors in self.geometry_actors.items():
            for actor in actors:
                if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                    self.mesh_display.renderer.RemoveActor(actor)

    if hasattr(self, 'geometry_actor') and self.geometry_actor:
        if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
            self.mesh_display.renderer.RemoveActor(self.geometry_actor)
        self.geometry_actor = None

    if hasattr(self, 'geometry_edges_actor') and self.geometry_edges_actor:
        if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
            self.mesh_display.renderer.RemoveActor(self.geometry_edges_actor)
        self.geometry_edges_actor = None

    self.geometry_actors = {}

    from fileIO.occ_to_vtk import create_vertex_actor, create_edge_actor, create_face_actor, create_solid_actor

    # 只显示属于可见部件的几何元素
    if 'geometry' in visible_elements:
        if 'vertices' in visible_elements['geometry']:
            self.geometry_actors['vertices'] = []
            for elem_index, elem_data in visible_elements['geometry']['vertices']:
                if visible_geometry_indices['vertices'] and elem_index in visible_geometry_indices['vertices']:
                    actor = create_vertex_actor(elem_data, color=(1.0, 0.0, 0.0), point_size=8.0)
                    self.geometry_actors['vertices'].append(actor)
                    if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                        self.mesh_display.renderer.AddActor(actor)

        if 'edges' in visible_elements['geometry']:
            self.geometry_actors['edges'] = []
            for elem_index, elem_data in visible_elements['geometry']['edges']:
                if visible_geometry_indices['edges'] and elem_index in visible_geometry_indices['edges']:
                    actor = create_edge_actor(elem_data, color=(0.0, 0.0, 1.0), line_width=2.0)
                    self.geometry_actors['edges'].append(actor)
                    if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                        self.mesh_display.renderer.AddActor(actor)

        if 'faces' in visible_elements['geometry']:
            self.geometry_actors['faces'] = []
            for elem_index, elem_data in visible_elements['geometry']['faces']:
                if visible_geometry_indices['faces'] and elem_index in visible_geometry_indices['faces']:
                    actor = create_face_actor(elem_data, color=(0.0, 1.0, 0.0), opacity=0.6)
                    self.geometry_actors['faces'].append(actor)
                    if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                        self.mesh_display.renderer.AddActor(actor)

        if 'bodies' in visible_elements['geometry']:
            self.geometry_actors['bodies'] = []
            for elem_index, elem_data in visible_elements['geometry']['bodies']:
                if visible_geometry_indices['bodies'] and elem_index in visible_geometry_indices['bodies']:
                    actor = create_solid_actor(elem_data, color=(0.8, 0.8, 0.9), opacity=0.5)
                    self.geometry_actors['bodies'].append(actor)
                    if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                        self.mesh_display.renderer.AddActor(actor)

    if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'render_window'):
        self.mesh_display.render_window.Render()
"""

GUI_MAIN_GEOMETRY_SELECTION_BACKUP = """
def on_geometry_element_selected(self, element_type, element_data, element_index):
    element_name = f"{element_type}_{element_index}"
    self.log_info(f"选中: {element_name}")

    if hasattr(self, 'props_text'):
        info_text = f"选中元素: {element_name}\\n"
        info_text += f"类型: {element_type}\\n"
        info_text += f"索引: {element_index}\\n"

        if element_type == "vertex":
            from OCC.Core.BRep import BRep_Tool
            pnt = BRep_Tool.Pnt(element_data)
            info_text += f"坐标: ({pnt.X():.3f}, {pnt.Y():.3f}, {pnt.Z():.3f})\\n"
        elif element_type == "edge":
            edge_length = self._get_edge_length(element_data)
            if edge_length is not None:
                info_text += f"长度: {edge_length:.3f}\\n"
        elif element_type == "face":
            face_area = self._get_face_area(element_data)
            if face_area is not None:
                info_text += f"面积: {face_area:.3f}\\n"
        elif element_type == "body":
            body_volume = self._get_solid_volume(element_data)
            if body_volume is not None:
                info_text += f"体积: {body_volume:.3f}\\n"

        self.props_text.setPlainText(info_text)

def on_model_tree_visibility_changed(self, *args):
    if len(args) == 2:
        category, visible = args

        if category == 'geometry':
            self._update_geometry_element_display()
        elif category == 'mesh':
            self._update_mesh_part_display()
        elif category == 'parts':
            self.refresh_display_all_parts()

    elif len(args) == 3:
        category, arg2, visible = args
        # 对于部件，arg2 是 element_index；对于其他类别，arg2 是 element_type
        if category == 'parts':
            element_index = arg2
            self.refresh_display_all_parts()
        else:
            element_type = arg2
            if category == 'geometry':
                self._update_geometry_element_display()
            elif category == 'mesh':
                self._update_mesh_part_display()

    elif len(args) == 4:
        category, element_type, element_index, visible = args

        if category == 'geometry':
            self._update_geometry_element_display()
        elif category == 'mesh':
            self._update_mesh_part_display()
        elif category == 'parts':
            self.refresh_display_all_parts()

def _get_edge_length(self, edge):
    from OCC.Core.GCPnts import GCPnts_AbscissaPoint
    from OCC.Core.BRep import BRep_Tool

    try:
        curve = BRep_Tool.Curve(edge)
        if curve:
            geom_curve, first, last = curve
            if geom_curve:
                length = GCPnts_AbscissaPoint.Length(geom_curve, first, last)
                return length
    except:
        pass
    return None

def _get_face_area(self, face):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    try:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        return props.Mass()
    except:
        return None

def _get_solid_volume(self, solid):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    try:
        props = GProp_GProps()
        brepgprop.VolumeProperties(solid, props)
        return props.Mass()
    except:
    return None
"""

GUI_MAIN_MODEL_TREE_SELECTION_BACKUP = """
def on_model_tree_selected(self, category, element_type, element_index, element_obj):
    if category == 'geometry':
        self.on_geometry_element_selected(element_type, element_obj, element_index)
    elif category == 'mesh':
        self.on_mesh_element_selected(element_type, element_obj, element_index)
    elif category == 'parts':
        self.on_part_selected(element_type, element_obj, element_index)

def on_mesh_element_selected(self, element_type, element_data, element_index):
    element_name = f"{element_type}_{element_index}"
    self.log_info(f"选中网格元素: {element_name}")

    if hasattr(self, 'props_text'):
        info_text = f"选中网格元素: {element_name}\\n"
        info_text += f"类型: {element_type}\\n"
        info_text += f"索引: {element_index}\\n"

        if isinstance(element_data, dict):
            if 'nodes' in element_data:
                info_text += f"节点数量: {len(element_data['nodes'])}\\n"
            if 'faces' in element_data:
                info_text += f"面数量: {len(element_data['faces'])}\\n"
            if 'edges' in element_data:
                info_text += f"边数量: {len(element_data['edges'])}\\n"

        self.props_text.setPlainText(info_text)

def on_part_selected(self, element_type, element_data, element_index):
    part_name = f"{element_type}_{element_index}"
    self.log_info(f"选中部件: {part_name}")

    if hasattr(self, 'props_text'):
        info_text = f"选中部件: {part_name}\\n"
        info_text += f"索引: {element_index}\\n"

        if isinstance(element_data, dict):
            bc_type = element_data.get('bc_type', '未知')
            info_text += f"边界条件: {bc_type}\\n"

            if 'faces' in element_data:
                info_text += f"面数量: {len(element_data['faces'])}\\n"

            if 'nodes' in element_data:
                info_text += f"节点数量: {len(element_data['nodes'])}\\n"

        self.props_text.setPlainText(info_text)
        self.update_status(f"已选中网格部件: {part_name}")
"""

GUI_MAIN_HANDLE_PART_VISIBILITY_BACKUP = """
def handle_part_visibility_change(self, part_name, is_visible):
    if not hasattr(self, 'mesh_display') or not self.mesh_display:
        return

    # Refresh the entire display to show/hide parts based on checkbox states
    self.refresh_display_all_parts()

    # Extract the actual part name from formatted text (e.g., "部件1 - Max Size: 1.0, Prism: True" -> "部件1")
    actual_part_name = part_name.split(' - ')[0] if ' - ' in part_name else part_name

    action = "显示" if is_visible else "隐藏"
    self.log_info(f"{action}部件: {actual_part_name}")
"""

GUI_MAIN_SHOW_PARTS_BACKUP = """
def show_selected_part(self):
    QMessageBox.information(self, "提示", "显示选中部件功能请使用模型树复选框")

def show_only_selected_part(self):
    QMessageBox.information(self, "提示", "只显示选中部件功能请使用模型树复选框")

def show_all_parts(self):
    self.log_info("显示所有部件")
    self.update_status("显示所有部件")

    if hasattr(self, 'model_tree_widget'):
        self.model_tree_widget.set_all_parts_visible(True)
        self.refresh_display_all_parts()
        self.log_info("成功显示所有部件")
    else:
        self.log_info("模型树组件未初始化")
"""

GUI_MAIN_TOGGLE_FULLSCREEN_BACKUP = """
def toggle_fullscreen(self):
    if self.isFullScreen():
        self.showNormal()
        self.update_status("退出全屏模式")
    else:
        self.showFullScreen()
        self.update_status("进入全屏模式")
"""

GUI_MAIN_ON_PART_SELECT_BACKUP = """
def on_part_select(self, index):
    self.log_info("部件选择事件已触发")
    self.update_status("已选中部件")
"""

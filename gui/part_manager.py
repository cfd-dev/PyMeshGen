from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt


class PartManager:
    """部件管理器 - 负责管理网格部件的创建、编辑、删除和显示"""

    def __init__(self, gui):
        self.gui = gui

    def add_part(self):
        """添加部件"""
        if not self.gui.params:
            QMessageBox.warning(self.gui, "警告", "请先创建或加载配置")
            return

        if hasattr(self.gui, 'model_tree_widget'):
            self.gui.model_tree_widget._create_part_dialog()
            self.gui.log_info("添加部件对话框已打开")
            self.gui.update_status("正在添加部件")
        else:
            QMessageBox.warning(self.gui, "警告", "模型树组件未初始化")

    def remove_part(self):
        """删除部件"""
        QMessageBox.information(self.gui, "提示", "删除部件功能暂未实现，请使用右键菜单删除部件")

    def edit_part(self):
        """编辑部件属性"""
        QMessageBox.information(self.gui, "提示", "编辑部件功能暂未实现，请使用右键菜单编辑部件")

    def edit_mesh_params(self):
        """编辑部件参数"""
        from PyQt5.QtWidgets import QDialog
        from gui.part_params_dialog import PartParamsDialog
        
        # 检查是否有导入的网格数据
        if not hasattr(self.gui, 'cas_parts_info') or not self.gui.cas_parts_info:
            QMessageBox.warning(self.gui, "警告", "请先导入网格文件以获取部件列表")
            self.gui.log_info("未检测到导入的网格数据，无法设置部件参数")
            self.gui.update_status("未检测到导入的网格数据")
            return
        
        # 从当前导入的网格数据中获取部件列表
        parts_params = []
        
        # 获取当前部件名称列表
        current_part_names = []
        if isinstance(self.gui.cas_parts_info, dict):
            current_part_names = list(self.gui.cas_parts_info.keys())
        elif isinstance(self.gui.cas_parts_info, list):
            current_part_names = [part_info.get('part_name', f'部件{self.gui.cas_parts_info.index(part_info)}') for part_info in self.gui.cas_parts_info]
        
        # 创建已保存参数的映射字典，按部件名称索引
        saved_params_map = {}
        if hasattr(self.gui, 'parts_params') and self.gui.parts_params:
            for param in self.gui.parts_params:
                if 'part_name' in param:
                    saved_params_map[param['part_name']] = param
            self.gui.log_info(f"使用已保存的部件参数，共 {len(saved_params_map)} 个部件")
        
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
        
        self.gui.log_info(f"已准备 {len(parts_params)} 个部件的参数")
        
        # 创建并显示对话框
        dialog = PartParamsDialog(self.gui, parts=parts_params)
        if dialog.exec_() == QDialog.Accepted:
            # 获取设置后的参数
            self.gui.parts_params = dialog.get_parts_params()
            self.gui.log_info(f"已更新部件参数，共 {len(self.gui.parts_params)} 个部件")
            self.gui.update_status("部件参数已更新")
        else:
            self.gui.log_info("取消设置部件参数")
            self.gui.update_status("已取消部件参数设置")

    def update_parts_list(self, update_status=True):
        """更新部件列表"""
        if not hasattr(self.gui, 'cas_parts_info') or not self.gui.cas_parts_info:
            self.gui.log_info("没有部件信息需要更新")
            return

        if hasattr(self.gui, 'model_tree_widget'):
            self.gui.model_tree_widget.load_parts(self.gui.cas_parts_info)
            self.gui.log_info("部件列表已更新")
            if update_status:
                self.gui.update_status("部件列表已更新")
        else:
            self.gui.log_info("模型树组件未初始化，无法更新部件列表")

    def update_parts_list_from_cas(self, parts_info=None, update_status=True):
        """从CAS数据更新部件列表"""
        if parts_info is not None:
            self.gui.cas_parts_info = parts_info

        if parts_info is None:
            if not hasattr(self.gui, 'cas_parts_info') or self.gui.cas_parts_info is None:
                self.gui.log_info("没有CAS部件数据")
                return

        if hasattr(self.gui, 'model_tree_widget'):
            self.gui.model_tree_widget.load_parts(self.gui.cas_parts_info)
            self.gui.log_info(f"已从CAS更新部件列表，共 {len(self.gui.cas_parts_info)} 个部件")
            if update_status:
                self.gui.update_status("部件列表已从CAS更新")
        else:
            self.gui.log_info("模型树组件未初始化，无法更新部件列表")

    def on_part_select(self, item):
        """部件选中事件处理"""
        part_name = item.text().split(' - ')[0] if ' - ' in item.text() else item.text()
        self.gui.log_info(f"选中部件: {part_name}")
        self.gui.update_status(f"已选中部件: {part_name}")

        if hasattr(self.gui, 'props_text'):
            info_text = f"选中部件: {part_name}\n"

            if hasattr(self.gui, 'cas_parts_info') and part_name in self.gui.cas_parts_info:
                part_data = self.gui.cas_parts_info[part_name]
                if isinstance(part_data, dict):
                    bc_type = part_data.get('bc_type', '未知')
                    info_text += f"边界条件: {bc_type}\n"

                    if 'faces' in part_data:
                        info_text += f"面数量: {len(part_data['faces'])}\n"

                    if 'nodes' in part_data:
                        info_text += f"节点数量: {len(part_data['nodes'])}\n"

                    if 'geometry_elements' in part_data:
                        geo_elems = part_data['geometry_elements']
                        info_text += f"几何元素:\n"
                        for elem_type, indices in geo_elems.items():
                            info_text += f"  {elem_type}: {len(indices)}\n"

            self.gui.props_text.setPlainText(info_text)

    def handle_part_visibility_change(self, item):
        """处理部件可见性改变"""
        part_name = item.text().split(' - ')[0] if ' - ' in item.text() else item.text()
        is_visible = item.checkState() == Qt.Checked

        if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display:
            if is_visible:
                self.gui.mesh_display.display_part(part_name, parts_info=self.gui.cas_parts_info)
                self.gui.log_info(f"显示部件: {part_name}")
            else:
                self.gui.mesh_display.hide_part(part_name)
                self.gui.log_info(f"隐藏部件: {part_name}")

        self.gui.update_status(f"部件可见性已改变: {part_name}")

    def on_part_created(self, part_name, geometry_elements=None, mesh_elements=None):
        """部件创建事件处理"""
        if not hasattr(self.gui, 'cas_parts_info'):
            self.gui.cas_parts_info = {}

        if not hasattr(self.gui, 'current_geometry') or not self.gui.current_geometry:
            QMessageBox.warning(self.gui, "警告", "请先导入几何模型")
            self.gui.log_info("未找到几何模型，无法创建部件")
            self.gui.update_status("未找到几何模型")
            return

        if not geometry_elements and not mesh_elements:
            QMessageBox.warning(self.gui, "警告", "请至少选择一个几何元素或网格元素")
            self.gui.log_info("未选择任何元素，无法创建部件")
            self.gui.update_status("未选择任何元素")
            return

        if geometry_elements is None:
            geometry_elements = {'vertices': [], 'edges': [], 'faces': [], 'bodies': []}

        if mesh_elements is None:
            mesh_elements = {'vertices': [], 'edges': [], 'faces': [], 'bodies': []}

        if 'DefaultPart' in self.gui.cas_parts_info:
            default_part = self.gui.cas_parts_info['DefaultPart']

            if 'mesh_elements' in default_part:
                default_mesh_elements = default_part['mesh_elements']
                for elem_type, elem_list in mesh_elements.items():
                    if elem_type in default_mesh_elements:
                        for elem_idx in elem_list:
                            if elem_idx in default_mesh_elements[elem_type]:
                                default_mesh_elements[elem_type].remove(elem_idx)

                default_part['num_vertices'] = len(default_mesh_elements.get('vertices', []))
                default_part['num_edges'] = len(default_mesh_elements.get('edges', []))
                default_part['num_faces'] = len(default_mesh_elements.get('faces', []))
                default_part['num_bodies'] = len(default_mesh_elements.get('bodies', []))

            if 'geometry_elements' in default_part:
                default_geo_elements = default_part['geometry_elements']
                for elem_type, elem_list in geometry_elements.items():
                    if elem_type in default_geo_elements:
                        for elem_idx in elem_list:
                            if elem_idx in default_geo_elements[elem_type]:
                                default_geo_elements[elem_type].remove(elem_idx)

                default_part['num_vertices'] = len(default_geo_elements.get('vertices', []))
                default_part['num_edges'] = len(default_geo_elements.get('edges', []))
                default_part['num_faces'] = len(default_geo_elements.get('faces', []))
                default_part['num_solids'] = len(default_geo_elements.get('bodies', []))

            self.gui.log_info(f"已从DefaultPart中移除分配给 {part_name} 的元素")

        converted_part_info = {
            'type': 'user_created',
            'bc_type': 'wall',
            'faces': [],
            'nodes': [],
            'cells': [],
            'geometry_elements': geometry_elements,
            'mesh_elements': mesh_elements
        }

        if hasattr(self.gui, 'current_mesh') and self.gui.current_mesh:
            if hasattr(self.gui.current_mesh, 'cells'):
                selected_faces = mesh_elements.get('faces', [])
                if selected_faces:
                    for face_idx in selected_faces:
                        if face_idx < len(self.gui.current_mesh.cells):
                            cell = self.gui.current_mesh.cells[face_idx]
                            converted_part_info['faces'].append({
                                'nodes': list(cell) if hasattr(cell, '__iter__') else [cell]
                            })

        self.gui.cas_parts_info[part_name] = converted_part_info

        if hasattr(self.gui, 'parts_list_widget'):
            self.gui.parts_list_widget.add_part_with_checkbox(part_name, True)

        if hasattr(self.gui, 'model_tree_widget') and 'DefaultPart' in self.gui.cas_parts_info:
            self.gui.model_tree_widget.load_parts({'parts_info': self.gui.cas_parts_info})

        self.gui.log_info(f"已创建部件: {part_name}")
        self.gui.update_status(f"部件已创建: {part_name}")

        self.refresh_display_all_parts()

    def refresh_display_all_parts(self):
        """刷新显示所有可见部件 - 优化版本，批量处理减少渲染次数"""
        if not hasattr(self.gui, 'mesh_display') or not self.gui.mesh_display:
            return

        # Clear current display
        self.gui.mesh_display.clear_mesh_actors()

        # Get all currently checked parts from geometry tree widget
        visible_parts = []
        if hasattr(self.gui, 'model_tree_widget'):
            visible_parts = self.gui.model_tree_widget.get_visible_parts()
        elif hasattr(self.gui, 'parts_list_widget'):
            for i in range(self.gui.parts_list_widget.parts_list.count()):
                item = self.gui.parts_list_widget.parts_list.item(i)
                part_name = item.text().split(' - ')[0] if ' - ' in item.text() else item.text()
                if item.checkState() == Qt.Checked:
                    visible_parts.append(part_name)

        # Display only visible parts - 批量处理以提高性能
        if visible_parts:
            # 批量显示所有部件，只渲染一次
            for part_name in visible_parts:
                self.gui.mesh_display.display_part(part_name, parts_info=self.gui.cas_parts_info, render_immediately=False)

            # 批量渲染一次，而不是每个部件都渲染
            self.gui.mesh_display.render_window.Render()
        else:
            # If no parts are visible, clear display
            self.gui.mesh_display.clear()

        # Re-add axes after displaying parts
        self.gui.mesh_display.add_axes()

        # 更新几何元素的显示（基于可见部件）
        self._update_geometry_display_for_parts(visible_parts)

    def switch_display_mode(self, mode):
        """切换显示模式"""
        if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display:
            self.gui.mesh_display.set_render_mode(mode)
            self.gui.update_status(f"显示模式已切换为: {mode}")
            self.refresh_display_all_parts()

    def on_geometry_visibility_changed(self, element_type, visible):
        """几何元素类别可见性改变时的回调"""
        self._update_geometry_element_display()

    def on_geometry_element_visibility_changed(self, element_type, element_index, visible):
        """单个几何元素可见性改变时的回调"""
        self._update_geometry_element_display()

    def on_mesh_part_visibility_changed(self, visible):
        """网格部件类别可见性改变时的回调"""
        self._update_mesh_part_display()

    def on_mesh_part_element_visibility_changed(self, part_index, visible):
        """单个网格部件可见性改变时的回调"""
        self._update_mesh_part_display()

    def on_mesh_part_selected(self, part_data, part_index):
        """网格部件被选中时的回调"""
        part_name = part_data.get('part_name', f'Part_{part_index}') if isinstance(part_data, dict) else f'Part_{part_index}'
        self.gui.log_info(f"选中网格部件: {part_name}")

        if hasattr(self.gui, 'props_text'):
            info_text = f"选中部件: {part_name}\n"
            info_text += f"索引: {part_index}\n"

            if isinstance(part_data, dict):
                bc_type = part_data.get('bc_type', '未知')
                info_text += f"边界条件: {bc_type}\n"

                if 'faces' in part_data:
                    info_text += f"面数量: {len(part_data['faces'])}\n"

                if 'nodes' in part_data:
                    info_text += f"节点数量: {len(part_data['nodes'])}\n"

            self.gui.props_text.setPlainText(info_text)
            self.gui.update_status(f"已选中网格部件: {part_name}")

    def _update_geometry_element_display(self):
        """更新几何元素的显示"""
        if not hasattr(self.gui, 'model_tree_widget') or not hasattr(self.gui, 'current_geometry'):
            return

        visible_elements = self.gui.model_tree_widget.get_visible_elements(category='geometry')

        if hasattr(self.gui, 'geometry_actors'):
            for elem_type, actors in self.gui.geometry_actors.items():
                for actor in actors:
                    if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                        self.gui.mesh_display.renderer.RemoveActor(actor)

        if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_actor)
            self.gui.geometry_actor = None

        if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_edges_actor)
            self.gui.geometry_edges_actor = None

        self.gui.geometry_actors = {}

        from fileIO.occ_to_vtk import create_vertex_actor, create_edge_actor, create_face_actor, create_solid_actor, create_shape_actor, create_geometry_edges_actor

        if 'geometry' in visible_elements and 'vertices' in visible_elements['geometry'] and visible_elements['geometry']['vertices']:
            self.gui.geometry_actors['vertices'] = []
            for elem_index, elem_data in visible_elements['geometry']['vertices']:
                actor = create_vertex_actor(elem_data, color=(1.0, 0.0, 0.0), point_size=8.0)
                self.gui.geometry_actors['vertices'].append(actor)
                if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                    self.gui.mesh_display.renderer.AddActor(actor)

        if 'geometry' in visible_elements and 'edges' in visible_elements['geometry'] and visible_elements['geometry']['edges']:
            self.gui.geometry_actors['edges'] = []
            for elem_index, elem_data in visible_elements['geometry']['edges']:
                actor = create_edge_actor(elem_data, color=(0.0, 0.0, 1.0), line_width=2.0)
                self.gui.geometry_actors['edges'].append(actor)
                if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                    self.gui.mesh_display.renderer.AddActor(actor)

        if 'geometry' in visible_elements and 'faces' in visible_elements['geometry'] and visible_elements['geometry']['faces']:
            self.gui.geometry_actors['faces'] = []
            for elem_index, elem_data in visible_elements['geometry']['faces']:
                actor = create_face_actor(elem_data, color=(0.0, 1.0, 0.0), opacity=0.6)
                self.gui.geometry_actors['faces'].append(actor)
                if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                    self.gui.mesh_display.renderer.AddActor(actor)

        if 'geometry' in visible_elements and 'bodies' in visible_elements['geometry'] and visible_elements['geometry']['bodies']:
            self.gui.geometry_actors['bodies'] = []
            for elem_index, elem_data in visible_elements['geometry']['bodies']:
                actor = create_solid_actor(elem_data, color=(0.8, 0.8, 0.9), opacity=0.5)
                self.gui.geometry_actors['bodies'].append(actor)
                if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                    self.gui.mesh_display.renderer.AddActor(actor)

        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
            self.gui.mesh_display.render_window.Render()

        render_mode = getattr(self.gui, 'render_mode', 'surface')
        
        if hasattr(self.gui, 'geometry_actors'):
            for elem_type, actors in self.gui.geometry_actors.items():
                for actor in actors:
                    if render_mode == "wireframe":
                        if elem_type == 'faces' or elem_type == 'bodies':
                            actor.SetVisibility(False)
                        elif elem_type == 'edges' or elem_type == 'vertices':
                            actor.SetVisibility(True)
                    elif render_mode == "surface-wireframe":
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
        
        if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
            if render_mode == "wireframe":
                self.gui.geometry_edges_actor.SetVisibility(True)
            elif render_mode == "surface-wireframe":
                self.gui.geometry_edges_actor.SetVisibility(True)
            else:
                self.gui.geometry_edges_actor.SetVisibility(False)
        
        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
            self.gui.mesh_display.render_window.Render()

    def _update_mesh_part_display(self):
        """更新网格部件的显示"""
        if not hasattr(self.gui, 'model_tree_widget') or not hasattr(self.gui, 'current_mesh'):
            return

        self.refresh_display_all_parts()

    def _update_geometry_display_for_parts(self, visible_parts):
        """根据可见部件更新几何元素的显示"""
        if not hasattr(self.gui, 'model_tree_widget') or not hasattr(self.gui, 'current_geometry'):
            return

        part_geometry_elements = {}
        if hasattr(self.gui, 'cas_parts_info') and self.gui.cas_parts_info:
            for part_name, part_data in self.gui.cas_parts_info.items():
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

        # 获取当前几何元素的可视性
        visible_elements = self.gui.model_tree_widget.get_visible_elements(category='geometry')

        # 移除所有几何元素actor
        if hasattr(self.gui, 'geometry_actors'):
            for elem_type, actors in self.gui.geometry_actors.items():
                for actor in actors:
                    if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                        self.gui.mesh_display.renderer.RemoveActor(actor)

        if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_actor)
            self.gui.geometry_actor = None

        if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                self.gui.mesh_display.renderer.RemoveActor(self.gui.geometry_edges_actor)
            self.gui.geometry_edges_actor = None

        self.gui.geometry_actors = {}

        from fileIO.occ_to_vtk import create_vertex_actor, create_edge_actor, create_face_actor, create_solid_actor

        # 只显示属于可见部件的几何元素
        if 'geometry' in visible_elements:
            if 'vertices' in visible_elements['geometry']:
                self.gui.geometry_actors['vertices'] = []
                for elem_index, elem_data in visible_elements['geometry']['vertices']:
                    if visible_geometry_indices['vertices'] and elem_index in visible_geometry_indices['vertices']:
                        actor = create_vertex_actor(elem_data, color=(1.0, 0.0, 0.0), point_size=8.0)
                        self.gui.geometry_actors['vertices'].append(actor)
                        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                            self.gui.mesh_display.renderer.AddActor(actor)

            if 'edges' in visible_elements['geometry']:
                self.gui.geometry_actors['edges'] = []
                for elem_index, elem_data in visible_elements['geometry']['edges']:
                    if visible_geometry_indices['edges'] and elem_index in visible_geometry_indices['edges']:
                        actor = create_edge_actor(elem_data, color=(0.0, 0.0, 1.0), line_width=2.0)
                        self.gui.geometry_actors['edges'].append(actor)
                        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                            self.gui.mesh_display.renderer.AddActor(actor)

            if 'faces' in visible_elements['geometry']:
                self.gui.geometry_actors['faces'] = []
                for elem_index, elem_data in visible_elements['geometry']['faces']:
                    if visible_geometry_indices['faces'] and elem_index in visible_geometry_indices['faces']:
                        actor = create_face_actor(elem_data, color=(0.0, 1.0, 0.0), opacity=0.6)
                        self.gui.geometry_actors['faces'].append(actor)
                        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                            self.gui.mesh_display.renderer.AddActor(actor)

            if 'bodies' in visible_elements['geometry']:
                self.gui.geometry_actors['bodies'] = []
                for elem_index, elem_data in visible_elements['geometry']['bodies']:
                    if visible_geometry_indices['bodies'] and elem_index in visible_geometry_indices['bodies']:
                        actor = create_solid_actor(elem_data, color=(0.8, 0.8, 0.9), opacity=0.5)
                        self.gui.geometry_actors['bodies'].append(actor)
                        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                            self.gui.mesh_display.renderer.AddActor(actor)

        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
            self.gui.mesh_display.render_window.Render()

    def on_geometry_element_selected(self, element_type, element_data, element_index):
        """几何元素被选中时的回调"""
        element_name = f"{element_type}_{element_index}"
        self.gui.log_info(f"选中: {element_name}")

        if hasattr(self.gui, 'props_text'):
            info_text = f"选中元素: {element_name}\n"
            info_text += f"类型: {element_type}\n"
            info_text += f"索引: {element_index}\n"

            if element_type == "vertex":
                from OCC.Core.BRep import BRep_Tool
                pnt = BRep_Tool.Pnt(element_data)
                info_text += f"坐标: ({pnt.X():.3f}, {pnt.Y():.3f}, {pnt.Z():.3f})\n"
            elif element_type == "edge":
                edge_length = self._get_edge_length(element_data)
                if edge_length is not None:
                    info_text += f"长度: {edge_length:.3f}\n"
            elif element_type == "face":
                face_area = self._get_face_area(element_data)
                if face_area is not None:
                    info_text += f"面积: {face_area:.3f}\n"
            elif element_type == "body":
                body_volume = self._get_solid_volume(element_data)
                if body_volume is not None:
                    info_text += f"体积: {body_volume:.3f}\n"

            self.gui.props_text.setPlainText(info_text)

    def on_model_tree_visibility_changed(self, *args):
        """模型树可见性改变的回调"""
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
        """获取边的长度"""
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
            from OCC.Core.GCPnts import GCPnts_AbscissaPoint
            adaptor = BRepAdaptor_Curve(edge)
            length = GCPnts_AbscissaPoint.Length(adaptor)
            return length
        except Exception:
            return None

    def _get_face_area(self, face):
        """获取面的面积"""
        try:
            from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
            from OCC.Core.GProp import GProp_GProps
            props = GProp_GProps()
            brepgprop_SurfaceProperties(face, props)
            return props.Mass()
        except Exception:
            return None

    def _get_solid_volume(self, solid):
        """获取实体的体积"""
        try:
            from OCC.Core.BRepGProp import brepgprop_VolumeProperties
            from OCC.Core.GProp import GProp_GProps
            props = GProp_GProps()
            brepgprop_VolumeProperties(solid, props)
            return props.Mass()
        except Exception:
            return None

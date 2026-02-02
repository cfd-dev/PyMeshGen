from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, QTimer

from gui.ui_utils import PARTS_INFO_RESERVED_KEYS


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

    def remove_part(self, part_name=None):
        """删除部件
        
        Args:
            part_name: 要删除的部件名称，如果为None则从部件列表获取当前选中的部件
        """
        # 如果没有提供部件名称，则从部件列表获取当前选中的部件
        if part_name is None:
            if hasattr(self.gui, 'parts_list_widget'):
                part_name = self.gui.parts_list_widget.get_selected_part_name()
            else:
                QMessageBox.warning(self.gui, "警告", "部件列表组件未初始化")
                return

        if not part_name:
            QMessageBox.warning(self.gui, "警告", "请先选择要删除的部件")
            return

        # 确认删除
        reply = QMessageBox.question(
            self.gui,
            "确认删除",
            f"确定要删除部件 '{part_name}' 吗？\n\n删除后，该部件中的所有元素将被移回默认部件。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.No:
            self.gui.log_info("已取消删除部件")
            self.gui.update_status("已取消删除部件")
            return

        # 检查部件是否存在
        if not hasattr(self.gui, 'cas_parts_info') or not self.gui.cas_parts_info:
            QMessageBox.warning(self.gui, "警告", "未找到部件信息")
            return

        if part_name not in self.gui.cas_parts_info:
            QMessageBox.warning(self.gui, "警告", f"未找到部件 '{part_name}'")
            return

        # 获取要删除的部件数据
        part_to_delete = self.gui.cas_parts_info[part_name]

        # 确保DefaultPart存在，如果不存在则创建
        if 'DefaultPart' not in self.gui.cas_parts_info:
            self.gui.cas_parts_info['DefaultPart'] = {
                'type': 'default',
                'bc_type': 'wall',
                'geometry_elements': {'vertices': [], 'edges': [], 'faces': [], 'bodies': []},
                'mesh_elements': {'vertices': [], 'edges': [], 'faces': [], 'bodies': []}
            }
            self.gui.log_info("已创建默认部件 DefaultPart")

            # 在部件列表中添加 DefaultPart（如果部件列表存在）
            if hasattr(self.gui, 'parts_list_widget'):
                # 检查 DefaultPart 是否已经在部件列表中
                parts_list = self.gui.parts_list_widget.parts_list
                default_part_exists = False
                for i in range(parts_list.count()):
                    item = parts_list.item(i)
                    if item and item.text() == 'DefaultPart':
                        default_part_exists = True
                        break

                # 如果不存在，则添加
                if not default_part_exists:
                    self.gui.parts_list_widget.add_part_with_checkbox('DefaultPart', True)

        # 将元素移回DefaultPart
        default_part = self.gui.cas_parts_info['DefaultPart']

        # 移回几何元素
        if 'geometry_elements' in part_to_delete:
            if 'geometry_elements' not in default_part:
                default_part['geometry_elements'] = {'vertices': [], 'edges': [], 'faces': [], 'bodies': []}
            self._merge_elements(default_part['geometry_elements'], part_to_delete['geometry_elements'])

        # 移回网格元素
        if 'mesh_elements' in part_to_delete:
            if 'mesh_elements' not in default_part:
                default_part['mesh_elements'] = {'vertices': [], 'edges': [], 'faces': [], 'bodies': []}
            self._merge_elements(default_part['mesh_elements'], part_to_delete['mesh_elements'])

        self.gui.log_info(f"已将部件 '{part_name}' 的元素移回默认部件")

        # 从部件信息中删除该部件
        del self.gui.cas_parts_info[part_name]

        # 从部件参数中删除该部件（如果存在）
        if hasattr(self.gui, 'parts_params') and self.gui.parts_params:
            self.gui.parts_params = [p for p in self.gui.parts_params if p.get('part_name') != part_name]

        # 从部件列表中删除该部件
        if hasattr(self.gui, 'parts_list_widget'):
            parts_list = self.gui.parts_list_widget.parts_list
            for i in range(parts_list.count()):
                item = parts_list.item(i)
                if item and item.text() == part_name:
                    parts_list.takeItem(i)
                    break

        # 更新模型树
        if hasattr(self.gui, 'model_tree_widget'):
            self.gui.model_tree_widget.load_parts(self.gui.cas_parts_info)

        # 刷新显示
        self.refresh_display_all_parts()

        self.gui.log_info(f"已成功删除部件: {part_name}")
        self.gui.update_status(f"部件已删除: {part_name}")

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

    def edit_mesh_params_for_part(self, part_name):
        """编辑指定部件的参数"""
        from PyQt5.QtWidgets import QDialog
        from gui.part_params_dialog import PartParamsDialog

        if not hasattr(self.gui, 'cas_parts_info') or not self.gui.cas_parts_info:
            QMessageBox.warning(self.gui, "警告", "请先导入网格文件以获取部件列表")
            self.gui.log_info("未检测到导入的网格数据，无法设置部件参数")
            self.gui.update_status("未检测到导入的网格数据")
            return

        current_part_names = []
        if isinstance(self.gui.cas_parts_info, dict):
            current_part_names = list(self.gui.cas_parts_info.keys())
        elif isinstance(self.gui.cas_parts_info, list):
            current_part_names = [part_info.get('part_name', f'部件{self.gui.cas_parts_info.index(part_info)}') for part_info in self.gui.cas_parts_info]

        if part_name not in current_part_names:
            self.gui.log_info(f"未找到部件: {part_name}")
            self.gui.update_status("未找到部件")
            return

        saved_params_map = {}
        if hasattr(self.gui, 'parts_params') and self.gui.parts_params:
            for param in self.gui.parts_params:
                if 'part_name' in param:
                    saved_params_map[param['part_name']] = param

        parts_params = []
        for name in current_part_names:
            if name in saved_params_map:
                parts_params.append(saved_params_map[name])
            else:
                parts_params.append({
                    "part_name": name,
                    "max_size": 1e6,
                    "PRISM_SWITCH": "off",
                    "first_height": 0.01,
                    "growth_rate": 1.2,
                    "max_layers": 5,
                    "full_layers": 5,
                    "multi_direction": False
                })

        current_index = current_part_names.index(part_name)
        dialog = PartParamsDialog(self.gui, parts=parts_params, current_part=current_index)
        if dialog.exec_() == QDialog.Accepted:
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

        # 更新部件列表（部件列表组件）
        if hasattr(self.gui, 'parts_list_widget'):
            parts_list = self.gui.parts_list_widget.parts_list

            # 获取当前部件列表中的所有部件名称
            existing_parts = set()
            for i in range(parts_list.count()):
                item = parts_list.item(i)
                if item:
                    existing_parts.add(item.text())

            # 获取需要显示的部件（包括DefaultPart）
            if isinstance(self.gui.cas_parts_info, dict):
                for part_name in self.gui.cas_parts_info.keys():
                    if part_name not in existing_parts:
                        self.gui.parts_list_widget.add_part_with_checkbox(part_name, True)
                        existing_parts.add(part_name)

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

        # 更新部件列表（部件列表组件）
        if hasattr(self.gui, 'parts_list_widget'):
            parts_list = self.gui.parts_list_widget.parts_list

            # 获取当前部件列表中的所有部件名称
            existing_parts = set()
            for i in range(parts_list.count()):
                item = parts_list.item(i)
                if item:
                    existing_parts.add(item.text())

            # 获取需要显示的部件（包括DefaultPart）
            if isinstance(self.gui.cas_parts_info, dict):
                for part_name in self.gui.cas_parts_info.keys():
                    if part_name not in existing_parts:
                        self.gui.parts_list_widget.add_part_with_checkbox(part_name, True)
                        existing_parts.add(part_name)

    def update_parts_list_from_generated_mesh(self, generated_mesh):
        """从生成的网格更新部件列表"""
        try:
            updated_parts_info = {}

            new_node_coords = None
            if hasattr(generated_mesh, 'node_coords'):
                new_node_coords = generated_mesh.node_coords

            # FIXME 对于初始边界重新离散了的情况，如含有match边界，保存的original_coords是旧的，而不是重新离散后的，导致映射后仍然是乱序的
            node_mapping = None
            if self.gui.original_node_coords and new_node_coords:
                node_mapping = self.gui._create_node_index_mapping(
                    self.gui.original_node_coords,
                    new_node_coords
                )
                if node_mapping:
                    self.gui.log_info(f"已创建节点索引映射，映射了 {len(node_mapping)} 个节点")

            if hasattr(self.gui, 'cas_parts_info') and self.gui.cas_parts_info:
                if node_mapping:
                    mapped_parts_info = self.gui._map_parts_info_to_new_mesh(
                        self.gui.cas_parts_info,
                        node_mapping
                    )
                    for part_name, part_data in mapped_parts_info.items():
                        if part_name not in PARTS_INFO_RESERVED_KEYS:
                            updated_parts_info[part_name] = part_data
                else:
                    for part_name, part_data in self.gui.cas_parts_info.items():
                        if part_name not in PARTS_INFO_RESERVED_KEYS:
                            updated_parts_info[part_name] = part_data

            if hasattr(generated_mesh, 'cell_container') and generated_mesh.cell_container:
                for cell in generated_mesh.cell_container:
                    part_name = getattr(cell, 'part_name', 'interior')
                    if part_name is None or part_name == '':
                        part_name = 'interior'

                    if part_name not in updated_parts_info:
                        updated_parts_info[part_name] = {
                            'part_name': part_name,
                            'bc_type': 'interior',
                            'node_count': 0,
                            'nodes': [],
                            'faces': []
                        }

                    if part_name in updated_parts_info:
                        updated_parts_info[part_name]['node_count'] += 1
                        if hasattr(cell, 'node_ids') and cell.node_ids:
                            updated_parts_info[part_name]['faces'].append({'nodes': cell.node_ids})

            if hasattr(generated_mesh, 'parts_info') and generated_mesh.parts_info:
                for part_name, part_data in generated_mesh.parts_info.items():
                    if part_name not in PARTS_INFO_RESERVED_KEYS:
                        if part_name not in updated_parts_info:
                            updated_parts_info[part_name] = part_data
                        else:
                            if isinstance(part_data, dict):
                                updated_parts_info[part_name].update(part_data)

            if hasattr(generated_mesh, 'boundary_nodes') and generated_mesh.boundary_nodes:
                extracted_parts = self.gui._extract_parts_from_boundary_nodes(generated_mesh.boundary_nodes)
                if extracted_parts:
                    for part_name, part_data in extracted_parts.items():
                        if (part_name not in updated_parts_info):
                            updated_parts_info[part_name] = part_data

            if updated_parts_info:
                self.update_parts_list_from_cas(parts_info=updated_parts_info, update_status=False)
                self.gui.log_info(f"已更新部件列表，检测到 {len(updated_parts_info)} 个部件: {list(updated_parts_info.keys())}")

                self.refresh_display_all_parts()
            else:
                self.gui.log_info("生成的网格中未检测到部件信息，保持原有部件列表")

        except Exception as e:
            self.gui.log_error(f"更新部件列表失败: {str(e)}")

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

    def show_selected_part(self):
        """显示选中部件（从右键菜单调用）"""
        QMessageBox.information(self.gui, "提示", "显示选中部件功能请使用模型树复选框")

    def show_only_selected_part(self):
        """只显示选中部件（从右键菜单调用）"""
        QMessageBox.information(self.gui, "提示", "只显示选中部件功能请使用模型树复选框")

    def show_all_parts(self):
        """显示所有部件（从右键菜单调用）"""
        self.gui.log_info("显示所有部件")
        self.gui.update_status("显示所有部件")

        if hasattr(self.gui, 'model_tree_widget'):
            self.gui.model_tree_widget.set_all_parts_visible(True)
            self.refresh_display_all_parts()
            self.gui.log_info("成功显示所有部件")
        else:
            self.gui.log_info("模型树组件未初始化")

    def handle_part_visibility_change(self, item_or_name, is_visible=None):
        """处理部件可见性改变"""
        if isinstance(item_or_name, str):
            part_name = item_or_name.split(' - ')[0] if ' - ' in item_or_name else item_or_name
            if is_visible is None:
                return
            if not hasattr(self.gui, 'mesh_display') or not self.gui.mesh_display:
                return
            self.refresh_display_all_parts()
            action = "显示" if is_visible else "隐藏"
            self.gui.log_info(f"{action}部件: {part_name}")
            return

        item = item_or_name
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

    def on_part_created(self, part_info_or_name, geometry_elements=None, mesh_elements=None):
        """部件创建事件处理"""
        skip_geometry_check = False
        skip_empty_check = False

        if isinstance(part_info_or_name, dict):
            part_info = part_info_or_name
            if not part_info:
                return
            part_name = part_info.get('part_name', '新部件')
            geometry_elements = part_info.get('geometry_elements', {}) or {}
            mesh_elements = part_info.get('mesh_elements', {}) or {}
            skip_geometry_check = True
            skip_empty_check = True
        else:
            part_name = part_info_or_name

        if not hasattr(self.gui, 'cas_parts_info'):
            self.gui.cas_parts_info = {}

        if not skip_geometry_check:
            if not hasattr(self.gui, 'current_geometry') or not self.gui.current_geometry:
                QMessageBox.warning(self.gui, "警告", "请先导入几何模型")
                self.gui.log_info("未找到几何模型，无法创建部件")
                self.gui.update_status("未找到几何模型")
                return

        if not skip_empty_check:
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

        if hasattr(self.gui, 'model_tree_widget'):
            self.gui.model_tree_widget.load_parts(self.gui.cas_parts_info)

        self.gui.log_info(f"已创建部件: {part_name}")
        self.gui.update_status(f"部件已创建: {part_name}")

        self.refresh_display_all_parts()

    def on_part_elements_added(self, part_info):
        """向已有部件添加元素"""
        if not isinstance(part_info, dict):
            return
        part_name = part_info.get('part_name')
        if not part_name:
            return
        if not hasattr(self.gui, 'cas_parts_info') or not self.gui.cas_parts_info:
            self.gui.log_info("未找到部件信息，无法添加元素")
            self.gui.update_status("未找到部件信息")
            return
        if part_name not in self.gui.cas_parts_info:
            self.gui.log_info(f"未找到部件 {part_name}")
            self.gui.update_status("未找到部件")
            return

        geometry_elements = part_info.get('geometry_elements', {}) or {}
        mesh_elements = part_info.get('mesh_elements', {}) or {}
        if not geometry_elements and not mesh_elements:
            self.gui.log_info("未选择任何元素")
            self.gui.update_status("未选择任何元素")
            return

        target_part = self.gui.cas_parts_info.get(part_name, {})
        target_part.setdefault('geometry_elements', {'vertices': [], 'edges': [], 'faces': [], 'bodies': []})
        target_part.setdefault('mesh_elements', {'vertices': [], 'edges': [], 'faces': [], 'bodies': []})

        self._merge_elements(target_part['geometry_elements'], geometry_elements)
        self._merge_elements(target_part['mesh_elements'], mesh_elements)

        if 'DefaultPart' in self.gui.cas_parts_info:
            default_part = self.gui.cas_parts_info['DefaultPart']
            if 'mesh_elements' in default_part:
                self._remove_elements(default_part['mesh_elements'], mesh_elements)
            if 'geometry_elements' in default_part:
                self._remove_elements(default_part['geometry_elements'], geometry_elements)

        self.gui.cas_parts_info[part_name] = target_part

        if hasattr(self.gui, 'model_tree_widget'):
            self.gui.model_tree_widget.load_parts(self.gui.cas_parts_info)

        self.gui.log_info(f"已向部件 {part_name} 添加元素")
        self.gui.update_status(f"部件已更新: {part_name}")
        self.refresh_display_all_parts()

    def _merge_elements(self, target, incoming):
        for key, items in incoming.items():
            if key not in target:
                target[key] = []
            for idx in items:
                if idx not in target[key]:
                    target[key].append(idx)

    def _remove_elements(self, target, incoming):
        for key, items in incoming.items():
            if key not in target:
                continue
            for idx in items:
                if idx in target[key]:
                    target[key].remove(idx)

    def refresh_display_all_parts(self):
        """刷新显示所有可见部件 - 优化版本，批量处理减少渲染次数"""
        if not hasattr(self.gui, 'mesh_display') or not self.gui.mesh_display:
            return

        # Clear current display
        self.gui.mesh_display.clear_mesh_actors()

        # Get all currently checked parts from geometry tree widget
        visible_parts = []
        parts_visible = True
        mesh_visible = True
        if hasattr(self.gui, 'model_tree_widget'):
            visible_parts = self.gui.model_tree_widget.get_visible_parts()
            parts_visible = self.gui.model_tree_widget.is_category_visible('parts')
            mesh_visible = self.gui.model_tree_widget.is_category_visible('mesh')
        elif hasattr(self.gui, 'parts_list_widget'):
            for i in range(self.gui.parts_list_widget.parts_list.count()):
                item = self.gui.parts_list_widget.parts_list.item(i)
                part_name = item.text().split(' - ')[0] if ' - ' in item.text() else item.text()
                if item.checkState() == Qt.Checked:
                    visible_parts.append(part_name)

        # Display only visible parts - 批量处理以提高性能
        if mesh_visible and parts_visible and visible_parts:
            # 批量显示所有部件，只渲染一次
            for part_name in visible_parts:
                self.gui.mesh_display.display_part(part_name, parts_info=self.gui.cas_parts_info, render_immediately=False)

            # 批量渲染一次，而不是每个部件都渲染
            self.gui.mesh_display.render_window.Render()
        else:
            # If no parts are visible, clear display
            self.gui.mesh_display.clear()

        # 刷新线网格显示（根据可见部件列表）
        self.gui.mesh_display.refresh_line_mesh_display(visible_parts)

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

    def on_mesh_element_selected(self, element_type, element_data, element_index):
        """网格元素被选中时的回调"""
        element_name = f"{element_type}_{element_index}"
        self.gui.log_info(f"选中网格元素: {element_name}")

        if hasattr(self.gui, 'props_text'):
            info_text = f"选中网格元素: {element_name}\n"
            info_text += f"类型: {element_type}\n"
            info_text += f"索引: {element_index}\n"

            if isinstance(element_data, dict):
                if 'nodes' in element_data:
                    info_text += f"节点数量: {len(element_data['nodes'])}\n"
                if 'faces' in element_data:
                    info_text += f"面数量: {len(element_data['faces'])}\n"
                if 'edges' in element_data:
                    info_text += f"边数量: {len(element_data['edges'])}\n"

            self.gui.props_text.setPlainText(info_text)

    def on_part_selected(self, element_type, element_data, element_index):
        """部件被选中时的回调"""
        part_name = f"{element_type}_{element_index}"
        self.gui.log_info(f"选中部件: {part_name}")

        if hasattr(self.gui, 'props_text'):
            info_text = f"选中部件: {part_name}\n"
            info_text += f"索引: {element_index}\n"

            if isinstance(element_data, dict):
                bc_type = element_data.get('bc_type', '未知')
                info_text += f"边界条件: {bc_type}\n"

                if 'faces' in element_data:
                    info_text += f"面数量: {len(element_data['faces'])}\n"

                if 'nodes' in element_data:
                    info_text += f"节点数量: {len(element_data['nodes'])}\n"

            self.gui.props_text.setPlainText(info_text)
            self.gui.update_status(f"已选中网格部件: {part_name}")

    def on_model_tree_selected(self, category, element_type, element_index, element_obj):
        """模型树元素被选中的回调"""
        if category == 'geometry':
            self.on_geometry_element_selected(element_type, element_obj, element_index)
        elif category == 'mesh':
            self.on_mesh_element_selected(element_type, element_obj, element_index)
        elif category == 'parts':
            self.on_part_selected(element_type, element_obj, element_index)

    def _get_geometry_type_states(self):
        """获取几何类型的勾选状态"""
        if hasattr(self.gui, 'model_tree_widget') and hasattr(self.gui.model_tree_widget, 'get_geometry_type_states'):
            return self.gui.model_tree_widget.get_geometry_type_states()
        return {'vertices': True, 'edges': True, 'faces': True, 'bodies': True}

    def _should_show_geometry_for_parts(self, visible_parts=None):
        """根据部件勾选状态判断是否显示几何"""
        if visible_parts is None:
            if not hasattr(self.gui, 'model_tree_widget'):
                return True
            parts_item = self.gui.model_tree_widget.tree.topLevelItem(2)
            if parts_item is None or parts_item.childCount() == 0:
                return True
            visible_parts = self.gui.model_tree_widget.get_visible_parts()

        if not visible_parts:
            return False

        if not hasattr(self.gui, 'cas_parts_info') or not isinstance(self.gui.cas_parts_info, dict):
            return True

        for part_name in visible_parts:
            part_data = self.gui.cas_parts_info.get(part_name)
            if isinstance(part_data, dict) and 'geometry_elements' in part_data:
                return True

        return False

    def _get_geometry_visibility_state(self, visible_parts=None, render_mode=None):
        """计算几何显示状态，统一供不同显示路径使用"""
        type_states = self._get_geometry_type_states()
        parts_visible = self._should_show_geometry_for_parts(visible_parts)
        render_mode = render_mode or getattr(self.gui, 'render_mode', 'surface')

        stats = getattr(self.gui, 'current_geometry_stats', None)
        face_count = stats.get('num_faces', 0) if stats else 0
        solid_count = stats.get('num_solids', 0) if stats else 0

        faces_checked = type_states.get('faces', True)
        bodies_checked = type_states.get('bodies', False)
        edges_checked = type_states.get('edges', False)
        vertices_checked = type_states.get('vertices', False)

        has_surface = (face_count > 0) or (solid_count > 0)
        if face_count > 0:
            surface_checked = faces_checked
        elif solid_count > 0:
            surface_checked = bodies_checked
        else:
            surface_checked = False

        if render_mode == "wireframe":
            show_faces = False
            show_edges = edges_checked and parts_visible
            show_vertices = vertices_checked and parts_visible
        elif render_mode == "surface-wireframe":
            show_faces = surface_checked and parts_visible
            show_edges = edges_checked and parts_visible
            show_vertices = vertices_checked and parts_visible
        else:
            show_faces = surface_checked and parts_visible
            if not has_surface:
                show_edges = edges_checked and parts_visible
                show_vertices = vertices_checked and parts_visible
            else:
                show_edges = False
                show_vertices = False

        return {
            'render_mode': render_mode,
            'parts_visible': parts_visible,
            'show_faces': show_faces,
            'show_edges': show_edges,
            'show_vertices': show_vertices
        }

    def _update_stl_geometry_visibility(self, visible_parts=None):
        """更新STL几何的可见性"""
        geometry_actor = getattr(self.gui, 'geometry_actor', None)
        edges_actor = getattr(self.gui, 'geometry_edges_actor', None)
        if not geometry_actor and not edges_actor:
            return

        state = self._get_geometry_visibility_state(visible_parts=visible_parts)
        show_faces = state['show_faces']
        show_edges = state['show_edges']

        if geometry_actor:
            geometry_actor.SetVisibility(show_faces)
        if edges_actor:
            edges_actor.SetVisibility(show_edges)

        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
            self.gui.mesh_display.render_window.Render()

    def _update_geometry_element_display(self):
        """更新几何元素的显示"""
        if not hasattr(self.gui, 'model_tree_widget') or not hasattr(self.gui, 'current_geometry'):
            return
        if self.gui.current_geometry is None:
            if hasattr(self.gui, '_geometry_cache_shape_id') and self.gui._geometry_cache_shape_id is not None:
                self.cleanup_geometry_actors()
                self.gui._geometry_cache_shape_id = None
            return
        shape_id = id(self.gui.current_geometry)
        if getattr(self.gui, '_geometry_cache_shape_id', None) != shape_id:
            self.cleanup_geometry_actors()
            self.gui._geometry_cache_shape_id = shape_id
        if not self.gui.model_tree_widget.is_category_visible('geometry'):
            self._hide_geometry_element_actors()
            if hasattr(self.gui, 'view_controller'):
                self.gui.view_controller._apply_render_mode_to_geometry(getattr(self.gui, 'render_mode', 'surface'))
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
                self.gui.mesh_display.render_window.Render()
            return

        if getattr(self.gui, 'geometry_display_source', None) == 'stl':
            self._update_stl_geometry_visibility()
            return
        display_mode = getattr(self.gui, 'display_mode', 'full')
        if display_mode != 'elements':
            if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
                self.gui.geometry_actor.SetVisibility(True)
            if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
                self.gui.geometry_edges_actor.SetVisibility(True)
            if hasattr(self.gui, 'geometry_points_actor') and self.gui.geometry_points_actor:
                self.gui.geometry_points_actor.SetVisibility(True)
            self._hide_geometry_element_actors()
            render_mode = getattr(self.gui, 'render_mode', 'surface')
            if hasattr(self.gui, 'view_controller'):
                self.gui.view_controller._apply_render_mode_to_geometry(render_mode)
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
                self.gui.mesh_display.render_window.Render()
            return

        # 初始化几何actors缓存（如果尚未初始化）
        if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
            self.gui.geometry_actor.SetVisibility(False)
        if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
            self.gui.geometry_edges_actor.SetVisibility(False)
        if hasattr(self.gui, 'geometry_points_actor') and self.gui.geometry_points_actor:
            self.gui.geometry_points_actor.SetVisibility(False)
        if not hasattr(self.gui, 'geometry_actors_cache'):
            self.gui.geometry_actors_cache = {}

        # 获取可见元素
        visible_elements = self.gui.model_tree_widget.get_visible_elements(category='geometry')
        visible_parts = self.gui.model_tree_widget.get_visible_parts()
        geometry_visible = self.gui.model_tree_widget.is_category_visible('geometry')

        # 从模型树获取所有几何元素（不仅仅是可见的）
        all_elements = self._get_all_geometry_elements_from_tree()

        # 更新actors缓存，确保每个元素都有对应的actor
        self._update_actors_cache(all_elements)

        # 根据可见部件过滤元素索引
        allowed_indices = None
        part_geometry_elements = {}
        if hasattr(self.gui, 'cas_parts_info') and self.gui.cas_parts_info:
            for part_name, part_data in self.gui.cas_parts_info.items():
                if isinstance(part_data, dict) and 'geometry_elements' in part_data:
                    part_geometry_elements[part_name] = part_data['geometry_elements']

        if geometry_visible and part_geometry_elements and not (len(part_geometry_elements) == 1 and 'DefaultPart' in part_geometry_elements and len(visible_parts) == len(part_geometry_elements)):
            allowed_indices = {'vertices': set(), 'edges': set(), 'faces': set(), 'bodies': set()}
            for part_name in visible_parts:
                if part_name in part_geometry_elements:
                    for elem_type, indices in part_geometry_elements[part_name].items():
                        if elem_type in allowed_indices:
                            allowed_indices[elem_type].update(indices)

        # 设置actors的可见性
        type_visibility = self._get_element_type_visibility(visible_parts=visible_parts)
        self._set_actors_visibility(visible_elements, allowed_indices=allowed_indices, type_visibility=type_visibility)

        # 应用渲染模式
        render_mode = getattr(self.gui, 'render_mode', 'surface')
        if hasattr(self.gui, 'view_controller'):
            self.gui.view_controller._apply_render_mode_to_geometry(render_mode)

        # 刷新渲染窗口
        if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
            self.gui.mesh_display.render_window.Render()

    def _get_all_geometry_elements_from_tree(self):
        """从模型树获取所有几何元素"""
        all_elements = {'vertices': [], 'edges': [], 'faces': [], 'bodies': []}

        geometry_item = None
        for i in range(self.gui.model_tree_widget.tree.topLevelItemCount()):
            item = self.gui.model_tree_widget.tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == 'geometry':
                geometry_item = item
                break

        if geometry_item:
            for j in range(geometry_item.childCount()):
                element_type_item = geometry_item.child(j)
                element_type_data = element_type_item.data(0, Qt.UserRole)

                if isinstance(element_type_data, tuple) and len(element_type_data) >= 2:
                    elem_category, elem_type = element_type_data[0], element_type_data[1]

                    if elem_type in all_elements:
                        for k in range(element_type_item.childCount()):
                            element_item = element_type_item.child(k)
                            element_data = element_item.data(0, Qt.UserRole)

                            if isinstance(element_data, tuple) and len(element_data) >= 4:
                                elem_category_data, elem_type_data, elem_obj, elem_index = element_data
                                if elem_type_data == elem_type:
                                    all_elements[elem_type].append((elem_index, elem_obj))

        return all_elements

    def _update_actors_cache(self, all_elements):
        """更新actors缓存，为新元素创建actors"""
        from fileIO.occ_to_vtk import create_vertex_actor, create_edge_actor, create_face_actor, create_solid_actor

        # 为每种类型的元素更新actors缓存
        for elem_type, elements in all_elements.items():
            if elem_type not in self.gui.geometry_actors_cache:
                self.gui.geometry_actors_cache[elem_type] = {}

            # 检查是否有新元素需要创建actor
            for elem_index, elem_obj in elements:
                if elem_index not in self.gui.geometry_actors_cache[elem_type]:
                    # 创建新的actor
                    if elem_type == 'vertices':
                        actor = create_vertex_actor(elem_obj, color=(1.0, 0.0, 0.0), point_size=8.0)
                    elif elem_type == 'edges':
                        actor = create_edge_actor(elem_obj, color=(0.0, 0.0, 1.0), line_width=2.0)
                    elif elem_type == 'faces':
                        actor = create_face_actor(elem_obj, color=(0.0, 1.0, 0.0), opacity=0.6)
                    elif elem_type == 'bodies':
                        actor = create_solid_actor(elem_obj, color=(0.8, 0.8, 0.9), opacity=0.5)
                    else:
                        continue

                    # 默认设为不可见，避免与整体显示重叠
                    actor.SetVisibility(False)

                    # 添加到渲染器
                    if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                        self.gui.mesh_display.renderer.AddActor(actor)

                    # 缓存actor
                    self.gui.geometry_actors_cache[elem_type][elem_index] = actor

    def _set_actors_visibility(self, visible_elements, allowed_indices=None, type_visibility=None):
        """设置actors的可见性"""
        # 首先将所有actors设为不可见
        for elem_type, actors_dict in self.gui.geometry_actors_cache.items():
            for actor in actors_dict.values():
                actor.SetVisibility(False)

        # 然后将可见的actors设为可见
        if 'geometry' in visible_elements:
            for elem_type, elements in visible_elements['geometry'].items():
                if type_visibility is not None and not type_visibility.get(elem_type, False):
                    continue
                if elem_type in self.gui.geometry_actors_cache:
                    for elem_index, elem_obj in elements:
                        if allowed_indices is not None and elem_index not in allowed_indices.get(elem_type, set()):
                            continue
                        if elem_index in self.gui.geometry_actors_cache[elem_type]:
                            actor = self.gui.geometry_actors_cache[elem_type][elem_index]
                            actor.SetVisibility(True)

    def _hide_geometry_element_actors(self):
        """隐藏几何元素actors"""
        if not hasattr(self.gui, 'geometry_actors_cache'):
            return
        for actors_dict in self.gui.geometry_actors_cache.values():
            for actor in actors_dict.values():
                actor.SetVisibility(False)

    def _get_element_type_visibility(self, visible_parts=None, render_mode=None):
        """获取元素类型可见性（与渲染模式和类型勾选一致）"""
        state = self._get_geometry_visibility_state(visible_parts=visible_parts, render_mode=render_mode)
        return {
            'faces': state['show_faces'],
            'bodies': state['show_faces'],
            'edges': state['show_edges'],
            'vertices': state['show_vertices']
        }

    def cleanup_geometry_actors(self):
        """清理几何actors缓存"""
        if hasattr(self.gui, 'geometry_actors_cache'):
            for elem_type, actors_dict in self.gui.geometry_actors_cache.items():
                for actor in actors_dict.values():
                    if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'renderer'):
                        self.gui.mesh_display.renderer.RemoveActor(actor)
            self.gui.geometry_actors_cache = {}
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
        if not self.gui.model_tree_widget.is_category_visible('geometry'):
            self._hide_geometry_element_actors()
            if hasattr(self.gui, 'view_controller'):
                self.gui.view_controller._apply_render_mode_to_geometry(getattr(self.gui, 'render_mode', 'surface'))
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
                self.gui.mesh_display.render_window.Render()
            return

        if getattr(self.gui, 'geometry_display_source', None) == 'stl':
            self._update_stl_geometry_visibility(visible_parts=visible_parts)
            return
        display_mode = getattr(self.gui, 'display_mode', 'full')
        if display_mode != 'elements':
            self._update_geometry_element_display()
            return

        # 检查是否存在部件几何元素映射
        part_geometry_elements = {}
        if hasattr(self.gui, 'cas_parts_info') and self.gui.cas_parts_info:
            for part_name, part_data in self.gui.cas_parts_info.items():
                if isinstance(part_data, dict) and 'geometry_elements' in part_data:
                    part_geometry_elements[part_name] = part_data['geometry_elements']

        # 没有可见部件时，隐藏所有几何元素显示
        if not visible_parts:
            self._hide_geometry_element_actors()
            render_mode = getattr(self.gui, 'render_mode', 'surface')
            if hasattr(self.gui, 'view_controller'):
                self.gui.view_controller._apply_render_mode_to_geometry(render_mode)
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
                self.gui.mesh_display.render_window.Render()
            return

        # 如果没有部件包含几何元素，或者只有DefaultPart且所有部件都可见，则显示所有几何元素
        if not part_geometry_elements or (len(part_geometry_elements) == 1 and 'DefaultPart' in part_geometry_elements and len(visible_parts) == len(part_geometry_elements)):
            # 没有部件几何元素映射，或所有部件都可见，使用标准的几何元素显示
            if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
                self.gui.geometry_actor.SetVisibility(False)
            if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
                self.gui.geometry_edges_actor.SetVisibility(False)
            if hasattr(self.gui, 'geometry_points_actor') and self.gui.geometry_points_actor:
                self.gui.geometry_points_actor.SetVisibility(False)
            if not hasattr(self.gui, 'geometry_actors_cache'):
                self.gui.geometry_actors_cache = {}
            visible_elements = self.gui.model_tree_widget.get_visible_elements(category='geometry')
            all_elements = self._get_all_geometry_elements_from_tree()
            self._update_actors_cache(all_elements)
            type_visibility = self._get_element_type_visibility(visible_parts=visible_parts)
            self._set_actors_visibility(visible_elements, type_visibility=type_visibility)
            if hasattr(self.gui, 'mesh_display') and hasattr(self.gui.mesh_display, 'render_window'):
                self.gui.mesh_display.render_window.Render()
            return

        # 收集所有可见部件的几何元素索引
        visible_geometry_indices = {'vertices': set(), 'edges': set(), 'faces': set(), 'bodies': set()}
        for part_name in visible_parts:
            if part_name in part_geometry_elements:
                for elem_type, indices in part_geometry_elements[part_name].items():
                    if elem_type in visible_geometry_indices:
                        visible_geometry_indices[elem_type].update(indices)

        # 使用缓存的actors机制来设置可见性
        if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
            self.gui.geometry_actor.SetVisibility(False)
        if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
            self.gui.geometry_edges_actor.SetVisibility(False)
        if hasattr(self.gui, 'geometry_points_actor') and self.gui.geometry_points_actor:
            self.gui.geometry_points_actor.SetVisibility(False)
        if not hasattr(self.gui, 'geometry_actors_cache'):
            self.gui.geometry_actors_cache = {}
        visible_elements = self.gui.model_tree_widget.get_visible_elements(category='geometry')
        all_elements = self._get_all_geometry_elements_from_tree()
        self._update_actors_cache(all_elements)
        type_visibility = self._get_element_type_visibility(visible_parts=visible_parts)
        self._set_actors_visibility(visible_elements, allowed_indices=visible_geometry_indices, type_visibility=type_visibility)

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
                pnt = self._get_vertex_point(element_data)
                if pnt is not None:
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

    def on_display_mode_changed(self, mode):
        """全局显示模式切换回调"""
        if hasattr(self.gui, 'view_controller'):
            self.gui.view_controller._apply_render_mode_to_geometry(getattr(self.gui, 'render_mode', 'surface'))
        self._update_geometry_element_display()
        self.refresh_display_all_parts()

        if mode == "elements":
            if hasattr(self.gui, 'geometry_actor') and self.gui.geometry_actor:
                self.gui.geometry_actor.SetVisibility(False)
            if hasattr(self.gui, 'geometry_edges_actor') and self.gui.geometry_edges_actor:
                self.gui.geometry_edges_actor.SetVisibility(False)
            if hasattr(self.gui, 'geometry_points_actor') and self.gui.geometry_points_actor:
                self.gui.geometry_points_actor.SetVisibility(False)
            if hasattr(self.gui, 'geometry_actors'):
                for actors in self.gui.geometry_actors.values():
                    for actor in actors:
                        actor.SetVisibility(False)

    def _get_vertex_point(self, vertex):
        """获取顶点的坐标"""
        from OCC.Core.BRep import BRep_Tool

        try:
            return BRep_Tool.Pnt(vertex)
        except Exception as e:
            import logging
            logging.error(f"获取顶点坐标失败: {e}")
            return None

    def _get_edge_length(self, edge):
        """获取边的长度"""
        from OCC.Core.GCPnts import GCPnts_AbscissaPoint
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

        try:
            adaptor = BRepAdaptor_Curve(edge)
            return GCPnts_AbscissaPoint.Length(adaptor)
        except Exception as e:
            import logging
            logging.error(f"获取边长度失败: {e}")
            return None

    def _get_face_area(self, face):
        """获取面的面积"""
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        try:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            return props.Mass()
        except:
            return None

    def _get_solid_volume(self, solid):
        """获取实体的体积"""
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        try:
            props = GProp_GProps()
            brepgprop.VolumeProperties(solid, props)
            return props.Mass()
        except:
            return None

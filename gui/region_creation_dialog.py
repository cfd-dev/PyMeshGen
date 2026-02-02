# -*- coding: utf-8 -*-
"""
区域创建对话框
支持选择多条Connector形成封闭区域，显示方向并支持翻转
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QListWidget, QListWidgetItem, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush
import numpy as np


class RegionCreationDialog(QDialog):
    """区域创建对话框"""
    
    region_created = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("创建区域")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        self.gui = parent
        self.selected_connectors = []
        self.connector_directions = {}
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 说明信息
        info_label = QLabel("选择多条Connector形成封闭区域。Connector的方向即为front_list的方向。")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(info_label)
        
        # Connector选择组
        self.create_connector_selection_group(layout)
        
        # 方向显示和翻转组
        self.create_direction_control_group(layout)
        
        # 区域信息组
        self.create_region_info_group(layout)
        
        # 按钮组
        self.create_button_group(layout)
        
    def create_connector_selection_group(self, parent_layout):
        """创建Connector选择组"""
        group = QGroupBox("Connector选择")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # 获取可用的Connector列表
        self.connector_list_widget = QListWidget()
        self.connector_list_widget.setMaximumHeight(150)
        self.connector_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.connector_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                background: white;
            }
            QListWidget::item {
                padding: 3px;
            }
            QListWidget::item:selected {
                background-color: #0078D4;
                color: white;
            }
        """)
        self.connector_list_widget.itemSelectionChanged.connect(self.on_connector_selection_changed)
        layout.addWidget(self.connector_list_widget)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self.on_select_all_clicked)
        button_layout.addWidget(self.btn_select_all)
        
        self.btn_clear_selection = QPushButton("清除选择")
        self.btn_clear_selection.clicked.connect(self.on_clear_selection_clicked)
        button_layout.addWidget(self.btn_clear_selection)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_direction_control_group(self, parent_layout):
        """创建方向控制组"""
        group = QGroupBox("Connector方向控制")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # 方向显示表格
        self.direction_table = QTableWidget()
        self.direction_table.setColumnCount(4)
        self.direction_table.setHorizontalHeaderLabels(["Connector名称", "起点", "终点", "方向"])
        self.direction_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.direction_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.direction_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.direction_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.direction_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.direction_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ccc;
                background: white;
            }
            QTableWidget::item {
                padding: 3px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 3px;
                border: 1px solid #ccc;
                font-weight: bold;
            }
        """)
        self.direction_table.setMaximumHeight(200)
        layout.addWidget(self.direction_table)
        
        # 翻转按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.btn_flip_selected = QPushButton("翻转选中Connector方向")
        self.btn_flip_selected.clicked.connect(self.on_flip_selected_clicked)
        button_layout.addWidget(self.btn_flip_selected)
        
        self.btn_flip_all = QPushButton("翻转所有Connector方向")
        self.btn_flip_all.clicked.connect(self.on_flip_all_clicked)
        button_layout.addWidget(self.btn_flip_all)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_region_info_group(self, parent_layout):
        """创建区域信息组"""
        group = QGroupBox("区域信息")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # 信息显示
        info_layout = QVBoxLayout()
        
        self.lbl_connector_count = QLabel("已选择Connector数量: 0")
        info_layout.addWidget(self.lbl_connector_count)
        
        self.lbl_region_closed = QLabel("区域状态: 未形成封闭区域")
        self.lbl_region_closed.setStyleSheet("color: red; font-weight: bold;")
        info_layout.addWidget(self.lbl_region_closed)
        
        self.lbl_total_fronts = QLabel("总Front数量: 0")
        info_layout.addWidget(self.lbl_total_fronts)
        
        layout.addLayout(info_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_button_group(self, parent_layout):
        """创建按钮组"""
        layout = QHBoxLayout()
        layout.setSpacing(10)
        
        layout.addStretch()
        
        self.btn_preview = QPushButton("预览区域")
        self.btn_preview.clicked.connect(self.on_preview_clicked)
        layout.addWidget(self.btn_preview)
        
        self.btn_create = QPushButton("创建区域")
        self.btn_create.clicked.connect(self.on_create_clicked)
        layout.addWidget(self.btn_create)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        layout.addWidget(self.btn_cancel)
        
        parent_layout.addLayout(layout)
        
    def load_available_connectors(self):
        """加载可用的Connector列表"""
        self.connector_list_widget.clear()
        
        if not hasattr(self.gui, 'line_connectors') or not self.gui.line_connectors:
            QMessageBox.warning(self, "警告", "没有可用的Connector！请先生成线网格。")
            return False
        
        for i, conn in enumerate(self.gui.line_connectors):
            item = QListWidgetItem(f"{i+1}. {conn.part_name} - {conn.curve_name}")
            item.setData(Qt.UserRole, i)
            self.connector_list_widget.addItem(item)
        
        return True
        
    def on_connector_selection_changed(self):
        """处理Connector选择变化"""
        selected_items = self.connector_list_widget.selectedItems()
        self.selected_connectors = []
        
        for item in selected_items:
            idx = item.data(Qt.UserRole)
            if 0 <= idx < len(self.gui.line_connectors):
                self.selected_connectors.append(self.gui.line_connectors[idx])
        
        self.update_direction_table()
        self.update_region_info()
        
    def on_select_all_clicked(self):
        """全选所有Connector"""
        for i in range(self.connector_list_widget.count()):
            item = self.connector_list_widget.item(i)
            item.setSelected(True)
            
    def on_clear_selection_clicked(self):
        """清除选择"""
        self.connector_list_widget.clearSelection()
        
    def update_direction_table(self):
        """更新方向显示表格"""
        self.direction_table.setRowCount(0)
        
        for i, conn in enumerate(self.selected_connectors):
            row = self.direction_table.rowCount()
            self.direction_table.insertRow(row)
            
            # Connector名称
            name_item = QTableWidgetItem(f"{conn.part_name} - {conn.curve_name}")
            self.direction_table.setItem(row, 0, name_item)
            
            # 获取起点和终点
            if conn.front_list:
                start_node = conn.front_list[0].node_elems[0]
                end_node = conn.front_list[-1].node_elems[1]
                
                start_coords = f"({start_node.coords[0]:.3f}, {start_node.coords[1]:.3f})"
                end_coords = f"({end_node.coords[0]:.3f}, {end_node.coords[1]:.3f})"
                
                # 检查是否被翻转过
                is_flipped = self.connector_directions.get(id(conn), False)
                
                if is_flipped:
                    start_coords, end_coords = end_coords, start_coords
                
                self.direction_table.setItem(row, 1, QTableWidgetItem(start_coords))
                self.direction_table.setItem(row, 2, QTableWidgetItem(end_coords))
                
                # 方向显示
                direction_text = "正向" if not is_flipped else "反向"
                direction_item = QTableWidgetItem(direction_text)
                direction_item.setForeground(QBrush(QColor("green" if not is_flipped else "red")))
                self.direction_table.setItem(row, 3, direction_item)
            else:
                self.direction_table.setItem(row, 1, QTableWidgetItem("无数据"))
                self.direction_table.setItem(row, 2, QTableWidgetItem("无数据"))
                self.direction_table.setItem(row, 3, QTableWidgetItem("无数据"))
        
    def on_flip_selected_clicked(self):
        """翻转选中的Connector方向"""
        selected_rows = set()
        for item in self.direction_table.selectedItems():
            selected_rows.add(item.row())
        
        for row in selected_rows:
            if row < len(self.selected_connectors):
                conn = self.selected_connectors[row]
                current_flipped = self.connector_directions.get(id(conn), False)
                self.connector_directions[id(conn)] = not current_flipped
        
        self.update_direction_table()
        self.update_region_info()
        
    def on_flip_all_clicked(self):
        """翻转所有Connector方向"""
        for conn in self.selected_connectors:
            current_flipped = self.connector_directions.get(id(conn), False)
            self.connector_directions[id(conn)] = not current_flipped
        
        self.update_direction_table()
        self.update_region_info()
        
    def update_region_info(self):
        """更新区域信息"""
        self.lbl_connector_count.setText(f"已选择Connector数量: {len(self.selected_connectors)}")
        
        # 计算总Front数量
        total_fronts = sum(len(conn.front_list) for conn in self.selected_connectors)
        self.lbl_total_fronts.setText(f"总Front数量: {total_fronts}")
        
        # 检查是否形成封闭区域
        is_closed = self.check_region_closed()
        if is_closed:
            self.lbl_region_closed.setText("区域状态: 已形成封闭区域 ✓")
            self.lbl_region_closed.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.lbl_region_closed.setText("区域状态: 未形成封闭区域")
            self.lbl_region_closed.setStyleSheet("color: red; font-weight: bold;")
        
        self.btn_create.setEnabled(is_closed)
        
    def check_region_closed(self):
        """检查是否形成封闭区域"""
        if len(self.selected_connectors) < 1:
            return False
        
        # 收集所有端点和连接关系
        endpoints = []
        point_connections = {}
        
        for conn_idx, conn in enumerate(self.selected_connectors):
            if not conn.front_list:
                return False
            
            is_flipped = self.connector_directions.get(id(conn), False)
            
            start_node = conn.front_list[0].node_elems[0]
            end_node = conn.front_list[-1].node_elems[1]
            
            if is_flipped:
                start_node, end_node = end_node, start_node
            
            start_coords = tuple(round(x, 6) for x in start_node.coords)
            end_coords = tuple(round(x, 6) for x in end_node.coords)
            
            endpoints.append((start_coords, end_coords, conn_idx))
            
            # 记录连接关系
            if start_coords not in point_connections:
                point_connections[start_coords] = []
            point_connections[start_coords].append(('start', conn_idx))
            
            if end_coords not in point_connections:
                point_connections[end_coords] = []
            point_connections[end_coords].append(('end', conn_idx))
        
        # 检查每个点的连接情况
        for point, connections in point_connections.items():
            # 每个点应该恰好有两个连接（一个起点，一个终点）
            if len(connections) != 2:
                return False
            
            # 检查是否一个起点一个终点
            has_start = any(c[0] == 'start' for c in connections)
            has_end = any(c[0] == 'end' for c in connections)
            if not (has_start and has_end):
                return False
        
        # 如果只有一个Connector，检查起点和终点是否重合
        if len(self.selected_connectors) == 1:
            start, end, _ = endpoints[0]
            return start == end
        
        # 如果有多个Connector，检查是否能形成闭环
        # 从第一个Connector的起点开始，沿着连接关系遍历
        tolerance = 1e-6
        visited_connectors = set()
        
        # 找到第一个Connector的起点
        current_point, next_point, current_conn_idx = endpoints[0]
        visited_connectors.add(current_conn_idx)
        
        # 沿着连接关系遍历所有Connector
        for _ in range(len(self.selected_connectors)):
            # 在当前终点处找到下一个Connector
            found_next = False
            for conn_type, conn_idx in point_connections.get(next_point, []):
                if conn_idx not in visited_connectors and conn_type == 'start':
                    # 找到下一个Connector
                    for start, end, idx in endpoints:
                        if idx == conn_idx:
                            current_point = start
                            next_point = end
                            visited_connectors.add(conn_idx)
                            found_next = True
                            break
                    break
            
            if not found_next:
                # 没有找到下一个Connector
                break
        
        # 检查是否所有Connector都被访问，并且最后一个Connector的终点回到第一个Connector的起点
        first_start = endpoints[0][0]
        last_end = next_point
        
        return len(visited_connectors) == len(self.selected_connectors) and first_start == last_end
        
    def on_preview_clicked(self):
        """预览区域"""
        if not self.selected_connectors:
            QMessageBox.warning(self, "警告", "请先选择Connector！")
            return
        
        # 清除之前的方向显示
        if hasattr(self.gui, 'direction_actors'):
            for actor in self.gui.direction_actors:
                self.gui.mesh_display.renderer.RemoveActor(actor)
            self.gui.direction_actors = []
        else:
            self.gui.direction_actors = []
        
        # 显示方向箭头
        import vtk
        for conn in self.selected_connectors:
            if not conn.front_list:
                continue
            
            is_flipped = self.connector_directions.get(id(conn), False)
            
            if is_flipped:
                start_node = conn.front_list[-1].node_elems[1]
                end_node = conn.front_list[0].node_elems[0]
            else:
                start_node = conn.front_list[0].node_elems[0]
                end_node = conn.front_list[-1].node_elems[1]
            
            # 创建方向箭头
            arrow_source = vtk.vtkArrowSource()
            arrow_source.SetShaftRadius(0.02)
            arrow_source.SetTipRadius(0.05)
            arrow_source.SetTipLength(0.3)
            arrow_source.Update()
            
            # 计算变换
            start_point = start_node.coords
            end_point = end_node.coords
            
            direction = np.array(end_point) - np.array(start_point)
            length = np.linalg.norm(direction)
            
            if length < 1e-10:
                continue
            
            direction = direction / length
            
            # 创建变换
            transform = vtk.vtkTransform()
            transform.Translate(start_point[0], start_point[1], start_point[2])
            transform.Scale(length, length, length)
            
            # 计算旋转矩阵
            z_axis = np.array([0, 0, 1])
            if np.allclose(direction, z_axis):
                pass
            elif np.allclose(direction, -z_axis):
                transform.RotateWXYZ(180, 1, 0, 0)
            else:
                rotation_axis = np.cross(z_axis, direction)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.degrees(np.arccos(np.dot(z_axis, direction)))
                transform.RotateWXYZ(angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
            
            # 应用变换
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputConnection(arrow_source.GetOutputPort())
            transform_filter.Update()
            
            # 创建mapper和actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transform_filter.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 0, 0)
            actor.GetProperty().SetOpacity(0.8)
            
            self.gui.mesh_display.renderer.AddActor(actor)
            self.gui.direction_actors.append(actor)
        
        self.gui.mesh_display.render_window.Render()
        
    def on_create_clicked(self):
        """创建区域"""
        if not self.check_region_closed():
            QMessageBox.warning(self, "警告", "请确保选择的Connector形成封闭区域！")
            return
        
        # 不再合并Connector，直接传递多个Connector
        # 但需要处理Connector的方向翻转
        for conn in self.selected_connectors:
            is_flipped = self.connector_directions.get(id(conn), False)
            
            if is_flipped:
                # 翻转front_list顺序
                conn.front_list = conn.front_list[::-1]
                # 翻转每个front的节点顺序
                for front in conn.front_list:
                    front.node_elems = [front.node_elems[1], front.node_elems[0]]
                    # 重新计算front的方向和法向量
                    node1, node2 = front.node_elems[0].coords, front.node_elems[1].coords
                    front.length = np.linalg.norm(np.array(node2) - np.array(node1))
                    front.center = [(a + b) / 2 for a, b in zip(node1, node2)]
                    front.direction = [(b - a) / front.length for a, b in zip(node1, node2)]
                    front.normal = [-front.direction[1], front.direction[0]]
        
        # 创建区域数据 - 直接传递多个Connector
        region_data = {
            'connectors': self.selected_connectors,
            'directions': self.connector_directions.copy(),
            'total_connectors': len(self.selected_connectors),
            'total_fronts': sum(len(conn.front_list) for conn in self.selected_connectors)
        }
        
        self.region_created.emit(region_data)
        self.accept()
        
    def showEvent(self, event):
        """显示事件"""
        super().showEvent(event)
        self.load_available_connectors()
        
    def closeEvent(self, event):
        """关闭事件"""
        # 清除方向显示
        if hasattr(self.gui, 'direction_actors'):
            for actor in self.gui.direction_actors:
                self.gui.mesh_display.renderer.RemoveActor(actor)
            self.gui.direction_actors = []
            self.gui.mesh_display.render_window.Render()
        super().closeEvent(event)
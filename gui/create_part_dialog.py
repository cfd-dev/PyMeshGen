#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建部件对话框
用于创建新部件并设置相关元素
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QCheckBox, QPushButton, QGroupBox, QLabel,
    QTreeWidget, QTreeWidgetItem, QWidget, QSplitter
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class CreatePartDialog(QDialog):
    """创建部件对话框"""
    
    def __init__(self, parent=None, geometry_data=None, mesh_data=None):
        """
        初始化创建部件对话框
        
        Args:
            parent: 父窗口
            geometry_data: 几何数据
            mesh_data: 网格数据
        """
        super().__init__(parent)
        self.setWindowTitle("创建部件")
        self.setModal(True)
        self.resize(700, 600)
        
        self.geometry_data = geometry_data
        self.mesh_data = mesh_data
        
        self.part_name = ""
        self.selected_geometry_elements = {
            "vertices": [],
            "edges": [],
            "faces": [],
            "bodies": []
        }
        self.selected_mesh_elements = {
            "vertices": [],
            "edges": [],
            "faces": [],
            "bodies": []
        }
        
        self._create_widgets()
        self._connect_signals()
        self._load_elements()
    
    def _create_widgets(self):
        """创建对话框部件"""
        main_layout = QVBoxLayout(self)
        
        # 部件名称区域
        name_group = QGroupBox("部件信息")
        name_layout = QFormLayout(name_group)
        
        self.part_name_edit = QLineEdit()
        self.part_name_edit.setPlaceholderText("请输入部件名称")
        self.part_name_edit.setStyleSheet("background-color: white;")
        name_layout.addRow("部件名称:", self.part_name_edit)
        
        main_layout.addWidget(name_group)
        
        # 使用分割器分隔几何元素和网格元素
        splitter = QSplitter(Qt.Horizontal)
        
        # 几何元素区域
        geometry_group = QGroupBox("几何元素")
        geometry_layout = QVBoxLayout(geometry_group)
        
        self.geometry_tree = QTreeWidget()
        self.geometry_tree.setHeaderLabels(["元素", "选择"])
        self.geometry_tree.setColumnWidth(0, 150)
        self.geometry_tree.setColumnWidth(1, 50)
        self.geometry_tree.setAlternatingRowColors(True)
        self.geometry_tree.itemChanged.connect(self._on_geometry_item_changed)
        
        geometry_layout.addWidget(self.geometry_tree)
        splitter.addWidget(geometry_group)
        
        # 网格元素区域
        mesh_group = QGroupBox("网格元素")
        mesh_layout = QVBoxLayout(mesh_group)
        
        self.mesh_tree = QTreeWidget()
        self.mesh_tree.setHeaderLabels(["元素", "选择"])
        self.mesh_tree.setColumnWidth(0, 150)
        self.mesh_tree.setColumnWidth(1, 50)
        self.mesh_tree.setAlternatingRowColors(True)
        self.mesh_tree.itemChanged.connect(self._on_mesh_item_changed)
        
        mesh_layout.addWidget(self.mesh_tree)
        splitter.addWidget(mesh_group)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_button = QPushButton("确定")
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #e6f7ff;
                border: 1px solid #0078d4;
                border-radius: 3px;
                padding: 6px 12px;
                color: #0078d4;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #cceeff;
            }
            QPushButton:pressed {
                background-color: #99ddff;
            }
        """)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
            QPushButton:pressed {
                background-color: #d9d9d9;
            }
        """)
        button_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(button_layout)
    
    def _connect_signals(self):
        """连接信号和槽"""
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
    
    def _load_elements(self):
        """加载几何和网格元素到树中"""
        self._load_geometry_elements()
        self._load_mesh_elements()
    
    def _load_geometry_elements(self):
        """加载几何元素"""
        self.geometry_tree.clear()
        self.geometry_tree.blockSignals(True)
        
        if self.geometry_data is None:
            self.geometry_tree.blockSignals(False)
            return
        
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
        
        # 点
        vertices_item = QTreeWidgetItem(self.geometry_tree)
        vertices_item.setText(0, "点")
        vertices_item.setCheckState(0, Qt.Unchecked)
        vertices_item.setExpanded(False)
        vertices_item.setData(0, Qt.UserRole, "geometry_vertices")
        
        explorer = TopExp_Explorer(self.geometry_data, TopAbs_VERTEX)
        vertex_count = 0
        while explorer.More():
            vertex = explorer.Current()
            vertex_item = QTreeWidgetItem(vertices_item)
            vertex_item.setText(0, f"点_{vertex_count}")
            vertex_item.setCheckState(0, Qt.Unchecked)
            vertex_item.setData(0, Qt.UserRole, ("vertex", vertex_count, vertex))
            vertex_count += 1
            explorer.Next()
        
        vertices_item.setText(1, f"{vertex_count}")
        
        # 线
        edges_item = QTreeWidgetItem(self.geometry_tree)
        edges_item.setText(0, "线")
        edges_item.setCheckState(0, Qt.Unchecked)
        edges_item.setExpanded(False)
        edges_item.setData(0, Qt.UserRole, "geometry_edges")
        
        explorer = TopExp_Explorer(self.geometry_data, TopAbs_EDGE)
        edge_count = 0
        while explorer.More():
            edge = explorer.Current()
            edge_item = QTreeWidgetItem(edges_item)
            edge_item.setText(0, f"线_{edge_count}")
            edge_item.setCheckState(0, Qt.Unchecked)
            edge_item.setData(0, Qt.UserRole, ("edge", edge_count, edge))
            edge_count += 1
            explorer.Next()
        
        edges_item.setText(1, f"{edge_count}")
        
        # 面
        faces_item = QTreeWidgetItem(self.geometry_tree)
        faces_item.setText(0, "面")
        faces_item.setCheckState(0, Qt.Unchecked)
        faces_item.setExpanded(False)
        faces_item.setData(0, Qt.UserRole, "geometry_faces")
        
        explorer = TopExp_Explorer(self.geometry_data, TopAbs_FACE)
        face_count = 0
        while explorer.More():
            face = explorer.Current()
            face_item = QTreeWidgetItem(faces_item)
            face_item.setText(0, f"面_{face_count}")
            face_item.setCheckState(0, Qt.Unchecked)
            face_item.setData(0, Qt.UserRole, ("face", face_count, face))
            face_count += 1
            explorer.Next()
        
        faces_item.setText(1, f"{face_count}")
        
        # 体
        bodies_item = QTreeWidgetItem(self.geometry_tree)
        bodies_item.setText(0, "体")
        bodies_item.setCheckState(0, Qt.Unchecked)
        bodies_item.setExpanded(False)
        bodies_item.setData(0, Qt.UserRole, "geometry_bodies")
        
        explorer = TopExp_Explorer(self.geometry_data, TopAbs_SOLID)
        body_count = 0
        while explorer.More():
            solid = explorer.Current()
            body_item = QTreeWidgetItem(bodies_item)
            body_item.setText(0, f"体_{body_count}")
            body_item.setCheckState(0, Qt.Unchecked)
            body_item.setData(0, Qt.UserRole, ("body", body_count, solid))
            body_count += 1
            explorer.Next()
        
        bodies_item.setText(1, f"{body_count}")
        
        self.geometry_tree.blockSignals(False)
    
    def _load_mesh_elements(self):
        """加载网格元素"""
        self.mesh_tree.clear()
        self.mesh_tree.blockSignals(True)
        
        if self.mesh_data is None:
            self.mesh_tree.blockSignals(False)
            return
        
        # 点
        vertices_item = QTreeWidgetItem(self.mesh_tree)
        vertices_item.setText(0, "点")
        vertices_item.setCheckState(0, Qt.Unchecked)
        vertices_item.setExpanded(False)
        vertices_item.setData(0, Qt.UserRole, "mesh_vertices")
        
        if hasattr(self.mesh_data, 'node_coords'):
            node_coords = self.mesh_data.node_coords
            vertex_count = len(node_coords)
            for i in range(min(vertex_count, 100)):
                vertex_item = QTreeWidgetItem(vertices_item)
                vertex_item.setText(0, f"点_{i}")
                vertex_item.setCheckState(0, Qt.Unchecked)
                vertex_item.setData(0, Qt.UserRole, ("vertex", i, node_coords[i]))
            
            if vertex_count > 100:
                summary_item = QTreeWidgetItem(vertices_item)
                summary_item.setText(0, f"... (还有 {vertex_count - 100} 个节点)")
                summary_item.setCheckState(0, Qt.Unchecked)
        
        vertices_item.setText(1, f"{vertex_count if hasattr(self.mesh_data, 'node_coords') else 0}")
        
        # 线
        edges_item = QTreeWidgetItem(self.mesh_tree)
        edges_item.setText(0, "线")
        edges_item.setCheckState(0, Qt.Unchecked)
        edges_item.setExpanded(False)
        edges_item.setData(0, Qt.UserRole, "mesh_edges")
        edges_item.setText(1, "0")
        
        # 面
        faces_item = QTreeWidgetItem(self.mesh_tree)
        faces_item.setText(0, "面")
        faces_item.setCheckState(0, Qt.Unchecked)
        faces_item.setExpanded(False)
        faces_item.setData(0, Qt.UserRole, "mesh_faces")
        
        if hasattr(self.mesh_data, 'cells'):
            cells = self.mesh_data.cells
            face_count = len(cells)
            for i in range(min(face_count, 100)):
                face_item = QTreeWidgetItem(faces_item)
                face_item.setText(0, f"面_{i}")
                face_item.setCheckState(0, Qt.Unchecked)
                face_item.setData(0, Qt.UserRole, ("face", i, cells[i]))
            
            if face_count > 100:
                summary_item = QTreeWidgetItem(faces_item)
                summary_item.setText(0, f"... (还有 {face_count - 100} 个单元)")
                summary_item.setCheckState(0, Qt.Unchecked)
        
        faces_item.setText(1, f"{face_count if hasattr(self.mesh_data, 'cells') else 0}")
        
        # 体
        bodies_item = QTreeWidgetItem(self.mesh_tree)
        bodies_item.setText(0, "体")
        bodies_item.setCheckState(0, Qt.Unchecked)
        bodies_item.setExpanded(False)
        bodies_item.setData(0, Qt.UserRole, "mesh_bodies")
        bodies_item.setText(1, "0")
        
        self.mesh_tree.blockSignals(False)
    
    def _on_geometry_item_changed(self, item, column):
        """几何元素项改变时处理"""
        if column != 0:
            return
        
        self.geometry_tree.blockSignals(True)
        
        data = item.data(0, Qt.UserRole)
        
        if isinstance(data, str) and data.startswith("geometry_"):
            element_type = data.replace("geometry_", "")
            
            for i in range(item.childCount()):
                child = item.child(i)
                child.setCheckState(0, item.checkState(0))
        elif isinstance(data, tuple) and len(data) >= 3:
            element_type, index, element = data[0], data[1], data[2]
            
            if element_type == "vertex":
                if item.checkState(0) == Qt.Checked:
                    if index not in self.selected_geometry_elements["vertices"]:
                        self.selected_geometry_elements["vertices"].append(index)
                else:
                    if index in self.selected_geometry_elements["vertices"]:
                        self.selected_geometry_elements["vertices"].remove(index)
            elif element_type == "edge":
                if item.checkState(0) == Qt.Checked:
                    if index not in self.selected_geometry_elements["edges"]:
                        self.selected_geometry_elements["edges"].append(index)
                else:
                    if index in self.selected_geometry_elements["edges"]:
                        self.selected_geometry_elements["edges"].remove(index)
            elif element_type == "face":
                if item.checkState(0) == Qt.Checked:
                    if index not in self.selected_geometry_elements["faces"]:
                        self.selected_geometry_elements["faces"].append(index)
                else:
                    if index in self.selected_geometry_elements["faces"]:
                        self.selected_geometry_elements["faces"].remove(index)
            elif element_type == "body":
                if item.checkState(0) == Qt.Checked:
                    if index not in self.selected_geometry_elements["bodies"]:
                        self.selected_geometry_elements["bodies"].append(index)
                else:
                    if index in self.selected_geometry_elements["bodies"]:
                        self.selected_geometry_elements["bodies"].remove(index)
        
        self.geometry_tree.blockSignals(False)
    
    def _on_mesh_item_changed(self, item, column):
        """网格元素项改变时处理"""
        if column != 0:
            return
        
        self.mesh_tree.blockSignals(True)
        
        data = item.data(0, Qt.UserRole)
        
        if isinstance(data, str) and data.startswith("mesh_"):
            element_type = data.replace("mesh_", "")
            
            for i in range(item.childCount()):
                child = item.child(i)
                child.setCheckState(0, item.checkState(0))
        elif isinstance(data, tuple) and len(data) >= 3:
            element_type, index, element = data[0], data[1], data[2]
            
            if element_type == "vertex":
                if item.checkState(0) == Qt.Checked:
                    if index not in self.selected_mesh_elements["vertices"]:
                        self.selected_mesh_elements["vertices"].append(index)
                else:
                    if index in self.selected_mesh_elements["vertices"]:
                        self.selected_mesh_elements["vertices"].remove(index)
            elif element_type == "face":
                if item.checkState(0) == Qt.Checked:
                    if index not in self.selected_mesh_elements["faces"]:
                        self.selected_mesh_elements["faces"].append(index)
                else:
                    if index in self.selected_mesh_elements["faces"]:
                        self.selected_mesh_elements["faces"].remove(index)
        
        self.mesh_tree.blockSignals(False)
    
    def accept(self):
        """接受对话框"""
        part_name = self.part_name_edit.text().strip()
        
        if not part_name:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请输入部件名称！")
            return
        
        self.part_name = part_name
        super().accept()
    
    def get_part_info(self):
        """获取部件信息"""
        return {
            "part_name": self.part_name,
            "geometry_elements": self.selected_geometry_elements,
            "mesh_elements": self.selected_mesh_elements
        }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
几何删除对话框
支持勾选或拾取几何元素并执行删除
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QTreeWidget, QTreeWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .ui_utils import UIStyles


class GeometryDeleteDialog(QDialog):
    """几何删除对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gui = parent
        self.setWindowTitle("删除几何")
        self.setModal(False)
        self.resize(520, 540)

        font = QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.setFont(font)

        self.geometry_data = getattr(self.gui, "current_geometry", None)
        self._picking_helper = None
        self.selected_geometry_elements = {
            "vertices": [],
            "edges": [],
            "faces": [],
            "bodies": []
        }

        self._create_widgets()
        self._connect_signals()
        self._load_geometry_elements()

    def _create_widgets(self):
        layout = QVBoxLayout(self)

        info_group = QGroupBox("操作")
        info_layout = QFormLayout(info_group)
        self.enable_pick_checkbox = QCheckBox("启用拾取")
        self.enable_pick_checkbox.setChecked(False)
        info_layout.addRow("拾取:", self.enable_pick_checkbox)
        layout.addWidget(info_group)

        geometry_group = QGroupBox("几何元素")
        geometry_layout = QVBoxLayout(geometry_group)

        self.geometry_tree = QTreeWidget()
        self.geometry_tree.setHeaderLabels(["元素", "选择"])
        self.geometry_tree.setColumnWidth(0, 260)
        self.geometry_tree.setColumnWidth(1, 60)
        self.geometry_tree.setAlternatingRowColors(True)
        self.geometry_tree.itemChanged.connect(self._on_geometry_item_changed)
        geometry_layout.addWidget(self.geometry_tree)
        layout.addWidget(geometry_group)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.delete_button = QPushButton("删除")
        self.delete_button.setStyleSheet(UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET)
        button_layout.addWidget(self.delete_button)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def _connect_signals(self):
        self.delete_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.enable_pick_checkbox.stateChanged.connect(self._on_pick_state_changed)

    def _load_geometry_elements(self):
        self.geometry_tree.clear()
        self.geometry_tree.blockSignals(True)

        if self.geometry_data is None:
            self.geometry_tree.blockSignals(False)
            return

        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID

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

    def _on_geometry_item_changed(self, item, column):
        if column != 0:
            return

        self.geometry_tree.blockSignals(True)
        data = item.data(0, Qt.UserRole)

        if isinstance(data, str) and data.startswith("geometry_"):
            for i in range(item.childCount()):
                child = item.child(i)
                child.setCheckState(0, item.checkState(0))
        elif isinstance(data, tuple) and len(data) >= 3:
            element_type, index = data[0], data[1]
            key_map = {
                "vertex": "vertices",
                "edge": "edges",
                "face": "faces",
                "body": "bodies"
            }
            key = key_map.get(element_type)
            if key:
                if item.checkState(0) == Qt.Checked:
                    if index not in self.selected_geometry_elements[key]:
                        self.selected_geometry_elements[key].append(index)
                else:
                    if index in self.selected_geometry_elements[key]:
                        self.selected_geometry_elements[key].remove(index)

        self.geometry_tree.blockSignals(False)

    def _on_pick_state_changed(self, state):
        if state == Qt.Checked:
            self._enable_picking()
        else:
            self._disable_picking()

    def _enable_picking(self):
        if self._picking_helper is not None:
            self._picking_helper.enable()
            return
        if not self.gui or not hasattr(self.gui, "mesh_display"):
            return
        from .geometry_picking import GeometryPickingHelper
        self._picking_helper = GeometryPickingHelper(
            self.gui.mesh_display,
            gui=self.gui,
            on_pick=self._on_geometry_pick,
            on_unpick=self._on_geometry_unpick,
        )
        self._picking_helper.enable()

    def _disable_picking(self):
        if self._picking_helper is None:
            return
        self._picking_helper.cleanup()
        self._picking_helper = None

    def _on_geometry_pick(self, element_type, element_obj, element_index):
        key_map = {
            "vertex": "vertices",
            "edge": "edges",
            "face": "faces",
            "body": "bodies",
        }
        key = key_map.get(element_type)
        if key is None:
            return
        tree_item = self._find_geometry_item(key, element_index)
        if tree_item is None:
            return
        self.geometry_tree.blockSignals(True)
        tree_item.setCheckState(0, Qt.Checked)
        self.geometry_tree.blockSignals(False)
        if element_index not in self.selected_geometry_elements[key]:
            self.selected_geometry_elements[key].append(element_index)

    def _on_geometry_unpick(self, element_type, element_obj, element_index):
        key_map = {
            "vertex": "vertices",
            "edge": "edges",
            "face": "faces",
            "body": "bodies",
        }
        key = key_map.get(element_type)
        if key is None:
            return
        tree_item = self._find_geometry_item(key, element_index)
        if tree_item is None:
            return
        self.geometry_tree.blockSignals(True)
        tree_item.setCheckState(0, Qt.Unchecked)
        self.geometry_tree.blockSignals(False)
        if element_index in self.selected_geometry_elements[key]:
            self.selected_geometry_elements[key].remove(element_index)

    def _find_geometry_item(self, element_key, element_index):
        root_map = {
            "vertices": "geometry_vertices",
            "edges": "geometry_edges",
            "faces": "geometry_faces",
            "bodies": "geometry_bodies",
        }
        root_key = root_map.get(element_key)
        if root_key is None:
            return None
        for i in range(self.geometry_tree.topLevelItemCount()):
            root_item = self.geometry_tree.topLevelItem(i)
            if root_item.data(0, Qt.UserRole) != root_key:
                continue
            for j in range(root_item.childCount()):
                child = root_item.child(j)
                data = child.data(0, Qt.UserRole)
                if isinstance(data, tuple) and len(data) >= 2:
                    if data[1] == element_index:
                        return child
        return None

    def _has_selection(self):
        return any(self.selected_geometry_elements.get(key) for key in self.selected_geometry_elements)

    def accept(self):
        if not self._has_selection():
            QMessageBox.warning(self, "警告", "请先选择要删除的几何元素")
            return
        if not self.gui or not hasattr(self.gui, "delete_geometry_elements"):
            QMessageBox.warning(self, "警告", "未找到删除几何功能入口")
            return
        success = self.gui.delete_geometry_elements(self.selected_geometry_elements)
        if success:
            self._disable_picking()
            super().accept()

    def reject(self):
        self._disable_picking()
        super().reject()

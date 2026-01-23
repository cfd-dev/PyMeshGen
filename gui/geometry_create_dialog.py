#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
几何创建对话框
支持创建点、直线、圆/圆弧、曲线（样条）
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QWidget, QComboBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .ui_utils import UIStyles


class GeometryCreateDialog(QDialog):
    """几何创建对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gui = parent
        self.setWindowTitle("创建几何")
        self.setModal(False)
        self.resize(720, 520)

        font = QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.setFont(font)

        self._picked_points = []
        self._create_widgets()
        self._connect_signals()

    def _create_widgets(self):
        layout = QVBoxLayout(self)

        mode_group = QGroupBox("创建类型")
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.addWidget(QLabel("类型:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["点", "直线", "圆/圆弧", "曲线"])
        self.mode_combo.setStyleSheet("background-color: white;")
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addWidget(mode_group)

        coord_group = QGroupBox("参数输入")
        coord_layout = QVBoxLayout(coord_group)

        self.point_form = QWidget()
        form = QFormLayout(self.point_form)
        self.px = self._create_spin()
        self.py = self._create_spin()
        self.pz = self._create_spin()
        form.addRow("X:", self.px)
        form.addRow("Y:", self.py)
        form.addRow("Z:", self.pz)
        coord_layout.addWidget(self.point_form)

        self.line_form = QWidget()
        line_form_layout = QFormLayout(self.line_form)
        self.lx1 = self._create_spin()
        self.ly1 = self._create_spin()
        self.lz1 = self._create_spin()
        self.lx2 = self._create_spin()
        self.ly2 = self._create_spin()
        self.lz2 = self._create_spin()
        line_form_layout.addRow("P1 X:", self.lx1)
        line_form_layout.addRow("P1 Y:", self.ly1)
        line_form_layout.addRow("P1 Z:", self.lz1)
        line_form_layout.addRow("P2 X:", self.lx2)
        line_form_layout.addRow("P2 Y:", self.ly2)
        line_form_layout.addRow("P2 Z:", self.lz2)
        coord_layout.addWidget(self.line_form)

        self.circle_form = QWidget()
        circle_form_layout = QFormLayout(self.circle_form)
        self.cx = self._create_spin()
        self.cy = self._create_spin()
        self.cz = self._create_spin()
        self.radius = self._create_spin(min_value=1e-6)
        self.start_angle = self._create_spin(min_value=-360.0, max_value=360.0, decimals=2)
        self.end_angle = self._create_spin(min_value=-360.0, max_value=360.0, decimals=2)
        circle_form_layout.addRow("圆心 X:", self.cx)
        circle_form_layout.addRow("圆心 Y:", self.cy)
        circle_form_layout.addRow("圆心 Z:", self.cz)
        circle_form_layout.addRow("半径:", self.radius)
        circle_form_layout.addRow("起始角度(°):", self.start_angle)
        circle_form_layout.addRow("终止角度(°):", self.end_angle)
        coord_layout.addWidget(self.circle_form)

        self.curve_form = QWidget()
        curve_layout = QVBoxLayout(self.curve_form)
        curve_layout.addWidget(QLabel("曲线点集 (至少2点):"))
        self.curve_table = QTableWidget(0, 3)
        self.curve_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.curve_table.horizontalHeader().setStretchLastSection(True)
        curve_layout.addWidget(self.curve_table)
        coord_layout.addWidget(self.curve_form)

        layout.addWidget(coord_group)

        pick_group = QGroupBox("拾取")
        pick_layout = QHBoxLayout(pick_group)
        self.pick_button = QPushButton("进入拾取")
        self.pick_button.setStyleSheet(UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET)
        self.pick_clear_button = QPushButton("清空拾取")
        self.pick_clear_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        pick_layout.addWidget(self.pick_button)
        pick_layout.addWidget(self.pick_clear_button)
        pick_layout.addStretch()
        layout.addWidget(pick_group)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.create_button = QPushButton("创建")
        self.create_button.setStyleSheet(UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET)
        self.cancel_button = QPushButton("关闭")
        self.cancel_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        button_layout.addWidget(self.create_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self._apply_mode_visibility()

    def _connect_signals(self):
        self.mode_combo.currentIndexChanged.connect(self._apply_mode_visibility)
        self.pick_button.clicked.connect(self._toggle_pick)
        self.pick_clear_button.clicked.connect(self._clear_picked)
        self.create_button.clicked.connect(self._create_geometry)
        self.cancel_button.clicked.connect(self.close)

    def _create_spin(self, min_value=-1e6, max_value=1e6, decimals=3):
        spin = QDoubleSpinBox()
        spin.setRange(min_value, max_value)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1)
        spin.setStyleSheet("background-color: white;")
        return spin

    def _apply_mode_visibility(self):
        mode = self.mode_combo.currentText()
        self.point_form.setVisible(mode == "点")
        self.line_form.setVisible(mode == "直线")
        self.circle_form.setVisible(mode == "圆/圆弧")
        self.curve_form.setVisible(mode == "曲线")

    def _toggle_pick(self):
        if not self.gui or not hasattr(self.gui, 'view_controller'):
            QMessageBox.warning(self, "警告", "未找到视图控制器")
            return
        self._picked_points.clear()
        if hasattr(self.gui.view_controller, 'start_point_pick'):
            if self.gui.view_controller.is_point_pick_active():
                self.gui.view_controller.stop_point_pick()
                self.pick_button.setText("进入拾取")
            else:
                self.gui.view_controller.start_point_pick(self._on_point_picked)
                self.pick_button.setText("拾取中...")
        else:
            QMessageBox.warning(self, "警告", "拾取功能不可用")

    def _clear_picked(self):
        self._picked_points.clear()
        self.curve_table.setRowCount(0)
        if self.gui and hasattr(self.gui, 'view_controller'):
            self.gui.view_controller.stop_point_pick()
            self.pick_button.setText("进入拾取")

    def _on_point_picked(self, point):
        self._picked_points.append(point)
        mode = self.mode_combo.currentText()
        if mode == "点":
            self.px.setValue(point[0])
            self.py.setValue(point[1])
            self.pz.setValue(point[2])
            if self.gui and hasattr(self.gui, 'view_controller'):
                self.gui.view_controller.stop_point_pick()
                self.pick_button.setText("进入拾取")
        elif mode == "直线":
            if len(self._picked_points) == 1:
                self.lx1.setValue(point[0])
                self.ly1.setValue(point[1])
                self.lz1.setValue(point[2])
            elif len(self._picked_points) >= 2:
                self.lx2.setValue(point[0])
                self.ly2.setValue(point[1])
                self.lz2.setValue(point[2])
                if self.gui and hasattr(self.gui, 'view_controller'):
                    self.gui.view_controller.stop_point_pick()
                    self.pick_button.setText("进入拾取")
        elif mode == "圆/圆弧":
            if len(self._picked_points) == 1:
                self.cx.setValue(point[0])
                self.cy.setValue(point[1])
                self.cz.setValue(point[2])
            elif len(self._picked_points) == 2:
                dx = point[0] - self.cx.value()
                dy = point[1] - self.cy.value()
                dz = point[2] - self.cz.value()
                self.radius.setValue((dx * dx + dy * dy + dz * dz) ** 0.5)
                if self.gui and hasattr(self.gui, 'view_controller'):
                    self.gui.view_controller.stop_point_pick()
                    self.pick_button.setText("进入拾取")
        elif mode == "曲线":
            row = self.curve_table.rowCount()
            self.curve_table.insertRow(row)
            self.curve_table.setItem(row, 0, QTableWidgetItem(f"{point[0]:.6f}"))
            self.curve_table.setItem(row, 1, QTableWidgetItem(f"{point[1]:.6f}"))
            self.curve_table.setItem(row, 2, QTableWidgetItem(f"{point[2]:.6f}"))

    def _create_geometry(self):
        if not self.gui or not hasattr(self.gui, 'geometry_operations'):
            QMessageBox.warning(self, "警告", "几何操作器未初始化")
            return
        mode = self.mode_combo.currentText()
        if mode == "点":
            points = [(self.px.value(), self.py.value(), self.pz.value())]
            self.gui.geometry_operations.create_geometry_from_points(points, mode="point")
        elif mode == "直线":
            points = [
                (self.lx1.value(), self.ly1.value(), self.lz1.value()),
                (self.lx2.value(), self.ly2.value(), self.lz2.value()),
            ]
            self.gui.geometry_operations.create_geometry_from_points(points, mode="line")
        elif mode == "圆/圆弧":
            center = (self.cx.value(), self.cy.value(), self.cz.value())
            radius = self.radius.value()
            self.gui.geometry_operations.create_geometry_circle(
                center,
                radius,
                start_angle=self.start_angle.value(),
                end_angle=self.end_angle.value()
            )
        elif mode == "曲线":
            points = self._get_curve_points()
            if len(points) < 2:
                QMessageBox.warning(self, "警告", "曲线至少需要两个点")
                return
            self.gui.geometry_operations.create_geometry_from_points(points, mode="curve")
        if self.gui and hasattr(self.gui, 'view_controller'):
            self.gui.view_controller.stop_point_pick()
            self.pick_button.setText("进入拾取")

    def _get_curve_points(self):
        points = []
        for row in range(self.curve_table.rowCount()):
            try:
                x = float(self.curve_table.item(row, 0).text())
                y = float(self.curve_table.item(row, 1).text())
                z = float(self.curve_table.item(row, 2).text())
                points.append((x, y, z))
            except Exception:
                continue
        return points

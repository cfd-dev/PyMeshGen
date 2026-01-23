#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
几何创建对话框
支持创建点、直线、圆/圆弧、曲线（样条）
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QWidget, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .ui_utils import UIStyles
from .icon_manager import get_icon


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
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.addWidget(QLabel("类型:"))
        row_2d_layout = QHBoxLayout()
        row_3d_layout = QHBoxLayout()
        self.point_mode_button = QPushButton("点")
        self.line_mode_button = QPushButton("直线")
        self.circle_mode_button = QPushButton("圆/圆弧")
        self.curve_mode_button = QPushButton("曲线")
        self.polyline_mode_button = QPushButton("多段线/折线")
        self.rectangle_mode_button = QPushButton("矩形")
        self.polygon_mode_button = QPushButton("多边形")
        self.ellipse_mode_button = QPushButton("椭圆/椭圆弧")
        self.box_mode_button = QPushButton("长方体")
        self.sphere_mode_button = QPushButton("圆球")
        self.cylinder_mode_button = QPushButton("圆柱")
        self.point_mode_button.setCheckable(True)
        self.line_mode_button.setCheckable(True)
        self.circle_mode_button.setCheckable(True)
        self.curve_mode_button.setCheckable(True)
        self.polyline_mode_button.setCheckable(True)
        self.rectangle_mode_button.setCheckable(True)
        self.polygon_mode_button.setCheckable(True)
        self.ellipse_mode_button.setCheckable(True)
        self.box_mode_button.setCheckable(True)
        self.sphere_mode_button.setCheckable(True)
        self.cylinder_mode_button.setCheckable(True)
        self.point_mode_button.setChecked(True)
        self.point_mode_button.setStyleSheet(UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET)
        self.line_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.circle_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.curve_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.polyline_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.rectangle_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.polygon_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.ellipse_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.box_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.sphere_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.cylinder_mode_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.polyline_mode_button.setMinimumWidth(
            self.fontMetrics().horizontalAdvance(self.polyline_mode_button.text()) + 20
        )
        self.ellipse_mode_button.setMinimumWidth(
            self.fontMetrics().horizontalAdvance(self.ellipse_mode_button.text()) + 20
        )
        self.point_mode_button.setIcon(get_icon("geom-point"))
        self.line_mode_button.setIcon(get_icon("geom-line"))
        self.circle_mode_button.setIcon(get_icon("geom-circle"))
        self.curve_mode_button.setIcon(get_icon("geom-curve"))
        self.polyline_mode_button.setIcon(get_icon("geom-polyline"))
        self.rectangle_mode_button.setIcon(get_icon("geom-rectangle"))
        self.polygon_mode_button.setIcon(get_icon("geom-polygon"))
        self.ellipse_mode_button.setIcon(get_icon("geom-ellipse"))
        self.box_mode_button.setIcon(get_icon("geom-box"))
        self.sphere_mode_button.setIcon(get_icon("geom-sphere"))
        self.cylinder_mode_button.setIcon(get_icon("geom-cylinder"))
        row_2d_layout.addWidget(self.point_mode_button)
        row_2d_layout.addWidget(self.line_mode_button)
        row_2d_layout.addWidget(self.circle_mode_button)
        row_2d_layout.addWidget(self.curve_mode_button)
        row_2d_layout.addWidget(self.polyline_mode_button)
        row_2d_layout.addWidget(self.rectangle_mode_button)
        row_2d_layout.addWidget(self.polygon_mode_button)
        row_2d_layout.addWidget(self.ellipse_mode_button)
        row_2d_layout.addStretch()
        row_3d_layout.addWidget(self.box_mode_button)
        row_3d_layout.addWidget(self.sphere_mode_button)
        row_3d_layout.addWidget(self.cylinder_mode_button)
        row_3d_layout.addStretch()
        mode_layout.addLayout(row_2d_layout)
        mode_layout.addLayout(row_3d_layout)
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

        self.polyline_form = QWidget()
        polyline_layout = QVBoxLayout(self.polyline_form)
        polyline_layout.addWidget(QLabel("多段线点集 (至少2点):"))
        self.polyline_table = QTableWidget(0, 3)
        self.polyline_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.polyline_table.horizontalHeader().setStretchLastSection(True)
        polyline_layout.addWidget(self.polyline_table)
        coord_layout.addWidget(self.polyline_form)

        self.rectangle_form = QWidget()
        rectangle_layout = QFormLayout(self.rectangle_form)
        self.rx1 = self._create_spin()
        self.ry1 = self._create_spin()
        self.rz1 = self._create_spin()
        self.rx2 = self._create_spin()
        self.ry2 = self._create_spin()
        self.rz2 = self._create_spin()
        rectangle_layout.addRow("P1 X:", self.rx1)
        rectangle_layout.addRow("P1 Y:", self.ry1)
        rectangle_layout.addRow("P1 Z:", self.rz1)
        rectangle_layout.addRow("P2 X:", self.rx2)
        rectangle_layout.addRow("P2 Y:", self.ry2)
        rectangle_layout.addRow("P2 Z:", self.rz2)
        coord_layout.addWidget(self.rectangle_form)

        self.polygon_form = QWidget()
        polygon_layout = QVBoxLayout(self.polygon_form)
        polygon_layout.addWidget(QLabel("多边形点集 (至少3点):"))
        self.polygon_table = QTableWidget(0, 3)
        self.polygon_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.polygon_table.horizontalHeader().setStretchLastSection(True)
        polygon_layout.addWidget(self.polygon_table)
        coord_layout.addWidget(self.polygon_form)

        self.ellipse_form = QWidget()
        ellipse_layout = QFormLayout(self.ellipse_form)
        self.ex = self._create_spin()
        self.ey = self._create_spin()
        self.ez = self._create_spin()
        self.major_radius = self._create_spin(min_value=1e-6)
        self.minor_radius = self._create_spin(min_value=1e-6)
        self.ellipse_start_angle = self._create_spin(min_value=-360.0, max_value=360.0, decimals=2)
        self.ellipse_end_angle = self._create_spin(min_value=-360.0, max_value=360.0, decimals=2)
        ellipse_layout.addRow("圆心 X:", self.ex)
        ellipse_layout.addRow("圆心 Y:", self.ey)
        ellipse_layout.addRow("圆心 Z:", self.ez)
        ellipse_layout.addRow("长半轴:", self.major_radius)
        ellipse_layout.addRow("短半轴:", self.minor_radius)
        ellipse_layout.addRow("起始角度(°):", self.ellipse_start_angle)
        ellipse_layout.addRow("终止角度(°):", self.ellipse_end_angle)
        coord_layout.addWidget(self.ellipse_form)

        self.box_form = QWidget()
        box_layout = QFormLayout(self.box_form)
        self.bx1 = self._create_spin()
        self.by1 = self._create_spin()
        self.bz1 = self._create_spin()
        self.bx2 = self._create_spin()
        self.by2 = self._create_spin()
        self.bz2 = self._create_spin()
        box_layout.addRow("P1 X:", self.bx1)
        box_layout.addRow("P1 Y:", self.by1)
        box_layout.addRow("P1 Z:", self.bz1)
        box_layout.addRow("P2 X:", self.bx2)
        box_layout.addRow("P2 Y:", self.by2)
        box_layout.addRow("P2 Z:", self.bz2)
        coord_layout.addWidget(self.box_form)

        self.sphere_form = QWidget()
        sphere_layout = QFormLayout(self.sphere_form)
        self.sx = self._create_spin()
        self.sy = self._create_spin()
        self.sz = self._create_spin()
        self.sr = self._create_spin(min_value=1e-6)
        sphere_layout.addRow("圆心 X:", self.sx)
        sphere_layout.addRow("圆心 Y:", self.sy)
        sphere_layout.addRow("圆心 Z:", self.sz)
        sphere_layout.addRow("半径:", self.sr)
        coord_layout.addWidget(self.sphere_form)

        self.cylinder_form = QWidget()
        cylinder_layout = QFormLayout(self.cylinder_form)
        self.cx3 = self._create_spin()
        self.cy3 = self._create_spin()
        self.cz3 = self._create_spin()
        self.cr = self._create_spin(min_value=1e-6)
        self.ch = self._create_spin(min_value=1e-6)
        cylinder_layout.addRow("底面圆心 X:", self.cx3)
        cylinder_layout.addRow("底面圆心 Y:", self.cy3)
        cylinder_layout.addRow("底面圆心 Z:", self.cz3)
        cylinder_layout.addRow("半径:", self.cr)
        cylinder_layout.addRow("高度:", self.ch)
        coord_layout.addWidget(self.cylinder_form)

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
        self.point_mode_button.clicked.connect(lambda: self._set_mode("点"))
        self.line_mode_button.clicked.connect(lambda: self._set_mode("直线"))
        self.circle_mode_button.clicked.connect(lambda: self._set_mode("圆/圆弧"))
        self.curve_mode_button.clicked.connect(lambda: self._set_mode("曲线"))
        self.polyline_mode_button.clicked.connect(lambda: self._set_mode("多段线/折线"))
        self.rectangle_mode_button.clicked.connect(lambda: self._set_mode("矩形"))
        self.polygon_mode_button.clicked.connect(lambda: self._set_mode("多边形"))
        self.ellipse_mode_button.clicked.connect(lambda: self._set_mode("椭圆/椭圆弧"))
        self.box_mode_button.clicked.connect(lambda: self._set_mode("长方体"))
        self.sphere_mode_button.clicked.connect(lambda: self._set_mode("圆球"))
        self.cylinder_mode_button.clicked.connect(lambda: self._set_mode("圆柱"))
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
        mode = self._current_mode()
        self.point_form.setVisible(mode == "点")
        self.line_form.setVisible(mode == "直线")
        self.circle_form.setVisible(mode == "圆/圆弧")
        self.curve_form.setVisible(mode == "曲线")
        self.polyline_form.setVisible(mode == "多段线/折线")
        self.rectangle_form.setVisible(mode == "矩形")
        self.polygon_form.setVisible(mode == "多边形")
        self.ellipse_form.setVisible(mode == "椭圆/椭圆弧")
        self.box_form.setVisible(mode == "长方体")
        self.sphere_form.setVisible(mode == "圆球")
        self.cylinder_form.setVisible(mode == "圆柱")

    def _set_mode(self, mode):
        button_map = {
            "点": self.point_mode_button,
            "直线": self.line_mode_button,
            "圆/圆弧": self.circle_mode_button,
            "曲线": self.curve_mode_button,
            "多段线/折线": self.polyline_mode_button,
            "矩形": self.rectangle_mode_button,
            "多边形": self.polygon_mode_button,
            "椭圆/椭圆弧": self.ellipse_mode_button,
            "长方体": self.box_mode_button,
            "圆球": self.sphere_mode_button,
            "圆柱": self.cylinder_mode_button,
        }
        for key, button in button_map.items():
            button.setChecked(key == mode)
        self.point_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "点"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.line_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "直线"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.circle_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "圆/圆弧"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.curve_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "曲线"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.polyline_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "多段线/折线"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.rectangle_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "矩形"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.polygon_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "多边形"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.ellipse_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "椭圆/椭圆弧"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.box_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "长方体"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.sphere_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "圆球"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self.cylinder_mode_button.setStyleSheet(
            UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET
            if mode == "圆柱"
            else UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET
        )
        self._apply_mode_visibility()

    def _current_mode(self):
        if self.line_mode_button.isChecked():
            return "直线"
        if self.circle_mode_button.isChecked():
            return "圆/圆弧"
        if self.curve_mode_button.isChecked():
            return "曲线"
        if self.polyline_mode_button.isChecked():
            return "多段线/折线"
        if self.rectangle_mode_button.isChecked():
            return "矩形"
        if self.polygon_mode_button.isChecked():
            return "多边形"
        if self.ellipse_mode_button.isChecked():
            return "椭圆/椭圆弧"
        if self.box_mode_button.isChecked():
            return "长方体"
        if self.sphere_mode_button.isChecked():
            return "圆球"
        if self.cylinder_mode_button.isChecked():
            return "圆柱"
        return "点"

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
        mode = self._current_mode()
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
        elif mode == "多段线/折线":
            row = self.polyline_table.rowCount()
            self.polyline_table.insertRow(row)
            self.polyline_table.setItem(row, 0, QTableWidgetItem(f"{point[0]:.6f}"))
            self.polyline_table.setItem(row, 1, QTableWidgetItem(f"{point[1]:.6f}"))
            self.polyline_table.setItem(row, 2, QTableWidgetItem(f"{point[2]:.6f}"))
        elif mode == "矩形":
            if len(self._picked_points) == 1:
                self.rx1.setValue(point[0])
                self.ry1.setValue(point[1])
                self.rz1.setValue(point[2])
            elif len(self._picked_points) >= 2:
                self.rx2.setValue(point[0])
                self.ry2.setValue(point[1])
                self.rz2.setValue(point[2])
                if self.gui and hasattr(self.gui, 'view_controller'):
                    self.gui.view_controller.stop_point_pick()
                    self.pick_button.setText("进入拾取")
        elif mode == "椭圆/椭圆弧":
            if len(self._picked_points) == 1:
                self.ex.setValue(point[0])
                self.ey.setValue(point[1])
                self.ez.setValue(point[2])
            elif len(self._picked_points) == 2:
                dx = point[0] - self.ex.value()
                dy = point[1] - self.ey.value()
                dz = point[2] - self.ez.value()
                self.major_radius.setValue((dx * dx + dy * dy + dz * dz) ** 0.5)
            elif len(self._picked_points) >= 3:
                dx = point[0] - self.ex.value()
                dy = point[1] - self.ey.value()
                dz = point[2] - self.ez.value()
                self.minor_radius.setValue((dx * dx + dy * dy + dz * dz) ** 0.5)
                if self.gui and hasattr(self.gui, 'view_controller'):
                    self.gui.view_controller.stop_point_pick()
                    self.pick_button.setText("进入拾取")
        elif mode == "长方体":
            if len(self._picked_points) == 1:
                self.bx1.setValue(point[0])
                self.by1.setValue(point[1])
                self.bz1.setValue(point[2])
            elif len(self._picked_points) >= 2:
                self.bx2.setValue(point[0])
                self.by2.setValue(point[1])
                self.bz2.setValue(point[2])
                if self.gui and hasattr(self.gui, 'view_controller'):
                    self.gui.view_controller.stop_point_pick()
                    self.pick_button.setText("进入拾取")
        elif mode == "圆球":
            if len(self._picked_points) == 1:
                self.sx.setValue(point[0])
                self.sy.setValue(point[1])
                self.sz.setValue(point[2])
            elif len(self._picked_points) >= 2:
                dx = point[0] - self.sx.value()
                dy = point[1] - self.sy.value()
                dz = point[2] - self.sz.value()
                self.sr.setValue((dx * dx + dy * dy + dz * dz) ** 0.5)
                if self.gui and hasattr(self.gui, 'view_controller'):
                    self.gui.view_controller.stop_point_pick()
                    self.pick_button.setText("进入拾取")
        elif mode == "圆柱":
            if len(self._picked_points) == 1:
                self.cx3.setValue(point[0])
                self.cy3.setValue(point[1])
                self.cz3.setValue(point[2])
            elif len(self._picked_points) == 2:
                dx = point[0] - self.cx3.value()
                dy = point[1] - self.cy3.value()
                dz = point[2] - self.cz3.value()
                self.cr.setValue((dx * dx + dy * dy + dz * dz) ** 0.5)
            elif len(self._picked_points) >= 3:
                self.ch.setValue(point[2] - self.cz3.value())
                if self.gui and hasattr(self.gui, 'view_controller'):
                    self.gui.view_controller.stop_point_pick()
                    self.pick_button.setText("进入拾取")
        elif mode == "多边形":
            row = self.polygon_table.rowCount()
            self.polygon_table.insertRow(row)
            self.polygon_table.setItem(row, 0, QTableWidgetItem(f"{point[0]:.6f}"))
            self.polygon_table.setItem(row, 1, QTableWidgetItem(f"{point[1]:.6f}"))
            self.polygon_table.setItem(row, 2, QTableWidgetItem(f"{point[2]:.6f}"))

    def _create_geometry(self):
        if not self.gui or not hasattr(self.gui, 'geometry_operations'):
            QMessageBox.warning(self, "警告", "几何操作器未初始化")
            return
        mode = self._current_mode()
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
        elif mode == "多段线/折线":
            points = self._get_table_points(self.polyline_table)
            if len(points) < 2:
                QMessageBox.warning(self, "警告", "多段线至少需要两个点")
                return
            self.gui.geometry_operations.create_geometry_polyline(points)
        elif mode == "矩形":
            p1 = (self.rx1.value(), self.ry1.value(), self.rz1.value())
            p2 = (self.rx2.value(), self.ry2.value(), self.rz2.value())
            self.gui.geometry_operations.create_geometry_rectangle(p1, p2)
        elif mode == "多边形":
            points = self._get_table_points(self.polygon_table)
            if len(points) < 3:
                QMessageBox.warning(self, "警告", "多边形至少需要三个点")
                return
            self.gui.geometry_operations.create_geometry_polygon(points)
        elif mode == "椭圆/椭圆弧":
            center = (self.ex.value(), self.ey.value(), self.ez.value())
            self.gui.geometry_operations.create_geometry_ellipse(
                center,
                self.major_radius.value(),
                self.minor_radius.value(),
                start_angle=self.ellipse_start_angle.value(),
                end_angle=self.ellipse_end_angle.value(),
            )
        elif mode == "长方体":
            p1 = (self.bx1.value(), self.by1.value(), self.bz1.value())
            p2 = (self.bx2.value(), self.by2.value(), self.bz2.value())
            self.gui.geometry_operations.create_geometry_box(p1, p2)
        elif mode == "圆球":
            center = (self.sx.value(), self.sy.value(), self.sz.value())
            self.gui.geometry_operations.create_geometry_sphere(center, self.sr.value())
        elif mode == "圆柱":
            base_center = (self.cx3.value(), self.cy3.value(), self.cz3.value())
            self.gui.geometry_operations.create_geometry_cylinder(base_center, self.cr.value(), self.ch.value())
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

    def _get_table_points(self, table):
        points = []
        for row in range(table.rowCount()):
            try:
                x = float(table.item(row, 0).text())
                y = float(table.item(row, 1).text())
                z = float(table.item(row, 2).text())
                points.append((x, y, z))
            except Exception:
                continue
        return points

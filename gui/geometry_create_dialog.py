#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
几何创建对话框
支持创建点、直线、圆/圆弧、曲线（样条）
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QWidget, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QMessageBox, QLineEdit
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

        self._current_pick_target = None
        self._continuous_pick_mode = None
        self._last_picked_point = None
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
        self.p_coord_widget, self.p_coord_pick_btn = self._create_coord_input_with_pick()
        form.addRow("坐标 (x, y, z):", self.p_coord_widget)
        coord_layout.addWidget(self.point_form)

        self.line_form = QWidget()
        line_form_layout = QFormLayout(self.line_form)
        self.l1_coord_widget, self.l1_coord_pick_btn = self._create_coord_input_with_pick()
        self.l2_coord_widget, self.l2_coord_pick_btn = self._create_coord_input_with_pick()
        line_form_layout.addRow("P1 (x, y, z):", self.l1_coord_widget)
        line_form_layout.addRow("P2 (x, y, z):", self.l2_coord_widget)
        coord_layout.addWidget(self.line_form)

        self.circle_form = QWidget()
        circle_form_layout = QFormLayout(self.circle_form)
        self.c_coord_widget, self.c_coord_pick_btn = self._create_coord_input_with_pick()
        self.radius = self._create_spin(min_value=1e-6)
        self.start_angle = self._create_spin(min_value=-360.0, max_value=360.0, decimals=2)
        self.end_angle = self._create_spin(min_value=-360.0, max_value=360.0, decimals=2)
        circle_form_layout.addRow("圆心 (x, y, z):", self.c_coord_widget)
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
        curve_button_layout = QHBoxLayout()
        self.curve_pick_button = QPushButton("进入拾取")
        self.curve_pick_button.setStyleSheet(UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET)
        self.curve_clear_button = QPushButton("清空")
        self.curve_clear_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        curve_button_layout.addWidget(self.curve_pick_button)
        curve_button_layout.addWidget(self.curve_clear_button)
        curve_button_layout.addStretch()
        curve_layout.addLayout(curve_button_layout)
        coord_layout.addWidget(self.curve_form)

        self.polyline_form = QWidget()
        polyline_layout = QVBoxLayout(self.polyline_form)
        polyline_layout.addWidget(QLabel("多段线点集 (至少2点):"))
        self.polyline_table = QTableWidget(0, 3)
        self.polyline_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.polyline_table.horizontalHeader().setStretchLastSection(True)
        polyline_layout.addWidget(self.polyline_table)
        polyline_button_layout = QHBoxLayout()
        self.polyline_pick_button = QPushButton("进入拾取")
        self.polyline_pick_button.setStyleSheet(UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET)
        self.polyline_clear_button = QPushButton("清空")
        self.polyline_clear_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        polyline_button_layout.addWidget(self.polyline_pick_button)
        polyline_button_layout.addWidget(self.polyline_clear_button)
        polyline_button_layout.addStretch()
        polyline_layout.addLayout(polyline_button_layout)
        coord_layout.addWidget(self.polyline_form)

        self.rectangle_form = QWidget()
        rectangle_layout = QFormLayout(self.rectangle_form)
        self.r1_coord_widget, self.r1_coord_pick_btn = self._create_coord_input_with_pick()
        self.r2_coord_widget, self.r2_coord_pick_btn = self._create_coord_input_with_pick()
        rectangle_layout.addRow("P1 (x, y, z):", self.r1_coord_widget)
        rectangle_layout.addRow("P2 (x, y, z):", self.r2_coord_widget)
        coord_layout.addWidget(self.rectangle_form)

        self.polygon_form = QWidget()
        polygon_layout = QVBoxLayout(self.polygon_form)
        polygon_layout.addWidget(QLabel("多边形点集 (至少3点):"))
        self.polygon_table = QTableWidget(0, 3)
        self.polygon_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.polygon_table.horizontalHeader().setStretchLastSection(True)
        polygon_layout.addWidget(self.polygon_table)
        polygon_button_layout = QHBoxLayout()
        self.polygon_pick_button = QPushButton("进入拾取")
        self.polygon_pick_button.setStyleSheet(UIStyles.DIALOG_PRIMARY_BUTTON_STYLESHEET)
        self.polygon_clear_button = QPushButton("清空")
        self.polygon_clear_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        polygon_button_layout.addWidget(self.polygon_pick_button)
        polygon_button_layout.addWidget(self.polygon_clear_button)
        polygon_button_layout.addStretch()
        polygon_layout.addLayout(polygon_button_layout)
        coord_layout.addWidget(self.polygon_form)

        self.ellipse_form = QWidget()
        ellipse_layout = QFormLayout(self.ellipse_form)
        self.e_coord_widget, self.e_coord_pick_btn = self._create_coord_input_with_pick()
        self.major_radius = self._create_spin(min_value=1e-6)
        self.minor_radius = self._create_spin(min_value=1e-6)
        self.ellipse_start_angle = self._create_spin(min_value=-360.0, max_value=360.0, decimals=2)
        self.ellipse_end_angle = self._create_spin(min_value=-360.0, max_value=360.0, decimals=2)
        ellipse_layout.addRow("圆心 (x, y, z):", self.e_coord_widget)
        ellipse_layout.addRow("长半轴:", self.major_radius)
        ellipse_layout.addRow("短半轴:", self.minor_radius)
        ellipse_layout.addRow("起始角度(°):", self.ellipse_start_angle)
        ellipse_layout.addRow("终止角度(°):", self.ellipse_end_angle)
        coord_layout.addWidget(self.ellipse_form)

        self.box_form = QWidget()
        box_layout = QFormLayout(self.box_form)
        self.b1_coord_widget, self.b1_coord_pick_btn = self._create_coord_input_with_pick()
        self.b2_coord_widget, self.b2_coord_pick_btn = self._create_coord_input_with_pick()
        box_layout.addRow("P1 (x, y, z):", self.b1_coord_widget)
        box_layout.addRow("P2 (x, y, z):", self.b2_coord_widget)
        coord_layout.addWidget(self.box_form)

        self.sphere_form = QWidget()
        sphere_layout = QFormLayout(self.sphere_form)
        self.s_coord_widget, self.s_coord_pick_btn = self._create_coord_input_with_pick()
        self.sr = self._create_spin(min_value=1e-6)
        sphere_layout.addRow("圆心 (x, y, z):", self.s_coord_widget)
        sphere_layout.addRow("半径:", self.sr)
        coord_layout.addWidget(self.sphere_form)

        self.cylinder_form = QWidget()
        cylinder_layout = QFormLayout(self.cylinder_form)
        self.cy3_coord_widget, self.cy3_coord_pick_btn = self._create_coord_input_with_pick()
        self.cr = self._create_spin(min_value=1e-6)
        self.ch = self._create_spin(min_value=1e-6)
        cylinder_layout.addRow("底面圆心 (x, y, z):", self.cy3_coord_widget)
        cylinder_layout.addRow("半径:", self.cr)
        cylinder_layout.addRow("高度:", self.ch)
        coord_layout.addWidget(self.cylinder_form)

        layout.addWidget(coord_group)

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

    def _create_coord_input_with_pick(self):
        """创建带拾取按钮的坐标输入框"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        line_edit = QLineEdit()
        line_edit.setPlaceholderText("x, y, z")
        line_edit.setStyleSheet("background-color: white; padding: 2px;")
        
        pick_btn = QPushButton("拾取")
        pick_btn.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        pick_btn.setMaximumWidth(60)
        
        layout.addWidget(line_edit, 1)
        layout.addWidget(pick_btn, 0)
        
        widget.line_edit = line_edit
        widget.pick_btn = pick_btn
        
        return widget, pick_btn

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
        self.create_button.clicked.connect(self._create_geometry)
        self.cancel_button.clicked.connect(self.close)

        self.curve_pick_button.clicked.connect(lambda: self._start_continuous_pick("curve"))
        self.curve_clear_button.clicked.connect(lambda: self._clear_table(self.curve_table))
        self.polyline_pick_button.clicked.connect(lambda: self._start_continuous_pick("polyline"))
        self.polyline_clear_button.clicked.connect(lambda: self._clear_table(self.polyline_table))
        self.polygon_pick_button.clicked.connect(lambda: self._start_continuous_pick("polygon"))
        self.polygon_clear_button.clicked.connect(lambda: self._clear_table(self.polygon_table))

        self.p_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.p_coord_widget.line_edit))
        self.l1_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.l1_coord_widget.line_edit))
        self.l2_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.l2_coord_widget.line_edit))
        self.c_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.c_coord_widget.line_edit))
        self.r1_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.r1_coord_widget.line_edit))
        self.r2_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.r2_coord_widget.line_edit))
        self.e_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.e_coord_widget.line_edit))
        self.b1_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.b1_coord_widget.line_edit))
        self.b2_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.b2_coord_widget.line_edit))
        self.s_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.s_coord_widget.line_edit))
        self.cy3_coord_pick_btn.clicked.connect(lambda: self._start_single_pick(self.cy3_coord_widget.line_edit))

    def _create_spin(self, min_value=-1e6, max_value=1e6, decimals=3):
        spin = QDoubleSpinBox()
        spin.setRange(min_value, max_value)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1)
        spin.setStyleSheet("background-color: white;")
        return spin

    def _parse_coord_input(self, text):
        """解析坐标输入，支持逗号或空格分隔，返回 (x, y, z) 元组"""
        if not text or not text.strip():
            return (0.0, 0.0, 0.0)
        try:
            text = text.strip()
            if ',' in text:
                parts = [p.strip() for p in text.split(',')]
            else:
                parts = text.split()
            
            x = float(parts[0]) if len(parts) > 0 else 0.0
            y = float(parts[1]) if len(parts) > 1 else 0.0
            z = float(parts[2]) if len(parts) > 2 else 0.0
            return (x, y, z)
        except (ValueError, IndexError):
            return (0.0, 0.0, 0.0)

    def _format_coord_output(self, x, y, z):
        """格式化坐标输出为字符串"""
        return f"{x:.6f}, {y:.6f}, {z:.6f}"

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

    def _start_single_pick(self, target_widget):
        """启动单个坐标拾取"""
        if not self.gui or not hasattr(self.gui, 'view_controller'):
            QMessageBox.warning(self, "警告", "未找到视图控制器")
            return
        self._current_pick_target = target_widget
        self.gui.view_controller.set_point_pick_callbacks(
            on_confirm=self._on_single_pick_confirm,
            on_cancel=self._on_single_pick_cancel,
            on_exit=self._on_single_pick_exit,
        )
        self.gui.view_controller.start_point_pick(self._on_single_point_picked)
        if hasattr(self.gui, 'log_info'):
            self.gui.log_info("点拾取已开启: 点击视图拾取点坐标")

    def _on_single_point_picked(self, point):
        """单个点拾取回调"""
        if self._current_pick_target:
            self._current_pick_target.setText(self._format_coord_output(point[0], point[1], point[2]))
            self._current_pick_target = None
        if self.gui and hasattr(self.gui, 'view_controller'):
            self.gui.view_controller.stop_point_pick()

    def _on_single_pick_confirm(self):
        """单个点拾取确认回调"""
        pass

    def _on_single_pick_cancel(self):
        """单个点拾取取消回调"""
        if self._current_pick_target:
            self._current_pick_target = None
        if self.gui and hasattr(self.gui, 'view_controller'):
            self.gui.view_controller.stop_point_pick()

    def _on_single_pick_exit(self):
        """单个点拾取退出回调"""
        if self._current_pick_target:
            self._current_pick_target = None

    def _start_continuous_pick(self, mode):
        """启动连续拾取模式"""
        if not self.gui or not hasattr(self.gui, 'view_controller'):
            QMessageBox.warning(self, "警告", "未找到视图控制器")
            return
        self._continuous_pick_mode = mode
        self.gui.view_controller.set_point_pick_callbacks(
            on_confirm=self._on_continuous_pick_confirm,
            on_cancel=self._on_continuous_pick_cancel,
            on_exit=self._on_continuous_pick_exit,
        )
        self.gui.view_controller.start_point_pick(self._on_continuous_point_picked)
        if hasattr(self.gui, 'log_info'):
            self.gui.log_info("连续拾取已开启: 左键选中，右键取消，Enter键确认添加点，Esc退出拾取模式")

    def _on_continuous_point_picked(self, point):
        """连续拾取点回调（左键选中）"""
        self._last_picked_point = point

    def _on_continuous_pick_confirm(self):
        """连续拾取确认回调（Enter键确认，添加点到表格）"""
        if self._last_picked_point is None:
            return
        
        point = self._last_picked_point
        table = None
        if self._continuous_pick_mode == "curve":
            table = self.curve_table
        elif self._continuous_pick_mode == "polyline":
            table = self.polyline_table
        elif self._continuous_pick_mode == "polygon":
            table = self.polygon_table
        
        if table:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(f"{point[0]:.6f}"))
            table.setItem(row, 1, QTableWidgetItem(f"{point[1]:.6f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{point[2]:.6f}"))
            if hasattr(self.gui, 'log_info'):
                self.gui.log_info(f"已添加点 ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")

    def _on_continuous_pick_cancel(self):
        """连续拾取取消回调（右键取消，移除最后一个点）"""
        table = None
        if self._continuous_pick_mode == "curve":
            table = self.curve_table
        elif self._continuous_pick_mode == "polyline":
            table = self.polyline_table
        elif self._continuous_pick_mode == "polygon":
            table = self.polygon_table
        
        if table and table.rowCount() > 0:
            table.removeRow(table.rowCount() - 1)
            if hasattr(self.gui, 'log_info'):
                self.gui.log_info("已移除最后一个点")

    def _on_continuous_pick_exit(self):
        """连续拾取退出回调"""
        self._continuous_pick_mode = None
        if self.gui and hasattr(self.gui, 'view_controller'):
            self.gui.view_controller.stop_point_pick()

    def _clear_table(self, table):
        """清空表格"""
        table.setRowCount(0)

    def _create_geometry(self):
        if not self.gui or not hasattr(self.gui, 'geometry_operations'):
            QMessageBox.warning(self, "警告", "几何操作器未初始化")
            return
        mode = self._current_mode()
        if mode == "点":
            points = [self._parse_coord_input(self.p_coord_widget.line_edit.text())]
            self.gui.geometry_operations.create_geometry_from_points(points, mode="point")
        elif mode == "直线":
            points = [
                self._parse_coord_input(self.l1_coord_widget.line_edit.text()),
                self._parse_coord_input(self.l2_coord_widget.line_edit.text()),
            ]
            self.gui.geometry_operations.create_geometry_from_points(points, mode="line")
        elif mode == "圆/圆弧":
            center = self._parse_coord_input(self.c_coord_widget.line_edit.text())
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
            p1 = self._parse_coord_input(self.r1_coord_widget.line_edit.text())
            p2 = self._parse_coord_input(self.r2_coord_widget.line_edit.text())
            self.gui.geometry_operations.create_geometry_rectangle(p1, p2)
        elif mode == "多边形":
            points = self._get_table_points(self.polygon_table)
            if len(points) < 3:
                QMessageBox.warning(self, "警告", "多边形至少需要三个点")
                return
            self.gui.geometry_operations.create_geometry_polygon(points)
        elif mode == "椭圆/椭圆弧":
            center = self._parse_coord_input(self.e_coord_widget.line_edit.text())
            self.gui.geometry_operations.create_geometry_ellipse(
                center,
                self.major_radius.value(),
                self.minor_radius.value(),
                start_angle=self.ellipse_start_angle.value(),
                end_angle=self.ellipse_end_angle.value(),
            )
        elif mode == "长方体":
            p1 = self._parse_coord_input(self.b1_coord_widget.line_edit.text())
            p2 = self._parse_coord_input(self.b2_coord_widget.line_edit.text())
            self.gui.geometry_operations.create_geometry_box(p1, p2)
        elif mode == "圆球":
            center = self._parse_coord_input(self.s_coord_widget.line_edit.text())
            self.gui.geometry_operations.create_geometry_sphere(center, self.sr.value())
        elif mode == "圆柱":
            base_center = self._parse_coord_input(self.cy3_coord_widget.line_edit.text())
            self.gui.geometry_operations.create_geometry_cylinder(base_center, self.cr.value(), self.ch.value())

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

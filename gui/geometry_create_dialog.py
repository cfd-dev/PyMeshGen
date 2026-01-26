#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
几何创建对话框
支持创建点、直线、圆/圆弧、曲线（样条）
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QWidget, QDoubleSpinBox,
    QMessageBox, QLineEdit, QCheckBox
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
        curve_form_layout = QFormLayout(self.curve_form)
        self.curve_num_points = self._create_spin(min_value=2, max_value=100, decimals=0)
        self.curve_num_points.setValue(2)
        curve_form_layout.addRow("点数量:", self.curve_num_points)
        self.curve_point_inputs_layout = QVBoxLayout()
        self.curve_point_widgets = []
        for i in range(2):  # 默认显示2个点
            widget, btn = self._create_coord_input_with_pick()
            self.curve_point_widgets.append((widget, btn))
            self.curve_point_inputs_layout.addWidget(widget)
        curve_form_layout.addRow("点坐标:", self.curve_point_inputs_layout)
        self.curve_add_point_button = QPushButton("增加点")
        self.curve_add_point_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.curve_remove_point_button = QPushButton("减少点")
        self.curve_remove_point_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        point_button_layout = QHBoxLayout()
        point_button_layout.addWidget(self.curve_add_point_button)
        point_button_layout.addWidget(self.curve_remove_point_button)
        point_button_layout.addStretch()
        curve_form_layout.addRow("", point_button_layout)
        coord_layout.addWidget(self.curve_form)

        self.polyline_form = QWidget()
        polyline_form_layout = QFormLayout(self.polyline_form)
        self.polyline_num_points = self._create_spin(min_value=2, max_value=100, decimals=0)
        self.polyline_num_points.setValue(2)
        polyline_form_layout.addRow("点数量:", self.polyline_num_points)
        self.polyline_point_inputs_layout = QVBoxLayout()
        self.polyline_point_widgets = []
        for i in range(2):  # 默认显示2个点
            widget, btn = self._create_coord_input_with_pick()
            self.polyline_point_widgets.append((widget, btn))
            self.polyline_point_inputs_layout.addWidget(widget)
        polyline_form_layout.addRow("点坐标:", self.polyline_point_inputs_layout)
        self.polyline_add_point_button = QPushButton("增加点")
        self.polyline_add_point_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.polyline_remove_point_button = QPushButton("减少点")
        self.polyline_remove_point_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        polyline_button_layout = QHBoxLayout()
        polyline_button_layout.addWidget(self.polyline_add_point_button)
        polyline_button_layout.addWidget(self.polyline_remove_point_button)
        polyline_button_layout.addStretch()
        polyline_form_layout.addRow("", polyline_button_layout)
        coord_layout.addWidget(self.polyline_form)

        self.rectangle_form = QWidget()
        rectangle_layout = QFormLayout(self.rectangle_form)
        self.r1_coord_widget, self.r1_coord_pick_btn = self._create_coord_input_with_pick()
        self.r2_coord_widget, self.r2_coord_pick_btn = self._create_coord_input_with_pick()
        rectangle_layout.addRow("P1 (x, y, z):", self.r1_coord_widget)
        rectangle_layout.addRow("P2 (x, y, z):", self.r2_coord_widget)
        coord_layout.addWidget(self.rectangle_form)

        self.polygon_form = QWidget()
        polygon_form_layout = QFormLayout(self.polygon_form)
        self.polygon_num_points = self._create_spin(min_value=3, max_value=100, decimals=0)
        self.polygon_num_points.setValue(3)
        polygon_form_layout.addRow("点数量:", self.polygon_num_points)
        self.polygon_point_inputs_layout = QVBoxLayout()
        self.polygon_point_widgets = []
        for i in range(3):  # 默认显示3个点
            widget, btn = self._create_coord_input_with_pick()
            self.polygon_point_widgets.append((widget, btn))
            self.polygon_point_inputs_layout.addWidget(widget)
        polygon_form_layout.addRow("点坐标:", self.polygon_point_inputs_layout)
        self.polygon_add_point_button = QPushButton("增加点")
        self.polygon_add_point_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        self.polygon_remove_point_button = QPushButton("减少点")
        self.polygon_remove_point_button.setStyleSheet(UIStyles.DIALOG_SECONDARY_BUTTON_STYLESHEET)
        polygon_button_layout = QHBoxLayout()
        polygon_button_layout.addWidget(self.polygon_add_point_button)
        polygon_button_layout.addWidget(self.polygon_remove_point_button)
        polygon_button_layout.addStretch()
        polygon_form_layout.addRow("", polygon_button_layout)
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

        options_group = QGroupBox("拾取选项")
        options_layout = QHBoxLayout(options_group)
        self.snap_checkbox = QCheckBox("启用磁吸")
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.setToolTip("启用后，拾取点时会自动吸附到附近的几何点")
        options_layout.addWidget(self.snap_checkbox)
        options_layout.addStretch()
        layout.addWidget(options_group)

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
        self.snap_checkbox.toggled.connect(self._on_snap_toggled)

        self.curve_num_points.valueChanged.connect(lambda: self._update_point_inputs("curve"))
        self.curve_add_point_button.clicked.connect(lambda: self._add_point_input("curve"))
        self.curve_remove_point_button.clicked.connect(lambda: self._remove_point_input("curve"))
        self.polyline_num_points.valueChanged.connect(lambda: self._update_point_inputs("polyline"))
        self.polyline_add_point_button.clicked.connect(lambda: self._add_point_input("polyline"))
        self.polyline_remove_point_button.clicked.connect(lambda: self._remove_point_input("polyline"))
        self.polygon_num_points.valueChanged.connect(lambda: self._update_point_inputs("polygon"))
        self.polygon_add_point_button.clicked.connect(lambda: self._add_point_input("polygon"))
        self.polygon_remove_point_button.clicked.connect(lambda: self._remove_point_input("polygon"))

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

        # 初始化曲线和折线的拾取按钮连接
        self._connect_curve_polyline_buttons()

    def _create_spin(self, min_value=-1e6, max_value=1e6, decimals=3):
        spin = QDoubleSpinBox()
        spin.setRange(min_value, max_value)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1)
        spin.setStyleSheet("background-color: white;")
        return spin

    def _connect_curve_polyline_buttons(self):
        """连接曲线、折线和多边形的拾取按钮"""
        # 连接初始的曲线拾取按钮
        for widget, btn in self.curve_point_widgets:
            btn.clicked.connect(lambda checked=False, w=widget: self._start_single_pick(w.line_edit))

        # 连接初始的折线拾取按钮
        for widget, btn in self.polyline_point_widgets:
            btn.clicked.connect(lambda checked=False, w=widget: self._start_single_pick(w.line_edit))

        # 连接初始的多边形拾取按钮
        for widget, btn in self.polygon_point_widgets:
            btn.clicked.connect(lambda checked=False, w=widget: self._start_single_pick(w.line_edit))

    def _update_point_inputs(self, mode):
        """根据点数更新输入控件"""
        if mode == "curve":
            widgets_list = self.curve_point_widgets
            inputs_layout = self.curve_point_inputs_layout
            num_points = int(self.curve_num_points.value())
        elif mode == "polyline":
            widgets_list = self.polyline_point_widgets
            inputs_layout = self.polyline_point_inputs_layout
            num_points = int(self.polyline_num_points.value())
        elif mode == "polygon":
            widgets_list = self.polygon_point_widgets
            inputs_layout = self.polygon_point_inputs_layout
            num_points = int(self.polygon_num_points.value())
        else:
            return

        current_count = len(widgets_list)

        # 添加缺少的输入控件
        while current_count < num_points:
            widget, btn = self._create_coord_input_with_pick()
            widgets_list.append((widget, btn))
            inputs_layout.addWidget(widget)
            # 连接新添加的拾取按钮
            btn.clicked.connect(lambda checked=False, w=widget: self._start_single_pick(w.line_edit))
            current_count += 1

        # 移除多余的输入控件
        while current_count > num_points:
            widget, btn = widgets_list.pop()
            # 断开信号连接
            try:
                btn.clicked.disconnect()
            except TypeError:
                # 如果没有连接信号，disconnect会抛出异常，忽略即可
                pass
            inputs_layout.removeWidget(widget)
            widget.setParent(None)  # 完全移除控件
            current_count -= 1

    def _add_point_input(self, mode):
        """增加一个点输入控件"""
        if mode == "curve":
            current_value = self.curve_num_points.value()
            self.curve_num_points.setValue(current_value + 1)
        elif mode == "polyline":
            current_value = self.polyline_num_points.value()
            self.polyline_num_points.setValue(current_value + 1)
        elif mode == "polygon":
            current_value = self.polygon_num_points.value()
            self.polygon_num_points.setValue(current_value + 1)

    def _remove_point_input(self, mode):
        """减少一个点输入控件（最少保留2个点）"""
        if mode == "curve":
            current_value = self.curve_num_points.value()
            if current_value > 2:
                self.curve_num_points.setValue(current_value - 1)
        elif mode == "polyline":
            current_value = self.polyline_num_points.value()
            if current_value > 2:
                self.polyline_num_points.setValue(current_value - 1)
        elif mode == "polygon":
            current_value = self.polygon_num_points.value()
            if current_value > 3:
                self.polygon_num_points.setValue(current_value - 1)

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

    def _on_snap_toggled(self, checked):
        """磁吸复选框切换回调"""
        if self.gui and hasattr(self.gui, 'view_controller'):
            self.gui.view_controller.set_point_snap_enabled(checked)
            status = "启用" if checked else "禁用"
            if hasattr(self.gui, 'log_info'):
                self.gui.log_info(f"磁吸功能已{status}")

    def _start_single_pick(self, target_widget):
        """启动单个坐标拾取"""
        if not self.gui or not hasattr(self.gui, 'view_controller'):
            QMessageBox.warning(self, "警告", "未找到视图控制器")
            return

        def on_point_picked(point):
            """单个点拾取回调"""
            # 检查是否拾取到了现有点
            is_existing_point = False
            if self.gui and hasattr(self.gui, 'view_controller') and hasattr(self.gui.view_controller, '_picking_helper'):
                picking_helper = self.gui.view_controller._picking_helper
                if picking_helper and hasattr(picking_helper, '_find_vertex_by_point'):
                    vertex_obj = picking_helper._find_vertex_by_point(point)
                    if vertex_obj:
                        is_existing_point = True
                        # 输出拾取到现有点的信息
                        if hasattr(self.gui, 'log_info'):
                            self.gui.log_info(f"拾取到现有点: {vertex_obj}")

            if not is_existing_point:
                # 输出拾取到新点的信息
                if hasattr(self.gui, 'log_info'):
                    self.gui.log_info(f"拾取到新点坐标: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")

            if target_widget:
                target_widget.setText(self._format_coord_output(point[0], point[1], point[2]))

        def on_confirm():
            """单个点拾取确认回调"""
            if self.gui and hasattr(self.gui, 'view_controller'):
                self.gui.view_controller.stop_point_pick()

        def on_cancel():
            """单个点拾取取消回调"""
            if target_widget:
                target_widget = None
            if self.gui and hasattr(self.gui, 'view_controller'):
                self.gui.view_controller.stop_point_pick()

        def on_exit():
            """单个点拾取退出回调"""
            if target_widget:
                target_widget = None
            if self.gui and hasattr(self.gui, 'view_controller'):
                self.gui.view_controller.stop_point_pick()

        self._start_general_point_pick(
            on_point_picked=on_point_picked,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            on_exit=on_exit,
            continuous_mode=False,
            target_widget=target_widget
        )

    def _start_general_point_pick(self, on_point_picked, on_confirm, on_cancel, on_exit, continuous_mode=False, target_widget=None):
        """通用的点拾取方法"""
        if not self.gui or not hasattr(self.gui, 'view_controller'):
            QMessageBox.warning(self, "警告", "未找到视图控制器")
            return

        # 设置拾取目标（如果是单个点拾取）
        if target_widget is not None:
            self._current_pick_target = target_widget

        snap_enabled = self.snap_checkbox.isChecked()
        self.gui.view_controller.set_point_snap_enabled(snap_enabled)

        self.gui.view_controller.set_point_pick_callbacks(
            on_pick=on_point_picked,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            on_exit=on_exit,
        )
        self.gui.view_controller.start_point_pick(on_point_picked)

        if hasattr(self.gui, 'log_info'):
            snap_status = "启用" if snap_enabled else "禁用"
            self.gui.log_info(f"点拾取已开启: 点击视图拾取点坐标 (磁吸{snap_status})")



    def close(self):
        """关闭对话框时停止点拾取模式"""
        if self.gui and hasattr(self.gui, 'view_controller'):
            if self.gui.view_controller.is_point_pick_active():
                self.gui.view_controller.stop_point_pick()
        super().close()

    def _create_geometry(self):
        if not self.gui or not hasattr(self.gui, 'geometry_operations'):
            QMessageBox.warning(self, "警告", "几何操作器未初始化")
            return
        
        if self.gui.view_controller.is_point_pick_active():
            self.gui.view_controller.stop_point_pick()
        
        mode = self._current_mode()
        result = None
        
        if mode == "点":
            points = [self._parse_coord_input(self.p_coord_widget.line_edit.text())]
            result = self.gui.geometry_operations.create_geometry_from_points(points, mode="point")
        elif mode == "直线":
            points = [
                self._parse_coord_input(self.l1_coord_widget.line_edit.text()),
                self._parse_coord_input(self.l2_coord_widget.line_edit.text()),
            ]
            result = self.gui.geometry_operations.create_geometry_from_points(points, mode="line")
        elif mode == "圆/圆弧":
            center = self._parse_coord_input(self.c_coord_widget.line_edit.text())
            radius = self.radius.value()
            result = self.gui.geometry_operations.create_geometry_circle(
                center,
                radius,
                start_angle=self.start_angle.value(),
                end_angle=self.end_angle.value()
            )
        elif mode == "曲线":
            points = []
            for widget, btn in self.curve_point_widgets:
                point = self._parse_coord_input(widget.line_edit.text())
                points.append(point)
            if len(points) < 2:
                QMessageBox.warning(self, "警告", "曲线至少需要两个点")
                return
            result = self.gui.geometry_operations.create_geometry_from_points(points, mode="curve")
        elif mode == "多段线/折线":
            points = []
            for widget, btn in self.polyline_point_widgets:
                point = self._parse_coord_input(widget.line_edit.text())
                points.append(point)
            if len(points) < 2:
                QMessageBox.warning(self, "警告", "多段线至少需要两个点")
                return
            result = self.gui.geometry_operations.create_geometry_polyline(points)
        elif mode == "矩形":
            p1 = self._parse_coord_input(self.r1_coord_widget.line_edit.text())
            p2 = self._parse_coord_input(self.r2_coord_widget.line_edit.text())
            result = self.gui.geometry_operations.create_geometry_rectangle(p1, p2)
        elif mode == "多边形":
            points = []
            for widget, btn in self.polygon_point_widgets:
                point = self._parse_coord_input(widget.line_edit.text())
                points.append(point)
            if len(points) < 3:
                QMessageBox.warning(self, "警告", "多边形至少需要三个点")
                return
            result = self.gui.geometry_operations.create_geometry_polygon(points)
        elif mode == "椭圆/椭圆弧":
            center = self._parse_coord_input(self.e_coord_widget.line_edit.text())
            result = self.gui.geometry_operations.create_geometry_ellipse(
                center,
                self.major_radius.value(),
                self.minor_radius.value(),
                start_angle=self.ellipse_start_angle.value(),
                end_angle=self.ellipse_end_angle.value(),
            )
        elif mode == "长方体":
            p1 = self._parse_coord_input(self.b1_coord_widget.line_edit.text())
            p2 = self._parse_coord_input(self.b2_coord_widget.line_edit.text())
            result = self.gui.geometry_operations.create_geometry_box(p1, p2)
        elif mode == "圆球":
            center = self._parse_coord_input(self.s_coord_widget.line_edit.text())
            result = self.gui.geometry_operations.create_geometry_sphere(center, self.sr.value())
        elif mode == "圆柱":
            base_center = self._parse_coord_input(self.cy3_coord_widget.line_edit.text())
            result = self.gui.geometry_operations.create_geometry_cylinder(base_center, self.cr.value(), self.ch.value())
        
        if result and hasattr(self.gui, 'view_controller'):
            self.gui.view_controller.fit_view()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部件参数设置对话框
用于设置网格生成的部件参数
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QPushButton, QGroupBox, QLabel, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .ui_utils import UIStyles


class PartParamsDialog(QDialog):
    """部件参数设置对话框"""
    
    def __init__(self, parent=None, parts=None, current_part=0):
        """初始化部件参数对话框"""
        super().__init__(parent)
        self.setWindowTitle("部件参数设置")
        self.setModal(True)
        self.resize(600, 500)
        
        # 设置字体
        font = QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.setFont(font)
        
        # 初始化部件数据
        self.parts = parts or [
            {"part_name": "part1", "max_size": 1.0, "PRISM_SWITCH": "off", 
             "first_height": 0.01, "growth_rate": 1.2, "max_layers": 5, 
             "full_layers": 5, "multi_direction": False},
            {"part_name": "part2", "max_size": 1.0, "PRISM_SWITCH": "off", 
             "first_height": 0.01, "growth_rate": 1.2, "max_layers": 5, 
             "full_layers": 5, "multi_direction": False},
            {"part_name": "part3", "max_size": 1.0, "PRISM_SWITCH": "off", 
             "first_height": 0.01, "growth_rate": 1.2, "max_layers": 5, 
             "full_layers": 5, "multi_direction": False},
            {"part_name": "part4", "max_size": 1.0, "PRISM_SWITCH": "off", 
             "first_height": 0.01, "growth_rate": 1.2, "max_layers": 5, 
             "full_layers": 5, "multi_direction": False}
        ]
        
        self.current_part_index = current_part
        self._create_widgets()
        self._connect_signals()
        self._update_param_fields()
    
    def _create_widgets(self):
        """创建对话框部件"""
        main_layout = QVBoxLayout(self)
        
        # 部件选择区域
        select_group = QGroupBox("部件选择")
        select_layout = QHBoxLayout(select_group)
        
        select_layout.addWidget(QLabel("选择部件:"))
        
        self.part_combo = QComboBox()
        for i, part in enumerate(self.parts):
            self.part_combo.addItem(f"{i+1}. {part['part_name']}")
        self.part_combo.setCurrentIndex(self.current_part_index)
        select_layout.addWidget(self.part_combo)
        select_layout.addStretch()
        
        main_layout.addWidget(select_group)
        
        # 参数设置区域
        params_group = QGroupBox("参数设置")
        params_layout = QFormLayout(params_group)
        
        # 部件名称
        self.part_name_edit = QWidget()
        name_layout = QHBoxLayout(self.part_name_edit)
        self.part_name_label = QLabel()
        name_layout.addWidget(self.part_name_label)
        name_layout.addStretch()
        params_layout.addRow("部件名称:", self.part_name_edit)
        
        # 最大网格尺寸
        self.max_size_spin = QDoubleSpinBox()
        self.max_size_spin.setRange(0.001, 100.0)
        self.max_size_spin.setDecimals(3)
        self.max_size_spin.setSingleStep(0.1)
        params_layout.addRow("最大网格尺寸:", self.max_size_spin)
        
        # PRISM_SWITCH
        self.prism_switch_combo = QComboBox()
        self.prism_switch_combo.addItems(["wall", "off"])
        params_layout.addRow("棱柱层开关:", self.prism_switch_combo)
        
        # 第一层高度
        self.first_height_spin = QDoubleSpinBox()
        self.first_height_spin.setRange(1e-6, 1.0)
        self.first_height_spin.setDecimals(6)
        self.first_height_spin.setSingleStep(0.001)
        self.first_height_spin.setSpecialValueText("自动")
        params_layout.addRow("第一层高度:", self.first_height_spin)
        
        # 增长率
        self.growth_rate_spin = QDoubleSpinBox()
        self.growth_rate_spin.setRange(1.01, 2.0)
        self.growth_rate_spin.setDecimals(2)
        self.growth_rate_spin.setSingleStep(0.05)
        params_layout.addRow("增长率:", self.growth_rate_spin)
        
        # 最大层数
        self.max_layers_spin = QSpinBox()
        self.max_layers_spin.setRange(1, 100)
        self.max_layers_spin.setSingleStep(1)
        params_layout.addRow("最大层数:", self.max_layers_spin)
        
        # 完整层数
        self.full_layers_spin = QSpinBox()
        self.full_layers_spin.setRange(1, 100)
        self.full_layers_spin.setSingleStep(1)
        params_layout.addRow("完整层数:", self.full_layers_spin)
        
        # 多方向
        self.multi_direction_check = QCheckBox()
        params_layout.addRow("多方向:", self.multi_direction_check)
        
        main_layout.addWidget(params_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.save_button = QPushButton("保存")
        # 使用现有的按钮样式
        self.save_button.setStyleSheet("""
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
        button_layout.addWidget(self.save_button)
        
        self.cancel_button = QPushButton("取消")
        # 使用现有的按钮样式
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
        self.part_combo.currentIndexChanged.connect(self._on_part_changed)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
    
    def _on_part_changed(self, index):
        """当部件选择改变时更新参数"""
        # 保存当前部件的参数
        self._save_current_params()
        # 更新当前部件索引
        self.current_part_index = index
        # 更新参数字段
        self._update_param_fields()
    
    def _update_param_fields(self):
        """更新参数字段"""
        current_part = self.parts[self.current_part_index]
        self.part_name_label.setText(current_part["part_name"])
        self.max_size_spin.setValue(current_part["max_size"])
        
        # 设置PRISM_SWITCH
        prism_index = self.prism_switch_combo.findText(current_part["PRISM_SWITCH"])
        if prism_index >= 0:
            self.prism_switch_combo.setCurrentIndex(prism_index)
        
        self.first_height_spin.setValue(current_part["first_height"])
        self.growth_rate_spin.setValue(current_part["growth_rate"])
        self.max_layers_spin.setValue(current_part["max_layers"])
        self.full_layers_spin.setValue(current_part["full_layers"])
        self.multi_direction_check.setChecked(current_part["multi_direction"])
    
    def _save_current_params(self):
        """保存当前部件的参数"""
        current_part = self.parts[self.current_part_index]
        current_part["max_size"] = self.max_size_spin.value()
        current_part["PRISM_SWITCH"] = self.prism_switch_combo.currentText()
        current_part["first_height"] = self.first_height_spin.value()
        current_part["growth_rate"] = self.growth_rate_spin.value()
        current_part["max_layers"] = self.max_layers_spin.value()
        current_part["full_layers"] = self.full_layers_spin.value()
        current_part["multi_direction"] = self.multi_direction_check.isChecked()
    
    def accept(self):
        """接受对话框"""
        # 保存当前部件参数
        self._save_current_params()
        super().accept()
    
    def get_parts_params(self):
        """获取部件参数"""
        return self.parts
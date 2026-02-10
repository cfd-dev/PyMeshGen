#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt GUI基础组件模块
提供通用的UI元素和基础功能
"""

import sys
import time
from PyQt5.QtWidgets import (QStatusBar, QToolBar,
                             QVBoxLayout, QHBoxLayout, QSplitter, QFrame, QGroupBox,
                             QLabel, QTextEdit, QScrollArea, QTabWidget, QToolButton,
                             QDockWidget, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette


class StatusBar:
    """状态栏类"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.status_bar = main_window.statusBar()
        # Set a larger minimum height to ensure text fits
        self.status_bar.setMinimumHeight(25)
        self.status_bar.showMessage("就绪")

        # 创建进度条
        from PyQt5.QtWidgets import QProgressBar, QWidget, QHBoxLayout
        self.progress_widget = QWidget()
        progress_layout = QHBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(2, 1, 2, 1)  # Reduced padding
        progress_layout.setSpacing(4)  # Reduced spacing

        # Create a styled progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMinimumWidth(100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)  # Show percentage text
        self.progress_bar.setMaximumHeight(20)
        # Apply custom styling to the progress bar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #888888;
                border-radius: 3px;
                text-align: center;
                color: black;
                font-size: 9pt;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                stop:0 #4CAF50, stop:1 #2E7D32);
                border-radius: 2px;
            }
        """)

        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        self.progress_label.setMaximumWidth(250)
        self.progress_label.setMinimumWidth(100)
        # Apply styling to the progress label to match status bar font
        self.progress_label.setStyleSheet("""
            QLabel {
                font-size: 9pt;
                color: #333333;
                padding: 1px 2px;
                border: 1px solid transparent;
                border-radius: 2px;
            }
        """)

        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)

        # 将进度条添加到状态栏右侧
        self.status_bar.addPermanentWidget(self.progress_widget)
        self._mesh_dim_label = QLabel("网格维度: 2D")
        self._mesh_dim_label.setStyleSheet("""
            QLabel {
                font-size: 9pt;
                color: #333333;
                padding: 1px 6px;
                border: 1px solid transparent;
            }
        """)
        self.status_bar.addPermanentWidget(self._mesh_dim_label)

    def update_status(self, message):
        """更新状态栏信息"""
        self.status_bar.showMessage(message)

        # Ensure the status bar can show the full text by setting appropriate font
        font = self.status_bar.font()
        font.setPixelSize(9)  # 9px font size
        self.status_bar.setFont(font)

    def show_progress(self, message, progress):
        """显示进度条

        Args:
            message: 进度消息
            progress: 进度值 (0-100)
        """
        self.progress_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_label.setText(message)
        self.progress_bar.setValue(progress)

    def hide_progress(self):
        """隐藏进度条"""
        self.progress_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

    def update_mesh_dimension(self, dimension):
        """更新网格维度显示"""
        label = "2D" if dimension == 2 else "3D"
        self._mesh_dim_label.setText(f"网格维度: {label}")


class InfoOutput:
    """信息输出窗口类"""

    _COLOR_MAP = {
        "WARNING": "#FFA500",
        "ERROR": "#FF0000",
        "DEBUG": "#808080",
        "VERBOSE": "#A0A0A0",
    }

    _LEVEL_PATTERN = r"^(?:\[\d{1,2}:\d{1,2}:\d{1,2}\] )?\[(INFO|WARNING|ERROR|DEBUG|VERBOSE)\] "

    def __init__(self, parent=None):
        self.parent = parent
        self.create_info_output_area()
        
    def create_info_output_area(self):
        """创建信息输出窗口"""
        # 创建信息输出框架
        self.frame = QGroupBox("信息输出")
        layout = QVBoxLayout()
        layout.setSpacing(3)  # Reduced spacing

        # 创建文本框 and set smaller minimum size
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(60)  # Reduced minimum height
        self.info_text.setAcceptRichText(True)
        layout.addWidget(self.info_text)

        self.frame.setLayout(layout)

        # 设置默认大小
        self.frame.setMinimumHeight(80)  # Reduced minimum height

        return self.frame
    
    def append_info_output(self, message):
        """添加信息到输出窗口，确保格式统一：[timestamp] [LEVEL] message"""
        import re

        match = re.match(self._LEVEL_PATTERN, message)
        if match:
            level = match.group(1)
            clean_message = message[len(match.group(0)):]
        else:
            level = "INFO"
            clean_message = message

        timestamp = time.strftime("%H:%M:%S")
        color = self._COLOR_MAP.get(level)
        formatted_message = f"[{timestamp}] [{level}] {clean_message}"
        if color:
            self.info_text.append(f'<span style="color: {color};">{formatted_message}</span>')
        else:
            self._append_plain_text(formatted_message)

    def _append_plain_text(self, message):
        default_color = self.info_text.palette().color(QPalette.Text)
        self.info_text.setTextColor(default_color)
        self.info_text.append(message)
    
    def log_info(self, message):
        """记录信息"""
        self.append_info_output(f"[INFO] {message}")

    def log_error(self, message):
        """记录错误"""
        self.append_info_output(f"[ERROR] {message}")

    def log_warning(self, message):
        """记录警告"""
        self.append_info_output(f"[WARNING] {message}")

    def log_debug(self, message):
        """记录调试信息"""
        self.append_info_output(f"[DEBUG] {message}")

    def log_verbose(self, message):
        """记录详细信息"""
        self.append_info_output(f"[VERBOSE] {message}")

    def clear_info_output(self):
        """清除信息输出"""
        self.info_text.clear()
        
    def clear(self):
        """清空信息输出窗口（与clear_info_output一致，供主界面按钮调用）"""
        self.clear_info_output()




class Splitter:
    """分割窗口类"""
    
    def __init__(self, parent=None, orientation=Qt.Horizontal):
        self.splitter = QSplitter(parent)
        self.splitter.setOrientation(orientation)
        
    def add_widget(self, widget):
        """添加组件到分割器"""
        self.splitter.addWidget(widget)
        
    def set_sizes(self, sizes):
        """设置分割器大小"""
        self.splitter.setSizes(sizes)
        
    def get_widget(self):
        """获取分割器"""
        return self.splitter


class DialogBase:
    """对话框基类"""
    
    def __init__(self, parent=None, title="对话框"):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox
        
        self.dialog = QDialog(parent)
        self.dialog.setWindowTitle(title)
        
        # 创建按钮框
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self.dialog
        )
        self.button_box.accepted.connect(self.ok)
        self.button_box.rejected.connect(self.cancel)
        
        # 创建布局
        main_layout = QVBoxLayout()
        self.dialog.setLayout(main_layout)
        
        # 结果变量
        self.result = None
    
    def ok(self):
        """确定按钮回调，子类应重写此方法"""
        self.dialog.accept()
    
    def cancel(self):
        """取消按钮回调"""
        self.dialog.reject()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI Components and Utilities
Provides reusable UI components and styling utilities for PyMeshGen GUI
"""

from PyQt5.QtWidgets import QToolButton, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon


class UIStyles:
    """Centralized UI style definitions"""

    RIBBON_TAB_STYLESHEET = """
        QTabWidget::pane {
            border: 0px;
            background-color: #f0f0f0;
        }
        QTabBar::tab {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                             stop: 0 #f0f0f0, stop: 1 #e0e0e0);
            color: #333333;
            border: 1px solid #b0b0b0;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 5px 10px;
            margin: 0px 2px 0px 0px;
            font-size: 9pt;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                             stop: 0 #ffffff, stop: 1 #f0f0f0);
            color: #000000;
            border-bottom: 2px solid #0078d4;
        }
        QTabBar::tab:hover {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                             stop: 0 #f8f8f8, stop: 1 #e8e8e8);
        }
    """

    LARGE_BUTTON_STYLESHEET = """
        QToolButton {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #ffffff, stop: 1 #e0e0e0);
            border: 1px solid #b0b0b0;
            border-radius: 4px;
            padding: 4px;
            font-size: 8pt;
            font-weight: normal;
            text-align: center;
        }
        QToolButton:hover {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #f0f0f0, stop: 1 #d0d0d0);
            border: 1px solid #909090;
        }
        QToolButton:pressed {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #d0d0d0, stop: 1 #b0b0b0);
            border: 1px solid #808080;
        }
        QToolButton:checked {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #cce5ff, stop: 1 #99ccff);
            border: 1px solid #0078d4;
        }
    """

    SMALL_BUTTON_STYLESHEET = """
        QToolButton {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #ffffff, stop: 1 #e0e0e0);
            border: 1px solid #b0b0b0;
            border-radius: 3px;
            padding: 2px 4px;
            font-size: 8pt;
            text-align: left;
        }
        QToolButton:hover {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #f0f0f0, stop: 1 #d0d0d0);
            border: 1px solid #909090;
        }
        QToolButton:pressed {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #d0d0d0, stop: 1 #b0b0b0);
            border: 1px solid #808080;
        }
        QToolButton:checked {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #cce5ff, stop: 1 #99ccff);
            border: 1px solid #0078d4;
        }
    """

    RIBBON_GROUP_STYLESHEET = """
        RibbonGroup {
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #f8f8f8, stop: 1 #e8e8e8);
            padding: 2px;
        }
    """

    MAIN_WINDOW_STYLESHEET = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QWidget {
            background-color: #f0f0f0;
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            font-size: 9pt;
            color: #333333;
        }
        QMenuBar {
            background-color: #e6e6e6;
            spacing: 5px;
        }
        QMenuBar::item {
            background: transparent;
            padding: 5px 10px;
            font-size: 9pt;
        }
        QMenuBar::item:selected {
            background: #0078d4;
            color: white;
        }
        QMenuBar::item:pressed {
            background: #005a9e;
            color: white;
        }
        QPushButton {
            background-color: #f5f5f5;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 6px 8px;
            min-width: 60px;
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            font-size: 9pt;
        }
        QPushButton:hover {
            background-color: #e6e6e6;
            border: 1px solid #999999;
        }
        QPushButton:pressed {
            background-color: #d9d9d9;
            border: 1px solid #666666;
        }
        QLabel {
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            font-size: 9pt;
        }
        QGroupBox {
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            font-size: 9pt;
            font-weight: bold;
        }
        QGroupBox::title {
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            font-size: 9pt;
            font-weight: bold;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
    """

    SPLITTER_STYLESHEET = """
        QSplitter::handle {
            background-color: #d0d0d0;
            border: 1px solid #b0b0b0;
            border-radius: 2px;
        }
        QSplitter::handle:horizontal {
            width: 6px;
        }
        QSplitter::handle:vertical {
            height: 6px;
        }
    """

    LIST_WIDGET_STYLESHEET = """
        QListWidget {
            border: 1px solid #cccccc;
            border-radius: 3px;
            background-color: #ffffff;
            color: #333333;
            alternate-background-color: #f0f0f0;
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            font-size: 9pt;
        }
        QListWidget::item {
            padding: 4px;
            border-bottom: 1px solid #e0e0e0;
        }
        QListWidget::item:selected {
            background-color: #0078d4;
            color: white;
        }
        QListWidget::item:hover {
            background-color: #cceeff;
        }
    """

    BUTTON_STYLESHEETS = {
        'add_button': """
            QPushButton {
                background-color: #e6f7ff;
                border: 1px solid #0078d4;
                border-radius: 3px;
                padding: 4px;
                color: #0078d4;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #cceeff;
            }
            QPushButton:pressed {
                background-color: #99ddff;
            }
        """,
        'remove_button': """
            QPushButton {
                background-color: #ffe6e6;
                border: 1px solid #cc0000;
                border-radius: 3px;
                padding: 4px;
                color: #cc0000;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffcccc;
            }
            QPushButton:pressed {
                background-color: #ff9999;
            }
        """,
        'edit_button': """
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 4px;
                color: #333333;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """
    }

    FRAME_STYLESHEET = """
        QFrame {
            border: 1px solid #cccccc;
            border-radius: 4px;
            background-color: #ffffff;
        }
    """

    GROUPBOX_STYLESHEET = """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 4px;
            margin-top: 1ex;
            padding-top: 10px;
            background-color: #f9f9f9;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #333333;
        }
    """

    TEXTEDIT_STYLESHEET = """
        QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 3px;
            background-color: #ffffff;
            color: #333333;
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            font-size: 9pt;
        }
    """

    TEXTEDIT_MONOSPACE_STYLESHEET = """
        QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 3px;
            background-color: #ffffff;
            color: #333333;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 9pt;
        }
    """


class LayoutConfig:
    """Centralized layout configuration"""

    RIBBON_TAB_MARGINS = (2, 2, 2, 2)
    RIBBIN_TAB_SPACING = 10

    RIBBON_GROUP_MARGINS = (2, 2, 2, 2)
    RIBBON_GROUP_SPACING = 2

    MAIN_LAYOUT_MARGINS = (0, 0, 0, 0)
    MAIN_LAYOUT_SPACING = 0


def create_tool_button(text, icon=None, callback=None, tooltip=None, 
                       button_style="large", size=None):
    """Create a tool button with common settings

    Args:
        text: Button text
        icon: QIcon for the button
        callback: Function to connect to clicked signal
        tooltip: Tooltip text
        button_style: 'large' or 'small' button style
        size: Tuple (width, height) for fixed size, or None for default

    Returns:
        QToolButton instance
    """
    button = QToolButton()
    button.setText(text)

    if button_style == "large":
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setIconSize(QSize(20, 20))
        button.setStyleSheet(UIStyles.LARGE_BUTTON_STYLESHEET)
        if size is None:
            size = (70, 60)
    else:
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setIconSize(QSize(16, 16))
        button.setStyleSheet(UIStyles.SMALL_BUTTON_STYLESHEET)
        if size is None:
            size = (90, 22)

    if icon:
        button.setIcon(icon)

    if callback:
        button.clicked.connect(callback)

    if tooltip:
        button.setToolTip(tooltip)

    button.setFixedSize(*size)
    return button


def create_horizontal_layout(margins=None, spacing=None, alignment=None):
    """Create a horizontal layout with common settings

    Args:
        margins: Tuple (left, top, right, bottom) for margins
        spacing: Spacing between items
        alignment: Qt alignment flag

    Returns:
        QHBoxLayout instance
    """
    layout = QHBoxLayout()

    if margins:
        layout.setContentsMargins(*margins)

    if spacing is not None:
        layout.setSpacing(spacing)

    if alignment:
        layout.setAlignment(alignment)

    return layout


def create_vertical_layout(margins=None, spacing=None, alignment=None):
    """Create a vertical layout with common settings

    Args:
        margins: Tuple (left, top, right, bottom) for margins
        spacing: Spacing between items
        alignment: Alignment for the layout

    Returns:
        QVBoxLayout instance
    """
    layout = QVBoxLayout()

    if margins:
        layout.setContentsMargins(*margins)

    if spacing is not None:
        layout.setSpacing(spacing)

    if alignment is not None:
        layout.setAlignment(alignment)

    return layout


class RibbonGroupContent(QWidget):
    """Reusable content widget for ribbon groups"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = create_horizontal_layout(
            margins=(0, 0, 0, 0),
            spacing=LayoutConfig.RIBBIN_TAB_SPACING,
            alignment=Qt.AlignLeft
        )
        self.setLayout(self.layout)

    def add_button(self, button):
        """Add a button to the content layout"""
        self.layout.addWidget(button)
        return button

    def add_widget(self, widget):
        """Add a widget to the content layout"""
        self.layout.addWidget(widget)
        return widget

    def add_button_column(self, buttons, rows=2):
        """Add buttons in a vertical column layout
        
        Args:
            buttons: List of buttons to add in column
            rows: Number of rows per column (default: 2)
        """
        column_layout = create_vertical_layout(
            margins=(0, 0, 0, 0),
            spacing=2,
            alignment=Qt.AlignTop
        )
        
        for button in buttons:
            column_layout.addWidget(button)
        
        column_widget = QWidget()
        column_widget.setLayout(column_layout)
        self.layout.addWidget(column_widget)
        return column_widget

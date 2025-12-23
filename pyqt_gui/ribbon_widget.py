#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt Ribbon Widget Implementation
Provides a Microsoft Office-style ribbon interface for PyMeshGen
"""

from PyQt5.QtWidgets import (QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFrame,
                             QToolButton, QButtonGroup, QScrollArea, QGridLayout,
                             QGroupBox, QLabel, QSizePolicy, QSplitter, QToolTip)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QSignalMapper, QTimer
from PyQt5.QtGui import QIcon, QFont, QPainter, QColor, QPalette


class RibbonTabBar(QTabWidget):
    """Custom ribbon tab bar with large buttons and icons"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QTabWidget.North)
        self.setElideMode(Qt.ElideNone)
        self.setDocumentMode(True)

        # Set custom style for ribbon tabs - more compact and visually appealing
        self.setStyleSheet("""
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
        """)


class RibbonGroup(QFrame):
    """A group of controls within a ribbon tab"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(1)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)  # Slightly more padding for visual appeal
        layout.setSpacing(6)  # More spacing for visual appeal

        # Title label
        self.title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(9)  # Slightly larger font for better visibility
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            color: #444444;
            font-weight: bold;
            border-bottom: 1px solid #cccccc;
            padding-bottom: 2px;
        """)
        # Ensure minimum size for proper text display
        self.title_label.setMinimumHeight(20)
        layout.addWidget(self.title_label)

        # Content area for buttons and controls
        self.content_widget = QWidget()
        self.content_layout = QGridLayout()
        self.content_layout.setSpacing(4)  # Better spacing for visual appeal
        self.content_widget.setLayout(self.content_layout)
        layout.addWidget(self.content_widget)

        self.setLayout(layout)

        # Style the group
        self.setStyleSheet("""
            RibbonGroup {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                stop: 0 #f8f8f8, stop: 1 #e8e8e8);
                padding: 2px;
            }
        """)

    def add_large_button(self, text, icon=None, callback=None, tooltip=None, row=0, col=0):
        """Add a large button to the ribbon group"""
        button = QToolButton()
        button.setText(text)
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setIconSize(QSize(20, 20))  # Smaller icon

        if icon:
            button.setIcon(icon)

        if callback:
            button.clicked.connect(callback)

        if tooltip:
            button.setToolTip(tooltip)
            # Enhance tooltip with custom styling
            button.setStyleSheet(button.styleSheet() + """
                QToolButton {
                    tooltip-duration: 10000;
                }
            """)

        button.setFixedSize(50, 50)  # Smaller size
        button.setStyleSheet("""
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
        """)

        self.content_layout.addWidget(button, row, col)
        return button

    def add_small_button(self, text, icon=None, callback=None, tooltip=None, row=0, col=0):
        """Add a small button to the ribbon group"""
        button = QToolButton()
        button.setText(text)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setIconSize(QSize(16, 16))  # Larger icon for better visibility

        if icon:
            button.setIcon(icon)

        if callback:
            button.clicked.connect(callback)

        if tooltip:
            button.setToolTip(tooltip)
            # Enhance tooltip with custom styling
            button.setStyleSheet(button.styleSheet() + """
                QToolButton {
                    tooltip-duration: 10000;
                }
            """)

        button.setFixedSize(90, 22)  # Slightly larger for better usability
        button.setStyleSheet("""
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
        """)

        self.content_layout.addWidget(button, row, col)
        return button


class RibbonWidget(QWidget):
    """Main ribbon widget that contains tabs and groups"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Keep minimal margins
        self.main_layout.setSpacing(0)  # No spacing between elements

        # Store references to buttons for later callback assignment
        self.buttons = {
            'file': {},
            'view': {},
            'config': {},
            'mesh': {},
            'help': {}
        }

        # Create ribbon tabs
        self.ribbon_tabs = RibbonTabBar()

        # Add file tab
        self.file_tab = self.create_file_tab()
        self.ribbon_tabs.addTab(self.file_tab, "文件")

        # Add view tab
        self.view_tab = self.create_view_tab()
        self.ribbon_tabs.addTab(self.view_tab, "视图")

        # Add config tab
        self.config_tab = self.create_config_tab()
        self.ribbon_tabs.addTab(self.config_tab, "配置")

        # Add mesh tab
        self.mesh_tab = self.create_mesh_tab()
        self.ribbon_tabs.addTab(self.mesh_tab, "网格")

        # Add help tab
        self.help_tab = self.create_help_tab()
        self.ribbon_tabs.addTab(self.help_tab, "帮助")

        self.main_layout.addWidget(self.ribbon_tabs)
        self.setLayout(self.main_layout)

        # Style the ribbon
        self.setStyleSheet("""
            RibbonWidget {
                background-color: #e6e6e6;
                border-bottom: 1px solid #cccccc;
            }
        """)

    def create_file_tab(self):
        """Create the file ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins
        layout.setSpacing(4)  # Reduced spacing

        # File operations group
        file_group = RibbonGroup("文件操作")
        self.buttons['file']['new'] = file_group.add_large_button("新建", tooltip="新建配置 (Ctrl+N)", row=0, col=0)
        self.buttons['file']['open'] = file_group.add_large_button("打开", tooltip="打开配置 (Ctrl+O)", row=0, col=1)
        self.buttons['file']['save'] = file_group.add_large_button("保存", tooltip="保存配置 (Ctrl+S)", row=0, col=2)
        layout.addWidget(file_group)

        # Import/Export group
        io_group = RibbonGroup("导入/导出")
        self.buttons['file']['import'] = io_group.add_large_button("导入", tooltip="导入网格 (Ctrl+I)", row=0, col=0)
        self.buttons['file']['export'] = io_group.add_large_button("导出", tooltip="导出网格 (Ctrl+E)", row=0, col=1)
        layout.addWidget(io_group)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_view_tab(self):
        """Create the view ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins
        layout.setSpacing(4)  # Reduced spacing

        # View operations group
        view_group = RibbonGroup("视图操作")
        self.buttons['view']['reset'] = view_group.add_large_button("重置", tooltip="重置视图 (R键)", row=0, col=0)
        self.buttons['view']['fit'] = view_group.add_large_button("适应", tooltip="适应视图 (F键)", row=0, col=1)
        layout.addWidget(view_group)

        # Zoom group
        zoom_group = RibbonGroup("缩放")
        self.buttons['view']['zoom_in'] = zoom_group.add_large_button("放大", tooltip="放大视图 (+键)", row=0, col=0)
        self.buttons['view']['zoom_out'] = zoom_group.add_large_button("缩小", tooltip="缩小视图 (-键)", row=0, col=1)
        layout.addWidget(zoom_group)

        # Display mode group
        display_group = RibbonGroup("显示模式")
        self.buttons['view']['surface'] = display_group.add_small_button("实体", tooltip="实体模式 (1键)", row=0, col=0)
        self.buttons['view']['wireframe'] = display_group.add_small_button("线框", tooltip="线框模式 (2键)", row=0, col=1)
        self.buttons['view']['mixed'] = display_group.add_small_button("混合", tooltip="混合模式 (3键)", row=1, col=0)
        self.buttons['view']['points'] = display_group.add_small_button("点云", tooltip="点云模式 (4键)", row=1, col=1)
        layout.addWidget(display_group)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_config_tab(self):
        """Create the config ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins
        layout.setSpacing(4)  # Reduced spacing

        # Parameter group
        param_group = RibbonGroup("参数设置")
        self.buttons['config']['params'] = param_group.add_large_button("参数", tooltip="编辑参数", row=0, col=0)
        self.buttons['config']['mesh_params'] = param_group.add_large_button("网格参数", tooltip="编辑网格参数", row=0, col=1)
        self.buttons['config']['boundary'] = param_group.add_large_button("边界条件", tooltip="编辑边界条件", row=0, col=2)
        layout.addWidget(param_group)

        # Configuration group
        config_group = RibbonGroup("配置管理")
        self.buttons['config']['import_config'] = config_group.add_large_button("导入", tooltip="导入配置", row=0, col=0)
        self.buttons['config']['export_config'] = config_group.add_large_button("导出", tooltip="导出配置", row=0, col=1)
        self.buttons['config']['reset'] = config_group.add_large_button("重置", tooltip="重置配置", row=0, col=2)
        layout.addWidget(config_group)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_mesh_tab(self):
        """Create the mesh ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins
        layout.setSpacing(4)  # Reduced spacing

        # Generation group
        gen_group = RibbonGroup("生成")
        self.buttons['mesh']['generate'] = gen_group.add_large_button("生成", tooltip="生成网格 (F5)", row=0, col=0)
        self.buttons['mesh']['display'] = gen_group.add_large_button("显示", tooltip="显示网格 (F6)", row=0, col=1)
        self.buttons['mesh']['clear'] = gen_group.add_large_button("清空", tooltip="清空网格", row=0, col=2)
        layout.addWidget(gen_group)

        # Quality group
        quality_group = RibbonGroup("质量")
        self.buttons['mesh']['quality'] = quality_group.add_large_button("质量", tooltip="检查网格质量", row=0, col=0)
        self.buttons['mesh']['smooth'] = quality_group.add_large_button("平滑", tooltip="平滑网格", row=0, col=1)
        self.buttons['mesh']['optimize'] = quality_group.add_large_button("优化", tooltip="优化网格", row=0, col=2)
        layout.addWidget(quality_group)

        # Analysis group
        analysis_group = RibbonGroup("分析")
        self.buttons['mesh']['statistics'] = analysis_group.add_large_button("统计", tooltip="网格统计", row=0, col=0)
        self.buttons['mesh']['report'] = analysis_group.add_large_button("报告", tooltip="导出报告", row=0, col=1)
        layout.addWidget(analysis_group)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_help_tab(self):
        """Create the help ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins
        layout.setSpacing(4)  # Reduced spacing

        # Documentation group
        doc_group = RibbonGroup("文档")
        self.buttons['help']['manual'] = doc_group.add_large_button("手册", tooltip="用户手册", row=0, col=0)
        self.buttons['help']['quick_start'] = doc_group.add_large_button("入门", tooltip="快速入门", row=0, col=1)
        layout.addWidget(doc_group)

        # Support group
        support_group = RibbonGroup("支持")
        self.buttons['help']['shortcuts'] = support_group.add_large_button("快捷键", tooltip="快捷键", row=0, col=0)
        self.buttons['help']['updates'] = support_group.add_large_button("更新", tooltip="检查更新", row=0, col=1)
        self.buttons['help']['about'] = support_group.add_large_button("关于", tooltip="关于", row=0, col=2)
        layout.addWidget(support_group)

        tab_widget.setLayout(layout)
        return tab_widget

    def set_callbacks(self, main_window):
        """Set callbacks for all ribbon buttons using the main window methods"""
        # File tab callbacks
        self.buttons['file']['new'].clicked.connect(main_window.new_config)
        self.buttons['file']['open'].clicked.connect(main_window.open_config)
        self.buttons['file']['save'].clicked.connect(main_window.save_config)
        self.buttons['file']['import'].clicked.connect(main_window.import_mesh)
        self.buttons['file']['export'].clicked.connect(main_window.export_mesh)

        # View tab callbacks
        self.buttons['view']['reset'].clicked.connect(main_window.reset_view)
        self.buttons['view']['fit'].clicked.connect(main_window.fit_view)
        self.buttons['view']['zoom_in'].clicked.connect(main_window.zoom_in)
        self.buttons['view']['zoom_out'].clicked.connect(main_window.zoom_out)
        self.buttons['view']['surface'].clicked.connect(lambda: main_window.set_render_mode("surface"))
        self.buttons['view']['wireframe'].clicked.connect(lambda: main_window.set_render_mode("wireframe"))
        self.buttons['view']['mixed'].clicked.connect(lambda: main_window.set_render_mode("mixed"))
        self.buttons['view']['points'].clicked.connect(lambda: main_window.set_render_mode("points"))

        # Config tab callbacks
        self.buttons['config']['params'].clicked.connect(main_window.edit_params)
        self.buttons['config']['mesh_params'].clicked.connect(main_window.edit_mesh_params)
        self.buttons['config']['boundary'].clicked.connect(main_window.edit_boundary_conditions)
        self.buttons['config']['import_config'].clicked.connect(main_window.import_config)
        self.buttons['config']['export_config'].clicked.connect(main_window.export_config)
        self.buttons['config']['reset'].clicked.connect(main_window.reset_config)

        # Mesh tab callbacks
        self.buttons['mesh']['generate'].clicked.connect(main_window.generate_mesh)
        self.buttons['mesh']['display'].clicked.connect(main_window.display_mesh)
        self.buttons['mesh']['clear'].clicked.connect(main_window.clear_mesh)
        self.buttons['mesh']['quality'].clicked.connect(main_window.check_mesh_quality)
        self.buttons['mesh']['smooth'].clicked.connect(main_window.smooth_mesh)
        self.buttons['mesh']['optimize'].clicked.connect(main_window.optimize_mesh)
        self.buttons['mesh']['statistics'].clicked.connect(main_window.show_mesh_statistics)
        self.buttons['mesh']['report'].clicked.connect(main_window.export_mesh_report)

        # Help tab callbacks
        self.buttons['help']['manual'].clicked.connect(main_window.show_user_manual)
        self.buttons['help']['quick_start'].clicked.connect(main_window.show_quick_start)
        self.buttons['help']['shortcuts'].clicked.connect(main_window.show_shortcuts)
        self.buttons['help']['updates'].clicked.connect(main_window.check_for_updates)
        self.buttons['help']['about'].clicked.connect(main_window.show_about)


class BubbleTooltip(QToolTip):
    """Custom bubble-style tooltip for ribbon buttons"""

    @staticmethod
    def show_bubble_tooltip(text, pos, parent=None, rect=None):
        """Show a bubble tooltip at the specified position"""
        QToolTip.showText(pos, text, parent, rect)
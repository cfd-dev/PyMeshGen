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
    
    # 添加标签栏点击信号
    tab_bar_clicked = pyqtSignal()

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
        
    def mousePressEvent(self, event):
        """重写鼠标点击事件，当点击标签栏时发出信号"""
        # Check if the click is on the tab bar area (not on the content pane)
        tab_bar = self.tabBar()
        if tab_bar.underMouse():
            # 发出标签栏点击信号
            self.tab_bar_clicked.emit()
        super().mousePressEvent(event)


class RibbonGroup(QFrame):
    """A group of controls within a ribbon tab"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(1)

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)  # Reduced padding
        layout.setSpacing(2)  # Reduced spacing

        # Content area for buttons and controls (no title)
        self.content_widget = QWidget()
        self.content_layout = QHBoxLayout()  # Changed to QHBoxLayout for left alignment
        self.content_layout.setSpacing(2)  # Reduced spacing
        self.content_layout.setAlignment(Qt.AlignLeft)  # Align content to the left
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
            # PyQt5不支持tooltip-duration属性，已移除

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

        self.content_layout.addWidget(button)  # Removed row, col parameters for QHBoxLayout
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
            # PyQt5不支持tooltip-duration属性，已移除

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

        self.content_layout.addWidget(button)  # Removed row, col parameters for QHBoxLayout
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

        # Add toggle button to the right of the ribbon tabs
        from PyQt5.QtWidgets import QPushButton
        self.toggle_button = QPushButton("^")
        self.toggle_button.setFixedSize(24, 24)
        self.toggle_button.setToolTip("折叠/展开功能区")

        # Create a horizontal layout to hold both ribbon tabs and toggle button
        # This will keep them on the same line
        self.tabs_and_toggle_layout = QHBoxLayout()
        self.tabs_and_toggle_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs_and_toggle_layout.setSpacing(0)
        
        # Add ribbon tabs to the layout, taking up most of the space
        self.tabs_and_toggle_layout.addWidget(self.ribbon_tabs, 1)  # Give ribbon tabs most space
        
        # Add toggle button to the layout
        self.tabs_and_toggle_layout.addWidget(self.toggle_button, 0, Qt.AlignTop)  # Align button to top
        
        # Add the combined layout to the main layout
        self.main_layout.addLayout(self.tabs_and_toggle_layout)

        self.setLayout(self.main_layout)

        # Track content visibility state
        self._content_visible = True
        
        # 连接标签页切换信号，确保在折叠状态下切换标签页也不会显示内容
        self.ribbon_tabs.currentChanged.connect(self._on_tab_changed)
        
        # 连接标签栏点击信号，用于在折叠状态下展开ribbon
        self.ribbon_tabs.tab_bar_clicked.connect(self._on_tab_bar_clicked)
    
    def _on_tab_changed(self, index):
        """当标签页切换时，根据当前可见性状态决定是否调整高度"""
        # 只在折叠状态下才调整高度，展开状态下不做任何操作
        if not self._content_visible:
            # 如果处于折叠状态，确保最大高度仍然限制在标签栏高度
            tab_bar_height = self.ribbon_tabs.tabBar().height()
            self.setMaximumHeight(tab_bar_height + 5)
        
        # 移除在这里设置样式表的代码，避免重复设置和信号干扰
        # 样式表应该只在初始化时设置一次
    
    def _on_tab_bar_clicked(self):
        """当标签栏被点击时，如果处于折叠状态则展开ribbon"""
        if not self._content_visible:
            # 如果处于折叠状态，点击标签栏则展开ribbon
            self.setMaximumHeight(16777215)  # Maximum allowed height for QWidget
            self.toggle_button.show()  # Show toggle button when expanded
            self.toggle_button.setText("^")  # Show up arrow when expanded
            self._content_visible = True
            # 恢复完整样式
            self.ribbon_tabs.setStyleSheet("""
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

            # Make sure the ribbon tabs are properly visible and update the layout
            self.ribbon_tabs.show()

            # Force a layout update to ensure the ribbon is properly displayed
            self.updateGeometry()
            if self.parent() and hasattr(self.parent(), 'update') and callable(getattr(self.parent(), 'update')):
                self.parent().update()

    def toggle_content_visibility(self):
        """Toggle visibility of ribbon content (tabs and groups)"""
        if self._content_visible:
            # Collapse the ribbon by reducing its maximum height
            # This keeps the tab headers visible but hides the content
            tab_bar_height = self.ribbon_tabs.tabBar().height()
            self.setMaximumHeight(tab_bar_height + 5)  # Just enough for tabs, no button space
            self.toggle_button.hide()  # Hide toggle button when collapsed
            self._content_visible = False
            # 更新样式以确保视觉一致性
            self.ribbon_tabs.setStyleSheet("""
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

            # Force a layout update after collapsing
            self.updateGeometry()
            if self.parent() and hasattr(self.parent(), 'update') and callable(getattr(self.parent(), 'update')):
                self.parent().update()
        else:
            # Expand the ribbon by removing maximum height restriction
            self.setMaximumHeight(16777215)  # Maximum allowed height for QWidget
            self.toggle_button.show()  # Show toggle button when expanded
            self.toggle_button.setText("^")  # Show up arrow when expanded
            self._content_visible = True
            # 恢复完整样式
            self.ribbon_tabs.setStyleSheet("""
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

            # Make sure the ribbon is visible after expanding
            self.show()

            # Force a layout update after expanding
            self.updateGeometry()
            if self.parent() and hasattr(self.parent(), 'update') and callable(getattr(self.parent(), 'update')):
                self.parent().update()

    def is_content_visible(self):
        """Check if ribbon content is visible"""
        return self._content_visible

    def create_file_tab(self):
        """Create the file ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)  # Further reduced margins
        layout.setSpacing(2)  # Further reduced spacing
        layout.setAlignment(Qt.AlignLeft)  # Align groups to the left

        # File operations group
        file_group = RibbonGroup("")
        self.buttons['file']['new'] = file_group.add_large_button("新建", tooltip="新建配置 (Ctrl+N)")
        self.buttons['file']['open'] = file_group.add_large_button("打开", tooltip="打开配置 (Ctrl+O)")
        self.buttons['file']['save'] = file_group.add_large_button("保存", tooltip="保存配置 (Ctrl+S)")
        layout.addWidget(file_group)

        # Import/Export group
        io_group = RibbonGroup("")
        self.buttons['file']['import'] = io_group.add_large_button("导入", tooltip="导入网格 (Ctrl+I)")
        self.buttons['file']['export'] = io_group.add_large_button("导出", tooltip="导出网格 (Ctrl+E)")
        layout.addWidget(io_group)

        # Add stretch to push groups to the left
        layout.addStretch(1)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_view_tab(self):
        """Create the view ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)  # Further reduced margins
        layout.setSpacing(2)  # Further reduced spacing
        layout.setAlignment(Qt.AlignLeft)  # Align groups to the left

        # View operations group
        view_group = RibbonGroup("")
        self.buttons['view']['reset'] = view_group.add_large_button("重置", tooltip="重置视图 (R键)")
        self.buttons['view']['fit'] = view_group.add_large_button("适应", tooltip="适应视图 (F键)")
        layout.addWidget(view_group)

        # Zoom group
        zoom_group = RibbonGroup("")
        self.buttons['view']['zoom_in'] = zoom_group.add_large_button("放大", tooltip="放大视图 (+键)")
        self.buttons['view']['zoom_out'] = zoom_group.add_large_button("缩小", tooltip="缩小视图 (-键)")
        layout.addWidget(zoom_group)

        # Display mode group
        display_group = RibbonGroup("")
        self.buttons['view']['surface'] = display_group.add_small_button("实体", tooltip="实体模式 (1键)")
        self.buttons['view']['wireframe'] = display_group.add_small_button("线框", tooltip="线框模式 (2键)")
        self.buttons['view']['mixed'] = display_group.add_small_button("混合", tooltip="混合模式 (3键)")
        self.buttons['view']['points'] = display_group.add_small_button("点云", tooltip="点云模式 (4键)")
        layout.addWidget(display_group)

        # Add stretch to push groups to the left
        layout.addStretch(1)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_config_tab(self):
        """Create the config ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)  # Further reduced margins
        layout.setSpacing(2)  # Further reduced spacing
        layout.setAlignment(Qt.AlignLeft)  # Align groups to the left

        # Parameter group
        param_group = RibbonGroup("")
        self.buttons['config']['params'] = param_group.add_large_button("参数", tooltip="编辑参数")
        self.buttons['config']['mesh_params'] = param_group.add_large_button("网格参数", tooltip="编辑网格参数")
        self.buttons['config']['boundary'] = param_group.add_large_button("边界条件", tooltip="编辑边界条件")
        layout.addWidget(param_group)

        # Configuration group
        config_group = RibbonGroup("")
        self.buttons['config']['import_config'] = config_group.add_large_button("导入", tooltip="导入配置")
        self.buttons['config']['export_config'] = config_group.add_large_button("导出", tooltip="导出配置")
        self.buttons['config']['reset'] = config_group.add_large_button("重置", tooltip="重置配置")
        layout.addWidget(config_group)

        # Add stretch to push groups to the left
        layout.addStretch(1)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_mesh_tab(self):
        """Create the mesh ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)  # Further reduced margins
        layout.setSpacing(2)  # Further reduced spacing
        layout.setAlignment(Qt.AlignLeft)  # Align groups to the left

        # Generation group
        gen_group = RibbonGroup("")
        self.buttons['mesh']['generate'] = gen_group.add_large_button("生成", tooltip="生成网格 (F5)")
        self.buttons['mesh']['display'] = gen_group.add_large_button("显示", tooltip="显示网格 (F6)")
        self.buttons['mesh']['clear'] = gen_group.add_large_button("清空", tooltip="清空网格")
        layout.addWidget(gen_group)

        # Quality group
        quality_group = RibbonGroup("")
        self.buttons['mesh']['quality'] = quality_group.add_large_button("质量", tooltip="检查网格质量")
        self.buttons['mesh']['smooth'] = quality_group.add_large_button("平滑", tooltip="平滑网格")
        self.buttons['mesh']['optimize'] = quality_group.add_large_button("优化", tooltip="优化网格")
        layout.addWidget(quality_group)

        # Analysis group
        analysis_group = RibbonGroup("")
        self.buttons['mesh']['statistics'] = analysis_group.add_large_button("统计", tooltip="网格统计")
        self.buttons['mesh']['report'] = analysis_group.add_large_button("报告", tooltip="导出报告")
        layout.addWidget(analysis_group)

        # Add stretch to push groups to the left
        layout.addStretch(1)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_help_tab(self):
        """Create the help ribbon tab"""
        tab_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)  # Further reduced margins
        layout.setSpacing(2)  # Further reduced spacing
        layout.setAlignment(Qt.AlignLeft)  # Align groups to the left

        # Documentation group
        doc_group = RibbonGroup("")
        self.buttons['help']['manual'] = doc_group.add_large_button("手册", tooltip="用户手册")
        self.buttons['help']['quick_start'] = doc_group.add_large_button("入门", tooltip="快速入门")
        layout.addWidget(doc_group)

        # Support group
        support_group = RibbonGroup("")
        self.buttons['help']['shortcuts'] = support_group.add_large_button("快捷键", tooltip="快捷键")
        self.buttons['help']['updates'] = support_group.add_large_button("更新", tooltip="检查更新")
        self.buttons['help']['about'] = support_group.add_large_button("关于", tooltip="关于")
        layout.addWidget(support_group)

        # Add stretch to push groups to the left
        layout.addStretch(1)

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
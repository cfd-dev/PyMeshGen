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

from .ui_utils import (UIStyles, LayoutConfig, create_tool_button,
                       create_horizontal_layout, create_vertical_layout,
                       RibbonGroupContent)


class RibbonTabBar(QTabWidget):
    """Custom ribbon tab bar with large buttons and icons"""

    tab_bar_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QTabWidget.North)
        self.setElideMode(Qt.ElideNone)
        self.setDocumentMode(True)
        self.setStyleSheet(UIStyles.RIBBON_TAB_STYLESHEET)

    def mousePressEvent(self, event):
        """重写鼠标点击事件，当点击标签栏时发出信号"""
        tab_bar = self.tabBar()
        if tab_bar.underMouse():
            self.tab_bar_clicked.emit()
        super().mousePressEvent(event)


class RibbonGroup(QFrame):
    """A group of controls within a ribbon tab"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(1)

        layout = create_vertical_layout(
            margins=LayoutConfig.RIBBON_GROUP_MARGINS,
            spacing=LayoutConfig.RIBBON_GROUP_SPACING
        )

        self.content_widget = RibbonGroupContent(self)
        layout.addWidget(self.content_widget)

        self.setLayout(layout)
        self.setStyleSheet(UIStyles.RIBBON_GROUP_STYLESHEET)

    def add_large_button(self, text, icon=None, callback=None, tooltip=None, row=0, col=0):
        """Add a large button to the ribbon group"""
        button = create_tool_button(text, icon, callback, tooltip, "large")
        self.content_widget.add_button(button)
        return button

    def add_small_button(self, text, icon=None, callback=None, tooltip=None, row=0, col=0):
        """Add a small button to the ribbon group"""
        button = create_tool_button(text, icon, callback, tooltip, "small")
        self.content_widget.add_button(button)
        return button


class RibbonWidget(QWidget):
    """Main ribbon widget that contains tabs and groups"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = create_vertical_layout(
            margins=LayoutConfig.MAIN_LAYOUT_MARGINS,
            spacing=LayoutConfig.MAIN_LAYOUT_SPACING
        )

        self.buttons = {
            'file': {},
            'view': {},
            'config': {},
            'mesh': {},
            'help': {}
        }

        self.ribbon_tabs = RibbonTabBar()

        self.file_tab = self.create_file_tab()
        self.ribbon_tabs.addTab(self.file_tab, "文件")

        self.view_tab = self.create_view_tab()
        self.ribbon_tabs.addTab(self.view_tab, "视图")

        self.config_tab = self.create_config_tab()
        self.ribbon_tabs.addTab(self.config_tab, "配置")

        self.mesh_tab = self.create_mesh_tab()
        self.ribbon_tabs.addTab(self.mesh_tab, "网格")

        self.help_tab = self.create_help_tab()
        self.ribbon_tabs.addTab(self.help_tab, "帮助")

        from PyQt5.QtWidgets import QPushButton
        self.toggle_button = QPushButton("^")
        self.toggle_button.setFixedSize(24, 24)
        self.toggle_button.setToolTip("折叠/展开功能区")

        self.tabs_and_toggle_layout = create_horizontal_layout(
            margins=(0, 0, 0, 0),
            spacing=0
        )
        self.tabs_and_toggle_layout.addWidget(self.ribbon_tabs, 1)
        self.tabs_and_toggle_layout.addWidget(self.toggle_button, 0, Qt.AlignTop)

        self.main_layout.addLayout(self.tabs_and_toggle_layout)
        self.setLayout(self.main_layout)

        self._content_visible = True
        self.ribbon_tabs.currentChanged.connect(self._on_tab_changed)
        self.ribbon_tabs.tab_bar_clicked.connect(self._on_tab_bar_clicked)
    
    def _on_tab_changed(self, index):
        """当标签页切换时，根据当前可见性状态决定是否调整高度"""
        if not self._content_visible:
            tab_bar_height = self.ribbon_tabs.tabBar().height()
            self.setMaximumHeight(tab_bar_height + 5)

    def _on_tab_bar_clicked(self):
        """当标签栏被点击时，如果处于折叠状态则展开ribbon"""
        if not self._content_visible:
            self.setMaximumHeight(16777215)
            self.toggle_button.show()
            self.toggle_button.setText("^")
            self._content_visible = True
            self.ribbon_tabs.setStyleSheet(UIStyles.RIBBON_TAB_STYLESHEET)
            self.ribbon_tabs.show()
            self.updateGeometry()
            if self.parent() and hasattr(self.parent(), 'update') and callable(getattr(self.parent(), 'update')):
                self.parent().update()

    def toggle_content_visibility(self):
        """Toggle visibility of ribbon content (tabs and groups)"""
        if self._content_visible:
            tab_bar_height = self.ribbon_tabs.tabBar().height()
            self.setMaximumHeight(tab_bar_height + 5)
            self.toggle_button.hide()
            self._content_visible = False
            self.ribbon_tabs.setStyleSheet(UIStyles.RIBBON_TAB_STYLESHEET)
            self.updateGeometry()
            if self.parent() and hasattr(self.parent(), 'update') and callable(getattr(self.parent(), 'update')):
                self.parent().update()
        else:
            self.setMaximumHeight(16777215)
            self.toggle_button.show()
            self.toggle_button.setText("^")
            self._content_visible = True
            self.ribbon_tabs.setStyleSheet(UIStyles.RIBBON_TAB_STYLESHEET)
            self.show()
            self.updateGeometry()
            if self.parent() and hasattr(self.parent(), 'update') and callable(getattr(self.parent(), 'update')):
                self.parent().update()

    def is_content_visible(self):
        """Check if ribbon content is visible"""
        return self._content_visible

    def create_file_tab(self):
        """Create the file ribbon tab"""
        tab_widget = QWidget()
        layout = create_horizontal_layout(
            margins=LayoutConfig.RIBBON_TAB_MARGINS,
            spacing=LayoutConfig.RIBBIN_TAB_SPACING,
            alignment=Qt.AlignLeft
        )

        file_group = RibbonGroup("")
        self.buttons['file']['new'] = file_group.add_large_button("新建工程", tooltip="新建工程 (Ctrl+N)")
        self.buttons['file']['open'] = file_group.add_large_button("打开工程", tooltip="打开工程 (Ctrl+O)")
        self.buttons['file']['save'] = file_group.add_large_button("保存工程", tooltip="保存工程 (Ctrl+S)")
        layout.addWidget(file_group)

        io_group = RibbonGroup("")
        self.buttons['file']['import'] = io_group.add_large_button("导入网格", tooltip="导入网格 (Ctrl+I)")
        self.buttons['file']['export'] = io_group.add_large_button("导出网格", tooltip="导出网格 (Ctrl+E)")
        layout.addWidget(io_group)

        layout.addStretch(1)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_view_tab(self):
        """Create the view ribbon tab"""
        tab_widget = QWidget()
        layout = create_horizontal_layout(
            margins=LayoutConfig.RIBBON_TAB_MARGINS,
            spacing=LayoutConfig.RIBBIN_TAB_SPACING,
            alignment=Qt.AlignLeft
        )

        view_group = RibbonGroup("")
        self.buttons['view']['reset'] = view_group.add_large_button("重置", tooltip="重置视图 (R键)")
        self.buttons['view']['fit'] = view_group.add_large_button("适应", tooltip="适应视图 (F键)")
        layout.addWidget(view_group)

        zoom_group = RibbonGroup("")
        self.buttons['view']['zoom_in'] = zoom_group.add_large_button("放大", tooltip="放大视图 (+键)")
        self.buttons['view']['zoom_out'] = zoom_group.add_large_button("缩小", tooltip="缩小视图 (-键)")
        layout.addWidget(zoom_group)

        display_group = RibbonGroup("")
        self.buttons['view']['surface'] = display_group.add_small_button("实体", tooltip="实体模式 (1键)")
        self.buttons['view']['wireframe'] = display_group.add_small_button("线框", tooltip="线框模式 (2键)")
        self.buttons['view']['mixed'] = display_group.add_small_button("混合", tooltip="混合模式 (3键)")
        self.buttons['view']['points'] = display_group.add_small_button("点云", tooltip="点云模式 (4键)")
        layout.addWidget(display_group)

        layout.addStretch(1)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_config_tab(self):
        """Create the config ribbon tab"""
        tab_widget = QWidget()
        layout = create_horizontal_layout(
            margins=LayoutConfig.RIBBON_TAB_MARGINS,
            spacing=LayoutConfig.RIBBIN_TAB_SPACING,
            alignment=Qt.AlignLeft
        )

        param_group = RibbonGroup("")
        self.buttons['config']['params'] = param_group.add_large_button("参数", tooltip="编辑参数")
        self.buttons['config']['mesh_params'] = param_group.add_large_button("网格参数", tooltip="编辑网格参数")
        self.buttons['config']['boundary'] = param_group.add_large_button("边界条件", tooltip="编辑边界条件")
        layout.addWidget(param_group)

        config_group = RibbonGroup("")
        self.buttons['config']['import_config'] = config_group.add_large_button("导入配置", tooltip="导入配置")
        self.buttons['config']['export_config'] = config_group.add_large_button("导出配置", tooltip="导出配置")
        self.buttons['config']['reset'] = config_group.add_large_button("重置配置", tooltip="重置配置")
        layout.addWidget(config_group)

        layout.addStretch(1)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_mesh_tab(self):
        """Create the mesh ribbon tab"""
        tab_widget = QWidget()
        layout = create_horizontal_layout(
            margins=LayoutConfig.RIBBON_TAB_MARGINS,
            spacing=LayoutConfig.RIBBIN_TAB_SPACING,
            alignment=Qt.AlignLeft
        )

        gen_group = RibbonGroup("")
        self.buttons['mesh']['generate'] = gen_group.add_large_button("生成", tooltip="生成网格 (F5)")
        self.buttons['mesh']['display'] = gen_group.add_large_button("显示", tooltip="显示网格 (F6)")
        self.buttons['mesh']['clear'] = gen_group.add_large_button("清空", tooltip="清空网格")
        layout.addWidget(gen_group)

        quality_group = RibbonGroup("")
        self.buttons['mesh']['quality'] = quality_group.add_large_button("质量", tooltip="检查网格质量")
        self.buttons['mesh']['smooth'] = quality_group.add_large_button("平滑", tooltip="平滑网格")
        self.buttons['mesh']['optimize'] = quality_group.add_large_button("优化", tooltip="优化网格")
        layout.addWidget(quality_group)

        analysis_group = RibbonGroup("")
        self.buttons['mesh']['statistics'] = analysis_group.add_large_button("统计", tooltip="网格统计")
        self.buttons['mesh']['report'] = analysis_group.add_large_button("报告", tooltip="导出报告")
        layout.addWidget(analysis_group)

        layout.addStretch(1)

        tab_widget.setLayout(layout)
        return tab_widget

    def create_help_tab(self):
        """Create the help ribbon tab"""
        tab_widget = QWidget()
        layout = create_horizontal_layout(
            margins=LayoutConfig.RIBBON_TAB_MARGINS,
            spacing=LayoutConfig.RIBBIN_TAB_SPACING,
            alignment=Qt.AlignLeft
        )

        doc_group = RibbonGroup("")
        self.buttons['help']['manual'] = doc_group.add_large_button("手册", tooltip="用户手册")
        self.buttons['help']['quick_start'] = doc_group.add_large_button("入门", tooltip="快速入门")
        layout.addWidget(doc_group)

        support_group = RibbonGroup("")
        self.buttons['help']['shortcuts'] = support_group.add_large_button("快捷键", tooltip="快捷键")
        self.buttons['help']['updates'] = support_group.add_large_button("更新", tooltip="检查更新")
        self.buttons['help']['about'] = support_group.add_large_button("关于", tooltip="关于")
        layout.addWidget(support_group)

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
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
from .icon_manager import get_icon


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

    def add_button_column(self, buttons):
        """Add buttons in a vertical column layout
        
        Args:
            buttons: List of buttons to add in column
        """
        return self.content_widget.add_button_column(buttons)


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
            'geometry': {},
            'view': {},
            'config': {},
            'mesh': {},
            'help': {}
        }

        self.ribbon_tabs = RibbonTabBar()

        self.file_tab = self.create_file_tab()
        self.ribbon_tabs.addTab(self.file_tab, "文件")

        self.geometry_tab = self.create_geometry_tab()
        self.ribbon_tabs.addTab(self.geometry_tab, "几何")

        self.config_tab = self.create_config_tab()
        self.ribbon_tabs.addTab(self.config_tab, "配置")

        self.mesh_tab = self.create_mesh_tab()
        self.ribbon_tabs.addTab(self.mesh_tab, "网格")

        self.view_tab = self.create_view_tab()
        self.ribbon_tabs.addTab(self.view_tab, "视图")

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

        geometry_group = RibbonGroup("")
        self.buttons['file']['import_geometry'] = geometry_group.add_large_button("导入几何", tooltip="导入几何文件 (IGES/STEP/STL等)")
        self.buttons['file']['export_geometry'] = geometry_group.add_large_button("导出几何", tooltip="导出几何文件 (IGES/STEP/STL等)")
        layout.addWidget(geometry_group)

        io_group = RibbonGroup("")
        self.buttons['file']['import'] = io_group.add_large_button("导入网格", tooltip="导入网格 (Ctrl+I)")
        self.buttons['file']['export'] = io_group.add_large_button("导出网格", tooltip="导出网格 (Ctrl+E)")
        layout.addWidget(io_group)

        dimension_group = RibbonGroup("")
        self.buttons['file']['mesh_dimension'] = dimension_group.add_large_button("网格维度", icon=get_icon('mesh-dimension'), tooltip="设置网格维度")
        layout.addWidget(dimension_group)

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

        camera_group = RibbonGroup("")
        self.buttons['view']['view_x_pos'] = camera_group.add_small_button("X+", icon=get_icon('view-x-pos'), tooltip="X轴正向视图")
        self.buttons['view']['view_x_neg'] = camera_group.add_small_button("X-", icon=get_icon('view-x-neg'), tooltip="X轴负向视图")
        self.buttons['view']['view_y_pos'] = camera_group.add_small_button("Y+", icon=get_icon('view-y-pos'), tooltip="Y轴正向视图")
        self.buttons['view']['view_y_neg'] = camera_group.add_small_button("Y-", icon=get_icon('view-y-neg'), tooltip="Y轴负向视图")
        self.buttons['view']['view_z_pos'] = camera_group.add_small_button("Z+", icon=get_icon('view-z-pos'), tooltip="Z轴正向视图")
        self.buttons['view']['view_z_neg'] = camera_group.add_small_button("Z-", icon=get_icon('view-z-neg'), tooltip="Z轴负向视图")
        self.buttons['view']['view_iso'] = camera_group.add_small_button("等轴测", icon=get_icon('view-iso'), tooltip="等轴测视图")
        
        camera_group.add_button_column([
            self.buttons['view']['view_x_pos'],
            self.buttons['view']['view_x_neg']
        ])
        camera_group.add_button_column([
            self.buttons['view']['view_y_pos'],
            self.buttons['view']['view_y_neg']
        ])
        camera_group.add_button_column([
            self.buttons['view']['view_z_pos'],
            self.buttons['view']['view_z_neg']
        ])
        camera_group.add_button_column([
            self.buttons['view']['view_iso']
        ])
        layout.addWidget(camera_group)

        display_group = RibbonGroup("")
        self.buttons['view']['surface'] = display_group.add_small_button("实体", tooltip="实体模式 (1键)")
        self.buttons['view']['wireframe'] = display_group.add_small_button("线框", tooltip="线框模式 (2键)")
        self.buttons['view']['surface-wireframe'] = display_group.add_small_button("实体+线框", tooltip="实体+线框模式 (3键)")
        
        display_group.add_button_column([
            self.buttons['view']['surface'],
            self.buttons['view']['wireframe']
        ])
        display_group.add_button_column([
            self.buttons['view']['surface-wireframe']
        ])
        layout.addWidget(display_group)

        background_group = RibbonGroup("")
        self.buttons['view']['background'] = background_group.add_small_button("背景色", tooltip="设置背景色")
        layout.addWidget(background_group)

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
        self.buttons['config']['params'] = param_group.add_large_button("全局参数", tooltip="编辑全局参数")
        self.buttons['config']['mesh_params'] = param_group.add_large_button("部件参数", tooltip="编辑部件参数")
        layout.addWidget(param_group)

        config_group = RibbonGroup("")
        self.buttons['config']['import_config'] = config_group.add_large_button("导入配置", tooltip="导入配置")
        self.buttons['config']['export_config'] = config_group.add_large_button("导出配置", tooltip="导出配置")
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

        region_group = RibbonGroup("")
        self.buttons['mesh']['create_region'] = region_group.add_large_button("创建区域", tooltip="创建区域 - 选择多条Connector形成封闭区域")
        layout.addWidget(region_group)

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

    def create_geometry_tab(self):
        """Create the geometry ribbon tab"""
        tab_widget = QWidget()
        layout = create_horizontal_layout(
            margins=LayoutConfig.RIBBON_TAB_MARGINS,
            spacing=LayoutConfig.RIBBIN_TAB_SPACING,
            alignment=Qt.AlignLeft
        )

        geometry_io_group = RibbonGroup("几何输入输出")
        self.buttons['geometry']['import_geometry'] = geometry_io_group.add_large_button("导入几何", tooltip="导入几何文件 (IGES/STEP/STL等)")
        self.buttons['geometry']['export_geometry'] = geometry_io_group.add_large_button("导出几何", tooltip="导出几何文件 (IGES/STEP/STL等)")
        layout.addWidget(geometry_io_group)
        

        geometry_group = RibbonGroup("几何操作")
        self.buttons['geometry']['create_geometry'] = geometry_group.add_large_button("创建几何", tooltip="创建点/线/圆弧/曲线")
        self.buttons['geometry']['delete_geometry'] = geometry_group.add_large_button("删除几何", tooltip="删除选中的几何元素")
        layout.addWidget(geometry_group)

        boundary_extract_group = RibbonGroup("边界网格提取")
        self.buttons['geometry']['import'] = boundary_extract_group.add_large_button("导入网格", tooltip="导入网格 (Ctrl+I)")
        self.buttons['geometry']['extract_boundary'] = boundary_extract_group.add_large_button("提取边界", icon=get_icon('extract_boundary'), tooltip="提取边界网格及部件信息")
        layout.addWidget(boundary_extract_group)

        line_mesh_group = RibbonGroup("线网格生成")
        self.buttons['geometry']['line_mesh'] = line_mesh_group.add_large_button("线网格", icon=get_icon('line-mesh-generate'), tooltip="生成线网格 (Ctrl+L)")
        layout.addWidget(line_mesh_group)

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

        # 小工具：安全记录信息到 InfoOutput 中（如果存在）
        def _log(message):
            try:
                if hasattr(main_window, 'info_output') and hasattr(main_window.info_output, 'log_info'):
                    main_window.info_output.log_info(message)
            except Exception:
                pass

        # 小工具：包装原有回调，调用后执行日志（不影响原有异常传播）
        def _wrap(func, message=None):
            def _wrapped():
                try:
                    func()
                finally:
                    if message:
                        _log(message)
            return _wrapped

        # File tab callbacks
        self.buttons['file']['new'].clicked.connect(_wrap(main_window.new_config, "新建工程"))
        self.buttons['file']['open'].clicked.connect(_wrap(main_window.open_config, "打开工程"))
        self.buttons['file']['save'].clicked.connect(_wrap(main_window.config_manager.save_config, "保存工程"))
        self.buttons['file']['import'].clicked.connect(_wrap(main_window.mesh_operations.import_mesh, "导入网格"))
        self.buttons['file']['export'].clicked.connect(_wrap(main_window.mesh_operations.export_mesh, "导出网格"))
        self.buttons['file']['import_geometry'].clicked.connect(_wrap(main_window.import_geometry, "导入几何"))
        self.buttons['file']['export_geometry'].clicked.connect(_wrap(main_window.export_geometry, "导出几何"))
        self.buttons['file']['mesh_dimension'].clicked.connect(_wrap(main_window.mesh_operations.set_mesh_dimension, "设置网格维度"))

        # Geometry tab callbacks
        self.buttons['geometry']['import_geometry'].clicked.connect(_wrap(main_window.import_geometry, "导入几何"))
        self.buttons['geometry']['export_geometry'].clicked.connect(_wrap(main_window.export_geometry, "导出几何"))
        self.buttons['geometry']['import'].clicked.connect(_wrap(main_window.mesh_operations.import_mesh, "导入网格"))
        self.buttons['geometry']['extract_boundary'].clicked.connect(_wrap(main_window.mesh_operations.extract_boundary_mesh_info, "提取边界网格"))
        self.buttons['geometry']['create_geometry'].clicked.connect(_wrap(main_window.open_geometry_create_dialog, "创建几何"))
        self.buttons['geometry']['delete_geometry'].clicked.connect(_wrap(main_window.open_geometry_delete_dialog, "删除几何"))
        self.buttons['geometry']['line_mesh'].clicked.connect(_wrap(main_window.open_line_mesh_dialog, "打开线网格生成对话框"))

        # View tab callbacks
        self.buttons['view']['reset'].clicked.connect(_wrap(main_window.view_controller.reset_view, "重置视图"))
        self.buttons['view']['fit'].clicked.connect(_wrap(main_window.view_controller.fit_view, "适应视图"))
        self.buttons['view']['zoom_in'].clicked.connect(_wrap(main_window.view_controller.zoom_in, "放大视图"))
        self.buttons['view']['zoom_out'].clicked.connect(_wrap(main_window.view_controller.zoom_out, "缩小视图"))
        self.buttons['view']['view_x_pos'].clicked.connect(_wrap(main_window.view_controller.set_view_x_positive, "X轴正向视图"))
        self.buttons['view']['view_x_neg'].clicked.connect(_wrap(main_window.view_controller.set_view_x_negative, "X轴负向视图"))
        self.buttons['view']['view_y_pos'].clicked.connect(_wrap(main_window.view_controller.set_view_y_positive, "Y轴正向视图"))
        self.buttons['view']['view_y_neg'].clicked.connect(_wrap(main_window.view_controller.set_view_y_negative, "Y轴负向视图"))
        self.buttons['view']['view_z_pos'].clicked.connect(_wrap(main_window.view_controller.set_view_z_positive, "Z轴正向视图"))
        self.buttons['view']['view_z_neg'].clicked.connect(_wrap(main_window.view_controller.set_view_z_negative, "Z轴负向视图"))
        self.buttons['view']['view_iso'].clicked.connect(_wrap(main_window.view_controller.set_view_isometric, "等轴测视图"))
        self.buttons['view']['surface'].clicked.connect(_wrap(lambda: main_window.view_controller.set_render_mode("surface"), "显示模式切换为 实体模式"))
        self.buttons['view']['wireframe'].clicked.connect(_wrap(lambda: main_window.view_controller.set_render_mode("wireframe"), "显示模式切换为 线框模式"))
        self.buttons['view']['surface-wireframe'].clicked.connect(_wrap(lambda: main_window.view_controller.set_render_mode("surface-wireframe"), "显示模式切换为 实体+线框模式"))
        self.buttons['view']['background'].clicked.connect(_wrap(main_window.view_controller.set_background_color, "设置背景色"))

        # Config tab callbacks
        self.buttons['config']['params'].clicked.connect(_wrap(main_window.config_manager.edit_params, "编辑全局参数"))
        self.buttons['config']['mesh_params'].clicked.connect(_wrap(main_window.part_manager.edit_mesh_params, "编辑部件参数"))
        self.buttons['config']['import_config'].clicked.connect(_wrap(main_window.config_manager.import_config, "导入配置"))
        self.buttons['config']['export_config'].clicked.connect(_wrap(main_window.config_manager.export_config, "导出配置"))

        # Mesh tab callbacks
        self.buttons['mesh']['generate'].clicked.connect(_wrap(main_window.mesh_operations.generate_mesh, "开始生成网格"))
        self.buttons['mesh']['display'].clicked.connect(_wrap(main_window.display_mesh, "显示网格"))
        self.buttons['mesh']['clear'].clicked.connect(_wrap(main_window.clear_mesh, "清空网格"))
        self.buttons['mesh']['create_region'].clicked.connect(_wrap(main_window.open_create_region_dialog, "打开创建区域对话框"))
        self.buttons['mesh']['quality'].clicked.connect(_wrap(main_window.mesh_operations.check_mesh_quality, "检查网格质量"))
        self.buttons['mesh']['smooth'].clicked.connect(_wrap(main_window.mesh_operations.smooth_mesh, "平滑网格"))
        self.buttons['mesh']['optimize'].clicked.connect(_wrap(main_window.mesh_operations.optimize_mesh, "优化网格"))
        self.buttons['mesh']['statistics'].clicked.connect(_wrap(main_window.mesh_operations.show_mesh_statistics, "网格统计"))
        self.buttons['mesh']['report'].clicked.connect(_wrap(main_window.mesh_operations.export_mesh_report, "导出网格报告"))

        # Help tab callbacks
        self.buttons['help']['manual'].clicked.connect(_wrap(main_window.help_module.show_user_manual, "查看用户手册"))
        self.buttons['help']['quick_start'].clicked.connect(_wrap(main_window.help_module.show_quick_start, "查看入门指南"))
        self.buttons['help']['shortcuts'].clicked.connect(_wrap(main_window.help_module.show_shortcuts, "查看快捷键"))
        self.buttons['help']['updates'].clicked.connect(_wrap(main_window.help_module.check_for_updates, "检查更新"))
        self.buttons['help']['about'].clicked.connect(_wrap(main_window.help_module.show_about, "关于本软件"))

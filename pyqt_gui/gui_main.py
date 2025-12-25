#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QLabel, QTextEdit, QPushButton,
    QListWidget, QTabWidget, QFrame, QMenuBar, QStatusBar,
    QToolBar, QAction, QFileDialog, QMessageBox, QScrollArea,
    QDockWidget, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SUB_DIRS = [
    os.path.join(PROJECT_ROOT, 'data_structure'),
    os.path.join(PROJECT_ROOT, 'fileIO'),
    os.path.join(PROJECT_ROOT, 'meshsize'),
    os.path.join(PROJECT_ROOT, 'visualization'),
    os.path.join(PROJECT_ROOT, 'adfront2'),
    os.path.join(PROJECT_ROOT, 'optimize'),
    os.path.join(PROJECT_ROOT, 'utils')
]

for dir_path in SUB_DIRS:
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

from pyqt_gui.gui_base import (
    StatusBar, InfoOutput, DialogBase,
    Splitter, PartListWidget
)
from pyqt_gui.ribbon_widget import RibbonWidget
from pyqt_gui.mesh_display import MeshDisplayArea
from pyqt_gui.ui_utils import UIStyles
from data_structure.parameters import Parameters
# 从core模块导入网格生成函数
import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from core import generate_mesh


class SimplifiedPyMeshGenGUI(QMainWindow):
    """PyQt版PyMeshGen GUI主类"""

    MIN_WINDOW_WIDTH = 1200
    MIN_WINDOW_HEIGHT = 800
    DEFAULT_FONT_FAMILY = "Microsoft YaHei"
    DEFAULT_FONT_SIZE = 9

    def __init__(self):
        super().__init__()
        self._setup_window()
        self._setup_fonts()
        self._initialize_modules()
        self._initialize_data()
        self._create_widgets()
        self.update_status("就绪")

    def _setup_window(self):
        """设置窗口大小和标题"""
        self.setWindowTitle("PyMeshGen - 基于Python的网格生成工具")

        icon_path = os.path.join(PROJECT_ROOT, "gui", "icons", "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        screen_width = screen_size.width()
        screen_height = screen_size.height()

        window_width, window_height = self._calculate_window_size(
            screen_width, screen_height
        )

        self.resize(window_width, window_height)

        frame_geometry = self.frameGeometry()
        center_point = screen.availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

        self.setMinimumSize(self.MIN_WINDOW_WIDTH, self.MIN_WINDOW_HEIGHT)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

    def _calculate_window_size(self, screen_width, screen_height):
        """根据屏幕分辨率计算窗口大小"""
        if screen_width >= 1920 and screen_height >= 1080:
            return int(screen_width * 0.75), int(screen_height * 0.75)
        elif screen_width >= 1366 and screen_height >= 768:
            return int(screen_width * 0.85), int(screen_height * 0.85)
        else:
            return int(screen_width * 0.95), int(screen_height * 0.95)

    def _setup_fonts(self):
        """设置字体"""
        font = QFont()
        font.setFamily(self.DEFAULT_FONT_FAMILY)
        font.setPointSize(self.DEFAULT_FONT_SIZE)
        font.setStyleHint(QFont.SansSerif)
        self.setFont(font)
        self._apply_stylesheet()

    def _apply_stylesheet(self):
        """应用全局样式表"""
        self.setStyleSheet(UIStyles.MAIN_WINDOW_STYLESHEET)

    def _initialize_modules(self):
        """初始化模块"""
        self.project_root = PROJECT_ROOT

    def _initialize_data(self):
        """初始化数据"""
        self.params = None
        self.mesh_generator = None
        self.current_mesh = None
        self.cas_parts_info = None
        self.parts_params = []  # 初始化部件参数列表
        self.render_mode = "wireframe"
        self.show_boundary = True

    def _create_widgets(self):
        """创建UI组件"""
        self._create_menu()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(2)

        self._create_main_layout(main_layout)

        self.status_bar = StatusBar(self)

    def _create_main_layout(self, main_layout):
        """创建主布局"""
        self.paned_window = QSplitter(Qt.Horizontal)

        self._create_left_panel()
        self._create_right_panel_layout()

        self.paned_window.setStretchFactor(0, 1)
        self.paned_window.setStretchFactor(1, 9)
        self.paned_window.setStyleSheet(UIStyles.SPLITTER_STYLESHEET)

        main_layout.addWidget(self.paned_window)

    def _create_left_panel(self):
        """创建左侧部件信息区域"""
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setSpacing(2)

        self._create_parts_list_widget()
        self._create_properties_panel()

        self.left_panel.setMinimumWidth(300)
        self.paned_window.addWidget(self.left_panel)

        self.parts_list_widget.parts_list.currentRowChanged.connect(self.on_part_select)

    def _create_parts_list_widget(self):
        """创建部件列表组件"""
        self.parts_list_widget = PartListWidget(
            parent=self,
            add_callback=self.add_part,
            remove_callback=self.remove_part,
            edit_callback=self.edit_part,
            show_callback=self.show_selected_part,
            show_only_callback=self.show_only_selected_part
        )

        self._apply_parts_list_styling()

        parts_frame_container = QGroupBox("部件列表")
        parts_layout = QVBoxLayout(parts_frame_container)
        parts_layout.setSpacing(2)

        parts_layout.addWidget(self.parts_list_widget.widget)

        left_layout = self.left_panel.layout()
        left_layout.addWidget(parts_frame_container)

    def _apply_parts_list_styling(self):
        """应用部件列表样式"""
        if hasattr(self.parts_list_widget, 'parts_list'):
            self.parts_list_widget.parts_list.setStyleSheet(UIStyles.LIST_WIDGET_STYLESHEET)

        for button_name, style in UIStyles.BUTTON_STYLESHEETS.items():
            if hasattr(self.parts_list_widget, button_name):
                getattr(self.parts_list_widget, button_name).setStyleSheet(style)

    def _create_properties_panel(self):
        """创建属性面板"""
        self.props_frame = QGroupBox("属性面板")
        self.props_frame.setStyleSheet(UIStyles.GROUPBOX_STYLESHEET)

        props_layout = QVBoxLayout()
        props_layout.setSpacing(2)

        self.props_text = QTextEdit()
        self.props_text.setReadOnly(True)
        self.props_text.setStyleSheet(UIStyles.TEXTEDIT_STYLESHEET)
        props_layout.addWidget(self.props_text)
        self.props_frame.setLayout(props_layout)

        left_layout = self.left_panel.layout()
        left_layout.addWidget(self.props_frame)

    def _create_right_panel_layout(self):
        """创建右侧布局"""
        self.right_paned = QSplitter(Qt.Vertical)

        self._create_mesh_display_area()
        self._create_status_output_area()

        self.paned_window.addWidget(self.right_paned)

        self.right_paned.setStretchFactor(0, 8)
        self.right_paned.setStretchFactor(1, 2)
        self.right_paned.setStyleSheet(UIStyles.SPLITTER_STYLESHEET)

    def _create_mesh_display_area(self):
        """创建网格显示区域"""
        self.main_mesh_display = MeshDisplayArea(self)
        self.main_mesh_display.frame.setStyleSheet(UIStyles.FRAME_STYLESHEET)

        self.mesh_display = self.main_mesh_display
        self.main_mesh_display.frame.setFocusPolicy(Qt.StrongFocus)
        self.main_mesh_display.frame.keyPressEvent = self.on_mesh_display_key

        self.right_paned.addWidget(self.main_mesh_display.frame)

    def _create_status_output_area(self):
        """创建状态输出区域"""
        self.info_output = InfoOutput(self)

        if hasattr(self.info_output, 'frame'):
            self.info_output.frame.setStyleSheet(UIStyles.GROUPBOX_STYLESHEET)

        if hasattr(self.info_output, 'info_text'):
            self.info_output.info_text.setStyleSheet(UIStyles.TEXTEDIT_MONOSPACE_STYLESHEET)

        self.right_paned.addWidget(self.info_output.frame)

    def _create_menu(self):
        """创建菜单"""
        self._create_ribbon()

    def _create_ribbon(self):
        """创建功能区"""
        self.ribbon = RibbonWidget(self)
        self._setup_ribbon_icons()
        self.ribbon.set_callbacks(self)
        self.setMenuWidget(self.ribbon)
        self.ribbon.toggle_button.clicked.connect(self.toggle_ribbon)

        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        shortcut.activated.connect(self.toggle_ribbon)

    def _setup_ribbon_icons(self):
        """设置功能区图标"""
        for button_name, button in self.ribbon.buttons.get('file', {}).items():
            button.setIcon(self._get_standard_icon({
                'new': 'document-new',
                'open': 'document-open',
                'save': 'document-save',
                'import': 'document-import',
                'export': 'document-export'
            }.get(button_name, 'document-new')))

        for button_name, button in self.ribbon.buttons.get('view', {}).items():
            button.setIcon(self._get_standard_icon({
                'reset': 'view-refresh',
                'fit': 'zoom-fit-best',
                'zoom_in': 'zoom-in',
                'zoom_out': 'zoom-out'
            }.get(button_name, 'view-refresh')))

        for button_name, button in self.ribbon.buttons.get('config', {}).items():
            button.setIcon(self._get_standard_icon({
                'params': 'configure',
                'mesh_params': 'configure',
                'boundary': 'configure',
                'import_config': 'document-properties',
                'export_config': 'document-properties',
                'reset': 'edit-clear'
            }.get(button_name, 'configure')))

        for button_name, button in self.ribbon.buttons.get('mesh', {}).items():
            button.setIcon(self._get_standard_icon({
                'generate': 'system-run',
                'display': 'view-fullscreen',
                'clear': 'edit-delete',
                'quality': 'tools-check-spelling',
                'smooth': 'tools-check-spelling',
                'optimize': 'tools-check-spelling',
                'statistics': 'x-office-spreadsheet',
                'report': 'x-office-spreadsheet'
            }.get(button_name, 'system-run')))

        for button_name, button in self.ribbon.buttons.get('help', {}).items():
            button.setIcon(self._get_standard_icon({
                'manual': 'help-contents',
                'quick_start': 'help-faq',
                'shortcuts': 'help-keyboard-shortcuts',
                'updates': 'help-about',
                'about': 'help-about'
            }.get(button_name, 'help-contents')))

    def _get_standard_icon(self, icon_name):
        """获取标准图标"""
        from PyQt5.QtWidgets import QStyle

        icon_map = {
            'document-new': QStyle.SP_FileIcon,
            'document-open': QStyle.SP_DirOpenIcon,
            'document-save': QStyle.SP_DialogSaveButton,
            'document-import': QStyle.SP_ArrowUp,
            'document-export': QStyle.SP_ArrowDown,
            'zoom-fit-best': QStyle.SP_ComputerIcon,
            'zoom-in': QStyle.SP_ArrowUp,
            'zoom-out': QStyle.SP_ArrowDown,
            'view-refresh': QStyle.SP_BrowserReload,
            'configure': QStyle.SP_ToolBarHorizontalExtensionButton,
            'document-properties': QStyle.SP_FileDialogDetailedView,
            'edit-clear': QStyle.SP_LineEditClearButton,
            'system-run': QStyle.SP_MediaPlay,
            'view-fullscreen': QStyle.SP_TitleBarMaxButton,
            'edit-delete': QStyle.SP_TrashIcon,
            'tools-check-spelling': QStyle.SP_DialogApplyButton,
            'x-office-spreadsheet': QStyle.SP_FileDialogContentsView,
            'help-contents': QStyle.SP_MessageBoxInformation,
            'help-faq': QStyle.SP_MessageBoxQuestion,
            'help-keyboard-shortcuts': QStyle.SP_DialogHelpButton,
            'help-about': QStyle.SP_MessageBoxQuestion,
        }

        return self.style().standardIcon(icon_map.get(icon_name, QStyle.SP_MessageBoxInformation))

    def toggle_ribbon(self):
        """切换功能区显示"""
        if hasattr(self, 'ribbon') and self.ribbon:
            self.ribbon.toggle_content_visibility()

    def set_render_mode(self, mode):
        """设置渲染模式"""
        self.render_mode = mode
        if hasattr(self, 'mesh_display'):
            self.mesh_display.set_render_mode(mode)

        mode_messages = {
            "surface": "渲染模式: 实体模式 (1键)",
            "wireframe": "渲染模式: 线框模式 (2键)",
            "mixed": "渲染模式: 混合模式 (3键)",
            "surface-wireframe": "渲染模式: 混合模式 (3键)",
            "points": "渲染模式: 点云模式 (4键)"
        }
        self.update_status(mode_messages.get(mode, f"渲染模式: {mode}"))

    def on_mesh_display_key(self, event):
        """处理网格显示区域的键盘事件"""
        key = event.key()
        key_actions = {
            Qt.Key_R: lambda: self.reset_view() or self.update_status("已重置视图 (R键)"),
            Qt.Key_F: lambda: self.fit_view() or self.update_status("已适应视图 (F键)"),
            Qt.Key_O: self._toggle_boundary_display,
            Qt.Key_1: lambda: self.set_render_mode("surface"),
            Qt.Key_2: lambda: self.set_render_mode("wireframe"),
            Qt.Key_3: lambda: self.set_render_mode("mixed"),
            Qt.Key_4: lambda: self.set_render_mode("points"),
        }

        action = key_actions.get(key)
        if action:
            action()

    def _toggle_boundary_display(self):
        """切换边界显示"""
        new_state = not self.show_boundary
        self.show_boundary = new_state
        self.mesh_display.toggle_boundary_display(new_state)
        self.update_status(f"边界显示: {'开启' if new_state else '关闭'} (O键)")

    def new_config(self):
        """新建工程"""
        self.log_info("新建工程功能暂未实现")
        self.update_status("新建工程功能暂未实现")

    def open_config(self):
        """打开工程"""
        self.log_info("打开工程功能暂未实现")
        self.update_status("打开工程功能暂未实现")

    def save_config(self):
        """保存工程"""
        self.log_info("保存工程功能暂未实现")
        self.update_status("保存工程功能暂未实现")

    def import_mesh(self):
        """导入网格"""
        from pyqt_gui.file_operations import FileOperations

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入网格文件",
            os.path.join(self.project_root, "meshes"),
            "网格文件 (*.vtk *.stl *.obj *.cas *.msh *.ply);;所有文件 (*.*)"
        )

        if file_path:
            try:
                file_ops = FileOperations(self.project_root)
                mesh_data = file_ops.import_mesh(file_path)
                self.current_mesh = mesh_data

                if hasattr(self, 'mesh_display'):
                    self.mesh_display.display_mesh(mesh_data)

                # 从MeshData对象中获取部件信息
                if hasattr(mesh_data, 'parts_info') and mesh_data.parts_info:
                    self.update_parts_list_from_cas(mesh_data.parts_info)

                self.log_info(f"已导入网格: {file_path}")
                self.update_status("已导入网格")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入网格失败: {str(e)}")
                self.log_error(f"导入网格失败: {str(e)}")

    def export_mesh(self):
        """导出网格"""
        from pyqt_gui.file_operations import FileOperations

        if not self.current_mesh:
            QMessageBox.warning(self, "警告", "没有可导出的网格")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出网格文件",
            os.path.join(self.project_root, "meshes", "mesh.vtk"),
            "网格文件 (*.vtk *.stl *.obj *.msh *.ply)"
        )

        if file_path:
            try:
                file_ops = FileOperations(self.project_root)
                vtk_poly_data = None
                if isinstance(self.current_mesh, dict):
                    vtk_poly_data = self.current_mesh.get('vtk_poly_data')

                if vtk_poly_data:
                    file_ops.export_mesh(vtk_poly_data, file_path)
                else:
                    if hasattr(self.current_mesh, 'node_coords') and hasattr(self.current_mesh, 'cell_container'):
                        if hasattr(self.current_mesh, 'save_to_vtkfile'):
                            self.current_mesh.save_to_vtkfile(file_path)
                        else:
                            QMessageBox.warning(self, "警告", "当前网格格式不支持直接保存，请使用VTK格式")
                            return
                    else:
                        QMessageBox.warning(self, "警告", "无法获取有效的VTK数据进行导出")
                        return

                self.log_info(f"已导出网格文件: {file_path}")
                self.update_status("已导出网格文件")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出网格失败: {str(e)}")
                self.log_error(f"导出网格失败: {str(e)}")

    def clear_mesh(self):
        """清空网格"""
        self.current_mesh = None
        if hasattr(self, 'mesh_display'):
            self.mesh_display.clear()
        self.log_info("已清空网格")
        self.update_status("已清空网格")

    def reset_view(self):
        """重置视图"""
        if hasattr(self, 'mesh_display'):
            self.mesh_display.reset_view()
        self.log_info("已重置视图")
        self.update_status("已重置视图")

    def fit_view(self):
        """适应视图"""
        if hasattr(self, 'mesh_display'):
            self.mesh_display.fit_view()
        self.log_info("已适应视图")
        self.update_status("已适应视图")

    def add_part(self):
        """添加部件"""
        if not self.params:
            QMessageBox.warning(self, "警告", "请先创建或加载配置")
            return

        part_name = f"部件{self.parts_list_widget.parts_list.count() + 1}"
        self.parts_list_widget.parts_list.addItem(part_name)
        self.log_info(f"已添加部件: {part_name}")
        self.update_status("已添加部件")

    def remove_part(self):
        """删除部件"""
        if not self.params:
            QMessageBox.warning(self, "警告", "请先创建或加载配置")
            return

        current_row = self.parts_list_widget.parts_list.currentRow()
        if current_row >= 0:
            item = self.parts_list_widget.parts_list.takeItem(current_row)
            part_name = item.text()
            self.log_info(f"已删除部件: {part_name}")
            self.update_status("已删除部件")
        else:
            QMessageBox.warning(self, "警告", "请先选择要删除的部件")

    def edit_part(self):
        """编辑部件参数（从右键菜单调用）"""
        from PyQt5.QtWidgets import QDialog
        from pyqt_gui.part_params_dialog import PartParamsDialog
        
        # 获取当前选中的部件索引
        current_row = self.parts_list_widget.parts_list.currentRow()
        if current_row < 0:
            return
        
        # 检查是否有导入的网格数据
        if not hasattr(self, 'current_mesh') or not self.current_mesh:
            QMessageBox.warning(self, "警告", "请先导入网格文件以获取部件列表")
            self.log_info("未检测到导入的网格数据，无法设置部件参数")
            self.update_status("未检测到导入的网格数据")
            return
        
        # 从当前导入的网格数据中获取部件列表
        parts_params = []
        
        # 获取部件信息
        parts_info = None
        if hasattr(self.current_mesh, 'parts_info') and self.current_mesh.parts_info:
            parts_info = self.current_mesh.parts_info
        
        # 获取当前部件名称列表
        current_part_names = []
        if parts_info:
            if isinstance(parts_info, dict):
                # 部件信息是字典格式: {part_name: part_info}
                current_part_names = list(parts_info.keys())
            elif isinstance(parts_info, list):
                # 部件信息是列表格式: [{part_name: "xxx", ...}, ...]
                current_part_names = [part_info.get('part_name', f'部件{parts_info.index(part_info)}') for part_info in parts_info]
        else:
            # 如果没有部件信息，获取当前选中部件的名称
            selected_part_item = self.parts_list_widget.parts_list.item(current_row)
            if selected_part_item:
                current_part_names = [selected_part_item.text()]
            else:
                QMessageBox.warning(self, "警告", "未找到选中的部件")
                return
        
        # 创建已保存参数的映射字典，按部件名称索引
        saved_params_map = {}
        if hasattr(self, 'parts_params') and self.parts_params:
            for param in self.parts_params:
                if 'part_name' in param:
                    saved_params_map[param['part_name']] = param
            self.log_info(f"使用已保存的部件参数，共 {len(saved_params_map)} 个部件")
        
        # 为每个当前部件创建参数，优先使用已保存的参数
        for part_name in current_part_names:
            if part_name in saved_params_map:
                # 使用已保存的参数
                parts_params.append(saved_params_map[part_name])
            else:
                # 使用默认参数
                parts_params.append({
                    "part_name": part_name,
                    "max_size": 1e6,
                    "PRISM_SWITCH": "off",
                    "first_height": 0.01,
                    "growth_rate": 1.2,
                    "max_layers": 5,
                    "full_layers": 5,
                    "multi_direction": False
                })
        
        self.log_info(f"已准备 {len(parts_params)} 个部件的参数")
        
        # 创建并显示对话框，默认选中当前部件
        dialog = PartParamsDialog(self, parts=parts_params, current_part=current_row)
        if dialog.exec_() == QDialog.Accepted:
            # 获取设置后的参数
            self.parts_params = dialog.get_parts_params()
            self.log_info(f"已更新部件参数，共 {len(self.parts_params)} 个部件")
            self.update_status("部件参数已更新")
        else:
            self.log_info("取消设置部件参数")
            self.update_status("已取消部件参数设置")
    
    def show_selected_part(self):
        """显示选中部件（从右键菜单调用）"""
        # 获取当前选中的部件索引
        current_row = self.parts_list_widget.parts_list.currentRow()
        if current_row < 0:
            return

        # 获取选中部件的名称
        selected_part_item = self.parts_list_widget.parts_list.item(current_row)
        if selected_part_item:
            selected_part_name = selected_part_item.text()

            # 在日志中显示部件信息
            self.log_info(f"显示选中部件: {selected_part_name}")

            # 更新状态栏
            self.update_status(f"已选中部件: {selected_part_name}")

            # 如果有3D显示区域，只显示选中的部件
            if hasattr(self, 'mesh_display'):
                try:
                    success = self.mesh_display.display_part(selected_part_name, parts_info=self.cas_parts_info)
                    if success:
                        self.log_info(f"成功显示部件: {selected_part_name}")
                    else:
                        self.log_error(f"显示部件失败: {selected_part_name}")
                except Exception as e:
                    self.log_error(f"显示部件失败: {str(e)}")



    def show_only_selected_part(self):
        """只显示选中部件，隐藏其他所有部件（从右键菜单调用）"""
        # 获取当前选中的部件索引
        current_row = self.parts_list_widget.parts_list.currentRow()
        if current_row < 0:
            return

        # 获取选中部件的名称
        selected_part_item = self.parts_list_widget.parts_list.item(current_row)
        if selected_part_item:
            selected_part_name = selected_part_item.text()

            # 在日志中显示部件信息
            self.log_info(f"只显示选中部件: {selected_part_name}")

            # 更新状态栏
            self.update_status(f"只显示部件: {selected_part_name}")

            # 如果有3D显示区域，只显示选中的部件
            if hasattr(self, 'mesh_display'):
                try:
                    success = self.mesh_display.show_only_selected_part(selected_part_name, parts_info=self.cas_parts_info)
                    if success:
                        self.log_info(f"成功只显示部件: {selected_part_name}")
                    else:
                        self.log_error(f"只显示部件失败: {selected_part_name}")
                except Exception as e:
                    self.log_error(f"只显示部件失败: {str(e)}")



    def update_params_display(self):
        """更新参数显示"""
        if not self.params:
            return
        self.update_parts_list()

    def update_parts_list(self):
        """更新部件列表"""
        self.parts_list_widget.parts_list.clear()

        if self.params:
            if hasattr(self.params, 'part_params') and self.params.part_params:
                for i, part in enumerate(self.params.part_params):
                    part_name = getattr(part, 'part_name', f"部件{i+1}")
                    if hasattr(part, 'part_params') and hasattr(part.part_params, 'part_name'):
                        part_name = part.part_params.part_name
                    self.parts_list_widget.parts_list.addItem(part_name)
            else:
                self.parts_list_widget.parts_list.addItem("默认部件")

    def update_parts_list_from_cas(self, parts_info):
        """从cas文件的部件信息更新部件列表"""
        self.parts_list_widget.parts_list.clear()
        self.cas_parts_info = parts_info

        if isinstance(parts_info, list):
            for part_info in parts_info:
                part_name = part_info.get('part_name', '未知部件')
                self.parts_list_widget.parts_list.addItem(part_name)
        elif isinstance(parts_info, dict):
            for part_name in parts_info.keys():
                if part_name not in ['type', 'node_coords', 'cells', 'num_points', 'num_cells', 'unstr_grid']:
                    self.parts_list_widget.parts_list.addItem(part_name)
        elif parts_info:
            self.parts_list_widget.parts_list.addItem(str(parts_info))

    def on_part_select(self, index):
        """处理部件列表选择事件"""
        if index < 0:
            self.props_text.setPlainText("未选择任何部件\n请从左侧列表中选择一个部件以查看其属性")
            self.update_status("未选择部件")
            # 清除高亮
            if hasattr(self, 'mesh_display'):
                self.mesh_display.highlight_part(None, False)
            return

        part_index = index
        selected_part_name = ""
        if self.parts_list_widget.parts_list.count() > index:
            selected_part_name = self.parts_list_widget.parts_list.item(index).text()

        if hasattr(self, 'cas_parts_info') and self.cas_parts_info:
            if isinstance(self.cas_parts_info, dict) and selected_part_name in self.cas_parts_info:
                part_info = self.cas_parts_info[selected_part_name]
                props_content = f"=== CAS部件属性 ===\n\n"
                props_content += f"部件名称: {selected_part_name}\n"
                props_content += f"边界条件类型: {part_info.get('type', '未知') if isinstance(part_info, dict) else '未知'}\n"
                props_content += f"面数量: {len(part_info.get('faces', []) if isinstance(part_info, dict) else [])}\n"
                if isinstance(part_info, dict) and 'faces' in part_info:
                    total_nodes = sum(len(face.get('nodes', [])) for face in part_info['faces'])
                    props_content += f"节点数量: {total_nodes}\n"
                else:
                    props_content += f"节点数量: 0\n"
                props_content += f"单元数量: 0\n"
                props_content += f"\n=== 状态信息 ===\n"
                props_content += f"选择时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                props_content += f"部件索引: {part_index}\n"
                props_content += f"总部件数: {len(self.cas_parts_info)}\n"
                props_content += f"数据来源: CAS文件\n"

                self.props_text.setPlainText(props_content)
                self.update_status(f"已选中CAS部件: {selected_part_name}")
                
                # 高亮显示选中的部件
                if hasattr(self, 'mesh_display'):
                    self.mesh_display.highlight_part(selected_part_name, highlight=True, parts_info=self.cas_parts_info)
                return
            elif isinstance(self.cas_parts_info, list) and part_index < len(self.cas_parts_info):
                part_info = self.cas_parts_info[part_index]
                part_name = part_info.get('part_name', f'部件{part_index}')
                props_content = f"=== CAS部件属性 ===\n\n"
                props_content += f"部件名称: {part_name}\n"
                props_content += f"边界条件类型: {part_info.get('bc_type', '未知')}\n"
                props_content += f"面数量: {part_info.get('face_count', 0)}\n"
                props_content += f"节点数量: {len(part_info.get('nodes', []))}\n"
                props_content += f"单元数量: {len(part_info.get('cells', []))}\n"
                props_content += f"\n=== 状态信息 ===\n"
                props_content += f"选择时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                props_content += f"部件索引: {part_index}\n"
                props_content += f"总部件数: {len(self.cas_parts_info)}\n"
                props_content += f"数据来源: CAS文件\n"

                self.props_text.setPlainText(props_content)
                self.update_status(f"已选中CAS部件: {part_name}")
                
                # 高亮显示选中的部件
                if hasattr(self, 'mesh_display'):
                    # 当cas_parts_info是列表时，需要构建一个临时的部件字典
                    temp_parts_info = {}
                    for info in self.cas_parts_info:
                        temp_parts_info[info.get('part_name', f'部件{self.cas_parts_info.index(info)}')] = info
                    self.mesh_display.highlight_part(part_name, highlight=True, parts_info=temp_parts_info)
                return

        if hasattr(self, 'params') and self.params:
            try:
                part_name = f"部件{part_index + 1}"
                props_content = f"=== 部件属性 ===\n\n"
                props_content += f"名称: {part_name}\n"
                props_content += f"类型: 默认类型\n"
                props_content += f"ID: {part_index}\n"
                props_content += f"\n=== 状态信息 ===\n"
                props_content += f"选择时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                props_content += f"部件索引: {part_index}\n"

                self.props_text.setPlainText(props_content)
                self.update_status(f"已选中部件: {part_name}")
                
                # 高亮显示选中的部件
                if hasattr(self, 'mesh_display'):
                    self.mesh_display.highlight_part(part_name, highlight=True)
            except Exception as e:
                self.log_error(f"显示部件属性时出错: {str(e)}")
                error_content = f"=== 错误信息 ===\n\n"
                error_content += f"显示部件属性时出错:\n{str(e)}\n"
                error_content += f"\n请检查部件数据是否正确\n"
                self.props_text.setPlainText(error_content)

    def log_info(self, message):
        """记录信息日志"""
        if hasattr(self, 'info_output'):
            self.info_output.log_info(message)

    def log_error(self, message):
        """记录错误日志"""
        if hasattr(self, 'info_output'):
            self.info_output.log_error(message)

    def update_status(self, message):
        """更新状态栏信息"""
        if hasattr(self, 'status_bar'):
            self.status_bar.update_status(message)

    def show_about(self):
        """显示关于对话框"""
        about_text = """PyMeshGen v1.0\n\n基于Python的网格生成工具\n\n© 2025 HighOrderMesh"""
        QMessageBox.about(self, "关于", about_text)

    def show_user_manual(self):
        """显示用户手册"""
        manual_text = """PyMeshGen 用户手册\n\n1. 文件菜单\n   - 新建工程：创建新的工程\n   - 打开工程：打开已保存的工程\n   - 保存工程：保存当前工程\n   - 导入网格：从外部文件导入网格数据\n   - 导出网格：将当前网格导出到文件\n\n2. 视图菜单\n   - 重置视图：将视图恢复到初始状态\n   - 适应视图：自动调整视图以适应整个网格\n   - 放大/缩小：缩放网格显示\n   - 显示工具栏：切换工具栏的显示/隐藏\n   - 显示状态栏：切换状态栏的显示/隐藏\n\n3. 配置菜单\n   - 参数设置：配置网格生成参数\n   - 网格参数：配置网格相关参数\n   - 边界条件：配置边界条件\n   - 导入配置：导入配置文件\n   - 导出配置：导出配置文件\n   - 重置配置：重置配置\n   - 清空网格：清除当前显示的网格\n\n4. 网格菜单\n   - 生成网格：根据当前配置生成网格\n   - 显示网格：显示/隐藏网格\n   - 清空网格：清除当前显示的网格\n   - 检查质量：检查网格质量\n   - 平滑网格：平滑网格\n   - 优化网格：优化网格\n   - 统计信息：显示网格统计信息\n   - 导出报告：导出网格报告\n\n5. 工具栏\n   - 提供常用功能的快速访问按钮\n\n6. 主界面\n   - 左侧：部件信息区域，包含部件列表和属性面板\n   - 右侧：网格视图交互区域，支持缩放、平移和选择操作\n   - 底部：状态栏显示系统状态和信息输出窗口"""
        QMessageBox.about(self, "用户手册", manual_text)

    def zoom_in(self):
        """放大视图"""
        if hasattr(self, 'mesh_display'):
            self.mesh_display.zoom_in()
        self.update_status("视图已放大")

    def zoom_out(self):
        """缩小视图"""
        if hasattr(self, 'mesh_display'):
            self.mesh_display.zoom_out()
        self.update_status("视图已缩小")

    def toggle_toolbar(self):
        """切换功能区显示"""
        if hasattr(self, 'ribbon') and self.ribbon:
            if self.ribbon.isVisible():
                self.ribbon.hide()
                self.update_status("功能区已隐藏")
            else:
                self.ribbon.show()
                self.update_status("功能区已显示")

    def toggle_statusbar(self):
        """切换状态栏显示"""
        if hasattr(self, 'status_bar') and hasattr(self, 'status_bar.status_bar'):
            if self.status_bar.status_bar.isVisible():
                self.status_bar.status_bar.hide()
                self.log_info("状态栏已隐藏")
            else:
                self.status_bar.status_bar.show()
                self.log_info("状态栏已显示")

    def edit_params(self):
        """编辑全局参数"""
        QMessageBox.information(self, "待开发", "全局参数配置功能正在开发中...")
        self.log_info("全局参数配置功能待开发")
        self.update_status("全局参数配置功能待开发")

    def edit_mesh_params(self):
        """编辑部件参数"""
        from PyQt5.QtWidgets import QDialog
        from pyqt_gui.part_params_dialog import PartParamsDialog
        
        # 检查是否有导入的网格数据
        if not hasattr(self, 'cas_parts_info') or not self.cas_parts_info:
            QMessageBox.warning(self, "警告", "请先导入网格文件以获取部件列表")
            self.log_info("未检测到导入的网格数据，无法设置部件参数")
            self.update_status("未检测到导入的网格数据")
            return
        
        # 从当前导入的网格数据中获取部件列表
        parts_params = []
        
        # 获取当前部件名称列表
        current_part_names = []
        if isinstance(self.cas_parts_info, dict):
            current_part_names = list(self.cas_parts_info.keys())
        elif isinstance(self.cas_parts_info, list):
            current_part_names = [part_info.get('part_name', f'部件{self.cas_parts_info.index(part_info)}') for part_info in self.cas_parts_info]
        
        # 创建已保存参数的映射字典，按部件名称索引
        saved_params_map = {}
        if hasattr(self, 'parts_params') and self.parts_params:
            for param in self.parts_params:
                if 'part_name' in param:
                    saved_params_map[param['part_name']] = param
            self.log_info(f"使用已保存的部件参数，共 {len(saved_params_map)} 个部件")
        
        # 为每个当前部件创建参数，优先使用已保存的参数
        for part_name in current_part_names:
            if part_name in saved_params_map:
                # 使用已保存的参数
                parts_params.append(saved_params_map[part_name])
            else:
                # 使用默认参数
                parts_params.append({
                    "part_name": part_name,
                    "max_size": 1e6,
                    "PRISM_SWITCH": "off",
                    "first_height": 0.01,
                    "growth_rate": 1.2,
                    "max_layers": 5,
                    "full_layers": 5,
                    "multi_direction": False
                })
        
        self.log_info(f"已准备 {len(parts_params)} 个部件的参数")
        
        # 创建并显示对话框
        dialog = PartParamsDialog(self, parts=parts_params)
        if dialog.exec_() == QDialog.Accepted:
            # 获取设置后的参数
            self.parts_params = dialog.get_parts_params()
            self.log_info(f"已更新部件参数，共 {len(self.parts_params)} 个部件")
            self.update_status("部件参数已更新")
        else:
            self.log_info("取消设置部件参数")
            self.update_status("已取消部件参数设置")

    def edit_boundary_conditions(self):
        """编辑边界条件"""
        self.log_info("编辑边界条件功能暂未实现")

    def import_config(self):
        """导入配置"""
        QMessageBox.information(self, "待开发", "导入配置功能正在开发中...")
        self.log_info("导入配置功能待开发")
        self.update_status("导入配置功能待开发")

    def export_config(self):
        """导出配置"""
        import json
        import os
        from PyQt5.QtWidgets import QFileDialog
        
        # 检查是否有需要导出的配置
        if not hasattr(self, 'parts_params') or not self.parts_params:
            QMessageBox.warning(self, "警告", "没有可导出的部件参数配置")
            self.log_info("没有可导出的部件参数配置")
            self.update_status("没有可导出的部件参数配置")
            return
        
        # 打开文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出配置文件",
            os.path.join(self.project_root, "config", "exported_config.json"),
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                # 创建配置数据结构
                config_data = {
                    "debug_level": 0,
                    "input_file": "",
                    "output_file": "./out/mesh.vtk",
                    "viz_enabled": True,
                    "parts": self.parts_params
                }
                
                # 保存配置到文件
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                self.log_info(f"配置已成功导出到: {file_path}")
                self.update_status(f"配置已成功导出到: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出配置失败: {str(e)}")
                self.log_error(f"导出配置失败: {str(e)}")

    def reset_config(self):
        """重置配置"""
        self.log_info("重置配置功能暂未实现")

    def generate_mesh(self):
        """生成网格"""
        try:
            # 检查是否有导入的网格文件
            if not hasattr(self, 'current_mesh') or not self.current_mesh:
                QMessageBox.warning(self, "警告", "请先导入网格文件")
                self.log_info("未导入网格文件，无法生成网格")
                self.update_status("未导入网格文件")
                return
            
            # 检查是否有配置好的部件参数
            if not hasattr(self, 'parts_params') or not self.parts_params:
                QMessageBox.warning(self, "警告", "请先配置部件参数")
                self.log_info("未配置部件参数，无法生成网格")
                self.update_status("未配置部件参数")
                return
            
            # 获取输入文件路径
            input_file = ""
            
            if isinstance(self.current_mesh, dict):
                if 'file_path' in self.current_mesh:
                    input_file = self.current_mesh['file_path']
            elif hasattr(self.current_mesh, 'file_path'):
                input_file = self.current_mesh.file_path
            
            # 确保文件路径是绝对路径
            if input_file:
                input_file = os.path.abspath(input_file)
            
            # 检查当前网格是否有网格数据
            has_mesh_data = False
            if isinstance(self.current_mesh, dict):
                has_mesh_data = 'node_coords' in self.current_mesh and 'cells' in self.current_mesh
            elif hasattr(self.current_mesh, 'node_coords') and hasattr(self.current_mesh, 'cells'):
                has_mesh_data = True
            
            # 只有当没有网格数据且没有有效输入文件时，才显示错误
            if not has_mesh_data and (not input_file or not os.path.exists(input_file)):
                QMessageBox.warning(self, "警告", "无法获取有效的输入网格文件路径")
                self.log_info("无法获取有效的输入网格文件路径")
                self.update_status("无法获取有效的输入网格文件路径")
                return
            
            # 构建配置数据
            config_data = {
                "debug_level": 0,
                "output_file": "./out/mesh.vtk",
                "viz_enabled": False,  # 禁用matplotlib可视化，使用VTK
                "parts": self.parts_params
            }

            # 如果有导入的部件信息，但没有设置参数，则使用默认参数
            if hasattr(self, 'cas_parts_info') and self.cas_parts_info and not self.parts_params:
                # 从导入的部件信息创建默认参数
                for part_name in self.cas_parts_info.keys():
                    config_data["parts"].append({
                        "part_name": part_name,
                        "max_size": 1e6,
                        "PRISM_SWITCH": "off",
                        "first_height": 0.01,
                        "growth_rate": 1.2,
                        "max_layers": 5,
                        "full_layers": 5,
                        "multi_direction": False
                    })
            
            # 总是添加input_file字段，即使为空字符串
            config_data["input_file"] = input_file if input_file else ""
            
            # 创建临时配置文件
            import json
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(config_data, temp_file, indent=2)
                temp_file_path = temp_file.name
            
            try:
                # 构建参数对象
                self.log_info("正在构建网格生成参数...")
                self.update_status("正在构建网格生成参数...")
                
                # 创建参数对象
                from data_structure.parameters import Parameters
                params = Parameters("FROM_CASE_JSON", temp_file_path)
                
                # 创建一个包含PyMeshGen所需属性的临时对象
                class GUITempObject:
                    def __init__(self, gui):
                        self.gui = gui
                        self.ax = None  # 添加ax属性，设置为None，因为我们使用VTK而不是matplotlib
                    
                    def append_info_output(self, message):
                        self.gui.log_info(message)
                
                # 创建临时GUI对象
                temp_gui_obj = GUITempObject(self)
                
                # 调用核心网格生成函数
                self.log_info("正在生成网格...")
                self.update_status("正在生成网格...")
                
                # 直接将当前网格数据传递给generate_mesh函数
                generate_mesh(params, self.current_mesh, temp_gui_obj)
                
                # 更新状态
                self.log_info("网格生成完成!")
                self.update_status("网格生成完成")
                
                # 加载生成的网格文件并显示
                self.log_info("正在加载生成的网格文件...")
                self.update_status("正在加载生成的网格文件...")
                
                # 从输出文件加载生成的网格
                from fileIO.vtk_io import parse_vtk_msh
                generated_mesh = parse_vtk_msh("./out/mesh.vtk")
                
                # 显示生成的网格
                if hasattr(self, 'mesh_display') and generated_mesh:
                    self.current_mesh = generated_mesh
                    self.mesh_display.display_mesh(generated_mesh)
                    self.log_info("已显示生成的网格")
                    self.update_status("已显示生成的网格")
            
            finally:
                # 删除临时文件
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成网格失败: {str(e)}")
            self.log_error(f"生成网格失败: {str(e)}")
            self.update_status("网格生成失败")

    def display_mesh(self):
        """显示网格"""
        self.log_info("显示网格功能暂未实现")

    def check_mesh_quality(self):
        """检查网格质量"""
        self.log_info("检查网格质量功能暂未实现")

    def smooth_mesh(self):
        """平滑网格"""
        self.log_info("平滑网格功能暂未实现")

    def optimize_mesh(self):
        """优化网格"""
        self.log_info("优化网格功能暂未实现")

    def show_mesh_statistics(self):
        """显示网格统计信息"""
        self.log_info("显示网格统计信息功能暂未实现")

    def export_mesh_report(self):
        """导出网格报告"""
        self.log_info("导出网格报告功能暂未实现")

    def show_quick_start(self):
        """显示快速入门"""
        quick_start_text = """PyMeshGen 快速入门\n\n1. 启动程序后，点击"新建配置"创建新项目\n2. 在左侧部件列表中添加几何部件\n3. 配置网格生成参数\n4. 点击"生成网格"按钮生成网格\n5. 使用视图工具查看和操作网格"""
        QMessageBox.about(self, "快速入门", quick_start_text)

    def check_for_updates(self):
        """检查更新"""
        self.log_info("检查更新功能暂未实现")

    def show_shortcuts(self):
        """显示快捷键"""
        shortcuts_text = """常用快捷键：\n\nCtrl+N: 新建工程\nCtrl+O: 打开工程\nCtrl+S: 保存工程\nCtrl+I: 导入网格\nCtrl+E: 导出网格\nF5: 生成网格\nF6: 显示网格\nF11: 全屏显示\nEsc: 退出全屏"""
        QMessageBox.about(self, "快捷键", shortcuts_text)

    def toggle_fullscreen(self):
        """切换全屏显示"""
        if self.isFullScreen():
            self.showNormal()
            self.update_status("退出全屏模式")
        else:
            self.showFullScreen()
            self.update_status("进入全屏模式")

    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(
            self,
            '退出',
            '确定要退出吗?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if hasattr(self, 'mesh_display') and self.mesh_display:
                try:
                    self.mesh_display.cleanup()
                except:
                    pass
            event.accept()
        else:
            event.ignore()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("PyMeshGen")
    app.setApplicationVersion("1.0")

    window = SimplifiedPyMeshGenGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

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
    QDockWidget, QSizePolicy, QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon

# Setup project paths before any imports that might depend on other modules
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
        sys.path.append(dir_path)

# Import modules after setting up paths
from gui.gui_base import (
    StatusBar, InfoOutput, DialogBase,
    Splitter, PartListWidget
)
from gui.model_tree import ModelTreeWidget
from gui.ribbon_widget import RibbonWidget
from gui.toolbar import ViewToolbar
from gui.mesh_display import MeshDisplayArea
from gui.ui_utils import UIStyles, PARTS_INFO_RESERVED_KEYS
from gui.view_controller import ViewController
from gui.mesh_operations import MeshOperations
from gui.part_manager import PartManager
from gui.config_manager import ConfigManager
from gui.geometry_operations import GeometryOperations
from gui.help_module import HelpModule
from gui.ui_helpers import UIHelpers
from data_structure.parameters import Parameters


def _register_qt_metatypes():
    from PyQt5 import QtCore, QtGui
    if hasattr(QtCore, "qRegisterMetaType"):
        QtCore.qRegisterMetaType(QtGui.QTextCursor, "QTextCursor")
    elif hasattr(QtCore, "QMetaType") and hasattr(QtCore.QMetaType, "registerType"):
        QtCore.QMetaType.registerType(QtGui.QTextCursor)

# 在模块导入时立即注册 QTextCursor 元类型，确保所有线程/信号在使用前已注册该类型
try:
    _register_qt_metatypes()
except Exception:
    pass


GLOBAL_MESH_DIMENSION = 2


class PyMeshGenGUI(QMainWindow):
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
        self.setWindowTitle("PyMeshGen V1.0 - 基于Python的网格生成工具")

        # Use only the docs/icon.png file as requested
        icon_path = os.path.join(PROJECT_ROOT, "docs", "icon.png")

        # Try to set the icon from the docs directory
        try:
            from PyQt5.QtGui import QIcon
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                # If the icon file doesn't exist, try alternative path
                alt_icon_path = os.path.join(PROJECT_ROOT, "..", "docs", "icon.png")
                if os.path.exists(alt_icon_path):
                    self.setWindowIcon(QIcon(alt_icon_path))
        except Exception as e:
            # If icon setting fails, log the error but continue
            print(f"Could not set application icon: {e}")

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

        self.view_controller = ViewController(self)
        self.mesh_operations = MeshOperations(self)
        self.part_manager = PartManager(self)
        self.config_manager = ConfigManager(self)
        self.geometry_operations = GeometryOperations(self)
        self.help_module = HelpModule(self)
        self.ui_helpers = UIHelpers(self)

    def _initialize_data(self):
        """初始化数据
        
        初始化GUI类中的所有数据成员，设置它们的初始值
        """
        global GLOBAL_MESH_DIMENSION
        self.params = None                    # 网格生成的参数配置
        self.mesh_generator = None            # 网格生成器实例
        self.current_mesh = None              # 当前生成的网格对象
        self.mesh_dimension = GLOBAL_MESH_DIMENSION  # 当前网格维度
        self.cas_parts_info = None            # CAS文件的部件信息
        self.original_node_coords = None      # 原始节点坐标
        self.parts_params = []                # 部件参数列表
        self.line_connectors = None           # 线网格生成的connectors列表
        self.line_parts = None                # 线网格生成的parts列表
        self.region_data = None               # 区域数据
        self.region_part = None               # 区域Part（包含多个Connector）
        self.direction_actors = []            # 方向箭头actor列表
        self.render_mode = "surface"           # 渲染模式（surface：表面渲染）
        self.show_boundary = True             # 是否显示边界
        self.mesh_generation_thread = None    # 网格生成的线程实例
        self.progress_dialog = None           # 进度对话框实例
        self.import_thread = None             # 导入操作的线程实例
        self._progress_cache = {}             # 进度日志缓存，用于节流输出
        self.geometry_display_source = None   # 几何显示来源（stl/occ）
        self.display_mode = "full"           # 显示模式（full/elements）
        self.geometry_element_actor_cache = {}  # 几何元素actor缓存
        self._delete_geometry_mode_active = False
        self._geometry_delete_elements_cache = {}

    def _create_widgets(self):
        """创建UI组件"""
        self._create_menu()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(2)

        self._create_main_layout(main_layout)

        self.status_bar = StatusBar(self)
        self.status_bar.update_mesh_dimension(self.mesh_dimension)

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

        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.setStyleSheet(UIStyles.SPLITTER_STYLESHEET)

        self._create_model_tree_widget()
        self._create_properties_panel()

        self.left_splitter.setStretchFactor(0, 3)
        self.left_splitter.setStretchFactor(1, 1)
        left_layout.addWidget(self.left_splitter)

        self.left_panel.setMinimumWidth(300)
        self.paned_window.addWidget(self.left_panel)

    def _create_model_tree_widget(self):
        """创建模型树组件"""
        self.model_tree_widget = ModelTreeWidget(parent=self)
        if hasattr(self.model_tree_widget, 'tree'):
            self.model_tree_widget.tree.itemSelectionChanged.connect(self._on_model_tree_selection_changed)

        model_tree_frame_container = QGroupBox("模型树")
        model_tree_layout = QVBoxLayout(model_tree_frame_container)
        model_tree_layout.setSpacing(2)

        model_tree_layout.addWidget(self.model_tree_widget.widget)

        self.model_tree_widget.widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if hasattr(self, 'left_splitter'):
            self.left_splitter.addWidget(model_tree_frame_container)
        else:
            left_layout = self.left_panel.layout()
            left_layout.addWidget(model_tree_frame_container)

    def _create_properties_panel(self):
        """创建属性面板"""
        self.props_frame = QGroupBox("属性面板")
        self.props_frame.setStyleSheet(UIStyles.GROUPBOX_STYLESHEET)

        props_layout = QVBoxLayout()
        props_layout.setSpacing(2)

        self.props_text = QTextEdit()
        self.props_text.setReadOnly(True)
        self.props_text.setStyleSheet(UIStyles.TEXTEDIT_STYLESHEET)
        # Ensure the properties text area expands to fill the available width
        self.props_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        props_layout.addWidget(self.props_text)
        self.props_frame.setLayout(props_layout)

        if hasattr(self, 'left_splitter'):
            self.left_splitter.addWidget(self.props_frame)
        else:
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
        self._create_view_toolbar()

    def _create_view_toolbar(self):
        """创建视图工具栏"""
        # Create the view toolbar
        self.view_toolbar = ViewToolbar(self)
        self.view_toolbar.add_view_toolbar_to_main_window(self)

        # Add the toolbar to the main window - docked to the right of the view area
        self.addToolBar(Qt.RightToolBarArea, self.view_toolbar)

        # Initially show the toolbar
        self.view_toolbar.show()

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

        import_geometry_shortcut = QShortcut(QKeySequence("Ctrl+G"), self)
        import_geometry_shortcut.activated.connect(self.import_geometry)

        export_geometry_shortcut = QShortcut(QKeySequence("Ctrl+Shift+E"), self)
        export_geometry_shortcut.activated.connect(self.export_geometry)

        line_mesh_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        line_mesh_shortcut.activated.connect(self.open_line_mesh_dialog)

    def _setup_ribbon_icons(self):
        """设置功能区图标"""
        from gui.icon_manager import get_icon

        for button_name, button in self.ribbon.buttons.get('file', {}).items():
            icon_name = {
                'new': 'document-new',
                'open': 'document-open',
                'save': 'document-save',
                'import': 'document-import',
                'export': 'document-export',
                'import_geometry': 'document-import',
                'export_geometry': 'document-export',
                'mesh_dimension': 'mesh-dimension'
            }.get(button_name, 'document-new')
            button.setIcon(get_icon(icon_name))

        for button_name, button in self.ribbon.buttons.get('view', {}).items():
            icon_name = {
                'reset': 'view-refresh',
                'fit': 'zoom-fit-best',
                'zoom_in': 'zoom-in',
                'zoom_out': 'zoom-out',
                'view_x_pos': 'view-x-pos',
                'view_x_neg': 'view-x-neg',
                'view_y_pos': 'view-y-pos',
                'view_y_neg': 'view-y-neg',
                'view_z_pos': 'view-z-pos',
                'view_z_neg': 'view-z-neg',
                'view_iso': 'view-iso',
                'surface': 'surface',
                'wireframe': 'wireframe',
                'surface-wireframe': 'surface-wireframe',
                'background': 'configure'
            }.get(button_name, 'view-refresh')
            button.setIcon(get_icon(icon_name))

            # Connect view buttons to their respective functions
            if button_name == 'surface':
                button.clicked.connect(lambda: self.view_controller.set_render_mode("surface"))
            elif button_name == 'wireframe':
                button.clicked.connect(lambda: self.view_controller.set_render_mode("wireframe"))
            elif button_name == 'surface-wireframe':
                button.clicked.connect(lambda: self.view_controller.set_render_mode("surface-wireframe"))

        for button_name, button in self.ribbon.buttons.get('config', {}).items():
            icon_name = {
                'params': 'configure',
                'mesh_params': 'configure',
                'boundary': 'configure',
                'import_config': 'document-properties',
                'export_config': 'document-properties',
                'reset': 'edit-clear'
            }.get(button_name, 'configure')
            button.setIcon(get_icon(icon_name))

        for button_name, button in self.ribbon.buttons.get('mesh', {}).items():
            # Use specific optimized icons for mesh operations
            icon_name = {
                'generate': 'mesh-generate',
                'display': 'view-fullscreen',
                'clear': 'edit-delete',
                'quality': 'mesh-quality',
                'smooth': 'mesh-smooth',
                'optimize': 'mesh-optimize',
                'statistics': 'statistics',
                'report': 'report'
            }.get(button_name, 'system-run')
            button.setIcon(get_icon(icon_name))

        for button_name, button in self.ribbon.buttons.get('geometry', {}).items():
            icon_name = {
                'import': 'document-import',
                'extract_boundary': 'extract-boundary',
                'create_geometry': 'geom-create',
                'delete_geometry': 'edit-delete',
                'line_mesh': 'mesh-generate',
                'line_mesh_params': 'configure'
            }.get(button_name, 'document-import')
            button.setIcon(get_icon(icon_name))

        for button_name, button in self.ribbon.buttons.get('help', {}).items():
            icon_name = {
                'manual': 'help-contents',
                'quick_start': 'help-faq',
                'shortcuts': 'help-keyboard-shortcuts',
                'updates': 'help-about',
                'about': 'help-about'
            }.get(button_name, 'help-contents')
            button.setIcon(get_icon(icon_name))

    def _get_standard_icon(self, icon_name):
        """获取标准图标 - 现在使用优化的图标管理器"""
        from gui.icon_manager import get_icon
        return get_icon(icon_name)

    def _set_ribbon_button_enabled(self, tab_key, button_key, enabled):
        """设置Ribbon按钮状态（启用/禁用）"""
        ribbon = getattr(self, 'ribbon', None)
        buttons = getattr(ribbon, 'buttons', None) if ribbon else None
        if not buttons:
            return
        button = buttons.get(tab_key, {}).get(button_key)
        if button:
            button.setEnabled(enabled)

    def _reset_progress_cache(self, context):
        """重置进度日志缓存，确保新任务从头记录"""
        if hasattr(self, '_progress_cache'):
            self._progress_cache.pop(context, None)

    def _update_progress(self, message, progress, context):
        """统一更新进度显示，并对日志输出进行节流"""
        self.status_bar.show_progress(message, progress)

        last_entry = self._progress_cache.get(context)
        last_progress = last_entry["progress"] if last_entry else None
        last_message = last_entry["message"] if last_entry else None

        log_needed = (
            last_entry is None
            or message != last_message
            or progress in (0, 100)
            or (last_progress is not None and abs(progress - last_progress) >= 5)
        )
        if log_needed:
            self.log_info(f"{message} ({progress}%)")
            self._progress_cache[context] = {"message": message, "progress": progress}

    def toggle_ribbon(self):
        """切换功能区显示"""
        if hasattr(self, 'ribbon') and self.ribbon:
            self.ribbon.toggle_content_visibility()

    def on_mesh_display_key(self, event):
        """处理网格显示区域的键盘事件"""
        key = event.key()
        if key == Qt.Key_Escape:
            if hasattr(self, 'view_controller'):
                if self.view_controller.is_point_pick_active():
                    self.view_controller._on_point_pick_exit(None, None)
                    return
                self.view_controller.set_picking_mode(False)
            if self._delete_geometry_mode_active:
                self._stop_delete_geometry_mode()
            return
        if key == Qt.Key_Delete:
            if self._delete_geometry_mode_active:
                self._delete_geometry_from_pick()
                return
            if self._delete_selected_geometry_from_tree():
                return
        key_actions = {
            Qt.Key_R: lambda: self.view_controller.reset_view() or self.update_status("已重置视图 (R键)"),
            Qt.Key_F: lambda: self.view_controller.fit_view() or self.update_status("已适应视图 (F键)"),
            Qt.Key_O: self.view_controller.toggle_boundary_display,
            Qt.Key_1: lambda: self.view_controller.set_render_mode("surface"),
            Qt.Key_2: lambda: self.view_controller.set_render_mode("wireframe"),
            Qt.Key_3: lambda: self.view_controller.set_render_mode("surface-wireframe"),
        }

        action = key_actions.get(key)
        if action:
            action()

    def new_config(self):
        """新建工程 - 清空当前所有数据，包括网格、配置参数等"""
        from PyQt5.QtWidgets import QMessageBox

        # 询问用户确认
        reply = QMessageBox.question(
            self,
            "确认新建工程",
            "确定要新建工程吗？当前所有数据（网格、配置参数等）将被清空。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # 清空当前网格数据
                if hasattr(self, 'current_mesh'):
                    self.current_mesh = None

                # 清空参数配置
                if hasattr(self, 'params'):
                    self.params = None

                # 清空部件参数
                if hasattr(self, 'parts_params'):
                    self.parts_params = []

                # 清空部件列表显示
                if hasattr(self, 'part_list_widget'):
                    self.part_list_widget.clear()

                # 清空模型树
                if hasattr(self, 'model_tree_widget'):
                    self.model_tree_widget.clear()

                # 清空网格显示
                if hasattr(self, 'mesh_visualizer'):
                    self.mesh_visualizer.clear_mesh()

                # 清空信息输出窗口
                if hasattr(self, 'info_output') and hasattr(self.info_output, 'info_text'):
                    self.info_output.info_text.clear()

                # 清空属性面板
                if hasattr(self, 'props_text'):
                    self.props_text.clear()

                # 清空状态栏
                self.statusBar().clearMessage()

                # 重置相关属性
                if hasattr(self, 'cas_parts_info'):
                    self.cas_parts_info = None

                if hasattr(self, 'original_node_coords'):
                    self.original_node_coords = None

                if hasattr(self, 'mesh_generator'):
                    self.mesh_generator = None

                if hasattr(self, 'json_config'):
                    self.json_config = {}

                if hasattr(self, 'render_mode'):
                    self.render_mode = "surface"

                if hasattr(self, 'show_boundary'):
                    self.show_boundary = True

                if hasattr(self, 'mesh_generation_thread'):
                    self.mesh_generation_thread = None

                if hasattr(self, 'progress_dialog'):
                    self.progress_dialog = None

                if hasattr(self, 'import_thread'):
                    self.import_thread = None

                if hasattr(self, 'geometry_display_source'):
                    self.geometry_display_source = None

                if hasattr(self, 'geometry_actor'):
                    if self.geometry_actor and hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                        try:
                            self.mesh_display.renderer.RemoveActor(self.geometry_actor)
                        except:
                            pass
                    self.geometry_actor = None

                if hasattr(self, 'geometry_edges_actor'):
                    if self.geometry_edges_actor and hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                        try:
                            self.mesh_display.renderer.RemoveActor(self.geometry_edges_actor)
                        except:
                            pass
                    self.geometry_edges_actor = None

                if hasattr(self, 'geometry_actors'):
                    for elem_type, actors in self.geometry_actors.items():
                        for actor in actors:
                            if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                                try:
                                    self.mesh_display.renderer.RemoveActor(actor)
                                except:
                                    pass
                    self.geometry_actors = {}

                if hasattr(self, 'current_geometry'):
                    self.current_geometry = None

                # 清空区域相关数据
                if hasattr(self, 'region_data'):
                    self.region_data = None
                if hasattr(self, 'region_part'):
                    self.region_part = None
                if hasattr(self, 'direction_actors'):
                    for actor in self.direction_actors:
                        if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                            try:
                                self.mesh_display.renderer.RemoveActor(actor)
                            except:
                                pass
                    self.direction_actors = []

                # 清空网格显示区域
                if hasattr(self, 'mesh_display'):
                    self.mesh_display.clear()
                    self.mesh_display.clear_mesh_actors()
                    self.mesh_display.clear_line_mesh_actors()  # 清除线网格
                    self.mesh_display.clear_boundary_actors()
                    self.mesh_display.clear_highlights()
                    self.mesh_display.mesh_data = None
                    self.mesh_display.mesh_actor = None
                    # 彻底清除渲染器中的所有 actors
                    if hasattr(self.mesh_display, 'renderer'):
                        try:
                            # 获取渲染器中的所有 actors
                            renderer = self.mesh_display.renderer
                            actors = renderer.GetActors()
                            actors.InitTraversal()
                            actor = actors.GetNextItem()
                            while actor:
                                renderer.RemoveActor(actor)
                                actor = actors.GetNextItem()
                        except:
                            pass
                    # 重新渲染窗口以确保所有元素都被清除
                    if hasattr(self.mesh_display, 'render_window'):
                        self.mesh_display.render_window.Render()

                # 清空模型树数据
                if hasattr(self, 'model_tree_widget'):
                    self.model_tree_widget.geometry_data = None
                    self.model_tree_widget.mesh_data = None
                    self.model_tree_widget.parts_data = None

                # 如果有画布，也清空
                if hasattr(self, 'canvas'):
                    self.canvas.figure.clear()
                    self.canvas.draw()

                # 记录操作
                self.log_info("已新建工程，所有数据已清空")
                self.update_status("新工程创建完成")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"新建工程失败：{str(e)}")
                self.log_info(f"新建工程失败：{str(e)}")
                self.update_status("新建工程失败")
        else:
            self.log_info("新建工程操作已取消")
            self.update_status("操作已取消")

    def open_config(self):
        """打开工程 - 读入相关参数和网格数据，并在视图区显示读入的网格"""
        try:
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            import json
            import os
            import pickle

            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "打开工程",
                os.path.join(self.project_root, "projects"),
                "PyMeshGen工程文件 (*.pymg)"
            )

            if not file_path:
                self.log_info("打开工程操作已取消")
                self.update_status("打开工程已取消")
                return

            # 读取项目文件
            with open(file_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            # 从项目文件中读取配置信息
            if "config" in project_data:
                config_info = project_data["config"]

                # 重建参数对象
                import tempfile

                # 创建临时配置文件用于加载参数
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
                    # 重构完整的配置数据
                    full_config = {
                        "debug_level": config_info.get("debug_level", 0),
                        "input_file": config_info.get("input_file", ""),
                        "output_file": config_info.get("output_file", ""),
                        "mesh_type": config_info.get("mesh_type", 1),
                        "viz_enabled": config_info.get("viz_enabled", False),
                        "parts": config_info.get("parts", [])
                    }
                    json.dump(full_config, temp_config, indent=2)
                    temp_config_path = temp_config.name

                # 加载参数
                self.params = Parameters("FROM_CASE_JSON", temp_config_path)

                # 清理临时文件
                os.unlink(temp_config_path)

                self.log_info("项目配置参数加载完成")
            else:
                self.log_info("项目文件中未找到配置信息")

            # 读取网格文件路径（如果存在）
            mesh_file_path = project_data.get("mesh_file_path", "")
            if mesh_file_path and os.path.exists(mesh_file_path):
                self.log_info(f"发现原始网格文件: {mesh_file_path}")
                # 可以根据需要导入原始网格文件
                # 但通常我们更关心生成的网格

            # 读取生成的网格数据（如果存在）
            generated_mesh_file = project_data.get("generated_mesh_file", "")
            if generated_mesh_file and os.path.exists(generated_mesh_file):
                try:
                    # 尝试加载生成的网格数据
                    if generated_mesh_file.endswith('.pymesh'):
                        # 使用pickle加载
                        with open(generated_mesh_file, 'rb') as f:
                            self.current_mesh = pickle.load(f)
                    elif generated_mesh_file.endswith('.vtk'):
                        # 如果是VTK文件，使用相应的加载方法
                        from fileIO.vtk_io import parse_vtk_msh
                        self.current_mesh = parse_vtk_msh(generated_mesh_file)
                    else:
                        # 尝试作为pickle文件加载
                        with open(generated_mesh_file, 'rb') as f:
                            self.current_mesh = pickle.load(f)

                    self.log_info(f"生成的网格数据已加载: {generated_mesh_file}")

                    # 在视图区显示网格
                    if hasattr(self, 'mesh_visualizer') and self.mesh_visualizer:
                        self.mesh_visualizer.update_mesh(self.current_mesh)
                        self.log_info("网格已在视图区显示")
                    elif hasattr(self, 'mesh_display'):
                        self.mesh_display.display_mesh(self.current_mesh, render_immediately=False)
                        self.log_info("网格已在视图区显示")

                        # Refresh display to show all parts with different colors
                        self.part_manager.refresh_display_all_parts()
                    else:
                        self.log_info("未找到网格显示组件")

                    # 刷新显示
                    if hasattr(self, 'canvas'):
                        self.canvas.draw()

                except Exception as e:
                    self.log_info(f"加载生成的网格数据失败: {str(e)}")
            else:
                if generated_mesh_file:
                    self.log_info(f"生成的网格文件不存在: {generated_mesh_file}")
                else:
                    self.log_info("项目文件中未找到生成的网格文件路径")

            # Also update parts_params if it exists
            if hasattr(self, 'params') and hasattr(self, 'parts_params'):
                if "config" in project_data and "parts" in project_data["config"]:
                    self.parts_params = []
                    for part in project_data["config"]["parts"]:
                        # Create part parameter dict in the expected format
                        part_param = {
                            "part_name": part["part_name"],
                            "max_size": part["max_size"],
                            "PRISM_SWITCH": part["PRISM_SWITCH"],
                            "first_height": part.get("first_height", 0.1),
                            "growth_rate": part.get("growth_rate", 1.2),
                            "growth_method": part.get("growth_method", "geometric"),
                            "max_layers": part.get("max_layers", 3),
                            "full_layers": part.get("full_layers", 0),
                            "multi_direction": part.get("multi_direction", False)
                        }
                        self.parts_params.append(part_param)

            # 更新状态
            self.update_status("工程打开完成")
            QMessageBox.information(self, "成功", f"工程已成功打开:\n{file_path}")

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"打开工程失败：{str(e)}")
            self.log_info(f"打开工程失败：{str(e)}")
            self.update_status("工程打开失败")

    def on_import_progress(self, message, progress):
        """导入进度更新回调"""
        self._update_progress(message, progress, "mesh")

    def _apply_mesh_dimension(self, dimension, update_mesh=True):
        """同步网格维度到全局变量、当前网格和状态栏"""
        if dimension not in (2, 3):
            return
        dimension = int(dimension)
        global GLOBAL_MESH_DIMENSION
        GLOBAL_MESH_DIMENSION = dimension
        self.mesh_dimension = dimension
        if update_mesh and hasattr(self, 'current_mesh') and self.current_mesh:
            if isinstance(self.current_mesh, dict):
                self.current_mesh['dimension'] = dimension
            else:
                if hasattr(self.current_mesh, 'dimension'):
                    self.current_mesh.dimension = dimension
        if hasattr(self, 'status_bar'):
            self.status_bar.update_mesh_dimension(dimension)

    def on_import_finished(self, mesh_data):
        """导入完成回调 - 优化版本，使用QTimer分步处理避免UI卡顿"""
        try:
            self.current_mesh = mesh_data
            mesh_dimension = None
            if isinstance(mesh_data, dict):
                mesh_dimension = mesh_data.get('dimension')
            elif hasattr(mesh_data, 'dimension'):
                mesh_dimension = mesh_data.dimension
            if mesh_dimension in (2, 3):
                self._apply_mesh_dimension(mesh_dimension)
            else:
                try:
                    from utils.geom_toolkit import detect_mesh_dimension_by_metadata
                    resolved_dim = detect_mesh_dimension_by_metadata(mesh_data, default_dim=self.mesh_dimension)
                    self._apply_mesh_dimension(resolved_dim)
                except Exception:
                    pass

            if hasattr(self, 'mesh_display'):
                self.update_status("正在显示网格...")
                # 使用延迟渲染，避免多次渲染
                self.mesh_display.display_mesh(mesh_data, render_immediately=False)

            # 保存原始节点坐标用于后续的节点映射
            if hasattr(mesh_data, 'node_coords'):
                self.update_status("正在保存节点坐标...")
                self.original_node_coords = [list(coord) for coord in mesh_data.node_coords]

            # 使用几何模型树加载网格数据 - 分步处理
            if hasattr(self, 'model_tree_widget'):
                self.update_status("正在加载模型树...")
                from PyQt5.QtCore import QTimer

                def _split_mesh_counts(mesh_obj):
                    surface_count = 0
                    volume_count = 0
                    line_count = 0
                    if hasattr(mesh_obj, 'cell_container'):
                        for cell in mesh_obj.cell_container:
                            if cell is None:
                                continue
                            cell_name = cell.__class__.__name__
                            if cell_name in ('Triangle', 'Quadrilateral'):
                                surface_count += 1
                            elif cell_name in ('Tetrahedron', 'Pyramid', 'Prism', 'Hexahedron'):
                                volume_count += 1
                            elif isinstance(cell, (list, tuple)):
                                node_count = len(cell)
                                if node_count in (3, 4) and getattr(mesh_obj, 'dimension', 2) == 2:
                                    surface_count += 1
                                elif node_count in (4, 5, 6, 8) and getattr(mesh_obj, 'dimension', 3) == 3:
                                    volume_count += 1
                                elif node_count == 2:
                                    line_count += 1
                    elif hasattr(mesh_obj, 'cells'):
                        surface_count = len(mesh_obj.cells)

                    if hasattr(mesh_obj, 'volume_cells') and mesh_obj.volume_cells:
                        volume_count = len(mesh_obj.volume_cells)
                        if surface_count == 0 and hasattr(mesh_obj, 'cells'):
                            surface_count = len(mesh_obj.cells)
                    if hasattr(mesh_obj, 'boundary_info') and mesh_obj.boundary_info:
                        boundary_lines = 0
                        for part_data in mesh_obj.boundary_info.values():
                            for face in part_data.get('faces', []):
                                nodes = face.get('nodes', [])
                                if len(nodes) == 2:
                                    boundary_lines += 1
                        if boundary_lines:
                            line_count = boundary_lines
                    if hasattr(mesh_obj, 'line_cells') and mesh_obj.line_cells:
                        if line_count == 0:
                            line_count = len(mesh_obj.line_cells)

                    return surface_count, volume_count, line_count
                
                # 创建分步处理函数 - 减少延迟时间
                def step1_load_mesh():
                    self.model_tree_widget.load_mesh(mesh_data, mesh_name="网格")
                    QTimer.singleShot(0, step2_load_parts)
                
                def step2_load_parts():
                    self.update_status("正在加载部件信息...")
                    self.model_tree_widget.load_parts(mesh_data)
                    # 如果网格文件本身包含部件信息，保留这些信息
                    if hasattr(mesh_data, 'parts_info') and mesh_data.parts_info:
                        self._merge_parts_info(mesh_data.parts_info)
                        self.log_info(f"已保留网格文件中的部件信息，共 {len(mesh_data.parts_info)} 个部件")
                    # 如果没有预设部件，自动创建Default部件
                    elif not hasattr(self, 'cas_parts_info') or not self.cas_parts_info:
                        self._create_default_part_for_mesh(mesh_data)
                    if hasattr(self, 'model_tree_widget') and hasattr(self, 'cas_parts_info') and self.cas_parts_info:
                        self.model_tree_widget.load_parts({'parts_info': self.cas_parts_info})
                    QTimer.singleShot(0, step3_refresh_display)
                
                def step3_refresh_display():
                    self.update_status("正在刷新显示...")
                    self.part_manager.refresh_display_all_parts()
                    QTimer.singleShot(0, step4_complete)
                
                def step4_complete():
                    self.log_info(f"已导入网格: {mesh_data.file_path}")
                    surface_count, volume_count, line_count = _split_mesh_counts(mesh_data)
                    if volume_count:
                        if line_count:
                            self.log_info(f"节点数: {len(mesh_data.node_coords)}, 线单元数: {line_count}, 面单元数: {surface_count}, 体单元数: {volume_count}")
                        else:
                            self.log_info(f"节点数: {len(mesh_data.node_coords)}, 面单元数: {surface_count}, 体单元数: {volume_count}")
                    else:
                        if line_count:
                            self.log_info(f"节点数: {len(mesh_data.node_coords)}, 线单元数: {line_count}, 面单元数: {surface_count}")
                        else:
                            self.log_info(f"节点数: {len(mesh_data.node_coords)}, 单元数: {surface_count}")
                    self.update_status("已导入网格")
                    self.status_bar.hide_progress()

                # 启动分步处理 - 使用0延迟
                QTimer.singleShot(0, step1_load_mesh)
            else:
                # Refresh display to show all parts with different colors
                self.update_status("正在刷新显示...")
                self.part_manager.refresh_display_all_parts()
                
                self.log_info(f"已导入网格: {mesh_data.file_path}")
                surface_count, volume_count, line_count = _split_mesh_counts(mesh_data)
                if volume_count:
                    if line_count:
                        self.log_info(f"节点数: {len(mesh_data.node_coords)}, 线单元数: {line_count}, 面单元数: {surface_count}, 体单元数: {volume_count}")
                    else:
                        self.log_info(f"节点数: {len(mesh_data.node_coords)}, 面单元数: {surface_count}, 体单元数: {volume_count}")
                else:
                    if line_count:
                        self.log_info(f"节点数: {len(mesh_data.node_coords)}, 线单元数: {line_count}, 面单元数: {surface_count}")
                    else:
                        self.log_info(f"节点数: {len(mesh_data.node_coords)}, 单元数: {surface_count}")
                self.update_status("已导入网格")
                self.status_bar.hide_progress()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理导入的网格失败: {str(e)}")
            self.log_error(f"处理导入的网格失败: {str(e)}")
            self.status_bar.hide_progress()

        finally:
            # 重新启用导入按钮
            self._set_ribbon_button_enabled('file', 'import', True)

    def on_import_failed(self, error_message):
        """导入失败回调"""
        QMessageBox.critical(self, "错误", error_message)
        self.log_error(error_message)
        self.update_status("导入失败")
        self.status_bar.hide_progress()

        # 重新启用导入按钮
        self._set_ribbon_button_enabled('file', 'import', True)

    def import_geometry(self):
        """导入几何文件（使用异步线程，避免GUI卡顿）"""
        self.geometry_operations.import_geometry()

    def open_geometry_create_dialog(self):
        """打开几何创建对话框"""
        from gui.geometry_create_dialog import GeometryCreateDialog
        existing_dialog = getattr(self, "_geometry_create_dialog", None)
        if existing_dialog and existing_dialog.isVisible():
            existing_dialog.raise_()
            existing_dialog.activateWindow()
            return
        dialog = GeometryCreateDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(lambda: setattr(self, "_geometry_create_dialog", None))
        self._geometry_create_dialog = dialog
        dialog.show()

    def open_geometry_delete_dialog(self):
        """进入几何删除拾取模式"""
        self._start_delete_geometry_mode()

    def open_line_mesh_dialog(self):
        """打开线网格生成对话框"""
        from gui.line_mesh_dialog import LineMeshGenerationDialog
        existing_dialog = getattr(self, "_line_mesh_dialog", None)
        if existing_dialog and existing_dialog.isVisible():
            existing_dialog.raise_()
            existing_dialog.activateWindow()
            return
        dialog = LineMeshGenerationDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(lambda: setattr(self, "_line_mesh_dialog", None))
        dialog.generation_requested.connect(self._on_line_mesh_generation_requested)
        self._line_mesh_dialog = dialog
        dialog.show()

    def open_create_region_dialog(self):
        """打开创建区域对话框"""
        from gui.region_creation_dialog import RegionCreationDialog
        existing_dialog = getattr(self, "_region_creation_dialog", None)
        if existing_dialog and existing_dialog.isVisible():
            existing_dialog.raise_()
            existing_dialog.activateWindow()
            return
        dialog = RegionCreationDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(lambda: setattr(self, "_region_creation_dialog", None))
        dialog.region_created.connect(self._on_region_created)
        self._region_creation_dialog = dialog
        dialog.show()

    def _on_region_created(self, region_data):
        """处理区域创建完成"""
        self.log_info(f"区域创建成功！包含 {region_data['total_connectors']} 个Connector，共 {region_data['total_fronts']} 个Front")
        self.update_status("区域创建成功")
        
        # 将区域数据保存到GUI实例中，供网格生成使用
        self.region_data = region_data
        
        # 直接传递多个Connector，不合并
        # 创建一个新的Part来包含这些Connector
        from data_structure.basic_elements import Part
        from data_structure.parameters import MeshParameters
        
        # 创建一个新的Part，包含所有选中的Connector
        region_part_params = MeshParameters(
            part_name="region",
            max_size=0.1,
            PRISM_SWITCH="off"
        )
        region_part = Part("region", region_part_params, region_data['connectors'])
        region_part.init_part_front_list()
        
        # 保存到GUI实例
        self.region_part = region_part
        
        self.log_info(f"已创建区域Part，包含 {len(region_data['connectors'])} 个Connector")
        self.log_info(f"区域Part可用于网格生成")

    def _on_line_mesh_generation_requested(self, params):
        """处理线网格生成请求"""
        from gui.line_mesh_generation import generate_line_mesh, convert_connectors_to_unstructured_grid, LineMeshParams

        if not params.get('edges'):
            self.log_warning("未选择几何线，请先拾取几何线")
            return

        self.log_info(f"开始处理 {len(params['edges'])} 条几何线的网格生成请求")

        edges_info = []
        for idx, edge in enumerate(params['edges']):
            try:
                edge_obj = edge.get('obj')
                if edge_obj is None:
                    self.log_warning(f"第 {idx+1} 条边的对象为空，已跳过")
                    continue
                
                self.log_info(f"第 {idx+1} 条边的对象类型: {type(edge_obj).__name__}")
                
                from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Shape
                from OCC.Core.TopExp import TopExp_Explorer
                from OCC.Core.TopAbs import TopAbs_VERTEX
                from OCC.Core.BRep import BRep_Tool
                from OCC.Core.BRepGProp import brepgprop
                from OCC.Core.GProp import GProp_GProps
                from OCC.Core.TopoDS import TopoDS_Vertex

                shape = None
                
                if isinstance(edge_obj, TopoDS_Edge):
                    self.log_info(f"第 {idx+1} 条边是 TopoDS_Edge 类型，直接使用")
                    shape = edge_obj
                elif hasattr(edge_obj, 'Shape'):
                    shape = edge_obj.Shape()
                    if shape.IsNull():
                        self.log_warning(f"第 {idx+1} 条边的形状为空，已跳过")
                        continue
                elif hasattr(edge_obj, 'geometry') or hasattr(edge_obj, 'geom'):
                    geom = getattr(edge_obj, 'geometry', None) or getattr(edge_obj, 'geom', None)
                    self.log_info(f"第 {idx+1} 条边有 geometry/geom 属性: {type(geom).__name__ if geom else 'None'}")
                    if geom and hasattr(geom, 'Shape'):
                        shape = geom.Shape()
                        if shape.IsNull():
                            self.log_warning(f"第 {idx+1} 条边的geometry形状为空，已跳过")
                            continue
                    else:
                        self.log_warning(f"第 {idx+1} 条边没有Shape或geometry属性，无法处理")
                        continue
                else:
                    self.log_warning(f"第 {idx+1} 条边类型 {type(edge_obj).__name__} 不支持，无法处理")
                    continue

                if shape is None or shape.IsNull():
                    self.log_warning(f"第 {idx+1} 条边的形状为空，已跳过")
                    continue

                v_exp = TopExp_Explorer(shape, TopAbs_VERTEX)
                vertices = []
                while v_exp.More():
                    v = TopoDS_Vertex(v_exp.Current())
                    pnt = BRep_Tool.Pnt(v)
                    vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
                    v_exp.Next()

                if len(vertices) < 2:
                    self.log_warning(f"第 {idx+1} 条边的顶点数量不足（{len(vertices)}个），已跳过")
                    continue

                gprop = GProp_GProps()
                brepgprop.LinearProperties(shape, gprop)

                # BRep_Tool.Curve 返回一个元组 (curve, first_param, last_param)
                curve_result = BRep_Tool.Curve(shape)
                if curve_result and len(curve_result) == 3:
                    curve, first_param, last_param = curve_result
                    print(f"提取曲线: 类型={type(curve)}, 参数范围=[{first_param}, {last_param}]")
                else:
                    curve = None
                    print(f"无法提取曲线: curve_result={curve_result}")

                edges_info.append({
                    'start_point': vertices[0],
                    'end_point': vertices[-1],
                    'curve': curve,
                    'length': gprop.Mass(),
                    'name': edge.get('name', f"edge_{len(edges_info)}"),
                    'obj': edge_obj
                })
                self.log_info(f"成功处理第 {idx+1} 条边: 长度={gprop.Mass():.4f}, 顶点数={len(vertices)}, 起点={vertices[0]}, 终点={vertices[-1]}")
            except Exception as e:
                import traceback
                error_msg = f"处理第 {idx+1} 条边时出错: {str(e)}"
                self.log_error(error_msg)
                traceback.print_exc()
                continue

        if not edges_info:
            self.log_warning("没有有效的几何线可用于生成网格，请检查选择的几何线")
            return

        self.log_info(f"共有 {len(edges_info)} 条有效几何线可用于生成网格")

        line_mesh_params = LineMeshParams(
            method=params.get('method', 'uniform'),
            num_elements=params.get('num_elements', 10),
            start_size=params.get('start_size', 0.1),
            end_size=params.get('end_size', 0.2),
            growth_rate=params.get('growth_rate', 1.2),
            tanh_factor=params.get('tanh_factor', 2.0),
            bc_type=params.get('bc_type', 'wall'),
            part_name=params.get('part_name', 'default_line')
        )

        try:
            connectors, parts = generate_line_mesh(edges_info, line_mesh_params)
            self.log_info(f"线网格生成完成: connectors={len(connectors) if connectors else 0}, parts={len(parts) if parts else 0}")

            # 保存connectors和parts，供后续网格生成使用
            self.line_connectors = connectors
            self.line_parts = parts
            self.log_info(f"已保存线网格数据: {len(connectors)} connectors, {len(parts)} parts")

            if connectors:
                # 转换为Unstructured_Grid
                unstr_grid = convert_connectors_to_unstructured_grid(connectors, grid_dimension=2)
                self.log_info(f"Unstructured_Grid创建成功: 节点数={unstr_grid.num_nodes}, 单元数={unstr_grid.num_cells}")

                # 更新网格显示
                self._update_mesh_display_with_unstructured_grid(unstr_grid)
                self.update_status(f"成功生成 {len(connectors)} 个线网格Connector")
            else:
                self.log_warning("未生成任何Connector，请检查参数设置")
        except Exception as e:
            import traceback
            error_msg = f"生成线网格时出错: {str(e)}"
            self.log_error(error_msg)
            traceback.print_exc()

    def _update_mesh_display_with_unstructured_grid(self, unstr_grid):
        """使用Unstructured_Grid更新网格显示（保留之前生成的线网格）"""
        if not hasattr(self, 'mesh_display') or not self.mesh_display:
            self.log_warning("网格显示对象不存在，无法显示生成的线网格")
            return

        try:
            # 添加线网格演员（保留之前生成的线网格）
            success = self.mesh_display.add_line_mesh_actor(unstr_grid, render_immediately=True)
            
            if success:
                self.log_info(f"线网格显示成功: 节点数={unstr_grid.num_nodes}, 单元数={unstr_grid.num_cells}")
                
                # 更新模型树
                if hasattr(self, 'model_tree_widget'):
                    self.model_tree_widget.load_mesh(unstr_grid, mesh_name="网格")
                    
                    # 合并部件信息（使用现成的合并方法，保留已有几何元素）
                    if hasattr(unstr_grid, 'parts_info') and unstr_grid.parts_info:
                        self._merge_parts_info(unstr_grid.parts_info)
                        self.log_info(f"已合并线网格部件信息，共 {len(unstr_grid.parts_info)} 个部件")
                    
                    # 更新模型树显示
                    if hasattr(self, 'cas_parts_info') and self.cas_parts_info:
                        self.model_tree_widget.load_parts({'parts_info': self.cas_parts_info})
            else:
                self.log_warning("线网格显示失败")
        except Exception as e:
            import traceback
            error_msg = f"显示线网格时出错: {str(e)}"
            self.log_error(error_msg)
            traceback.print_exc()

    def on_geometry_import_progress(self, message, progress):
        """几何导入进度更新回调"""
        self._update_progress(message, progress, "geometry")

    def on_geometry_import_finished(self, result):
        """几何导入完成回调"""
        try:
            from fileIO.occ_to_vtk import create_shape_actor

            shape = result['shape']
            stats = result['stats']
            file_path = result['file_path']
            unit = result.get('unit', 'mm')
            unit_source = result.get('unit_source', 'manual')
            auto_detected = result.get('auto_detected', True)
            file_ext = os.path.splitext(file_path)[1].lower()

            self.current_geometry_stats = stats

            self.log_info(f"几何统计信息:")
            self.log_info(f"  - 顶点数: {stats['num_vertices']}")
            self.log_info(f"  - 边数: {stats['num_edges']}")
            self.log_info(f"  - 面数: {stats['num_faces']}")
            self.log_info(f"  - 实体数: {stats['num_solids']}")

            bbox_min, bbox_max = stats['bounding_box']
            self.log_info(f"  - 边界框: ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}) 到 ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f})")

            # 如果之前有几何，清理其actors缓存
            if hasattr(self, 'part_manager') and hasattr(self.part_manager, 'cleanup_geometry_actors'):
                self.part_manager.cleanup_geometry_actors()

            self.current_geometry = shape
            self.geometry_actors = {}
            self.geometry_actor = None
            self.geometry_edges_actor = None
            self.geometry_points_actor = None

            # 初始化几何actors缓存
            if hasattr(self, 'part_manager'):
                if not hasattr(self, 'geometry_actors_cache'):
                    self.geometry_actors_cache = {}

            # 对于STL文件，若单位为mm则直接VTK显示，否则使用OCC显示
            if file_ext == ".stl" and unit == "mm":
                self.update_status("正在创建几何显示...")

                if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                    from fileIO.occ_to_vtk import create_geometry_edges_actor
                    stl_displayed = False

                    try:
                        import vtk

                        reader = vtk.vtkSTLReader()
                        reader.SetFileName(file_path)
                        reader.Update()

                        polydata = reader.GetOutput()
                        if polydata and polydata.GetNumberOfPoints() > 0:
                            mapper = vtk.vtkPolyDataMapper()
                            mapper.SetInputData(polydata)

                            self.geometry_actor = vtk.vtkActor()
                            self.geometry_actor.SetMapper(mapper)
                            self.geometry_actor.GetProperty().SetColor(0.8, 0.8, 0.9)
                            self.geometry_actor.GetProperty().SetOpacity(0.8)

                            edges_filter = vtk.vtkExtractEdges()
                            edges_filter.SetInputData(polydata)
                            edges_filter.Update()

                            edges_mapper = vtk.vtkPolyDataMapper()
                            edges_mapper.SetInputData(edges_filter.GetOutput())

                            self.geometry_edges_actor = vtk.vtkActor()
                            self.geometry_edges_actor.SetMapper(edges_mapper)
                            self.geometry_edges_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
                            self.geometry_edges_actor.GetProperty().SetLineWidth(1.5)

                            self.mesh_display.renderer.AddActor(self.geometry_actor)
                            self.mesh_display.renderer.AddActor(self.geometry_edges_actor)
                            self.geometry_actors['main'] = [self.geometry_actor]
                            self.geometry_actors['edges'] = [self.geometry_edges_actor]

                            stl_displayed = True
                    except Exception as e:
                        self.log_warning(f"STL显示失败，回退到OCC显示: {str(e)}")

                    if not stl_displayed:
                        has_surface = (stats.get('num_faces', 0) + stats.get('num_solids', 0)) > 0
                        has_edges = stats.get('num_edges', 0) > 0
                        has_vertices = stats.get('num_vertices', 0) > 0

                        if has_surface:
                            self.geometry_actor = create_shape_actor(
                                shape,
                                mesh_quality=2.0,
                                display_mode='surface',
                                color=(0.8, 0.8, 0.9),
                                opacity=0.8
                            )
                            self.mesh_display.renderer.AddActor(self.geometry_actor)
                            self.geometry_actors['main'] = [self.geometry_actor]

                        if has_edges:
                            self.geometry_edges_actor = create_geometry_edges_actor(
                                shape,
                                color=(0.0, 0.0, 0.0),
                                line_width=1.5,
                                sample_rate=0.01,
                                max_points_per_edge=500
                            )
                            self.mesh_display.renderer.AddActor(self.geometry_edges_actor)
                            self.geometry_actors['edges'] = [self.geometry_edges_actor]

                        if has_vertices:
                            self.geometry_points_actor = create_shape_actor(
                                shape,
                                mesh_quality=2.0,
                                display_mode='points',
                                color=(1.0, 0.0, 0.0),
                                opacity=1.0
                            )
                            self.mesh_display.renderer.AddActor(self.geometry_points_actor)
                            self.geometry_actors['points'] = [self.geometry_points_actor]

                    self.geometry_display_source = "stl" if stl_displayed else "occ"

                    if hasattr(self, 'view_controller'):
                        self.view_controller._apply_render_mode_to_geometry(self.render_mode)

                    self.mesh_display.fit_view()
            else:
                # 对于非STL文件，不创建初始显示，等待模型树加载完成后显示
                self.geometry_display_source = "occ"
                self.log_info(f"非STL文件，跳过初始显示，等待模型树加载...")

            if unit_source == "auto" and not auto_detected:
                self.log_warning("未能从数模文件中读取单位，已按毫米(mm)处理")
            self.log_info(f"已导入几何: {file_path} (单位: {unit})")
            self.update_status("已导入几何")
            self.status_bar.hide_progress()

            if hasattr(self, 'model_tree_widget'):
                self.update_status("正在加载模型树...")
                self.log_info("正在后台加载模型树...")
                QTimer.singleShot(100, lambda: self._load_geometry_tree_async(shape, file_path, stats))

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理导入的几何失败: {str(e)}")
            self.log_error(f"处理导入的几何失败: {str(e)}")
            self.status_bar.hide_progress()

        finally:
            self._set_ribbon_button_enabled('file', 'import_geometry', True)

    def _load_geometry_tree_async(self, shape, file_path, stats):
        """异步加载几何模型树"""
        try:
            if hasattr(self, 'model_tree_widget'):
                self.model_tree_widget.load_geometry(shape, os.path.basename(file_path))

                if hasattr(self, 'cas_parts_info'):
                    # self.log_info(f"导入几何前cas_parts_info状态: {type(self.cas_parts_info)}, 内容: {self.cas_parts_info}")
                    self.log_info(f"导入几何前cas_parts_info状态: {type(self.cas_parts_info)}")
                else:
                    self.log_info(f"导入几何前cas_parts_info不存在")

                if not self._has_geometry_parts_info():
                    self.log_info("开始创建Default部件...")
                    self._create_default_part_for_geometry(shape, stats)
                else:
                    self.log_info("已存在几何部件信息，跳过Default部件创建")

                # 检查文件扩展名，如果是非STL文件，在模型树加载完成后显示几何
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext != ".stl":
                    # 延迟一点时间，确保模型树完全加载
                    QTimer.singleShot(200, lambda: self._show_non_stl_geometry_after_tree_loaded(shape, stats))

                self.update_status("已导入几何")
                self.log_info("模型树加载完成")
        except Exception as e:
            self.log_error(f"加载模型树失败: {str(e)}")
            self.update_status("模型树加载失败")

    def _show_non_stl_geometry_after_tree_loaded(self, shape, stats):
        """在模型树加载完成后显示非STL文件的几何"""
        try:
            self.update_status("正在创建几何显示...")

            if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                from fileIO.occ_to_vtk import create_shape_actor, create_geometry_edges_actor

                # 清除之前的几何actors（如果有的话）
                if hasattr(self, 'geometry_actors') and self.geometry_actors:
                    for actor_list in self.geometry_actors.values():
                        for actor in actor_list:
                            self.mesh_display.renderer.RemoveActor(actor)

                # 重新初始化几何actors
                self.geometry_actors = {}
                self.geometry_actor = None
                self.geometry_edges_actor = None
                self.geometry_points_actor = None

                # 创建新的几何actors
                has_surface = (stats.get('num_faces', 0) + stats.get('num_solids', 0)) > 0
                has_edges = stats.get('num_edges', 0) > 0
                has_vertices = stats.get('num_vertices', 0) > 0

                if has_surface:
                    self.geometry_actor = create_shape_actor(
                        shape,
                        mesh_quality=2.0,
                        display_mode='surface',
                        color=(0.8, 0.8, 0.9),
                        opacity=0.8
                    )
                    self.mesh_display.renderer.AddActor(self.geometry_actor)
                    self.geometry_actors['main'] = [self.geometry_actor]

                if has_edges:
                    self.geometry_edges_actor = create_geometry_edges_actor(
                        shape,
                        color=(0.0, 0.0, 0.0),
                        line_width=1.5,
                        sample_rate=0.01,
                        max_points_per_edge=500
                    )
                    self.mesh_display.renderer.AddActor(self.geometry_edges_actor)
                    self.geometry_actors['edges'] = [self.geometry_edges_actor]

                if has_vertices:
                    self.geometry_points_actor = create_shape_actor(
                        shape,
                        mesh_quality=8.0,
                        display_mode='points',
                        color=(1.0, 0.0, 0.0),
                        opacity=1.0
                    )
                    self.mesh_display.renderer.AddActor(self.geometry_points_actor)
                    self.geometry_actors['points'] = [self.geometry_points_actor]

                self.geometry_display_source = "occ"

                if hasattr(self, 'view_controller'):
                    self.view_controller._apply_render_mode_to_geometry(self.render_mode)

                self.mesh_display.fit_view()

                self.update_status("几何显示已创建")
                self.log_info("非STL文件几何显示已创建")
        except Exception as e:
            self.log_error(f"创建非STL几何显示失败: {str(e)}")
            self.update_status("几何显示创建失败")

    def _ensure_cas_parts_info(self):
        """确保cas_parts_info已初始化"""
        if not hasattr(self, 'cas_parts_info') or self.cas_parts_info is None:
            self.cas_parts_info = {}

    def _merge_parts_info(self, parts_info):
        """合并部件信息，保留已有几何元素"""
        self._ensure_cas_parts_info()
        if not parts_info:
            return

        parts_iter = []
        if isinstance(parts_info, dict):
            for part_name, part_data in parts_info.items():
                if isinstance(part_data, dict):
                    parts_iter.append((part_name, part_data))
        elif isinstance(parts_info, list):
            for idx, part_data in enumerate(parts_info):
                if isinstance(part_data, dict):
                    part_name = part_data.get('part_name', f'部件_{idx}')
                    parts_iter.append((part_name, part_data))

        for part_name, part_data in parts_iter:
            existing_part = self.cas_parts_info.get(part_name)
            if isinstance(existing_part, dict):
                for key, value in part_data.items():
                    if key == 'geometry_elements' and existing_part.get('geometry_elements'):
                        continue
                    if key == 'part_name':
                        existing_part.setdefault('part_name', value)
                    else:
                        existing_part[key] = value
                existing_part.setdefault('part_name', part_name)
                self.cas_parts_info[part_name] = existing_part
            else:
                part_copy = part_data.copy()
                part_copy.setdefault('part_name', part_name)
                self.cas_parts_info[part_name] = part_copy

    def _has_geometry_parts_info(self):
        """检查是否已有几何部件信息"""
        if not hasattr(self, 'cas_parts_info') or not self.cas_parts_info:
            return False

        if isinstance(self.cas_parts_info, dict):
            for part_data in self.cas_parts_info.values():
                if isinstance(part_data, dict):
                    geometry_elements = part_data.get('geometry_elements') or {}
                    if any(geometry_elements.values()):
                        return True
            return False

        if isinstance(self.cas_parts_info, list):
            for part_data in self.cas_parts_info:
                if isinstance(part_data, dict):
                    geometry_elements = part_data.get('geometry_elements') or {}
                    if any(geometry_elements.values()):
                        return True
            return False

        return False

    def _register_default_part(self, part_name, element_key, elements, counts, log_message):
        """注册默认部件并更新模型树显示"""
        self._ensure_cas_parts_info()

        existing_part = self.cas_parts_info.get(part_name) if isinstance(self.cas_parts_info, dict) else None
        if isinstance(existing_part, dict):
            part_info = existing_part
            part_info.setdefault('part_name', part_name)
            part_info.setdefault('bc_type', '')
            part_info[element_key] = elements
            part_info.update(counts)
        else:
            part_info = {
                'part_name': part_name,
                'bc_type': '',
                element_key: elements
            }
            part_info.update(counts)

        self.cas_parts_info[part_name] = part_info

        if hasattr(self, 'model_tree_widget'):
            self.model_tree_widget.load_parts({'parts_info': self.cas_parts_info})

        self.log_info(log_message)

    def _create_default_part_for_mesh(self, mesh_data):
        """
        为导入的网格创建Default部件
        
        Args:
            mesh_data: 网格数据对象
        """
        try:
            # 收集所有网格元素的索引
            mesh_elements = {
                "vertices": [],
                "edges": [],
                "faces": [],
                "bodies": []
            }
            
            # 提取顶点索引
            if hasattr(mesh_data, 'node_coords'):
                num_vertices = len(mesh_data.node_coords)
                mesh_elements["vertices"] = list(range(num_vertices))
            
            # 提取面索引
            if hasattr(mesh_data, 'cells'):
                num_faces = len(mesh_data.cells)
                mesh_elements["faces"] = list(range(num_faces))
            
            # 创建Default部件信息
            part_name = "DefaultPart"
            counts = {
                'num_vertices': len(mesh_elements["vertices"]),
                'num_edges': len(mesh_elements["edges"]),
                'num_faces': len(mesh_elements["faces"]),
                'num_bodies': len(mesh_elements["bodies"])
            }
            self._register_default_part(
                part_name,
                'mesh_elements',
                mesh_elements,
                counts,
                "已自动创建Default部件，包含所有网格元素"
            )
            
        except Exception as e:
            self.log_error(f"创建Default部件失败: {str(e)}")

    def _create_default_part_for_geometry(self, shape, stats):
        """
        为导入的几何创建Default部件
        
        Args:
            shape: OpenCASCADE TopoDS_Shape对象
            stats: 几何统计信息
        """
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
            
            # 收集所有几何元素的索引
            geometry_elements = {
                "vertices": [],
                "edges": [],
                "faces": [],
                "bodies": []
            }
            
            # 提取顶点索引
            vertex_index = 0
            explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
            while explorer.More():
                geometry_elements["vertices"].append(vertex_index)
                vertex_index += 1
                explorer.Next()
            
            # 提取边索引
            edge_index = 0
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while explorer.More():
                geometry_elements["edges"].append(edge_index)
                edge_index += 1
                explorer.Next()
            
            # 提取面索引
            face_index = 0
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                geometry_elements["faces"].append(face_index)
                face_index += 1
                explorer.Next()
            
            # 提取体索引
            body_index = 0
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while explorer.More():
                geometry_elements["bodies"].append(body_index)
                body_index += 1
                explorer.Next()
            
            # 记录收集到的元素数量
            self.log_info(f"Default部件元素收集:")
            self.log_info(f"  - 顶点: {len(geometry_elements['vertices'])} 个")
            self.log_info(f"  - 边: {len(geometry_elements['edges'])} 个")
            self.log_info(f"  - 面: {len(geometry_elements['faces'])} 个")
            self.log_info(f"  - 体: {len(geometry_elements['bodies'])} 个")
            
            # 创建Default部件信息
            part_name = "DefaultPart"
            counts = {
                'num_vertices': len(geometry_elements["vertices"]),
                'num_edges': len(geometry_elements["edges"]),
                'num_faces': len(geometry_elements["faces"]),
                'num_solids': len(geometry_elements["bodies"])
            }
            self._register_default_part(
                part_name,
                'geometry_elements',
                geometry_elements,
                counts,
                "已自动创建Default部件，包含所有几何元素"
            )
            
        except Exception as e:
            self.log_error(f"创建Default部件失败: {str(e)}")
            import traceback
            self.log_error(traceback.format_exc())

    def delete_geometry_elements(self, element_map):
        """删除几何元素（由对话框调用）"""
        if not hasattr(self, 'geometry_operations'):
            return False
        return self.geometry_operations.delete_geometry(element_map)

    def _get_selected_geometry_elements_from_tree(self):
        if not hasattr(self, 'model_tree_widget'):
            return {}
        tree = getattr(self.model_tree_widget, 'tree', None)
        if tree is None:
            return {}
        if hasattr(self, 'view_controller'):
            helper = getattr(self.view_controller, '_picking_helper', None)
            if helper is not None and hasattr(helper, 'get_selected_elements'):
                picked = helper.get_selected_elements()
                if picked and any(picked.values()):
                    return picked
        selected = {"vertices": set(), "edges": set(), "faces": set(), "bodies": set()}
        selected_items = tree.selectedItems()
        type_map = {
            "vertices": "vertices",
            "edges": "edges",
            "faces": "faces",
            "bodies": "bodies",
        }
        for item in selected_items:
            data = item.data(0, Qt.UserRole)
            if not (isinstance(data, tuple) and len(data) >= 4):
                continue
            category, element_type, element_obj = data[0], data[1], data[2]
            if category != "geometry":
                continue
            key = type_map.get(element_type)
            if key:
                selected[key].add(element_obj)
        return {key: list(values) for key, values in selected.items()}

    def _delete_selected_geometry_from_tree(self):
        element_map = self._get_selected_geometry_elements_from_tree()
        if not element_map or not any(element_map.values()):
            self.update_status("删除几何: 未选中元素")
            return False
        return self.delete_geometry_elements(element_map)

    def _on_model_tree_selection_changed(self):
        if not self._delete_geometry_mode_active:
            return
        element_map = self._get_selected_geometry_elements_from_tree()
        self._geometry_delete_elements_cache = {}
        for key, values in element_map.items():
            if values:
                self._geometry_delete_elements_cache[key] = set(values)

    def _start_delete_geometry_mode(self):
        if self._delete_geometry_mode_active:
            self.update_status("删除几何: 已在拾取模式")
            return
        if not hasattr(self, 'view_controller'):
            return
        if hasattr(self, 'model_tree_widget') and hasattr(self.model_tree_widget, 'tree'):
            self.model_tree_widget.tree.clearSelection()
        self._delete_geometry_mode_active = True
        self._geometry_delete_elements_cache = {}
        self.view_controller.start_geometry_pick(
            on_pick=self._on_delete_geometry_pick,
            on_unpick=self._on_delete_geometry_unpick,
            on_confirm=self._on_delete_geometry_confirm,
            on_cancel=self._on_delete_geometry_cancel,
            on_delete=self._delete_geometry_from_pick,
        )
        hint = "删除几何: 左键拾取，右键取消，Enter键确认删除，Delete键删除已选元素，Esc退出"
        self.log_info(hint)
        self.update_status(hint)

    def _stop_delete_geometry_mode(self):
        if not self._delete_geometry_mode_active:
            return
        self._delete_geometry_mode_active = False
        self._geometry_delete_elements_cache = {}
        if hasattr(self, 'view_controller'):
            self.view_controller.stop_geometry_pick(restore_display_mode=True)
        self.update_status("删除几何: 已退出")

    def _on_delete_geometry_pick(self, element_type, element_obj, element_index):
        key_map = {
            "vertex": "vertices",
            "edge": "edges",
            "face": "faces",
            "body": "bodies",
        }
        key = key_map.get(element_type)
        if key is None:
            return
        self._geometry_delete_elements_cache.setdefault(key, set()).add(element_obj)

    def _on_delete_geometry_unpick(self, element_type, element_obj, element_index):
        key_map = {
            "vertex": "vertices",
            "edge": "edges",
            "face": "faces",
            "body": "bodies",
        }
        key = key_map.get(element_type)
        if key is None:
            return
        if key in self._geometry_delete_elements_cache:
            self._geometry_delete_elements_cache[key].discard(element_obj)

    def _on_delete_geometry_confirm(self):
        self._delete_geometry_from_pick()

    def _on_delete_geometry_cancel(self):
        self._stop_delete_geometry_mode()

    def _collect_picked_geometry_elements(self):
        if not hasattr(self, 'view_controller'):
            return {}
        helper = getattr(self.view_controller, '_picking_helper', None)
        if helper is None or not hasattr(helper, 'get_selected_elements'):
            return {}
        return helper.get_selected_elements()

    def _delete_geometry_from_pick(self):
        if not self._delete_geometry_mode_active:
            self._start_delete_geometry_mode()
        element_map = self._collect_picked_geometry_elements()
        if not element_map or not any(element_map.values()):
            element_map = {key: list(values) for key, values in self._geometry_delete_elements_cache.items()}
        if not element_map or not any(element_map.values()):
            self.update_status("删除几何: 未选中元素")
            return False
        success = self.delete_geometry_elements(element_map)
        if success and hasattr(self, 'view_controller'):
            helper = getattr(self.view_controller, '_picking_helper', None)
            if helper is not None and hasattr(helper, 'clear_selection'):
                helper.clear_selection()
            self._geometry_delete_elements_cache = {}
        return success

    def delete_geometry_selected_elements(self):
        return self._delete_selected_geometry_from_tree()

    def _rebuild_parts_for_geometry(self, old_shape, new_shape, removed_shapes):
        """根据删除结果重建部件几何索引映射"""
        if not hasattr(self, 'cas_parts_info') or not self.cas_parts_info:
            return

        if new_shape is None or (hasattr(new_shape, 'IsNull') and new_shape.IsNull()):
            self.log_info(f"new_shape 为空或无效，跳过部件重建")
            return

        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID

        type_pairs = [
            ("vertices", TopAbs_VERTEX),
            ("edges", TopAbs_EDGE),
            ("faces", TopAbs_FACE),
            ("bodies", TopAbs_SOLID),
        ]

        old_maps = {}
        for key, occ_type in type_pairs:
            idx_map = {}
            explorer = TopExp_Explorer(old_shape, occ_type)
            idx = 0
            while explorer.More():
                shape = explorer.Current()
                idx_map[idx] = shape
                idx += 1
                explorer.Next()
            old_maps[key] = idx_map

        new_maps = {}
        for key, occ_type in type_pairs:
            shape_to_index = {}
            explorer = TopExp_Explorer(new_shape, occ_type)
            idx = 0
            while explorer.More():
                shape = explorer.Current()
                shape_to_index[shape] = idx
                idx += 1
                explorer.Next()
            new_maps[key] = shape_to_index

        def _remap_indices(indices, key):
            updated = []
            for old_idx in indices:
                shape = old_maps.get(key, {}).get(old_idx)
                if shape is None:
                    continue
                if shape in removed_shapes:
                    continue
                new_idx = new_maps.get(key, {}).get(shape)
                if new_idx is None:
                    continue
                updated.append(new_idx)
            return sorted(set(updated))

        if isinstance(self.cas_parts_info, dict):
            for part_name, part_data in self.cas_parts_info.items():
                if not isinstance(part_data, dict):
                    continue
                geometry_elements = part_data.get('geometry_elements') or {}
                new_elements = {}
                for key, _ in type_pairs:
                    new_elements[key] = _remap_indices(geometry_elements.get(key, []), key)
                part_data['geometry_elements'] = new_elements
                part_data['num_vertices'] = len(new_elements.get('vertices', []))
                part_data['num_edges'] = len(new_elements.get('edges', []))
                part_data['num_faces'] = len(new_elements.get('faces', []))
                part_data['num_solids'] = len(new_elements.get('bodies', []))
        else:
            return

        if self.cas_parts_info:
            has_geometry = False
            for part_data in self.cas_parts_info.values():
                if isinstance(part_data, dict):
                    geometry_elements = part_data.get('geometry_elements') or {}
                    if any(geometry_elements.values()):
                        has_geometry = True
                        break
            if not has_geometry:
                self._create_default_part_for_geometry(new_shape, getattr(self, 'current_geometry_stats', {}))

        if hasattr(self, 'model_tree_widget'):
            self.model_tree_widget.load_parts({'parts_info': self.cas_parts_info})

    def on_geometry_import_failed(self, error_message):
        """几何导入失败回调"""
        QMessageBox.critical(self, "错误", error_message)
        self.log_error(error_message)
        self.update_status("导入几何失败")
        self.status_bar.hide_progress()

        # 重新启用导入几何按钮
        self._set_ribbon_button_enabled('file', 'import_geometry', True)

    def export_geometry(self):
        """导出几何文件"""
        self.geometry_operations.export_geometry()


    def clear_mesh(self):
        """清空网格"""
        self.current_mesh = None
        self.line_connectors = None
        self.line_parts = None
        self._apply_mesh_dimension(GLOBAL_MESH_DIMENSION, update_mesh=False)
        if hasattr(self, 'mesh_display'):
            self.mesh_display.clear()
        self.log_info("已清空网格")
        self.update_status("已清空网格")

    def _execute_part_operation(self, operation_name, display_method, selected_part_name):
        """执行部件操作的通用方法

        Args:
            operation_name: 操作名称，用于日志和状态显示
            display_method: 要调用的显示方法
            selected_part_name: 选中的部件名称
        """
        # Extract the actual part name from formatted text (e.g., "部件1 - Max Size: 1.0, Prism: True" -> "部件1")
        actual_part_name = selected_part_name.split(' - ')[0] if ' - ' in selected_part_name else selected_part_name

        # 在日志中显示部件信息
        self.log_info(f"{operation_name}: {selected_part_name} (实际名称: {actual_part_name})")

        # 更新状态栏
        self.update_status(f"已{operation_name.split(' ')[0]}部件: {actual_part_name}")

        # 如果有3D显示区域，执行操作
        if hasattr(self, 'mesh_display'):
            try:
                # First try with cas_parts_info if available
                if hasattr(self, 'cas_parts_info') and self.cas_parts_info:
                    success = display_method(actual_part_name, parts_info=self.cas_parts_info)
                else:
                    # If cas_parts_info is not available, try with no parts_info to let the display method handle fallbacks
                    success = display_method(actual_part_name, parts_info=None)

                if success:
                    self.log_info(f"成功{operation_name.split(' ')[0]}部件: {actual_part_name}")
                else:
                    self.log_error(f"{operation_name.split(' ')[0]}部件失败: {actual_part_name}")
                    # If operation fails, try to highlight the part instead as a fallback
                    self.log_info(f"尝试高亮部件作为备选方案: {actual_part_name}")
                    if hasattr(self, 'cas_parts_info') and self.cas_parts_info:
                        self.mesh_display.highlight_part(actual_part_name, highlight=True, parts_info=self.cas_parts_info)
                    else:
                        self.mesh_display.highlight_part(actual_part_name, highlight=True)
            except Exception as e:
                self.log_error(f"{operation_name.split(' ')[0]}部件失败: {str(e)}")

    def update_params_display(self):
        """更新参数显示"""
        if not self.params:
            return
        self.part_manager.update_parts_list(update_status=False)

    def on_part_select(self, item):
        """处理部件列表选择事件"""
        if hasattr(item, 'text'):
            self.part_manager.on_part_select(item)
            return

        self.log_info("部件选择事件已触发")
        self.update_status("已选中部件")


    def log_info(self, message):
        """记录信息日志"""
        self.ui_helpers.log_info(message)

    def log_error(self, message):
        """记录错误日志"""
        self.ui_helpers.log_error(message)

    def log_warning(self, message):
        """记录警告日志"""
        self.ui_helpers.log_warning(message)

    def log_debug(self, message):
        """记录调试日志"""
        self.ui_helpers.log_debug(message)

    def log_verbose(self, message):
        """记录详细日志"""
        self.ui_helpers.log_verbose(message)

    def append_info_output(self, message):
        """添加信息到输出窗口，供 utils.message 模块调用"""
        if hasattr(self, 'info_output'):
            self.info_output.append_info_output(message)

    def update_status(self, message):
        """更新状态栏信息"""
        self.ui_helpers.update_status(message)

    def reset_config(self):
        """重置配置"""
        self.config_manager.reset_config()

    def _extract_parts_from_boundary_nodes(self, boundary_nodes):
        """从边界节点提取部件信息"""
        try:
            # 创建一个字典来存储每个部件的节点信息
            parts_dict = {}

            for node in boundary_nodes:
                # 获取节点的部件名称
                part_name = getattr(node, 'part_name', 'Unknown')
                if part_name is None or part_name == '':
                    part_name = 'Unknown'

                # 如果部件名称不存在于字典中，创建新的条目
                if part_name not in parts_dict:
                    parts_dict[part_name] = {
                        'nodes': [],
                        'bc_type': getattr(node, 'bc_type', 'boundary'),
                        'count': 0
                    }

                # 添加节点到对应部件
                parts_dict[part_name]['nodes'].append(node)
                parts_dict[part_name]['count'] += 1

            # 转换为GUI期望的格式
            extracted_parts_info = {}
            for part_name, part_data in parts_dict.items():
                extracted_parts_info[part_name] = {
                    'part_name': part_name,
                    'bc_type': part_data['bc_type'],
                    'node_count': part_data['count'],
                    'nodes': part_data['nodes']
                }

            return extracted_parts_info

        except Exception as e:
            self.log_error(f"从边界节点提取部件信息失败: {str(e)}")
            return None

    def _create_node_index_mapping(self, original_coords, new_coords, tolerance=1e-6):
        """
        基于坐标创建原始节点到新节点的索引映射

        Args:
            original_coords: 原始节点坐标列表
            new_coords: 新节点坐标列表
            tolerance: 坐标匹配的容差（未使用，保留参数以保持兼容性）

        Returns:
            字典：{original_index: new_index}，如果找不到匹配则返回None
        """
        if not original_coords or not new_coords:
            return None

        from data_structure.basic_elements import NodeElement

        # 确保所有节点都是3D坐标 FIXME
        if original_coords and len(original_coords[0]) == 2:
            original_coords = [list(coord) + [0.0] for coord in original_coords]
        
        if new_coords and len(new_coords[0]) == 2:
            new_coords = [list(coord) + [0.0] for coord in new_coords]
            
        # 将原始坐标和新坐标转换为 NodeElement 对象
        original_nodes = [NodeElement(coord, idx) for idx, coord in enumerate(original_coords)]
        new_nodes = [NodeElement(coord, idx) for idx, coord in enumerate(new_coords)]

        # 创建新节点的 hash 到索引的映射
        new_node_hash_map = {}
        for new_node in new_nodes:
            new_node_hash_map[new_node.hash] = new_node.idx

        # 创建原始节点到新节点的索引映射
        mapping = {}
        for orig_idx, orig_node in enumerate(original_nodes):
            if orig_node.hash in new_node_hash_map:
                mapping[orig_idx] = new_node_hash_map[orig_node.hash]

        return mapping

    def _map_parts_info_to_new_mesh(self, original_parts_info, node_mapping):
        """
        将原始部件信息中的节点索引映射到新网格的节点索引

        Args:
            original_parts_info: 原始部件信息
            node_mapping: 节点索引映射字典 {original_index: new_index}

        Returns:
            映射后的部件信息
        """
        if not original_parts_info or not node_mapping:
            return original_parts_info

        mapped_parts_info = {}

        for part_name, part_data in original_parts_info.items():
            if part_name in PARTS_INFO_RESERVED_KEYS:
                mapped_parts_info[part_name] = part_data
                continue

            if not isinstance(part_data, dict):
                mapped_parts_info[part_name] = part_data
                continue

            mapped_part_data = part_data.copy()

            # 映射节点索引
            if 'nodes' in part_data and isinstance(part_data['nodes'], list):
                mapped_nodes = []
                for node_idx in part_data['nodes']:
                    try:
                        if isinstance(node_idx, int) and node_idx in node_mapping:
                            mapped_nodes.append(node_mapping[node_idx])
                        else:
                            mapped_nodes.append(node_idx)
                    except Exception as e:
                        self.log_warning(f"映射节点索引 {node_idx} 时出错: {str(e)}")
                        mapped_nodes.append(node_idx)
                mapped_part_data['nodes'] = mapped_nodes

            # 映射面中的节点索引
            if 'faces' in part_data and isinstance(part_data['faces'], list):
                mapped_faces = []
                for face in part_data['faces']:
                    if isinstance(face, dict) and 'nodes' in face:
                        mapped_face = face.copy()
                        mapped_face_nodes = []
                        if isinstance(face['nodes'], list):
                            for node_idx in face['nodes']:
                                try:
                                    if isinstance(node_idx, int) and node_idx in node_mapping:
                                        mapped_face_nodes.append(node_mapping[node_idx])
                                    else:
                                        mapped_face_nodes.append(node_idx)
                                except Exception as e:
                                    self.log_warning(f"映射面节点索引 {node_idx} 时出错: {str(e)}")
                                    mapped_face_nodes.append(node_idx)
                        else:
                            mapped_face_nodes = face['nodes']
                        mapped_face['nodes'] = mapped_face_nodes
                        mapped_faces.append(mapped_face)
                    else:
                        mapped_faces.append(face)
                mapped_part_data['faces'] = mapped_faces

            mapped_parts_info[part_name] = mapped_part_data

        return mapped_parts_info

    def display_mesh(self):
        """显示网格"""
        if hasattr(self, 'current_mesh') and self.current_mesh:
            if hasattr(self, 'mesh_display'):
                self.mesh_display.display_mesh(self.current_mesh, render_immediately=False)
                self.log_info("已显示网格")
                self.update_status("已显示网格")
            else:
                self.log_info("未找到网格显示组件")
                self.update_status("未找到网格显示组件")
        else:
            self.log_info("未找到网格数据")
            self.update_status("未找到网格数据")

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 检查是否有网格生成任务正在运行
        if self.mesh_generation_thread and self.mesh_generation_thread.isRunning():
            reply = QMessageBox.question(
                self,
                '网格生成中',
                '网格生成任务正在进行中，确定要退出吗？任务将被强制终止。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # 停止网格生成任务
                self.mesh_generation_thread.stop()
            else:
                event.ignore()
                return

        # 检查是否有网格导入任务正在运行
        if self.import_thread and self.import_thread.isRunning():
            reply = QMessageBox.question(
                self,
                '网格导入中',
                '网格导入任务正在进行中，确定要退出吗？任务将被强制终止。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # 停止网格导入任务
                self.import_thread.stop()
            else:
                event.ignore()
                return

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

    window = PyMeshGenGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

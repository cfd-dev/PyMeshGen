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
        sys.path.insert(0, dir_path)

# Import modules after setting up paths
from gui.gui_base import (
    StatusBar, InfoOutput, DialogBase,
    Splitter, PartListWidget
)
from gui.model_tree import ModelTreeWidget
from gui.ribbon_widget import RibbonWidget
from gui.toolbar import ViewToolbar
from gui.mesh_display import MeshDisplayArea
from gui.ui_utils import UIStyles
from gui.view_controller import ViewController
from gui.mesh_operations import MeshOperations
from gui.part_manager import PartManager
from gui.config_manager import ConfigManager
from gui.geometry_operations import GeometryOperations
from gui.help_module import HelpModule
from gui.ui_helpers import UIHelpers
from data_structure.parameters import Parameters


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
        self.params = None                    # 网格生成的参数配置
        self.mesh_generator = None            # 网格生成器实例
        self.current_mesh = None              # 当前生成的网格对象
        self.cas_parts_info = None            # CAS文件的部件信息
        self.original_node_coords = None      # 原始节点坐标
        self.parts_params = []                # 部件参数列表
        self.render_mode = "surface"           # 渲染模式（surface：表面渲染）
        self.show_boundary = True             # 是否显示边界
        self.mesh_generation_thread = None    # 网格生成的线程实例
        self.progress_dialog = None           # 进度对话框实例
        self.import_thread = None             # 导入操作的线程实例
        self._progress_cache = {}             # 进度日志缓存，用于节流输出

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

        self._create_model_tree_widget()
        self._create_properties_panel()

        self.left_panel.setMinimumWidth(300)
        self.paned_window.addWidget(self.left_panel)

    def _create_model_tree_widget(self):
        """创建模型树组件"""
        self.model_tree_widget = ModelTreeWidget(parent=self)

        model_tree_frame_container = QGroupBox("模型树")
        model_tree_layout = QVBoxLayout(model_tree_frame_container)
        model_tree_layout.setSpacing(2)

        model_tree_layout.addWidget(self.model_tree_widget.widget)

        self.model_tree_widget.widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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
                'export_geometry': 'document-export'
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
                button.clicked.connect(lambda: self.set_render_mode("surface"))
            elif button_name == 'wireframe':
                button.clicked.connect(lambda: self.set_render_mode("wireframe"))
            elif button_name == 'surface-wireframe':
                button.clicked.connect(lambda: self.set_render_mode("surface-wireframe"))

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
                'extract_boundary': 'extract-boundary'
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

    def set_render_mode(self, mode):
        """设置渲染模式"""
        self.view_controller.set_render_mode(mode)

    def on_mesh_display_key(self, event):
        """处理网格显示区域的键盘事件"""
        key = event.key()
        key_actions = {
            Qt.Key_R: lambda: self.reset_view() or self.update_status("已重置视图 (R键)"),
            Qt.Key_F: lambda: self.fit_view() or self.update_status("已适应视图 (F键)"),
            Qt.Key_O: self._toggle_boundary_display,
            Qt.Key_1: lambda: self.set_render_mode("surface"),
            Qt.Key_2: lambda: self.set_render_mode("wireframe"),
            Qt.Key_3: lambda: self.set_render_mode("surface-wireframe"),
        }

        action = key_actions.get(key)
        if action:
            action()

    def _toggle_boundary_display(self):
        """切换边界显示"""
        self.view_controller.toggle_boundary_display()

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

                # 清空网格显示区域
                if hasattr(self, 'mesh_display'):
                    self.mesh_display.clear()
                    self.mesh_display.clear_mesh_actors()
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
                        self.refresh_display_all_parts()
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

    def save_config(self):
        """保存工程 - 将JSON配置文件和导入网格文件的路径保存到.pymg工程文件中"""
        self.config_manager.save_config()

    def import_mesh(self):
        """导入网格（使用异步线程，避免GUI卡顿）"""
        from gui.file_operations import FileOperations
        from gui.import_thread import MeshImportThread

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入网格文件",
            os.path.join(self.project_root, "meshes"),
            "网格文件 (*.vtk *.vtu *.stl *.obj *.cas *.msh *.ply *.xdmf *.xmf *.off *.med *.mesh *.meshb *.bdf *.fem *.nas *.inp *.e *.exo *.ex2 *.su2 *.cgns *.avs *.vol *.mdpa *.h5m *.f3grid *.dat *.tec *.ugrid *.ele *.node *.xml *.post *.wkt *.hmf);;所有文件 (*.*)"
        )

        if file_path:
            try:
                file_ops = FileOperations(self.project_root, log_callback=self.log_info)

                # 创建导入线程
                self.import_thread = MeshImportThread(file_path, file_ops)

                # 连接信号
                self.import_thread.progress_updated.connect(self.on_import_progress)
                self.import_thread.import_finished.connect(self.on_import_finished)
                self.import_thread.import_failed.connect(self.on_import_failed)

                # 禁用导入按钮，防止重复导入
                self._reset_progress_cache("mesh")
                self._set_ribbon_button_enabled('file', 'import', False)

                # 启动线程
                self.import_thread.start()
                self.log_info(f"开始导入网格: {file_path}")
                self.update_status("正在导入网格...")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入网格失败: {str(e)}")
                self.log_error(f"导入网格失败: {str(e)}")

    def on_import_progress(self, message, progress):
        """导入进度更新回调"""
        self._update_progress(message, progress, "mesh")

    def on_import_finished(self, mesh_data):
        """导入完成回调 - 优化版本，使用QTimer分步处理避免UI卡顿"""
        try:
            self.current_mesh = mesh_data

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
                
                # 创建分步处理函数 - 减少延迟时间
                def step1_load_mesh():
                    self.model_tree_widget.load_mesh(mesh_data, mesh_name="网格")
                    QTimer.singleShot(0, step2_load_parts)
                
                def step2_load_parts():
                    self.update_status("正在加载部件信息...")
                    self.model_tree_widget.load_parts(mesh_data)
                    # 如果网格文件本身包含部件信息，保留这些信息
                    if hasattr(mesh_data, 'parts_info') and mesh_data.parts_info:
                        self.cas_parts_info = mesh_data.parts_info.copy()
                        self.log_info(f"已保留网格文件中的部件信息，共 {len(mesh_data.parts_info)} 个部件")
                    # 如果没有预设部件，自动创建Default部件
                    elif not hasattr(self, 'cas_parts_info') or not self.cas_parts_info:
                        self._create_default_part_for_mesh(mesh_data)
                    QTimer.singleShot(0, step3_refresh_display)
                
                def step3_refresh_display():
                    self.update_status("正在刷新显示...")
                    self.refresh_display_all_parts()
                    QTimer.singleShot(0, step4_complete)
                
                def step4_complete():
                    self.log_info(f"已导入网格: {mesh_data.file_path}")
                    self.log_info(f"节点数: {len(mesh_data.node_coords)}, 单元数: {len(mesh_data.cells)}")
                    self.update_status("已导入网格")
                    self.status_bar.hide_progress()

                # 启动分步处理 - 使用0延迟
                QTimer.singleShot(0, step1_load_mesh)
            else:
                # Refresh display to show all parts with different colors
                self.update_status("正在刷新显示...")
                self.refresh_display_all_parts()
                
                self.log_info(f"已导入网格: {mesh_data.file_path}")
                self.log_info(f"节点数: {len(mesh_data.node_coords)}, 单元数: {len(mesh_data.cells)}")
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

            self.log_info(f"几何统计信息:")
            self.log_info(f"  - 顶点数: {stats['num_vertices']}")
            self.log_info(f"  - 边数: {stats['num_edges']}")
            self.log_info(f"  - 面数: {stats['num_faces']}")
            self.log_info(f"  - 实体数: {stats['num_solids']}")

            bbox_min, bbox_max = stats['bounding_box']
            self.log_info(f"  - 边界框: ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}) 到 ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f})")

            self.current_geometry = shape
            self.geometry_actors = {}
            self.geometry_actor = None
            self.geometry_edges_actor = None

            self.update_status("正在创建几何显示...")

            if hasattr(self, 'mesh_display') and hasattr(self.mesh_display, 'renderer'):
                from fileIO.occ_to_vtk import create_geometry_edges_actor
                
                self.geometry_actor = create_shape_actor(
                    shape,
                    mesh_quality=8.0,
                    display_mode='surface',
                    color=(0.8, 0.8, 0.9),
                    opacity=0.8
                )
                self.mesh_display.renderer.AddActor(self.geometry_actor)
                self.geometry_actors['main'] = [self.geometry_actor]
                
                self.geometry_edges_actor = create_geometry_edges_actor(
                    shape,
                    color=(0.0, 0.0, 0.0),
                    line_width=1.5,
                    sample_rate=0.5,
                    max_points_per_edge=20
                )
                self.mesh_display.renderer.AddActor(self.geometry_edges_actor)
                self.geometry_actors['edges'] = [self.geometry_edges_actor]
                
                if self.render_mode == "wireframe":
                    self.geometry_actor.SetVisibility(False)
                    self.geometry_edges_actor.SetVisibility(True)
                elif self.render_mode == "surface-wireframe":
                    self.geometry_actor.SetVisibility(True)
                    self.geometry_edges_actor.SetVisibility(True)
                else:
                    self.geometry_actor.SetVisibility(True)
                    self.geometry_edges_actor.SetVisibility(False)
                
                self.mesh_display.renderer.ResetCamera()
                self.mesh_display.render_window.Render()

            self.log_info(f"已导入几何: {file_path}")
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
                    self.log_info(f"导入几何前cas_parts_info状态: {type(self.cas_parts_info)}, 内容: {self.cas_parts_info}")
                else:
                    self.log_info(f"导入几何前cas_parts_info不存在")
                
                if not hasattr(self, 'cas_parts_info') or not self.cas_parts_info:
                    self.log_info(f"开始创建Default部件...")
                    self._create_default_part_for_geometry(shape, stats)
                else:
                    self.log_info(f"已存在预设部件，跳过Default部件创建")
                
                self.update_status("已导入几何")
                self.log_info("模型树加载完成")
        except Exception as e:
            self.log_error(f"加载模型树失败: {str(e)}")
            self.update_status("模型树加载失败")

    def _create_default_part_for_mesh(self, mesh_data):
        """
        为导入的网格创建Default部件
        
        Args:
            mesh_data: 网格数据对象
        """
        try:
            # 初始化cas_parts_info
            if not hasattr(self, 'cas_parts_info') or self.cas_parts_info is None:
                self.cas_parts_info = {}
            
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
            part_info = {
                'part_name': part_name,
                'bc_type': '',
                'mesh_elements': mesh_elements,
                'num_vertices': len(mesh_elements["vertices"]),
                'num_edges': len(mesh_elements["edges"]),
                'num_faces': len(mesh_elements["faces"]),
                'num_bodies': len(mesh_elements["bodies"])
            }
            
            # 添加到cas_parts_info
            self.cas_parts_info[part_name] = part_info
            
            # 更新模型树中的部件显示
            if hasattr(self, 'model_tree_widget'):
                self.model_tree_widget.load_parts({'parts_info': self.cas_parts_info})
            
            self.log_info(f"已自动创建Default部件，包含所有网格元素")
            
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
            
            # 初始化cas_parts_info
            if not hasattr(self, 'cas_parts_info') or self.cas_parts_info is None:
                self.cas_parts_info = {}
            
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
            part_info = {
                'part_name': part_name,
                'bc_type': '',
                'geometry_elements': geometry_elements,
                'num_vertices': len(geometry_elements["vertices"]),
                'num_edges': len(geometry_elements["edges"]),
                'num_faces': len(geometry_elements["faces"]),
                'num_solids': len(geometry_elements["bodies"])
            }
            
            # 添加到cas_parts_info
            self.cas_parts_info[part_name] = part_info
            
            # 更新模型树中的部件显示
            if hasattr(self, 'model_tree_widget'):
                self.model_tree_widget.load_parts({'parts_info': self.cas_parts_info})
            
            self.log_info(f"已自动创建Default部件，包含所有几何元素")
            
        except Exception as e:
            self.log_error(f"创建Default部件失败: {str(e)}")
            import traceback
            self.log_error(traceback.format_exc())

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

    def export_mesh(self):
        """导出网格"""
        self.geometry_operations.export_mesh()


    def clear_mesh(self):
        """清空网格"""
        self.current_mesh = None
        if hasattr(self, 'mesh_display'):
            self.mesh_display.clear()
        self.log_info("已清空网格")
        self.update_status("已清空网格")

    def reset_view(self):
        """重置视图"""
        self.view_controller.reset_view()

    def fit_view(self):
        """适应视图"""
        self.view_controller.fit_view()

    def set_view_x_positive(self):
        """设置X轴正向视图"""
        self.view_controller.set_view_x_positive()

    def set_view_x_negative(self):
        """设置X轴负向视图"""
        self.view_controller.set_view_x_negative()

    def set_view_y_positive(self):
        """设置Y轴正向视图"""
        self.view_controller.set_view_y_positive()

    def set_view_y_negative(self):
        """设置Y轴负向视图"""
        self.view_controller.set_view_y_negative()

    def set_view_z_positive(self):
        """设置Z轴正向视图"""
        self.view_controller.set_view_z_positive()

    def set_view_z_negative(self):
        """设置Z轴负向视图"""
        self.view_controller.set_view_z_negative()

    def set_view_isometric(self):
        """设置等轴测视图"""
        self.view_controller.set_view_isometric()

    def add_part(self):
        """添加部件"""
        self.part_manager.add_part()

    def remove_part(self):
        """删除部件"""
        self.part_manager.remove_part()

    def edit_part(self):
        """编辑部件参数（从右键菜单调用）"""
        self.part_manager.edit_mesh_params()
    
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

    def show_selected_part(self):
        """显示选中部件（从右键菜单调用）"""
        self.part_manager.show_selected_part()



    def show_only_selected_part(self):
        """只显示选中部件，隐藏其他所有部件（从右键菜单调用）"""
        self.part_manager.show_only_selected_part()


    def show_all_parts(self):
        """显示所有部件（从右键菜单调用）"""
        self.part_manager.show_all_parts()


    def update_params_display(self):
        """更新参数显示"""
        if not self.params:
            return
        self.update_parts_list()

    def update_parts_list(self):
        """更新部件列表"""
        self.part_manager.update_parts_list(update_status=False)

    def update_parts_list_from_cas(self, parts_info):
        """从cas文件的部件信息更新部件列表"""
        self.part_manager.update_parts_list_from_cas(parts_info=parts_info, update_status=False)

    def on_part_select(self, item):
        """处理部件列表选择事件"""
        if hasattr(item, 'text'):
            self.part_manager.on_part_select(item)
            return

        self.log_info("部件选择事件已触发")
        self.update_status("已选中部件")

    def handle_part_visibility_change(self, part_name, is_visible):
        """处理部件可见性变化"""
        self.part_manager.handle_part_visibility_change(part_name, is_visible)

    def on_part_created(self, part_info):
        """
        处理新建部件的回调
        
        Args:
            part_info: 部件信息字典，包含:
                - part_name: 部件名称
                - geometry_elements: 几何元素字典
                - mesh_elements: 网格元素字典
        """
        self.part_manager.on_part_created(part_info)

    def refresh_display_all_parts(self):
        """刷新显示所有可见部件 - 优化版本，批量处理减少渲染次数"""
        self.part_manager.refresh_display_all_parts()

    def switch_display_mode(self, mode):
        """切换显示模式"""
        self.part_manager.switch_display_mode(mode)

    def log_info(self, message):
        """记录信息日志"""
        self.ui_helpers.log_info(message)

    def log_error(self, message):
        """记录错误日志"""
        self.ui_helpers.log_error(message)

    def log_warning(self, message):
        """记录警告日志"""
        self.ui_helpers.log_warning(message)

    def update_status(self, message):
        """更新状态栏信息"""
        self.ui_helpers.update_status(message)

    def on_geometry_visibility_changed(self, element_type, visible):
        """几何元素类别可见性改变时的回调"""
        self.part_manager.on_geometry_visibility_changed(element_type, visible)

    def on_geometry_element_visibility_changed(self, element_type, element_index, visible):
        """单个几何元素可见性改变时的回调"""
        self.part_manager.on_geometry_element_visibility_changed(element_type, element_index, visible)

    def on_mesh_part_visibility_changed(self, visible):
        """网格部件类别可见性改变时的回调"""
        self.part_manager.on_mesh_part_visibility_changed(visible)

    def on_mesh_part_element_visibility_changed(self, part_index, visible):
        """单个网格部件可见性改变时的回调"""
        self.part_manager.on_mesh_part_element_visibility_changed(part_index, visible)

    def on_mesh_part_selected(self, part_data, part_index):
        """网格部件被选中时的回调"""
        self.part_manager.on_mesh_part_selected(part_data, part_index)

    def _update_geometry_element_display(self):
        """更新几何元素的显示"""
        self.part_manager._update_geometry_element_display()

    def _update_mesh_part_display(self):
        """更新网格部件的显示"""
        self.part_manager._update_mesh_part_display()

    def _update_geometry_display_for_parts(self, visible_parts):
        """根据可见部件更新几何元素的显示"""
        self.part_manager._update_geometry_display_for_parts(visible_parts)

    def on_geometry_element_selected(self, element_type, element_data, element_index):
        """几何元素被选中时的回调"""
        self.part_manager.on_geometry_element_selected(element_type, element_data, element_index)

    def on_model_tree_visibility_changed(self, *args):
        """
        模型树可见性改变的回调
        
        Args:
            可以是以下几种形式:
            - (category, visible): 整个类别的可见性改变
            - (category, element_type, visible): 类别下特定元素类型的可见性改变
            - (category, element_type, element_index, visible): 特定元素的可见性改变
        """
        self.part_manager.on_model_tree_visibility_changed(*args)

    def on_model_tree_selected(self, category, element_type, element_index, element_obj):
        """
        模型树元素被选中的回调
        
        Args:
            category: 类别（"geometry", "mesh", "parts"）
            element_type: 元素类型（"vertices", "edges", "faces", "bodies"）
            element_index: 元素索引
            element_obj: 元素对象
        """
        self.part_manager.on_model_tree_selected(category, element_type, element_index, element_obj)

    def on_mesh_element_selected(self, element_type, element_data, element_index):
        """网格元素被选中时的回调"""
        self.part_manager.on_mesh_element_selected(element_type, element_data, element_index)

    def on_part_selected(self, element_type, element_data, element_index):
        """部件被选中时的回调"""
        self.part_manager.on_part_selected(element_type, element_data, element_index)

    def _get_edge_length(self, edge):
        """获取边的长度"""
        return self.part_manager._get_edge_length(edge)

    def _get_face_area(self, face):
        """获取面的面积"""
        return self.part_manager._get_face_area(face)

    def _get_solid_volume(self, solid):
        """获取体的体积"""
        return self.part_manager._get_solid_volume(solid)

    def show_about(self):
        """显示关于对话框"""
        self.help_module.show_about()

    def show_user_manual(self):
        """显示用户手册 - 打开UserGuide.pdf或UserGuide.md文件"""
        self.help_module.show_user_manual()

    def zoom_in(self):
        """放大视图"""
        self.view_controller.zoom_in()

    def zoom_out(self):
        """缩小视图"""
        self.view_controller.zoom_out()

    def toggle_toolbar(self):
        """切换功能区显示"""
        self.view_controller.toggle_toolbar()

    def toggle_statusbar(self):
        """切换状态栏显示"""
        self.view_controller.toggle_statusbar()

    def edit_params(self):
        """编辑全局参数"""
        self.ui_helpers.edit_params()

    def edit_mesh_params(self):
        """编辑部件参数"""
        self.part_manager.edit_mesh_params()

    def edit_boundary_conditions(self):
        """编辑边界条件"""
        self.mesh_operations.edit_boundary_conditions()

    def import_config(self):
        """导入配置"""
        self.config_manager.import_config()

    def export_config(self):
        """导出配置"""
        self.config_manager.export_config()

    def reset_config(self):
        """重置配置"""
        self.config_manager.reset_config()

    def generate_mesh(self):
        """生成网格 - 使用异步线程避免UI冻结"""
        self.mesh_operations.generate_mesh()

    def _update_parts_list_from_generated_mesh(self, generated_mesh):
        """从生成的网格更新部件列表"""
        try:
            # 从新生成的网格中提取实际的部件信息
            updated_parts_info = {}

            # 获取新网格的节点坐标
            new_node_coords = None
            if hasattr(generated_mesh, 'node_coords'):
                new_node_coords = generated_mesh.node_coords

            # 创建节点索引映射（如果存在原始节点坐标）
            node_mapping = None
            if self.original_node_coords and new_node_coords:
                node_mapping = self._create_node_index_mapping(
                    self.original_node_coords,
                    new_node_coords
                )
                if node_mapping:
                    self.log_info(f"已创建节点索引映射，映射了 {len(node_mapping)} 个节点")

            # 首先，尝试从原始导入的部件信息中获取边界部件信息（这些应该保留原始名称如"wall"）
            # 使用节点映射将原始部件信息映射到新网格
            if hasattr(self, 'cas_parts_info') and self.cas_parts_info:
                if node_mapping:
                    # 使用映射后的部件信息
                    mapped_parts_info = self._map_parts_info_to_new_mesh(
                        self.cas_parts_info,
                        node_mapping
                    )
                    for part_name, part_data in mapped_parts_info.items():
                        if part_name not in ['type', 'node_coords', 'cells', 'num_points', 'num_cells', 'unstr_grid']:
                            updated_parts_info[part_name] = part_data
                else:
                    # 如果没有映射，直接使用原始部件信息
                    for part_name, part_data in self.cas_parts_info.items():
                        if part_name not in ['type', 'node_coords', 'cells', 'num_points', 'num_cells', 'unstr_grid']:
                            updated_parts_info[part_name] = part_data

            # 方法1: 从网格的单元中提取部件信息
            if hasattr(generated_mesh, 'cell_container') and generated_mesh.cell_container:
                for cell in generated_mesh.cell_container:
                    part_name = getattr(cell, 'part_name', 'interior')
                    if part_name is None or part_name == '':
                        part_name = 'interior'

                    if part_name not in updated_parts_info:
                        updated_parts_info[part_name] = {
                            'part_name': part_name,
                            'bc_type': 'interior',
                            'node_count': 0,
                            'nodes': []
                        }

                    if part_name in updated_parts_info:
                        updated_parts_info[part_name]['node_count'] += 1

            # 方法2: 如果有parts_info属性，也合并进来
            if hasattr(generated_mesh, 'parts_info') and generated_mesh.parts_info:
                for part_name, part_data in generated_mesh.parts_info.items():
                    if part_name not in ['type', 'node_coords', 'cells', 'num_points', 'num_cells', 'unstr_grid']:
                        if part_name not in updated_parts_info:
                            updated_parts_info[part_name] = part_data
                        else:
                            if isinstance(part_data, dict):
                                updated_parts_info[part_name].update(part_data)

            # 方法3: 如果没有从单元中提取到信息，从边界节点提取
            if hasattr(generated_mesh, 'boundary_nodes') and generated_mesh.boundary_nodes:
                extracted_parts = self._extract_parts_from_boundary_nodes(generated_mesh.boundary_nodes)
                if extracted_parts:
                    for part_name, part_data in extracted_parts.items():
                        if (part_name not in updated_parts_info):
                            updated_parts_info[part_name] = part_data

            if updated_parts_info:
                # 更新cas_parts_info以包含新网格的实际部件信息
                # self.cas_parts_info = updated_parts_info
                # 更新部件列表显示
                self.update_parts_list_from_cas(updated_parts_info)
                self.log_info(f"已更新部件列表，检测到 {len(updated_parts_info)} 个部件: {list(updated_parts_info.keys())}")

                # Refresh display to show all parts with different colors
                self.refresh_display_all_parts()
            else:
                # 如果没有部件信息，保持原有部件列表
                self.log_info("生成的网格中未检测到部件信息，保持原有部件列表")

        except Exception as e:
            self.log_error(f"更新部件列表失败: {str(e)}")

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
            if part_name in ['type', 'node_coords', 'cells', 'num_points', 'num_cells', 'unstr_grid']:
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

    def check_mesh_quality(self):
        """检查网格质量 - 显示网格质量skewness直方图"""
        self.mesh_operations.check_mesh_quality()

    def smooth_mesh(self):
        """平滑网格 - 使用laplacian光滑算法"""
        self.mesh_operations.smooth_mesh()

    def optimize_mesh(self):
        """优化网格 - 使用edge_swap和laplacian_smooth算法"""
        self.mesh_operations.optimize_mesh()

    def show_mesh_statistics(self):
        """显示网格统计信息 - 包括网格单元信息和质量统计"""
        self.mesh_operations.show_mesh_statistics()

    def extract_boundary_mesh_info(self):
        """提取边界网格及部件信息"""
        self.mesh_operations.extract_boundary_mesh_info()


    def export_mesh_report(self):
        """导出网格报告 - 将网格生成的主要参数、部件参数和生成结果写到md文档中"""
        self.mesh_operations.export_mesh_report()


    def show_quick_start(self):
        """显示快速入门"""
        self.help_module.show_quick_start()

    def check_for_updates(self):
        """检查更新"""
        self.help_module.check_for_updates()

    def show_shortcuts(self):
        """显示快捷键"""
        self.help_module.show_shortcuts()

    def set_background_color(self):
        """设置视图区背景色"""
        self.view_controller.set_background_color()

    def toggle_fullscreen(self):
        """切换全屏显示"""
        self.view_controller.toggle_fullscreen()

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

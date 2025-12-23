#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重构后的PyQt GUI主模块
整合各个功能模块，提供统一的用户界面
"""

import os
import sys
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QGroupBox, QLabel, QTextEdit, QPushButton, 
                             QListWidget, QTabWidget, QFrame, QMenuBar, QStatusBar, 
                             QToolBar, QAction, QFileDialog, QMessageBox, QScrollArea,
                             QDockWidget, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加必要的子目录到路径 (for compatibility with main script approach)
data_structure_dir = os.path.join(project_root, 'data_structure')
fileio_dir = os.path.join(project_root, 'fileIO')
meshsize_dir = os.path.join(project_root, 'meshsize')
visualization_dir = os.path.join(project_root, 'visualization')
adfront2_dir = os.path.join(project_root, 'adfront2')
optimize_dir = os.path.join(project_root, 'optimize')
utils_dir = os.path.join(project_root, 'utils')

for dir_path in [data_structure_dir, fileio_dir, meshsize_dir, visualization_dir, adfront2_dir, optimize_dir, utils_dir]:
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

# 导入自定义模块
from pyqt_gui.gui_base import BaseWidget, MenuBar, StatusBar, InfoOutput, DialogBase, ToolBar, Splitter, PartListWidget
from pyqt_gui.mesh_display import MeshDisplayArea
from parameters import Parameters
from PyMeshGen import PyMeshGen as MeshGenerator


class SimplifiedPyMeshGenGUI(QMainWindow):
    """PyQt版的PyMeshGen GUI主类"""
    
    def __init__(self):
        super().__init__()
        self.setup_window()
        self.setup_fonts()
        self.initialize_modules()
        self.initialize_data()
        self.create_widgets()
        self.update_status("就绪")
        
    def setup_window(self):
        """设置窗口大小和标题"""
        self.setWindowTitle("PyMeshGen - 高性能网格生成工具 (PyQt)")
        
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        screen_width = screen_size.width()
        screen_height = screen_size.height()
        
        # 根据屏幕分辨率设置窗口大小
        if screen_width >= 1920 and screen_height >= 1080:
            # 高分辨率屏幕
            window_width = int(screen_width * 0.75)
            window_height = int(screen_height * 0.75)
        elif screen_width >= 1366 and screen_height >= 768:
            # 中等分辨率屏幕
            window_width = int(screen_width * 0.85)
            window_height = int(screen_height * 0.85)
        else:
            # 低分辨率屏幕
            window_width = int(screen_width * 0.95)
            window_height = int(screen_height * 0.95)
        
        # 设置最小窗口大小
        min_width = 1024
        min_height = 768
        
        # 确保窗口大小不小于最小值
        window_width = max(window_width, min_width)
        window_height = max(window_height, min_height)
        
        self.resize(window_width, window_height)
        
        # 居中窗口
        frame_geometry = self.frameGeometry()
        center_point = screen.availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())
        
        # 设置最小窗口大小
        self.setMinimumSize(min_width, min_height)
    
    def setup_fonts(self):
        """设置字体"""
        # 应用系统默认字体
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)
    
    def initialize_modules(self):
        """初始化模块"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def initialize_data(self):
        """初始化数据"""
        self.params = None
        self.mesh_generator = None
        self.current_mesh = None
        self.cas_parts_info = None  # 初始化cas_parts_info属性
        # 初始化视图控制变量
        self.render_mode = "surface"
        self.show_boundary = True
    
    def create_widgets(self):
        """创建UI组件"""
        # 创建菜单栏
        self.create_menu()

        # 创建工具栏（位于菜单栏下方）
        self.create_toolbar()

        # 创建中央小工具
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建左右两栏布局（3:7比例）
        self.paned_window = QSplitter(Qt.Horizontal)
        
        # 左侧部件信息区域（3/10宽度）
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        
        # Initialize the part list widget first
        self.create_left_panel()
        
        # 部件列表分组
        parts_frame_container = QGroupBox("部件列表")
        parts_layout = QVBoxLayout(parts_frame_container)
        
        # 部件列表带滚动条
        parts_layout.addWidget(self.parts_list_widget.widget)
        
        # 部件操作按钮
        parts_button_layout = QHBoxLayout()
        add_part_btn = QPushButton("添加")
        add_part_btn.clicked.connect(self.add_part)
        remove_part_btn = QPushButton("删除")
        remove_part_btn.clicked.connect(self.remove_part)
        edit_part_btn = QPushButton("编辑")
        edit_part_btn.clicked.connect(self.edit_part)
        
        parts_button_layout.addWidget(add_part_btn)
        parts_button_layout.addWidget(remove_part_btn)
        parts_button_layout.addWidget(edit_part_btn)
        parts_layout.addLayout(parts_button_layout)
        
        left_layout.addWidget(parts_frame_container)
        left_layout.addWidget(self.props_frame)
        self.left_panel.setMinimumWidth(300)
        
        # 连接部件列表选择事件
        self.parts_list_widget.parts_list.currentRowChanged.connect(self.on_part_select)
        
        self.paned_window.addWidget(self.left_panel)
        
        # 右侧网格视图交互区域（7/10宽度）
        right_main_widget = QWidget()
        right_main_layout = QVBoxLayout(right_main_widget)
        
        # 在右侧区域中创建垂直分割窗格（网格显示和状态输出）
        self.right_paned = QSplitter(Qt.Vertical)
        
        # 上半部分：网格视图交互区域
        self.right_panel = QWidget()
        right_panel_layout = QVBoxLayout(self.right_panel)
        self.create_right_panel()
        right_panel_layout.addWidget(self.main_mesh_display.frame)
        self.right_paned.addWidget(self.right_panel)
        
        # 下半部分：状态输出面板区域
        self.bottom_frame = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_frame)
        self.create_status_output_panel()
        bottom_layout.addWidget(self.status_output_paned)
        self.right_paned.addWidget(self.bottom_frame)
        
        right_main_layout.addWidget(self.right_paned)
        self.paned_window.addWidget(right_main_widget)
        
        # 设置拉伸比例
        self.paned_window.setStretchFactor(0, 3)
        self.paned_window.setStretchFactor(1, 7)
        
        # 设置右侧部分的拉伸
        self.right_paned.setStretchFactor(0, 1)
        self.right_paned.setStretchFactor(1, 0)
        
        main_layout.addWidget(self.paned_window)
        
        # 创建状态栏
        self.status_bar = StatusBar(self)
    
    def create_toolbar(self):
        """创建工具栏"""
        self.toolbar = ToolBar(self)
        
        # 导入Qt图标
        from PyQt5.QtGui import QIcon
        from PyQt5.QtWidgets import QActionGroup, QAction
        
        # 获取图标路径
        icon_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui", "icons")
        
        # 创建工具栏按钮，使用现有图标
        def get_icon(icon_name):
            icon_path = os.path.join(icon_dir, f"{icon_name}.png")
            if os.path.exists(icon_path):
                return QIcon(icon_path)
            return None
        
        # 文件操作按钮
        self.toolbar.add_action("新建", get_icon("new"), self.new_config, "创建新的网格配置\n快捷键: Ctrl+N")
        self.toolbar.add_action("打开", get_icon("open"), self.open_config, "打开已保存的网格配置文件\n支持格式: .json, .cfg\n快捷键: Ctrl+O")
        self.toolbar.add_action("保存", get_icon("save"), self.save_config, "保存当前网格配置到文件\n格式: .json\n快捷键: Ctrl+S")
        self.toolbar.add_separator()
        self.toolbar.add_action("导入", get_icon("import"), self.import_mesh, "从外部文件导入网格数据\n支持格式: .vtk, .cas, .msh\n快捷键: Ctrl+I")
        self.toolbar.add_action("导出", get_icon("export"), self.export_mesh, "将当前网格导出到文件\n支持格式: .vtk, .obj, .stl\n快捷键: Ctrl+E")
        self.toolbar.add_separator()
        
        # 网格操作按钮
        self.toolbar.add_action("生成", get_icon("generate"), self.generate_mesh, "根据当前配置生成网格\n支持三角形、四边形和混合网格\n快捷键: F5")
        self.toolbar.add_action("显示", get_icon("display"), self.display_mesh, "在显示区域显示当前网格\n支持缩放、旋转和平移操作\n快捷键: F6")
        self.toolbar.add_action("清空", get_icon("clear"), self.clear_mesh, "清空当前网格数据\n注意: 此操作不可撤销\n快捷键: Delete")
        self.toolbar.add_separator()
        
        # 显示模式切换按钮组
        self.display_mode_group = QActionGroup(self)
        self.display_mode_group.setExclusive(True)
        
        # 实体模式
        self.surface_mode_action = QAction("实体模式", self)
        self.surface_mode_action.setCheckable(True)
        self.surface_mode_action.setChecked(True)
        self.surface_mode_action.setToolTip("实体模式：仅显示模型表面 (1键)")
        self.surface_mode_action.triggered.connect(lambda: self.set_render_mode("surface"))
        self.display_mode_group.addAction(self.surface_mode_action)
        self.toolbar.toolbar.addAction(self.surface_mode_action)
        
        # 线框模式
        self.wireframe_mode_action = QAction("线框模式", self)
        self.wireframe_mode_action.setCheckable(True)
        self.wireframe_mode_action.setToolTip("线框模式：仅显示模型边缘线条 (2键)")
        self.wireframe_mode_action.triggered.connect(lambda: self.set_render_mode("wireframe"))
        self.display_mode_group.addAction(self.wireframe_mode_action)
        self.toolbar.toolbar.addAction(self.wireframe_mode_action)
        
        # 混合模式
        self.mixed_mode_action = QAction("混合模式", self)
        self.mixed_mode_action.setCheckable(True)
        self.mixed_mode_action.setToolTip("混合模式：同时显示模型表面和边缘线条 (3键)")
        self.mixed_mode_action.triggered.connect(lambda: self.set_render_mode("mixed"))
        self.display_mode_group.addAction(self.mixed_mode_action)
        self.toolbar.toolbar.addAction(self.mixed_mode_action)
    
    def create_menu(self):
        """创建菜单"""
        # 创建菜单栏对象
        self.menu_bar = MenuBar(self)
        
        # 文件菜单
        file_commands = {
            "新建配置": self.new_config,
            "打开配置": self.open_config,
            "保存配置": self.save_config,
            "另存为": self.save_config_as,
            "---": None,
            "导入网格": self.import_mesh,
            "导出网格": self.export_mesh,
            "---": None,
            "最近文件": self.show_recent_files,
            "---": None,
            "退出": self.close
        }
        self.menu_bar.create_file_menu(file_commands)
        
        # 视图菜单
        view_commands = {
            "重置视图": self.reset_view,
            "适应视图": self.fit_view,
            "---": None,
            "放大": self.zoom_in,
            "缩小": self.zoom_out,
            "平移": self.pan_view,
            "旋转": self.rotate_view,
            "---": None,
            "显示工具栏": self.toggle_toolbar,
            "显示状态栏": self.toggle_statusbar,
            "显示部件列表": self.toggle_parts_list,
            "显示属性面板": self.toggle_properties_panel,
            "---": None,
            "全屏显示": self.toggle_fullscreen
        }
        self.menu_bar.create_view_menu(view_commands)
        
        # 配置菜单
        config_commands = {
            "参数设置": self.edit_params,
            "网格参数": self.edit_mesh_params,
            "边界条件": self.edit_boundary_conditions,
            "---": None,
            "导入配置": self.import_config,
            "导出配置": self.export_config,
            "---": None,
            "清空网格": self.clear_mesh,
            "重置配置": self.reset_config
        }
        self.menu_bar.create_config_menu(config_commands)
        
        # 网格菜单
        mesh_commands = {
            "生成网格": self.generate_mesh,
            "显示网格": self.display_mesh,
            "网格质量": self.check_mesh_quality,
            "---": None,
            "平滑网格": self.smooth_mesh,
            "优化网格": self.optimize_mesh,
            "---": None,
            "网格统计": self.show_mesh_statistics,
            "导出报告": self.export_mesh_report
        }
        self.menu_bar.create_mesh_menu(mesh_commands)
        
        # 帮助菜单
        help_commands = {
            "用户手册": self.show_user_manual,
            "快速入门": self.show_quick_start,
            "---": None,
            "快捷键": self.show_shortcuts,
            "检查更新": self.check_for_updates,
            "---": None,
            "关于": self.show_about
        }
        self.menu_bar.create_help_menu(help_commands)
    
    def create_left_panel(self):
        """创建左侧部件信息区域（分组更清晰，带滚动）"""
        # 部件列表分组
        self.parts_list_widget = PartListWidget()
        
        # 属性面板分组
        self.props_frame = QGroupBox("属性面板")
        props_layout = QVBoxLayout()
        
        self.props_text = QTextEdit()
        self.props_text.setReadOnly(True)
        props_layout.addWidget(self.props_text)
        self.props_frame.setLayout(props_layout)
    
    def create_right_panel(self):
        """创建右侧网格视图交互区域（最优化版，移除所有标签，仅保留纯净的网格显示）"""
        # 创建网格显示区域
        self.main_mesh_display = MeshDisplayArea(self.right_panel)
        # 这里我们 need to get the actual frame that the MeshDisplayArea creates
        # Store reference to the actual VTK widget
        self.mesh_display = self.main_mesh_display
        
        # Set up keyboard event handling for the VTK widget
        # For now we'll just ensure the widget can receive focus
        self.main_mesh_display.frame.setFocusPolicy(Qt.StrongFocus)
        
        # Add keyboard event handling
        self.main_mesh_display.frame.keyPressEvent = self.on_mesh_display_key
    
    def set_render_mode(self, mode):
        """设置渲染模式"""
        self.render_mode = mode
        if hasattr(self, 'mesh_display'):
            self.mesh_display.set_render_mode(mode)
        
        # 更新UI状态
        if mode == "surface":
            self.surface_mode_action.setChecked(True)
            self.update_status("渲染模式: 实体模式 (1键)")
        elif mode == "wireframe":
            self.wireframe_mode_action.setChecked(True)
            self.update_status("渲染模式: 线框模式 (2键)")
        elif mode == "mixed" or mode == "surface-wireframe":
            self.mixed_mode_action.setChecked(True)
            self.update_status("渲染模式: 混合模式 (3键)")
        
    def on_mesh_display_key(self, event):
        """处理网格显示区域的键盘事件"""
        key = event.key()
        if key == Qt.Key_R:  # 重置视图
            self.reset_view()
            self.update_status("已重置视图 (R键)")
        elif key == Qt.Key_F:  # 适应视图
            self.fit_view()
            self.update_status("已适应视图 (F键)")
        elif key == Qt.Key_O:  # 切换边界显示
            new_state = not self.show_boundary
            self.show_boundary = new_state
            self.mesh_display.toggle_boundary_display(new_state)
            self.update_status(f"边界显示: {'开启' if new_state else '关闭'} (O键)")
        elif key == Qt.Key_1:  # 切换到实体模式
            self.set_render_mode("surface")
        elif key == Qt.Key_2:  # 切换到线框模式
            self.set_render_mode("wireframe")
        elif key == Qt.Key_3:  # 切换到混合模式
            self.set_render_mode("mixed")
        elif key == Qt.Key_4:  # 切换到点云模式
            self.render_mode = "points"
            self.mesh_display.set_render_mode("points")
            self.update_status("渲染模式: 点云 (4键)")
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:  # 放大
            self.zoom_in()
            self.update_status("视图已放大 (+键)")
        elif key == Qt.Key_Minus:  # 缩小
            self.zoom_out()
            self.update_status("视图已缩小 (-键)")
        else:
            # Call the original event handler for other keys
            QWidget.keyPressEvent(self.main_mesh_display.frame, event)
    
    def create_status_output_panel(self):
        """创建状态栏和信息输出面板（左右4:6布局，信息输出区带清空按钮）"""
        # 创建左右分割的状态和输出面板
        self.status_output_paned = QSplitter(Qt.Horizontal)
        
        # 左侧状态显示区域（4/10宽度）
        self.status_panel = QGroupBox("状态信息")
        status_layout = QVBoxLayout(self.status_panel)
        
        # 创建状态文本框和滚动条
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        
        # 添加到分割器
        self.status_output_paned.addWidget(self.status_panel)
        
        # 右侧信息输出区域（6/10宽度）
        self.info_output = InfoOutput(self.status_output_paned)
        self.status_output_paned.addWidget(self.info_output.frame)
        
        # 设置拉伸比例
        self.status_output_paned.setStretchFactor(0, 4)
        self.status_output_paned.setStretchFactor(1, 6)
    
    def update_status(self, message):
        """更新状态栏和状态文本框"""
        # 更新状态栏
        if hasattr(self, 'status_bar'):
            self.status_bar.update_status(message)
        
        # 更新状态文本框
        if hasattr(self, 'status_text'):
            # 添加时间戳
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_text.append(f"[{timestamp}] {message}")
    
    def log_info(self, message):
        """记录信息"""
        if hasattr(self, 'info_output'):
            self.info_output.append_info_output(f"[INFO] {message}")
    
    def log_warning(self, message):
        """记录警告"""
        if hasattr(self, 'info_output'):
            self.info_output.append_info_output(f"[WARNING] {message}")
    
    def log_error(self, message):
        """记录错误"""
        if hasattr(self, 'info_output'):
            self.info_output.append_info_output(f"[ERROR] {message}")
    
    def new_config(self):
        """新建配置"""
        # 创建默认配置字典
        default_config = {
            "debug_level": 0,
            "input_file": "",
            "output_file": "",
            "mesh_type": 1,
            "viz_enabled": True,
            "parts": []
        }
        
        # 创建临时配置文件
        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(default_config, f, indent=2)
            temp_file = f.name
        
        # 使用临时文件创建参数对象
        self.params = Parameters("FROM_CASE_JSON", temp_file)
        
        # 删除临时文件
        import os
        os.unlink(temp_file)
        
        self.update_params_display()
        self.log_info("已创建新配置")
        self.update_status("已创建新配置")
    
    def open_config(self):
        """打开配置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择配置文件",
            "",
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                # For now, just create a basic parameters object
                self.params = Parameters("FROM_CASE_JSON", file_path)
                self.update_params_display()
                self.log_info(f"已加载配置文件: {file_path}")
                self.update_status("已加载配置文件")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载配置文件失败: {str(e)}")
                self.log_error(f"加载配置文件失败: {str(e)}")
    
    def save_config(self):
        """保存配置"""
        if not self.params:
            QMessageBox.warning(self, "警告", "没有可保存的配置")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存配置文件",
            "",
            "JSON文件 (*.json);;所有文件 (*.*)",
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                # For now, just save a basic config
                config = {
                    "debug_level": getattr(self.params, 'debug_level', 0),
                    "input_file": getattr(self.params, 'input_file', ""),
                    "output_file": getattr(self.params, 'output_file', ""),
                    "mesh_type": getattr(self.params, 'mesh_type', 1),
                    "viz_enabled": getattr(self.params, 'viz_enabled', True)
                }
                
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                self.log_info(f"已保存配置文件: {file_path}")
                self.update_status("已保存配置文件")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存配置文件失败: {str(e)}")
                self.log_error(f"保存配置文件失败: {str(e)}")
    
    def toggle_boundary_display(self):
        """切换边界显示"""
        if hasattr(self, 'mesh_display'):
            self.mesh_display.toggle_boundary_display(self.show_boundary)
            self.update_status(f"边界显示: {'开启' if self.show_boundary else '关闭'}")
        else:
            self.update_status("边界显示功能不可用")
    
    def toggle_parts_list(self):
        """切换部件列表显示"""
        if hasattr(self, 'left_panel'):
            if self.left_panel.isVisible():
                self.left_panel.hide()
                self.update_status("部件列表已隐藏")
            else:
                self.left_panel.show()
                self.update_status("部件列表已显示")
                
    def toggle_properties_panel(self):
        """切换属性面板显示"""
        if hasattr(self, 'props_frame'):
            if self.props_frame.isVisible():
                self.props_frame.hide()
                self.update_status("属性面板已隐藏")
            else:
                self.props_frame.show()
                self.update_status("属性面板已显示")
    
    def edit_params(self):
        """编辑参数"""
        if not self.params:
            QMessageBox.warning(self, "警告", "请先创建或加载配置")
            return
            
        try:
            # 创建配置管理器
            from gui.config_manager import ConfigManager
            config_manager = ConfigManager(self.project_root)
            
            # 从参数创建配置
            config = config_manager.create_config_from_params(self.params)
            
            # 创建配置对话框
            from pyqt_gui.gui_base import ConfigDialog
            dialog = ConfigDialog(self, config)
            
            # 显示对话框
            if dialog.dialog.exec_() == dialog.dialog.Accepted:
                # 应用修改后的配置
                new_config = dialog.result
                if new_config:
                    # 从新配置创建参数对象
                    self.params = config_manager.create_params_from_config(new_config)
                    self.update_params_display()
                    self.log_info("已更新参数配置")
                    self.update_status("已更新参数配置")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"编辑参数失败: {str(e)}")
            self.log_error(f"编辑参数失败: {str(e)}")
    
    def generate_mesh(self):
        """生成网格"""
        if not self.params:
            QMessageBox.warning(self, "警告", "请先创建或加载配置")
            return
            
        try:
            self.update_status("正在生成网格...")
            self.log_info("开始生成网格")
            
            # 创建网格生成器实例
            from PyMeshGen import PyMeshGen as MeshGenerator
            self.mesh_generator = MeshGenerator(self.params)
            
            # 生成网格
            self.current_mesh = self.mesh_generator.generate()
            
            # 获取网格信息用于日志记录
            node_count = 0
            element_count = 0
            if hasattr(self.current_mesh, 'nodes') and hasattr(self.current_mesh, 'elements'):
                node_count = len(self.current_mesh.nodes)
                element_count = len(self.current_mesh.elements)
                self.log_info(f"生成网格: 节点数={node_count}, 单元数={element_count}")
            elif isinstance(self.current_mesh, dict):
                node_count = self.current_mesh.get('num_points', 0)
                element_count = self.current_mesh.get('num_cells', 0)
                self.log_info(f"生成网格: 节点数={node_count}, 单元数={element_count}")
            
            self.log_info("网格生成完成")
            self.update_status("网格生成完成")
            
            # 自动显示生成的网格
            self.display_mesh()
            
        except ImportError as e:
            QMessageBox.critical(self, "错误", f"导入PyMeshGen模块失败: {str(e)}")
            self.log_error(f"导入PyMeshGen模块失败: {str(e)}")
            self.update_status("导入PyMeshGen模块失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成网格失败: {str(e)}")
            self.log_error(f"生成网格失败: {str(e)}")
            self.update_status("生成网格失败")
    
    def display_mesh(self):
        """显示网格"""
        # Check if we have a generated mesh or if we need to load from output file
        mesh_to_display = None
        if hasattr(self, 'current_mesh') and self.current_mesh:
            mesh_to_display = self.current_mesh
        elif hasattr(self, 'mesh_data') and self.mesh_data:  # Check for mesh_data set by PyMeshGen
            mesh_to_display = self.mesh_data
        elif self.params and hasattr(self.params, 'output_file') and self.params.output_file:
            # Try to load mesh from output file
            try:
                from fileIO.vtk_io import parse_vtk_msh
                mesh_to_display = parse_vtk_msh(self.params.output_file)
            except Exception as e:
                self.log_error(f"无法从输出文件加载网格: {str(e)}")
        
        if not mesh_to_display:
            QMessageBox.warning(self, "警告", "请先生成网格或导入网格文件")
            return
            
        try:
            self.update_status("正在显示网格...")
            self.log_info("开始显示网格")
            
            # Display mesh in the VTK widget
            if hasattr(self, 'mesh_display'):
                self.mesh_display.display_mesh(mesh_to_display)
            
            self.log_info("网格显示完成")
            self.update_status("网格显示完成")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示网格失败: {str(e)}")
            self.log_error(f"显示网格失败: {str(e)}")
            self.update_status("显示网格失败")
    
    def import_mesh(self):
        """导入网格"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择网格文件",
            "",
            "所有支持的文件 (*.vtk *.stl *.obj *.ply *.cas);;VTK文件 (*.vtk);;STL文件 (*.stl);;OBJ文件 (*.obj);;PLY文件 (*.ply);;CAS文件 (*.cas);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                # 创建文件操作对象
                from gui.file_operations import FileOperations
                file_ops = FileOperations(self.project_root)
                
                # 导入网格文件
                self.current_mesh = file_ops.import_mesh(file_path)
                
                # Update mesh display area with mesh data
                if hasattr(self, 'mesh_display'):
                    self.mesh_display.set_mesh_data(self.current_mesh)
                    self.mesh_display.display_mesh()
                
                # Get grid info
                node_count = self.current_mesh.get('num_points', 0)
                element_count = self.current_mesh.get('num_cells', 0)
                
                # Log grid info 
                self.log_info(f"导入网格: 节点数={node_count}, 单元数={element_count}")
                
                # 更新部件列表（如果是CAS文件）
                if self.current_mesh.get('type') == 'cas' and 'parts_info' in self.current_mesh:
                    self.update_parts_list_from_cas(self.current_mesh['parts_info'])
                
                self.log_info(f"已导入网格文件: {file_path}")
                self.update_status("已导入网格文件")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入网格失败: {str(e)}")
                self.log_error(f"导入网格失败: {str(e)}")
                self.update_status("导入网格失败")
    
    def export_mesh(self):
        """导出网格"""
        if not self.current_mesh:
            QMessageBox.warning(self, "警告", "没有可导出的网格")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存网格文件",
            "",
            "所有支持的文件 (*.vtk *.stl *.obj *.ply);;VTK文件 (*.vtk);;STL文件 (*.stl);;OBJ文件 (*.obj);;PLY文件 (*.ply);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                # 创建文件操作对象
                from gui.file_operations import FileOperations
                file_ops = FileOperations(self.project_root)
                
                # 获取要导出的VTK PolyData对象
                vtk_poly_data = None
                if isinstance(self.current_mesh, dict):
                    # 如果是字典类型，尝试获取vtk_poly_data
                    vtk_poly_data = self.current_mesh.get('vtk_poly_data')
                
                if vtk_poly_data:
                    # 导出网格文件
                    file_ops.export_mesh(vtk_poly_data, file_path)
                else:
                    # 如果没有vtk_poly_data，检查是否是Unstructured_Grid对象
                    if hasattr(self.current_mesh, 'node_coords') and hasattr(self.current_mesh, 'cell_container'):
                        # 对于Unstructured_Grid对象，使用save_to_vtkfile方法（如果可用）
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
            
        # For now, just add a default part
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
        """编辑部件"""
        if not self.params:
            QMessageBox.warning(self, "警告", "请先创建或加载配置")
            return
            
        current_row = self.parts_list_widget.parts_list.currentRow()
        if current_row >= 0:
            part_name = self.parts_list_widget.parts_list.item(current_row).text()
            QMessageBox.information(self, "信息", f"编辑部件: {part_name}")
        else:
            QMessageBox.warning(self, "警告", "请先选择要编辑的部件")
    
    def update_params_display(self):
        """更新参数显示"""
        if not self.params:
            return
            
        # Update parts list
        self.update_parts_list()
    
    def update_parts_list(self):
        """更新部件列表"""
        # Clear the list
        self.parts_list_widget.parts_list.clear()
        
        if self.params:
            # Add actual parts from params object
            if hasattr(self.params, 'part_params') and self.params.part_params:
                for i, part in enumerate(self.params.part_params):
                    # Get part name from part_params if available
                    part_name = getattr(part, 'part_name', f"部件{i+1}")
                    if hasattr(part, 'part_params') and hasattr(part.part_params, 'part_name'):
                        part_name = part.part_params.part_name
                    self.parts_list_widget.parts_list.addItem(part_name)
            else:
                # If no parts available, add a default part
                self.parts_list_widget.parts_list.addItem("默认部件")
    
    def update_parts_list_from_cas(self, parts_info):
        """从cas文件的部件信息更新部件列表"""
        # Clear the list
        self.parts_list_widget.parts_list.clear()
        
        # Store cas parts info for selection
        self.cas_parts_info = parts_info
        
        # Add parts to list
        for part_info in parts_info:
            part_name = part_info.get('part_name', '未知部件')
            self.parts_list_widget.parts_list.addItem(part_name)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """PyMeshGen v1.0\n\n高性能网格生成工具\n\n© 2023 HighOrderMesh"""
        QMessageBox.about(self, "关于", about_text)
    
    def show_user_manual(self):
        """显示用户手册"""
        manual_text = """PyMeshGen 用户手册\n\n1. 文件菜单\n   - 新建配置：创建新的网格配置\n   - 打开配置：加载已保存的配置文件\n   - 保存配置：保存当前配置到文件\n   - 导入网格：从外部文件导入网格数据\n   - 导出网格：将当前网格导出到文件\n\n2. 视图菜单\n   - 重置视图：将视图恢复到初始状态\n   - 适应视图：自动调整视图以适应整个网格\n   - 放大/缩小：缩放网格显示\n   - 显示工具栏：切换工具栏的显示/隐藏\n   - 显示状态栏：切换状态栏的显示/隐藏\n\n3. 配置菜单\n   - 参数设置：配置网格生成参数\n   - 清空网格：清除当前显示的网格\n\n4. 网格菜单\n   - 生成网格：根据当前配置生成网格\n   - 显示网格：显示/隐藏网格\n\n5. 工具栏\n   - 提供常用功能的快速访问按钮\n\n6. 主界面\n   - 左侧：部件信息区域，包含部件列表和属性面板\n   - 右侧：网格视图交互区域，支持缩放、平移和选择操作\n   - 底部：状态栏显示系统状态和信息输出窗口"""
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
        """切换工具栏显示"""
        if hasattr(self, 'toolbar') and hasattr(self, 'toolbar.toolbar'):
            if self.toolbar.toolbar.isVisible():
                self.toolbar.toolbar.hide()
                self.update_status("工具栏已隐藏")
            else:
                self.toolbar.toolbar.show()
                self.update_status("工具栏已显示")
    
    def toggle_statusbar(self):
        """切换状态栏显示"""
        if hasattr(self, 'status_bar') and hasattr(self, 'status_bar.status_bar'):
            if self.status_bar.status_bar.isVisible():
                self.status_bar.status_bar.hide()
                self.log_info("状态栏已隐藏")
            else:
                self.status_bar.status_bar.show()
                self.log_info("状态栏已显示")
    
    def on_part_select(self, index):
        """处理部件列表选择事件"""
        if index < 0:
            # Clear properties panel
            self.props_text.setPlainText("未选择任何部件\n请从左侧列表中选择一个部件以查看其属性")
            self.update_status("未选择部件")
            return
            
        # Get selected part index
        part_index = index
        
        # Check if it's cas file parts
        if hasattr(self, 'cas_parts_info') and self.cas_parts_info and part_index < len(self.cas_parts_info):
            # Show cas part properties
            part_info = self.cas_parts_info[part_index]
            
            # Clear properties text
            props_content = f"=== CAS部件属性 ===\n\n"
            props_content += f"部件名称: {part_info.get('part_name', '未知')}\n"
            props_content += f"边界条件类型: {part_info.get('bc_type', '未知')}\n"
            props_content += f"面数量: {part_info.get('face_count', 0)}\n"
            props_content += f"节点数量: {len(part_info.get('nodes', []))}\n"
            props_content += f"单元数量: {len(part_info.get('cells', []))}\n"
            
            # Add status info
            props_content += f"\n=== 状态信息 ===\n"
            props_content += f"选择时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            props_content += f"部件索引: {part_index}\n"
            props_content += f"总部件数: {len(self.cas_parts_info)}\n"
            props_content += f"数据来源: CAS文件\n"
            
            self.props_text.setPlainText(props_content)
            self.update_status(f"已选中CAS部件: {part_info.get('part_name', f'部件{part_index}')}")
            return
        
        # If we have params, show selected part properties
        if hasattr(self, 'params') and self.params:
            try:
                # For now, just show dummy part info
                part_name = f"部件{part_index + 1}"
                
                # Clear properties text
                props_content = f"=== 部件属性 ===\n\n"
                props_content += f"名称: {part_name}\n"
                props_content += f"类型: 默认类型\n"
                props_content += f"ID: {part_index}\n"
                
                # Add status info
                props_content += f"\n=== 状态信息 ===\n"
                props_content += f"选择时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                props_content += f"部件索引: {part_index}\n"
                
                self.props_text.setPlainText(props_content)
                self.update_status(f"已选中部件: {part_name}")
            except Exception as e:
                self.log_error(f"显示部件属性时出错: {str(e)}")
                # In properties panel show error info
                error_content = f"=== 错误信息 ===\n\n"
                error_content += f"显示部件属性时出错:\n{str(e)}\n"
                error_content += f"\n请检查部件数据是否正确\n"
                self.props_text.setPlainText(error_content)
    
    def save_config_as(self):
        """配置另存为"""
        QMessageBox.information(self, "信息", "配置另存为功能")
        
    def show_recent_files(self):
        """显示最近文件"""
        QMessageBox.information(self, "信息", "最近文件功能")
        
    def pan_view(self):
        """平移视图"""
        self.update_status("视图平移模式")
        
    def rotate_view(self):
        """旋转视图"""
        self.update_status("视图旋转模式")
    
    def toggle_fullscreen(self):
        """切换全屏显示"""
        if self.isFullScreen():
            self.showNormal()
            self.update_status("退出全屏模式")
        else:
            self.showFullScreen()
            self.update_status("进入全屏模式")
    
    def edit_mesh_params(self):
        """编辑网格参数"""
        QMessageBox.information(self, "信息", "编辑网格参数功能")
        
    def edit_boundary_conditions(self):
        """编辑边界条件"""
        QMessageBox.information(self, "信息", "编辑边界条件功能")
        
    def import_config(self):
        """导入配置"""
        QMessageBox.information(self, "信息", "导入配置功能")
        
    def export_config(self):
        """导出配置"""
        QMessageBox.information(self, "信息", "导出配置功能")
        
    def reset_config(self):
        """重置配置"""
        QMessageBox.information(self, "信息", "重置配置功能")
        
    def check_mesh_quality(self):
        """检查网格质量"""
        QMessageBox.information(self, "信息", "检查网格质量功能")
        
    def smooth_mesh(self):
        """平滑网格"""
        QMessageBox.information(self, "信息", "平滑网格功能")
        
    def optimize_mesh(self):
        """优化网格"""
        QMessageBox.information(self, "信息", "优化网格功能")
        
    def show_mesh_statistics(self):
        """显示网格统计"""
        QMessageBox.information(self, "信息", "显示网格统计功能")
        
    def export_mesh_report(self):
        """导出网格报告"""
        QMessageBox.information(self, "信息", "导出网格报告功能")
        
    def show_quick_start(self):
        """显示快速入门"""
        QMessageBox.information(self, "信息", "快速入门功能")
        
    def show_shortcuts(self):
        """显示快捷键"""
        shortcuts_text = """常用快捷键：\n\nCtrl+N: 新建配置\nCtrl+O: 打开配置\nCtrl+S: 保存配置\nCtrl+I: 导入网格\nCtrl+E: 导出网格\nF5: 生成网格\nF6: 显示网格\nF11: 全屏显示\nEsc: 退出全屏"""
        QMessageBox.about(self, "快捷键", shortcuts_text)
        
    def check_for_updates(self):
        """检查更新"""
        QMessageBox.information(self, "信息", "检查更新功能")

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
            # Cleanup VTK resources
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
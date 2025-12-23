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
from pyqt_gui.gui_base import BaseWidget, StatusBar, InfoOutput, DialogBase, Splitter, PartListWidget
from pyqt_gui.ribbon_widget import RibbonWidget
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
        self.setWindowTitle("PyMeshGen - 基于Python的网格生成工具")

        # 设置窗口图标 if available
        try:
            import os
            icon_path = os.path.join(self.project_root, "gui", "icons", "app_icon.png")
            if os.path.exists(icon_path):
                from PyQt5.QtGui import QIcon
                self.setWindowIcon(QIcon(icon_path))
        except:
            pass  # 如果图标不存在则跳过

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
        min_width = 1200
        min_height = 800

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

        # 设置窗口透明度效果
        self.setAttribute(Qt.WA_TranslucentBackground, False)
    
    def setup_fonts(self):
        """设置字体"""
        # 应用系统默认字体
        font = QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)  # Slightly smaller font to prevent truncation
        font.setStyleHint(QFont.SansSerif)
        self.setFont(font)

        # Apply global stylesheet for consistent appearance
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QWidget {
                background-color: #f0f0f0;
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
                font-size: 9pt;  /* Slightly smaller font to prevent truncation */
                color: #333333;
            }
            QMenuBar {
                background-color: #e6e6e6;
                spacing: 5px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 5px 10px;  /* More padding to prevent text truncation */
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
                padding: 6px 8px;  /* More padding to prevent text truncation */
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
        """)
    
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
        self.render_mode = "wireframe"
        self.show_boundary = True
    
    def create_widgets(self):
        """创建UI组件"""
        # 创建功能区（替代菜单栏和工具栏）
        self.create_menu()

        # 创建中央小工具
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(2)  # Reduced spacing
        
        # 创建左右两栏布局（3:7比例）
        self.paned_window = QSplitter(Qt.Horizontal)
        
        # 左侧部件信息区域（3/10宽度）
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setSpacing(2)  # Reduced spacing
        
        # Initialize the part list widget first
        self.create_left_panel()
        
        # 部件列表分组
        parts_frame_container = QGroupBox("部件列表")
        parts_layout = QVBoxLayout(parts_frame_container)
        parts_layout.setSpacing(2)  # Reduced spacing

        # 部件列表带滚动条 - includes buttons in the widget
        parts_layout.addWidget(self.parts_list_widget.widget)
        
        left_layout.addWidget(parts_frame_container)
        left_layout.addWidget(self.props_frame)
        self.left_panel.setMinimumWidth(300)
        
        # 连接部件列表选择事件
        self.parts_list_widget.parts_list.currentRowChanged.connect(self.on_part_select)
        
        self.paned_window.addWidget(self.left_panel)
        
        # 右侧网格视图交互区域（7/10宽度）
        right_main_widget = QWidget()
        right_main_layout = QVBoxLayout(right_main_widget)
        right_main_layout.setSpacing(2)  # Reduced spacing
        
        # 在右侧区域中创建垂直分割窗格（网格显示和状态输出）
        self.right_paned = QSplitter(Qt.Vertical)
        
        # 上半部分：网格视图交互区域
        self.right_panel = QWidget()
        right_panel_layout = QVBoxLayout(self.right_panel)
        right_panel_layout.setSpacing(2)  # Reduced spacing
        self.create_right_panel()
        right_panel_layout.addWidget(self.main_mesh_display.frame)
        self.right_paned.addWidget(self.right_panel)
        
        # 下半部分：状态输出面板区域
        self.bottom_frame = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_frame)
        bottom_layout.setSpacing(2)  # Reduced spacing
        self.create_status_output_panel()
        bottom_layout.addWidget(self.status_output_paned)
        self.right_paned.addWidget(self.bottom_frame)
        
        right_main_layout.addWidget(self.right_paned)
        self.paned_window.addWidget(right_main_widget)
        
        # 设置拉伸比例 - improved proportions for larger view area
        self.paned_window.setStretchFactor(0, 1)  # Left panel: 10% (reduced from 20%)
        self.paned_window.setStretchFactor(1, 9)  # Right panel: 90% (increased from 80%)

        # Style the splitters
        self.paned_window.setStyleSheet("""
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
        """)

        # 设置右侧部分的拉伸 - improved proportions for larger view area
        self.right_paned.setStretchFactor(0, 8)  # Mesh display: 80% (increased from 70%)
        self.right_paned.setStretchFactor(1, 2)  # Status output: 20% (reduced from 30%)

        # Style the right splitter too
        self.right_paned.setStyleSheet("""
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
        """)
        
        main_layout.addWidget(self.paned_window)
        
        # 创建状态栏
        self.status_bar = StatusBar(self)
    
    def create_ribbon(self):
        """创建功能区"""
        self.ribbon = RibbonWidget(self)
        # Set up icons for the ribbon buttons
        self.setup_ribbon_icons()
        self.ribbon.set_callbacks(self)
        self.setMenuWidget(self.ribbon)

        # Connect the toggle button to the same toggle function
        self.ribbon.toggle_button.clicked.connect(self.toggle_ribbon)

        # Also connect a keyboard shortcut
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        shortcut.activated.connect(self.toggle_ribbon)

    def setup_ribbon_icons(self):
        """设置功能区按钮图标"""
        # Try to load icons from the icons directory
        icon_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui", "icons")

        # Function to get icon with fallback
        def get_icon(icon_name):
            icon_path = os.path.join(icon_dir, f"{icon_name}.png")
            if os.path.exists(icon_path):
                from PyQt5.QtGui import QIcon
                return QIcon(icon_path)
            # If file doesn't exist, return a standard icon
            from PyQt5.QtGui import QIcon
            return QIcon.fromTheme(icon_name, QIcon())  # Fallback to system theme

        # Add icons to file buttons
        if hasattr(self.ribbon, 'buttons') and 'file' in self.ribbon.buttons:
            for button_name, button in self.ribbon.buttons['file'].items():
                if button_name == 'new':
                    button.setIcon(self.get_standard_icon('document-new'))
                elif button_name == 'open':
                    button.setIcon(self.get_standard_icon('document-open'))
                elif button_name == 'save':
                    button.setIcon(self.get_standard_icon('document-save'))
                elif button_name == 'import':
                    button.setIcon(self.get_standard_icon('document-import'))
                elif button_name == 'export':
                    button.setIcon(self.get_standard_icon('document-export'))

        # Add icons to view buttons
        if hasattr(self.ribbon, 'buttons') and 'view' in self.ribbon.buttons:
            for button_name, button in self.ribbon.buttons['view'].items():
                if button_name == 'reset':
                    button.setIcon(self.get_standard_icon('view-refresh'))
                elif button_name == 'fit':
                    button.setIcon(self.get_standard_icon('zoom-fit-best'))
                elif button_name == 'zoom_in':
                    button.setIcon(self.get_standard_icon('zoom-in'))
                elif button_name == 'zoom_out':
                    button.setIcon(self.get_standard_icon('zoom-out'))
                elif button_name in ['surface', 'wireframe', 'mixed', 'points']:
                    button.setIcon(self.get_standard_icon('applications-graphics'))

        # Add icons to config buttons
        if hasattr(self.ribbon, 'buttons') and 'config' in self.ribbon.buttons:
            for button_name, button in self.ribbon.buttons['config'].items():
                if button_name in ['params', 'mesh_params', 'boundary']:
                    button.setIcon(self.get_standard_icon('configure'))
                elif button_name in ['import_config', 'export_config']:
                    button.setIcon(self.get_standard_icon('document-properties'))
                elif button_name == 'reset':
                    button.setIcon(self.get_standard_icon('edit-clear'))

        # Add icons to mesh buttons
        if hasattr(self.ribbon, 'buttons') and 'mesh' in self.ribbon.buttons:
            for button_name, button in self.ribbon.buttons['mesh'].items():
                if button_name == 'generate':
                    button.setIcon(self.get_standard_icon('system-run'))
                elif button_name == 'display':
                    button.setIcon(self.get_standard_icon('view-fullscreen'))
                elif button_name == 'clear':
                    button.setIcon(self.get_standard_icon('edit-delete'))
                elif button_name in ['quality', 'smooth', 'optimize']:
                    button.setIcon(self.get_standard_icon('tools-check-spelling'))
                elif button_name in ['statistics', 'report']:
                    button.setIcon(self.get_standard_icon('x-office-spreadsheet'))

        # Add icons to help buttons
        if hasattr(self.ribbon, 'buttons') and 'help' in self.ribbon.buttons:
            for button_name, button in self.ribbon.buttons['help'].items():
                if button_name == 'manual':
                    button.setIcon(self.get_standard_icon('help-contents'))
                elif button_name == 'quick_start':
                    button.setIcon(self.get_standard_icon('help-faq'))
                elif button_name == 'shortcuts':
                    button.setIcon(self.get_standard_icon('help-keyboard-shortcuts'))
                elif button_name == 'updates':
                    button.setIcon(self.get_standard_icon('help-about'))
                elif button_name == 'about':
                    button.setIcon(self.get_standard_icon('help-about'))

    def get_standard_icon(self, icon_name):
        """获取标准图标"""
        from PyQt5.QtWidgets import QStyle
        from PyQt5.QtGui import QIcon

        # Map common icon names to Qt standard icons
        icon_map = {
            'document-new': QStyle.SP_FileIcon,
            'document-open': QStyle.SP_DirOpenIcon,
            'document-save': QStyle.SP_DialogSaveButton,
            'document-import': QStyle.SP_ArrowUp,
            'document-export': QStyle.SP_ArrowDown,
            'zoom-fit-best': QStyle.SP_ComputerIcon,
            'zoom-in': QStyle.SP_ArrowUp,
            'zoom-out': QStyle.SP_ArrowDown,
            'applications-graphics': QStyle.SP_DesktopIcon,
            'configure': QStyle.SP_ToolBarHorizontalExtensionButton,
            'system-run': QStyle.SP_MediaPlay,
            'view-fullscreen': QStyle.SP_TitleBarMaxButton,
            'edit-delete': QStyle.SP_TrashIcon,
            'edit-clear': QStyle.SP_LineEditClearButton,
            'tools-check-spelling': QStyle.SP_DialogApplyButton,
            'x-office-spreadsheet': QStyle.SP_FileDialogContentsView,
            'help-contents': QStyle.SP_MessageBoxInformation,
            'help-faq': QStyle.SP_MessageBoxQuestion,
            'help-keyboard-shortcuts': QStyle.SP_DialogHelpButton,
            'help-about': QStyle.SP_MessageBoxQuestion,
            'view-refresh': QStyle.SP_BrowserReload,
        }

        if icon_name in icon_map:
            return self.style().standardIcon(icon_map[icon_name])
        else:
            # Return a generic icon if not found
            return self.style().standardIcon(QStyle.SP_MessageBoxInformation)
    
    def create_menu(self):
        """创建菜单 - 使用功能区替代传统菜单"""
        # 创建功能区替代传统菜单和工具栏
        self.create_ribbon()


    def toggle_ribbon(self):
        """Toggle ribbon content visibility"""
        if hasattr(self, 'ribbon') and self.ribbon:
            # Instead of hiding the entire ribbon, we'll collapse the content inside
            # This will be handled by a method in the ribbon widget
            self.ribbon.toggle_content_visibility()
    
    def create_left_panel(self):
        """创建左侧部件信息区域（分组更清晰，带滚动）"""
        # 部件列表分组
        self.parts_list_widget = PartListWidget(
            parent=self,
            add_callback=self.add_part,
            remove_callback=self.remove_part,
            edit_callback=self.edit_part
        )

        # Apply styling to the parts list widget
        if hasattr(self.parts_list_widget, 'parts_list') and self.parts_list_widget.parts_list:
            self.parts_list_widget.parts_list.setStyleSheet("""
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
            """)

        # Apply styling to the buttons
        if hasattr(self.parts_list_widget, 'add_button'):
            self.parts_list_widget.add_button.setStyleSheet("""
                QPushButton {
                    background-color: #e6f3ff;
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
            """)

        if hasattr(self.parts_list_widget, 'remove_button'):
            self.parts_list_widget.remove_button.setStyleSheet("""
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
            """)

        if hasattr(self.parts_list_widget, 'edit_button'):
            self.parts_list_widget.edit_button.setStyleSheet("""
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
            """)

        # 属性面板分组
        self.props_frame = QGroupBox("属性面板")
        self.props_frame.setStyleSheet("""
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
        """)
        props_layout = QVBoxLayout()
        props_layout.setSpacing(2)  # Reduced spacing

        self.props_text = QTextEdit()
        self.props_text.setReadOnly(True)
        self.props_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #ffffff;
                color: #333333;
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
                font-size: 9pt;
            }
        """)
        props_layout.addWidget(self.props_text)
        self.props_frame.setLayout(props_layout)
    
    def create_right_panel(self):
        """创建右侧网格视图交互区域（最优化版，移除所有标签，仅保留纯净的网格显示）"""
        # 创建网格显示区域
        self.main_mesh_display = MeshDisplayArea(self.right_panel)

        # Apply styling to the mesh display container
        self.main_mesh_display.frame.setStyleSheet("""
            QFrame {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff;
            }
        """)

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
            self.update_status("渲染模式: 实体模式 (1键)")
        elif mode == "wireframe":
            self.update_status("渲染模式: 线框模式 (2键)")
        elif mode == "mixed" or mode == "surface-wireframe":
            self.update_status("渲染模式: 混合模式 (3键)")
        elif mode == "points":
            self.update_status("渲染模式: 点云模式 (4键)")
        
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
        """创建信息输出面板（移除状态窗口，仅保留信息输出区）"""
        # 直接创建信息输出区域，不再使用左右分割
        self.info_output = InfoOutput(self)

        # Apply styling to the info output panel
        if hasattr(self.info_output, 'frame') and self.info_output.frame:
            self.info_output.frame.setStyleSheet("""
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
            """)

        # Apply styling to the info text area
        if hasattr(self.info_output, 'info_text') and self.info_output.info_text:
            self.info_output.info_text.setStyleSheet("""
                QTextEdit {
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    background-color: #ffffff;
                    color: #333333;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 9pt;
                }
            """)

        # Set the info_output frame as the main panel
        self.status_output_paned = self.info_output.frame
    
    def update_status(self, message):
        """更新状态栏（移除状态文本框）"""
        # 更新状态栏
        if hasattr(self, 'status_bar'):
            self.status_bar.update_status(message)
    
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
            from pyqt_gui.config_manager import ConfigManager
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
                from pyqt_gui.file_operations import FileOperations
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
                from pyqt_gui.file_operations import FileOperations
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

        # Handle both list and dictionary formats for parts_info
        if isinstance(parts_info, list):
            # If parts_info is a list, iterate directly
            for part_info in parts_info:
                part_name = part_info.get('part_name', '未知部件')
                self.parts_list_widget.parts_list.addItem(part_name)
        elif isinstance(parts_info, dict):
            # If parts_info is a dictionary (like boundary_info), use the keys as part names
            for part_name in parts_info.keys():
                # Only add if it's a valid part name (not a general info key)
                if part_name not in ['type', 'node_coords', 'cells', 'num_points', 'num_cells', 'unstr_grid']:
                    self.parts_list_widget.parts_list.addItem(part_name)
        elif parts_info:
            # If parts_info is neither a list nor a dict but is truthy, try to handle as single item
            self.parts_list_widget.parts_list.addItem(str(parts_info))
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """PyMeshGen v1.0\n\n基于Python的网格生成工具\n\n© 2025 HighOrderMesh"""
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
        """切换工具栏显示 - 现在切换功能区显示"""
        if hasattr(self, 'ribbon') and hasattr(self, 'ribbon'):
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
        # Get the part name from the list widget
        selected_part_name = ""
        if self.parts_list_widget.parts_list.count() > index:
            selected_part_name = self.parts_list_widget.parts_list.item(index).text()

        # Check if it's cas file parts - handle both dict and list formats
        if hasattr(self, 'cas_parts_info') and self.cas_parts_info:
            # Check if cas_parts_info is a dictionary (most common case for boundary_info)
            if isinstance(self.cas_parts_info, dict) and selected_part_name in self.cas_parts_info:
                part_info = self.cas_parts_info[selected_part_name]

                # Clear properties text
                props_content = f"=== CAS部件属性 ===\n\n"
                props_content += f"部件名称: {selected_part_name}\n"
                props_content += f"边界条件类型: {part_info.get('type', '未知') if isinstance(part_info, dict) else '未知'}\n"
                props_content += f"面数量: {len(part_info.get('faces', []) if isinstance(part_info, dict) else [])}\n"
                if isinstance(part_info, dict) and 'faces' in part_info:
                    total_nodes = 0
                    for face in part_info['faces']:
                        total_nodes += len(face.get('nodes', []))
                    props_content += f"节点数量: {total_nodes}\n"
                else:
                    props_content += f"节点数量: 0\n"
                props_content += f"单元数量: 0\n"  # Assuming no cell info in this format

                # Add status info
                props_content += f"\n=== 状态信息 ===\n"
                props_content += f"选择时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                props_content += f"部件索引: {part_index}\n"
                props_content += f"总部件数: {len(self.cas_parts_info)}\n"
                props_content += f"数据来源: CAS文件\n"

                self.props_text.setPlainText(props_content)
                self.update_status(f"已选中CAS部件: {selected_part_name}")
                return
            # Original logic for when cas_parts_info is a list
            elif isinstance(self.cas_parts_info, list) and part_index < len(self.cas_parts_info):
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
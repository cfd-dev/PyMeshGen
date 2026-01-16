#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt Toolbar Implementation
Provides a draggable and dockable toolbar with view functionality
"""

from PyQt5.QtWidgets import (QWidget, QToolBar, QAction, QToolButton, QButtonGroup,
                             QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QMenu)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QPoint
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor

from .icon_manager import get_icon


class ViewToolbar(QToolBar):
    """Draggable and dockable view toolbar with display mode and orientation controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ViewToolbar")
        self.setWindowTitle("视图工具栏")
        self.setAllowedAreas(Qt.AllToolBarAreas)
        self.setMovable(True)  # Enable dragging and docking
        self.setFloatable(True)  # Allow floating as a separate window
        
        # Store references to actions for external access
        self.actions = {}
        
        # Create toolbar content
        self._create_toolbar_content()
        
        # Set initial size
        self.setIconSize(QSize(24, 24))
        
    def _create_toolbar_content(self):
        """Create the toolbar content with view controls"""
        
        # Display mode actions
        self.addSeparator()
        
        # Surface mode
        surface_action = QAction(get_icon('surface'), "实体模式", self)
        surface_action.setStatusTip("切换到实体显示模式")
        surface_action.triggered.connect(lambda: self._emit_mode_change("surface"))
        self.addAction(surface_action)
        self.actions['surface'] = surface_action
        
        # Wireframe mode
        wireframe_action = QAction(get_icon('wireframe'), "线框模式", self)
        wireframe_action.setStatusTip("切换到线框显示模式")
        wireframe_action.triggered.connect(lambda: self._emit_mode_change("wireframe"))
        self.addAction(wireframe_action)
        self.actions['wireframe'] = wireframe_action
        
        # Surface + Wireframe mode
        surface_wireframe_action = QAction(get_icon('surface-wireframe'), "实体+线框模式", self)
        surface_wireframe_action.setStatusTip("切换到实体+线框显示模式")
        surface_wireframe_action.triggered.connect(lambda: self._emit_mode_change("surface-wireframe"))
        self.addAction(surface_wireframe_action)
        self.actions['surface-wireframe'] = surface_wireframe_action
        
        self.addSeparator()
        
        # View orientation actions
        # Reset view
        reset_action = QAction(get_icon('view-refresh'), "重置视图", self)
        reset_action.setStatusTip("重置视图为默认方向")
        reset_action.triggered.connect(lambda: self._emit_view_change("reset"))
        self.addAction(reset_action)
        self.actions['reset'] = reset_action
        
        # Fit view
        fit_action = QAction(get_icon('zoom-fit-best'), "适应视图", self)
        fit_action.setStatusTip("适应视图以显示全部内容")
        fit_action.triggered.connect(lambda: self._emit_view_change("fit"))
        self.addAction(fit_action)
        self.actions['fit'] = fit_action
        
        self.addSeparator()
        
        # Orientation views
        x_pos_action = QAction(get_icon('view-x-pos'), "X+", self)
        x_pos_action.setStatusTip("X轴正向视图")
        x_pos_action.triggered.connect(lambda: self._emit_view_change("x_pos"))
        self.addAction(x_pos_action)
        self.actions['x_pos'] = x_pos_action
        
        x_neg_action = QAction(get_icon('view-x-neg'), "X-", self)
        x_neg_action.setStatusTip("X轴负向视图")
        x_neg_action.triggered.connect(lambda: self._emit_view_change("x_neg"))
        self.addAction(x_neg_action)
        self.actions['x_neg'] = x_neg_action
        
        y_pos_action = QAction(get_icon('view-y-pos'), "Y+", self)
        y_pos_action.setStatusTip("Y轴正向视图")
        y_pos_action.triggered.connect(lambda: self._emit_view_change("y_pos"))
        self.addAction(y_pos_action)
        self.actions['y_pos'] = y_pos_action
        
        y_neg_action = QAction(get_icon('view-y-neg'), "Y-", self)
        y_neg_action.setStatusTip("Y轴负向视图")
        y_neg_action.triggered.connect(lambda: self._emit_view_change("y_neg"))
        self.addAction(y_neg_action)
        self.actions['y_neg'] = y_neg_action
        
        z_pos_action = QAction(get_icon('view-z-pos'), "Z+", self)
        z_pos_action.setStatusTip("Z轴正向视图")
        z_pos_action.triggered.connect(lambda: self._emit_view_change("z_pos"))
        self.addAction(z_pos_action)
        self.actions['z_pos'] = z_pos_action
        
        z_neg_action = QAction(get_icon('view-z-neg'), "Z-", self)
        z_neg_action.setStatusTip("Z轴负向视图")
        z_neg_action.triggered.connect(lambda: self._emit_view_change("z_neg"))
        self.addAction(z_neg_action)
        self.actions['z_neg'] = z_neg_action
        
        iso_action = QAction(get_icon('view-iso'), "等轴测", self)
        iso_action.setStatusTip("等轴测视图")
        iso_action.triggered.connect(lambda: self._emit_view_change("iso"))
        self.addAction(iso_action)
        self.actions['iso'] = iso_action
        
        self.addSeparator()
        

    def _emit_mode_change(self, mode):
        """Emit signal for mode change - to be connected externally"""
        view_controller = None
        if hasattr(self.parent(), 'view_controller'):
            view_controller = self.parent().view_controller
        elif hasattr(self.window(), 'view_controller'):
            view_controller = self.window().view_controller

        if view_controller and hasattr(view_controller, 'set_render_mode'):
            view_controller.set_render_mode(mode)
        elif hasattr(self.parent(), 'set_render_mode'):
            self.parent().set_render_mode(mode)
        elif hasattr(self.window(), 'set_render_mode'):
            self.window().set_render_mode(mode)

    def _emit_view_change(self, view_type):
        """Emit signal for view change - to be connected externally"""
        # Map view types to appropriate methods in the main window
        view_methods = {
            'reset': 'reset_view',
            'fit': 'fit_view',
            'x_pos': 'set_view_x_positive',
            'x_neg': 'set_view_x_negative',
            'y_pos': 'set_view_y_positive',
            'y_neg': 'set_view_y_negative',
            'z_pos': 'set_view_z_positive',
            'z_neg': 'set_view_z_negative',
            'iso': 'set_view_isometric',
            'zoom_in': 'zoom_in',
            'zoom_out': 'zoom_out'
        }
        
        method_name = view_methods.get(view_type)
        if method_name:
            view_controller = None
            if hasattr(self.parent(), 'view_controller'):
                view_controller = self.parent().view_controller
            elif hasattr(self.window(), 'view_controller'):
                view_controller = self.window().view_controller

            if view_controller and hasattr(view_controller, method_name):
                getattr(view_controller, method_name)()
            elif hasattr(self.parent(), method_name):
                getattr(self.parent(), method_name)()
            elif hasattr(self.window(), method_name):
                getattr(self.window(), method_name)()

    def add_view_toolbar_to_main_window(self, main_window):
        """Connect the toolbar to the main window methods
        
        Note: The actions are already connected in _create_toolbar_content() via _emit_view_change()
        This method is kept for potential future use but no longer adds duplicate connections.
        """
        pass

    def contextMenuEvent(self, event):
        """Override context menu to provide toolbar options"""
        menu = QMenu(self)
        
        # Toggle visibility action
        toggle_action = menu.addAction("隐藏工具栏")
        toggle_action.triggered.connect(self.toggleViewAction().trigger)
        
        # Dock options
        dock_menu = menu.addMenu("停靠位置")
        
        left_dock = dock_menu.addAction("左侧")
        left_dock.triggered.connect(lambda: self.setFloating(False) or self.parent().addToolBar(Qt.LeftToolBarArea, self))
        
        right_dock = dock_menu.addAction("右侧")  
        right_dock.triggered.connect(lambda: self.setFloating(False) or self.parent().addToolBar(Qt.RightToolBarArea, self))
        
        top_dock = dock_menu.addAction("顶部")
        top_dock.triggered.connect(lambda: self.setFloating(False) or self.parent().addToolBar(Qt.TopToolBarArea, self))
        
        bottom_dock = dock_menu.addAction("底部")
        bottom_dock.triggered.connect(lambda: self.setFloating(False) or self.parent().addToolBar(Qt.BottomToolBarArea, self))
        
        menu.exec_(event.globalPos())


class DraggableToolbarWidget(QFrame):
    """A wrapper widget that makes the toolbar draggable and dockable"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(1)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create the actual toolbar
        self.toolbar = ViewToolbar(self)
        
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        
        # Enable dragging
        self.setMouseTracking(True)
        self.drag_start_position = None
        self.is_dragging = False
        
    def mousePressEvent(self, event):
        """Handle mouse press for dragging"""
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging"""
        if not self.drag_start_position:
            return
            
        if event.buttons() == Qt.LeftButton:
            # Move the window
            self.move(event.globalPos() - self.drag_start_position)
            event.accept()
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        self.drag_start_position = None
        event.accept()

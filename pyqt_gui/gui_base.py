#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt GUI基础组件模块
提供通用的UI元素和基础功能
"""

import sys
from PyQt5.QtWidgets import (QWidget, QMainWindow, QMenuBar, QStatusBar, QToolBar, 
                             QVBoxLayout, QHBoxLayout, QSplitter, QFrame, QGroupBox, 
                             QLabel, QTextEdit, QScrollArea, QTabWidget, QToolButton,
                             QDockWidget, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class BaseWidget:
    """基础窗口组件类，提供通用功能"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.widget = QWidget(parent)
        
    def show(self):
        """显示组件"""
        self.widget.show()
        
    def hide(self):
        """隐藏组件"""
        self.widget.hide()


class MenuBar:
    """菜单栏类"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.menubar = main_window.menuBar()
        
    def create_file_menu(self, commands):
        """创建文件菜单"""
        file_menu = self.menubar.addMenu('文件')
        
        for label, command in commands.items():
            if label == "---":
                file_menu.addSeparator()
            else:
                action = file_menu.addAction(label)
                action.triggered.connect(command)
                
        return file_menu
    
    def create_view_menu(self, commands):
        """创建视图菜单"""
        view_menu = self.menubar.addMenu('视图')
        
        for label, command in commands.items():
            if label == "---":
                view_menu.addSeparator()
            else:
                action = view_menu.addAction(label)
                action.triggered.connect(command)
                
        return view_menu
    
    def create_config_menu(self, commands):
        """创建配置菜单"""
        config_menu = self.menubar.addMenu('配置')
        
        for label, command in commands.items():
            if label == "---":
                config_menu.addSeparator()
            else:
                action = config_menu.addAction(label)
                action.triggered.connect(command)
            
        return config_menu
    
    def create_mesh_menu(self, commands):
        """创建网格菜单"""
        mesh_menu = self.menubar.addMenu('网格')
        
        for label, command in commands.items():
            if label == "---":
                mesh_menu.addSeparator()
            else:
                action = mesh_menu.addAction(label)
                action.triggered.connect(command)
                
        return mesh_menu
    
    def create_help_menu(self, commands):
        """创建帮助菜单"""
        help_menu = self.menubar.addMenu('帮助')
        
        for label, command in commands.items():
            if label == "---":
                help_menu.addSeparator()
            else:
                action = help_menu.addAction(label)
                action.triggered.connect(command)
            
        return help_menu


class StatusBar:
    """状态栏类"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.status_bar = main_window.statusBar()
        self.status_bar.showMessage("就绪")
        
    def update_status(self, message):
        """更新状态栏信息"""
        self.status_bar.showMessage(message)


class InfoOutput:
    """信息输出窗口类"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.create_info_output_area()
        
    def create_info_output_area(self):
        """创建信息输出窗口"""
        # 创建信息输出框架
        self.frame = QGroupBox("信息输出")
        layout = QVBoxLayout()
        
        # 创建文本框和滚动条
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
        self.frame.setLayout(layout)
        
        # 设置默认大小
        self.frame.setMinimumHeight(100)
        
        return self.frame
    
    def append_info_output(self, message):
        """添加信息到输出窗口"""
        # 在主线程中执行GUI更新
        self.info_text.append(message)
    
    def clear_info_output(self):
        """清除信息输出"""
        self.info_text.clear()
        
    def clear(self):
        """清空信息输出窗口（与clear_info_output一致，供主界面按钮调用）"""
        self.clear_info_output()


class ToolBar:
    """工具栏类"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.toolbar = main_window.addToolBar('工具栏')
        
    def add_action(self, text, icon=None, callback=None, tooltip=None):
        """添加动作到工具栏"""
        if icon:
            action = self.toolbar.addAction(icon, text)
        else:
            action = self.toolbar.addAction(text)
        if callback:
            action.triggered.connect(callback)
        if tooltip:
            action.setToolTip(tooltip)
        return action
        
    def add_separator(self):
        """添加分隔符"""
        self.toolbar.addSeparator()


class Splitter:
    """分割窗口类"""
    
    def __init__(self, parent=None, orientation=Qt.Horizontal):
        self.splitter = QSplitter(parent)
        self.splitter.setOrientation(orientation)
        
    def add_widget(self, widget):
        """添加组件到分割器"""
        self.splitter.addWidget(widget)
        
    def set_sizes(self, sizes):
        """设置分割器大小"""
        self.splitter.setSizes(sizes)
        
    def get_widget(self):
        """获取分割器"""
        return self.splitter


class DialogBase:
    """对话框基类"""
    
    def __init__(self, parent=None, title="对话框"):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox
        
        self.dialog = QDialog(parent)
        self.dialog.setWindowTitle(title)
        
        # 创建按钮框
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self.dialog
        )
        self.button_box.accepted.connect(self.ok)
        self.button_box.rejected.connect(self.cancel)
        
        # 创建布局
        main_layout = QVBoxLayout()
        self.dialog.setLayout(main_layout)
        
        # 添加按钮到布局
        main_layout.addWidget(self.button_box)
        
        # 结果变量
        self.result = None
    
    def ok(self):
        """确定按钮回调，子类应重写此方法"""
        self.dialog.accept()
    
    def cancel(self):
        """取消按钮回调"""
        self.dialog.reject()


class PartListWidget:
    """部件列表组件"""
    
    def __init__(self, parent=None):
        from PyQt5.QtWidgets import QListWidget, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
        from PyQt5.QtCore import pyqtSignal
        
        self.parent = parent
        self.widget = QWidget()
        
        layout = QVBoxLayout()
        self.parts_list = QListWidget()
        layout.addWidget(self.parts_list)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("添加")
        self.remove_button = QPushButton("删除")
        self.edit_button = QPushButton("编辑")
        
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.edit_button)
        
        layout.addLayout(button_layout)
        self.widget.setLayout(layout)
        
        # 信号连接
        self.add_button.clicked.connect(self.add_part)
        self.remove_button.clicked.connect(self.remove_part)
        self.edit_button.clicked.connect(self.edit_part)
        
    def add_part(self):
        """添加部件 - 需要在主类中重写"""
        pass
        
    def remove_part(self):
        """删除部件 - 需要在主类中重写"""
        pass
        
    def edit_part(self):
        """编辑部件 - 需要在主类中重写"""
        pass
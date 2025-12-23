#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt GUI基础组件模块
提供通用的UI元素和基础功能
"""

import sys
from PyQt5.QtWidgets import (QWidget, QMainWindow, QStatusBar, QToolBar,
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




class StatusBar:
    """状态栏类"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.status_bar = main_window.statusBar()
        # Set a larger minimum height to ensure text fits
        self.status_bar.setMinimumHeight(25)
        self.status_bar.showMessage("就绪")

    def update_status(self, message):
        """更新状态栏信息"""
        self.status_bar.showMessage(message)

        # Ensure the status bar can show the full text by setting appropriate font
        font = self.status_bar.font()
        font.setPointSize(9)  # Slightly smaller font to fit more text
        self.status_bar.setFont(font)


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
        layout.setSpacing(3)  # Reduced spacing

        # 创建文本框 and set smaller minimum size
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(60)  # Reduced minimum height
        layout.addWidget(self.info_text)

        self.frame.setLayout(layout)

        # 设置默认大小
        self.frame.setMinimumHeight(80)  # Reduced minimum height

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
        
        # 结果变量
        self.result = None
    
    def ok(self):
        """确定按钮回调，子类应重写此方法"""
        self.dialog.accept()
    
    def cancel(self):
        """取消按钮回调"""
        self.dialog.reject()


class ConfigDialog(DialogBase):
    """配置编辑对话框"""
    
    def __init__(self, parent=None, config=None):
        super().__init__(parent, "编辑配置")
        
        # 设置配置，默认空配置
        self.config = config.copy() if config else {
            "debug_level": 0,
            "input_file": "",
            "output_file": "",
            "mesh_type": 1,
            "viz_enabled": True,
            "parts": []
        }
        
        # 创建布局和控件
        self.create_widgets()
        
        # 显示对话框
        self.dialog.resize(700, 600)
    
    def create_widgets(self):
        """创建对话框组件"""
        from PyQt5.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QComboBox,
            QLineEdit, QRadioButton, QCheckBox, QPushButton, QTreeView, QStandardItemModel,
            QHeaderView, QSplitter, QWidget, QFileDialog, QMessageBox
        )
        from PyQt5.QtCore import Qt, QModelIndex
        from PyQt5.QtGui import QStandardItem
        
        # 创建主布局
        main_layout = self.dialog.layout()
        
        # 创建参数设置分组框
        params_group = QGroupBox("基本参数")
        params_layout = QGridLayout(params_group)
        
        # 调试级别
        params_layout.addWidget(QLabel("调试级别:"), 0, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self.debug_level_combo = QComboBox()
        self.debug_level_combo.addItems(["0", "1", "2"])
        self.debug_level_combo.setCurrentText(str(self.config.get("debug_level", 0)))
        params_layout.addWidget(self.debug_level_combo, 0, 1, 1, 2)
        
        # 输入文件（只读）
        params_layout.addWidget(QLabel("输入文件:"), 1, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self.input_file_edit = QLineEdit(self.config.get("input_file", ""))
        self.input_file_edit.setReadOnly(True)
        params_layout.addWidget(self.input_file_edit, 1, 1, 1, 2)
        
        # 输出文件
        params_layout.addWidget(QLabel("输出文件:"), 2, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self.output_file_edit = QLineEdit(self.config.get("output_file", ""))
        params_layout.addWidget(self.output_file_edit, 2, 1)
        
        # 浏览按钮
        browse_output_btn = QPushButton("浏览")
        browse_output_btn.clicked.connect(self.browse_output_file)
        params_layout.addWidget(browse_output_btn, 2, 2)
        
        # 网格类型
        params_layout.addWidget(QLabel("网格类型:"), 3, 0, Qt.AlignLeft | Qt.AlignVCenter)
        mesh_type_widget = QWidget()
        mesh_type_layout = QHBoxLayout(mesh_type_widget)
        mesh_type_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mesh_type = self.config.get("mesh_type", 1)
        self.mesh_type_radio1 = QRadioButton("三角形")
        self.mesh_type_radio1.setChecked(self.mesh_type == 1)
        self.mesh_type_radio2 = QRadioButton("直角三角形")
        self.mesh_type_radio2.setChecked(self.mesh_type == 2)
        self.mesh_type_radio3 = QRadioButton("混合网格")
        self.mesh_type_radio3.setChecked(self.mesh_type == 3)
        
        mesh_type_layout.addWidget(self.mesh_type_radio1)
        mesh_type_layout.addWidget(self.mesh_type_radio2)
        mesh_type_layout.addWidget(self.mesh_type_radio3)
        params_layout.addWidget(mesh_type_widget, 3, 1, 1, 2)
        
        # 可视化开关
        self.viz_checkbox = QCheckBox("启用可视化")
        self.viz_checkbox.setChecked(self.config.get("viz_enabled", True))
        params_layout.addWidget(self.viz_checkbox, 4, 0, 1, 3, Qt.AlignLeft)
        
        # 添加按钮
        self.generate_btn = QPushButton("生成配置文件")
        self.generate_btn.clicked.connect(self.generate_config_file)
        params_layout.addWidget(self.generate_btn, 5, 0)
        
        self.open_btn = QPushButton("打开配置文件")
        self.open_btn.clicked.connect(self.open_config_file)
        params_layout.addWidget(self.open_btn, 5, 1)
        
        # 添加基本参数分组到主布局
        main_layout.insertWidget(0, params_group)
        
        # 创建部件配置分组框
        parts_group = QGroupBox("部件配置")
        parts_layout = QVBoxLayout(parts_group)
        parts_layout.setSpacing(2)  # Reduced spacing
        
        # 创建部件配置的树形视图
        self.parts_tree = QTreeView()
        self.parts_model = QStandardItemModel()
        self.parts_model.setHorizontalHeaderLabels(["部件名称", "最大尺寸", "第一层高度", "增长率", "最大层数"])
        self.parts_tree.setModel(self.parts_model)
        self.parts_tree.header().setSectionResizeMode(QHeaderView.Stretch)
        
        # 填充部件数据
        self.populate_parts_tree()
        
        # 添加树形视图到布局
        parts_layout.addWidget(self.parts_tree)
        
        # 添加部件操作按钮
        parts_buttons_layout = QHBoxLayout()
        parts_buttons_layout.setSpacing(2)  # Reduced spacing
        self.add_part_btn = QPushButton("添加部件")
        self.add_part_btn.clicked.connect(self.add_part)
        parts_buttons_layout.addWidget(self.add_part_btn)

        self.remove_part_btn = QPushButton("删除部件")
        self.remove_part_btn.clicked.connect(self.remove_part)
        parts_buttons_layout.addWidget(self.remove_part_btn)

        parts_layout.addLayout(parts_buttons_layout)
        
        # 添加部件配置分组到主布局
        main_layout.insertWidget(1, parts_group)
        
    def populate_parts_tree(self):
        """填充部件配置树形视图"""
        from PyQt5.QtGui import QStandardItem
        
        # 清空现有数据
        self.parts_model.removeRows(0, self.parts_model.rowCount())
        
        # 添加部件数据
        for i, part in enumerate(self.config.get("parts", [])):
            # 获取部件参数
            part_params = part.get("part_params", {})
            part_name = part.get("part_name", f"部件{i+1}")
            max_size = part_params.get("max_size", 1.0)
            first_height = part_params.get("first_height", 0.1)
            growth_rate = part_params.get("growth_rate", 1.2)
            max_layers = part_params.get("max_layers", 3)
            
            # 创建行项目
            items = [
                QStandardItem(part_name),
                QStandardItem(str(max_size)),
                QStandardItem(str(first_height)),
                QStandardItem(str(growth_rate)),
                QStandardItem(str(max_layers))
            ]
            
            # 设置可编辑
            for item in items:
                item.setEditable(True)
            
            # 添加到模型
            self.parts_model.appendRow(items)
    
    def browse_output_file(self):
        """浏览输出文件"""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.dialog,
            "保存网格文件",
            "",
            "VTK文件 (*.vtk);;所有文件 (*.*)"
        )
        if file_path:
            self.output_file_edit.setText(file_path)
    
    def generate_config_file(self):
        """生成配置文件"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import json
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.dialog,
            "保存配置文件",
            "",
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        if file_path:
            try:
                # 获取当前配置
                current_config = self.get_current_config()
                
                # 保存到文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(current_config, f, indent=4, ensure_ascii=False)
                
                QMessageBox.information(self.dialog, "成功", f"配置文件已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self.dialog, "错误", f"保存配置文件失败: {str(e)}")
    
    def open_config_file(self):
        """打开配置文件"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import json
        
        file_path, _ = QFileDialog.getOpenFileName(
            self.dialog,
            "选择配置文件",
            "",
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        if file_path:
            try:
                # 读取配置文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_config = json.load(f)
                
                # 更新界面
                self.config = new_config
                self.debug_level_combo.setCurrentText(str(new_config.get("debug_level", 0)))
                self.input_file_edit.setText(new_config.get("input_file", ""))
                self.output_file_edit.setText(new_config.get("output_file", ""))
                self.viz_checkbox.setChecked(new_config.get("viz_enabled", True))
                
                # 更新网格类型
                mesh_type = new_config.get("mesh_type", 1)
                self.mesh_type_radio1.setChecked(mesh_type == 1)
                self.mesh_type_radio2.setChecked(mesh_type == 2)
                self.mesh_type_radio3.setChecked(mesh_type == 3)
                
                # 更新部件配置
                self.populate_parts_tree()
                
                QMessageBox.information(self.dialog, "成功", f"已加载配置文件: {file_path}")
            except Exception as e:
                QMessageBox.critical(self.dialog, "错误", f"加载配置文件失败: {str(e)}")
    
    def get_current_config(self):
        """获取当前配置"""
        # 获取基本参数
        mesh_type = 1
        if self.mesh_type_radio2.isChecked():
            mesh_type = 2
        elif self.mesh_type_radio3.isChecked():
            mesh_type = 3
        
        # 获取部件配置
        parts = []
        for row in range(self.parts_model.rowCount()):
            part_name = self.parts_model.item(row, 0).text()
            
            # 尝试转换数值，使用默认值如果转换失败
            try:
                max_size = float(self.parts_model.item(row, 1).text())
            except ValueError:
                max_size = 1.0
            
            try:
                first_height = float(self.parts_model.item(row, 2).text())
            except ValueError:
                first_height = 0.1
            
            try:
                growth_rate = float(self.parts_model.item(row, 3).text())
            except ValueError:
                growth_rate = 1.2
            
            try:
                max_layers = int(self.parts_model.item(row, 4).text())
            except ValueError:
                max_layers = 3
            
            parts.append({
                "part_name": part_name,
                "part_params": {
                    "max_size": max_size,
                    "first_height": first_height,
                    "growth_rate": growth_rate,
                    "max_layers": max_layers
                }
            })
        
        return {
            "debug_level": int(self.debug_level_combo.currentText()),
            "input_file": self.input_file_edit.text(),
            "output_file": self.output_file_edit.text(),
            "mesh_type": mesh_type,
            "viz_enabled": self.viz_checkbox.isChecked(),
            "parts": parts
        }
    
    def add_part(self):
        """添加新部件"""
        from PyQt5.QtGui import QStandardItem
        
        # 添加默认部件
        items = [
            QStandardItem(f"部件{self.parts_model.rowCount() + 1}"),
            QStandardItem("1.0"),
            QStandardItem("0.1"),
            QStandardItem("1.2"),
            QStandardItem("3")
        ]
        
        for item in items:
            item.setEditable(True)
        
        self.parts_model.appendRow(items)
    
    def remove_part(self):
        """删除选中的部件"""
        from PyQt5.QtWidgets import QMessageBox
        
        selected_indexes = self.parts_tree.selectionModel().selectedRows()
        if not selected_indexes:
            QMessageBox.warning(self.dialog, "警告", "请先选择要删除的部件")
            return
        
        # 删除选中的行
        for index in sorted(selected_indexes, key=lambda x: x.row(), reverse=True):
            self.parts_model.removeRow(index.row())
    
    def ok(self):
        """确定按钮回调"""
        try:
            # 获取当前配置
            self.result = self.get_current_config()
            self.dialog.accept()
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self.dialog, "错误", f"配置数据格式错误: {str(e)}")


class PartListWidget:
    """部件列表组件"""

    def __init__(self, parent=None, add_callback=None, remove_callback=None, edit_callback=None):
        from PyQt5.QtWidgets import QListWidget, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
        from PyQt5.QtCore import pyqtSignal

        self.parent = parent
        self.widget = QWidget()

        layout = QVBoxLayout()
        layout.setSpacing(2)  # Reduced spacing
        self.parts_list = QListWidget()
        layout.addWidget(self.parts_list)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(2)  # Reduced spacing
        self.add_button = QPushButton("添加")
        self.remove_button = QPushButton("删除")
        self.edit_button = QPushButton("编辑")

        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.edit_button)

        layout.addLayout(button_layout)
        self.widget.setLayout(layout)

        # 信号连接 - use provided callbacks or default to empty methods
        self.add_callback = add_callback or self.add_part
        self.remove_callback = remove_callback or self.remove_part
        self.edit_callback = edit_callback or self.edit_part

        self.add_button.clicked.connect(self.add_callback)
        self.remove_button.clicked.connect(self.remove_callback)
        self.edit_button.clicked.connect(self.edit_callback)

    def add_part(self):
        """添加部件 - 需要在主类中重写"""
        pass

    def remove_part(self):
        """删除部件 - 需要在主类中重写"""
        pass

    def edit_part(self):
        """编辑部件 - 需要在主类中重写"""
        pass
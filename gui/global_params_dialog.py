from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QCheckBox, 
                             QLineEdit, QComboBox, QPushButton, QFileDialog, QGridLayout)
from PyQt5.QtCore import Qt
import os

class GlobalParamsDialog(QDialog):
    def __init__(self, parent=None, params=None):
        super().__init__(parent)
        self.setWindowTitle("全局参数设置")
        self.setMinimumWidth(500)
        
        # 初始化参数
        self.params = params or {}
        
        # 创建UI
        self._init_ui()
        
        # 设置初始值
        self._load_params()
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # 创建参数组
        # 1. 自动输出网格设置
        auto_output_group = QGroupBox("自动输出网格设置")
        auto_output_layout = QGridLayout()
        
        self.auto_output_check = QCheckBox("启用自动输出网格")
        self.auto_output_check.stateChanged.connect(self._on_auto_output_toggled)
        auto_output_layout.addWidget(self.auto_output_check, 0, 0, 1, 2)
        
        # 输出路径标签
        self.output_path_label = QLabel("输出路径:")
        auto_output_layout.addWidget(self.output_path_label, 1, 0)
        
        # 输出路径编辑框和浏览按钮
        path_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        # 设置白色背景
        self.output_path_edit.setStyleSheet("background-color: white;")
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self._browse_output_path)
        path_layout.addWidget(self.output_path_edit)
        path_layout.addWidget(self.browse_btn)
        auto_output_layout.addLayout(path_layout, 1, 1)
        
        auto_output_group.setLayout(auto_output_layout)
        main_layout.addWidget(auto_output_group)
        
        # 2. 信息输出等级设置
        verbosity_group = QGroupBox("信息输出设置")
        verbosity_layout = QHBoxLayout()
        verbosity_layout.addWidget(QLabel("VERBOSITY等级:"))
        self.verbosity_combo = QComboBox()
        # 设置白色背景
        self.verbosity_combo.setStyleSheet("background-color: white;")
        self.verbosity_combo.addItems(["0 (静默)", "1 (基本信息)", "2 (详细信息)"])
        verbosity_layout.addWidget(self.verbosity_combo)
        verbosity_layout.addStretch()
        verbosity_group.setLayout(verbosity_layout)
        main_layout.addWidget(verbosity_group)
        
        # 3. 网格类型设置
        mesh_type_group = QGroupBox("网格类型设置")
        mesh_type_layout = QHBoxLayout()
        mesh_type_layout.addWidget(QLabel("生成的网格类型:"))
        self.mesh_type_combo = QComboBox()
        # 设置白色背景
        self.mesh_type_combo.setStyleSheet("background-color: white;")
        self.mesh_type_combo.addItems(["三角形网格", "三角形/四边形混合网格"])
        mesh_type_layout.addWidget(self.mesh_type_combo)
        mesh_type_layout.addStretch()
        mesh_type_group.setLayout(mesh_type_layout)
        main_layout.addWidget(mesh_type_group)
        
        # 4. 全局网格尺寸设置
        global_size_group = QGroupBox("全局网格尺寸设置")
        global_size_layout = QHBoxLayout()
        global_size_layout.addWidget(QLabel("全局最大网格尺寸:"))
        self.global_size_edit = QLineEdit()
        self.global_size_edit.setPlaceholderText("输入数值")
        # 设置白色背景
        self.global_size_edit.setStyleSheet("background-color: white;")
        global_size_layout.addWidget(self.global_size_edit)
        global_size_layout.addWidget(QLabel("(单位: 模型单位)"))
        global_size_layout.addStretch()
        global_size_group.setLayout(global_size_layout)
        main_layout.addWidget(global_size_group)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(button_layout)
    
    def _browse_output_path(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择输出文件路径", "./out/mesh.vtk", "VTK Files (*.vtk)")
        if file_path:
            self.output_path_edit.setText(file_path)
    
    def _on_auto_output_toggled(self, state):
        # 当自动输出网格选项被切换时，显示或隐藏输出路径相关控件
        is_checked = state == Qt.Checked
        self.output_path_label.setVisible(is_checked)
        self.output_path_edit.setVisible(is_checked)
        self.browse_btn.setVisible(is_checked)
    
    def _load_params(self):
        # 加载现有参数
        # 1. 自动输出网格设置
        auto_output = self.params.get("auto_output", True)
        self.auto_output_check.setChecked(auto_output)
        
        output_path = self.params.get("output_file", "./out/mesh.vtk")
        # 如果output_path是列表，取第一个元素
        if isinstance(output_path, list):
            output_path = output_path[0] if output_path else "./out/mesh.vtk"
        self.output_path_edit.setText(output_path)
        
        # 根据初始状态设置控件可见性
        self._on_auto_output_toggled(Qt.Checked if auto_output else Qt.Unchecked)
        
        # 2. 信息输出等级设置
        verbosity = self.params.get("debug_level", 1)
        self.verbosity_combo.setCurrentIndex(verbosity)
        
        # 3. 网格类型设置
        mesh_type = self.params.get("mesh_type", 1)
        # 转换为对话框中的索引：1-三角形(0), 3-混合(1)
        combo_index = 0 if mesh_type == 1 else 1
        self.mesh_type_combo.setCurrentIndex(combo_index)
        
        # 4. 全局网格尺寸设置
        # 从部件参数中获取全局尺寸，如果没有则使用默认值
        global_size = self.params.get("global_max_size", 1e6)
        self.global_size_edit.setText(str(global_size))
    
    def get_params(self):
        # 获取用户设置的参数
        params = {}
        
        # 1. 自动输出网格设置
        params["auto_output"] = self.auto_output_check.isChecked()
        params["output_file"] = [self.output_path_edit.text()]
        
        # 2. 信息输出等级设置
        params["debug_level"] = self.verbosity_combo.currentIndex()
        
        # 3. 网格类型设置
        combo_index = self.mesh_type_combo.currentIndex()
        # 转换为内部表示：0-三角形(1), 1-混合(3)
        params["mesh_type"] = 1 if combo_index == 0 else 3
        
        # 4. 全局网格尺寸设置
        try:
            params["global_max_size"] = float(self.global_size_edit.text())
        except ValueError:
            params["global_max_size"] = 1.0
        
        return params

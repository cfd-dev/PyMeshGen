from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QCheckBox,
                             QLineEdit, QComboBox, QPushButton, QFileDialog, QGridLayout, QDoubleSpinBox,
                             QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt
import os


class GlobalParamsDialog(QDialog):
    def __init__(self, parent=None, params=None):
        super().__init__(parent)
        self.setWindowTitle("全局参数设置")
        self.setMinimumWidth(500)

        # 初始化参数
        self.params = params or {}

        # 创建 UI
        self._init_ui()

        # 设置初始值
        self._load_params()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

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
        verbosity_layout.addWidget(QLabel("VERBOSITY 等级:"))
        self.verbosity_combo = QComboBox()
        self.verbosity_combo.setStyleSheet("background-color: white;")
        self.verbosity_combo.addItems(["0 (基本信息)", "1 (调试信息)", "2 (详细信息)"])
        verbosity_layout.addWidget(self.verbosity_combo)
        verbosity_layout.addStretch()
        verbosity_group.setLayout(verbosity_layout)
        main_layout.addWidget(verbosity_group)

        # 3. 网格类型设置
        mesh_type_group = QGroupBox("网格类型设置")
        mesh_type_layout = QGridLayout()
        mesh_type_layout.addWidget(QLabel("生成的网格类型:"), 0, 0)
        self.mesh_type_combo = QComboBox()
        self.mesh_type_combo.setStyleSheet("background-color: white;")
        self.mesh_type_combo.addItems(["三角形网格", "三角形/四边形混合网格"])
        self.mesh_type_combo.currentIndexChanged.connect(self._on_mesh_type_changed)
        mesh_type_layout.addWidget(self.mesh_type_combo, 0, 1)

        # 网格生成算法选择（与网格类型联动）
        self.mesh_algorithm_label = QLabel("网格生成算法:")
        mesh_type_layout.addWidget(self.mesh_algorithm_label, 1, 0)
        self.mesh_algorithm_combo = QComboBox()
        self.mesh_algorithm_combo.setStyleSheet("background-color: white;")
        self.mesh_algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        mesh_type_layout.addWidget(self.mesh_algorithm_combo, 1, 1)

        # 三角形合并算法选择（仅在混合网格模式下可用）
        self.triangle_to_quad_label = QLabel("三角形合并算法:")
        mesh_type_layout.addWidget(self.triangle_to_quad_label, 2, 0)
        self.triangle_to_quad_combo = QComboBox()
        self.triangle_to_quad_combo.setStyleSheet("background-color: white;")
        self.triangle_to_quad_combo.addItems(["greedy_merge", "q_morph"])
        self.triangle_to_quad_combo.setToolTip("greedy_merge: 贪婪合并算法\nq_morph: Q-Morph 算法")
        self.triangle_to_quad_combo.currentIndexChanged.connect(self._on_triangle_to_quad_changed)
        mesh_type_layout.addWidget(self.triangle_to_quad_combo, 2, 1)

        # Delaunay backend 单选（仅在三角+Delaunay算法下可见）
        self.delaunay_backend_label = QLabel("Delaunay Backend:")
        mesh_type_layout.addWidget(self.delaunay_backend_label, 3, 0)
        backend_layout = QHBoxLayout()
        self.bw_radio = QRadioButton("Bowyer-Watson")
        self.triangle_radio = QRadioButton("Triangle")
        self.backend_group = QButtonGroup(self)
        self.backend_group.addButton(self.bw_radio)
        self.backend_group.addButton(self.triangle_radio)
        self.bw_radio.setChecked(True)
        backend_layout.addWidget(self.bw_radio)
        backend_layout.addWidget(self.triangle_radio)
        backend_layout.addStretch()
        mesh_type_layout.addLayout(backend_layout, 3, 1)

        mesh_type_layout.setColumnStretch(2, 1)
        mesh_type_group.setLayout(mesh_type_layout)
        main_layout.addWidget(mesh_type_group)

        # 4. 全局网格尺寸设置
        global_size_group = QGroupBox("全局网格尺寸设置")
        global_size_layout = QHBoxLayout()
        global_size_layout.addWidget(QLabel("全局最大网格尺寸:"))
        self.global_size_edit = QLineEdit()
        self.global_size_edit.setPlaceholderText("输入数值")
        self.global_size_edit.setStyleSheet("background-color: white;")
        global_size_layout.addWidget(self.global_size_edit)
        global_size_layout.addWidget(QLabel("(单位：模型单位)"))
        global_size_layout.addStretch()
        global_size_group.setLayout(global_size_layout)
        main_layout.addWidget(global_size_group)

        # 5. 尺寸场衰减参数设置
        decay_group = QGroupBox("尺寸场衰减参数")
        decay_layout = QHBoxLayout()
        decay_layout.addWidget(QLabel("Sizing Decay:"))
        self.sizing_decay_spin = QDoubleSpinBox()
        self.sizing_decay_spin.setRange(0.0, 10.0)
        self.sizing_decay_spin.setDecimals(3)
        self.sizing_decay_spin.setSingleStep(0.05)
        self.sizing_decay_spin.setStyleSheet("background-color: white;")
        self.sizing_decay_spin.setToolTip("decay=1.0 基本不随距离增长；>1 越大增长越快；<1 将跳过decay传播")
        decay_layout.addWidget(self.sizing_decay_spin)
        decay_layout.addWidget(QLabel("(建议 >= 1.0)"))
        decay_layout.addStretch()
        decay_group.setLayout(decay_layout)
        main_layout.addWidget(decay_group)

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
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件路径", "./out/mesh.vtk", "VTK Files (*.vtk)")
        if file_path:
            self.output_path_edit.setText(file_path)

    def _on_auto_output_toggled(self, state):
        """当自动输出网格选项被切换时，显示或隐藏输出路径相关控件"""
        is_checked = state == Qt.Checked
        self.output_path_label.setVisible(is_checked)
        self.output_path_edit.setVisible(is_checked)
        self.browse_btn.setVisible(is_checked)

    def _on_mesh_type_changed(self, index):
        """当网格类型改变时，联动算法和可见控件"""
        # 索引 0=三角形网格，1=三角形/四边形混合网格
        self.mesh_algorithm_combo.blockSignals(True)
        self.mesh_algorithm_combo.clear()
        if index == 0:
            self.mesh_algorithm_combo.addItem("前沿推进 (Adfront2)", "advancing_front")
            self.mesh_algorithm_combo.addItem("Delaunay 三角剖分", "delaunay")
        else:
            self.mesh_algorithm_combo.addItem("混合前沿推进 (Adfront2Hybrid)", "hybrid")
            self.mesh_algorithm_combo.addItem("Q-Morph (先三角后重构)", "q_morph")
        self.mesh_algorithm_combo.setCurrentIndex(0)
        self.mesh_algorithm_combo.blockSignals(False)
        self._update_algorithm_dependent_controls()

    def _on_algorithm_changed(self, _index):
        self._update_algorithm_dependent_controls()

    def _on_triangle_to_quad_changed(self, index):
        # 混合网格时，保持算法选择与三角形合并策略一致
        if self.mesh_type_combo.currentIndex() != 1:
            return
        target_algo = "q_morph" if index == 1 else "hybrid"
        algo_index = self.mesh_algorithm_combo.findData(target_algo)
        if algo_index >= 0 and algo_index != self.mesh_algorithm_combo.currentIndex():
            self.mesh_algorithm_combo.blockSignals(True)
            self.mesh_algorithm_combo.setCurrentIndex(algo_index)
            self.mesh_algorithm_combo.blockSignals(False)
        self._update_algorithm_dependent_controls()

    def _update_algorithm_dependent_controls(self):
        is_mixed = (self.mesh_type_combo.currentIndex() == 1)
        selected_algo = self.mesh_algorithm_combo.currentData()
        is_delaunay = (selected_algo == "delaunay")

        # 混合网格显示三角形合并策略
        self.triangle_to_quad_label.setVisible(is_mixed)
        self.triangle_to_quad_combo.setVisible(is_mixed)

        # 若算法下拉选择了 q_morph/hybrid，同步 triangle_to_quad_method
        if is_mixed:
            if selected_algo == "q_morph":
                self.triangle_to_quad_combo.blockSignals(True)
                self.triangle_to_quad_combo.setCurrentText("q_morph")
                self.triangle_to_quad_combo.blockSignals(False)
            elif selected_algo == "hybrid":
                self.triangle_to_quad_combo.blockSignals(True)
                self.triangle_to_quad_combo.setCurrentText("greedy_merge")
                self.triangle_to_quad_combo.blockSignals(False)

        # Delaunay backend 仅在三角网格 + Delaunay 算法可见
        self.delaunay_backend_label.setVisible((not is_mixed) and is_delaunay)
        self.bw_radio.setVisible((not is_mixed) and is_delaunay)
        self.triangle_radio.setVisible((not is_mixed) and is_delaunay)

    def _load_params(self):
        """加载现有参数"""
        # 1. 自动输出网格设置
        auto_output = self.params.get("auto_output", True)
        self.auto_output_check.setChecked(auto_output)

        output_path = self.params.get("output_file", "./out/mesh.vtk")
        if isinstance(output_path, list):
            output_path = output_path[0] if output_path else "./out/mesh.vtk"
        self.output_path_edit.setText(output_path)

        self._on_auto_output_toggled(Qt.Checked if auto_output else Qt.Unchecked)

        # 2. 信息输出等级设置
        verbosity = self.params.get("debug_level", 0)
        self.verbosity_combo.setCurrentIndex(verbosity)

        # 3. 网格类型 / 算法设置
        mesh_type = self.params.get("mesh_type", 1)
        combo_index = 0 if mesh_type in (1, 4) else 1
        self.mesh_type_combo.setCurrentIndex(combo_index)

        # 三角形合并算法
        triangle_to_quad_method = self.params.get("triangle_to_quad_method", "q_morph")
        method_index = self.triangle_to_quad_combo.findText(triangle_to_quad_method)
        if method_index >= 0:
            self.triangle_to_quad_combo.setCurrentIndex(method_index)

        # Delaunay backend
        delaunay_backend = self.params.get("delaunay_backend", "bowyer_watson")
        if delaunay_backend == "triangle":
            self.triangle_radio.setChecked(True)
        else:
            self.bw_radio.setChecked(True)

        # 按 mesh_type 设置算法下拉
        if combo_index == 0:
            algo = "delaunay" if mesh_type == 4 else "advancing_front"
        else:
            algo = "q_morph" if triangle_to_quad_method == "q_morph" else "hybrid"
        algo_index = self.mesh_algorithm_combo.findData(algo)
        if algo_index >= 0:
            self.mesh_algorithm_combo.setCurrentIndex(algo_index)

        self._update_algorithm_dependent_controls()

        # 4. 全局网格尺寸设置
        global_size = self.params.get("global_max_size", 1e6)
        self.global_size_edit.setText(str(global_size))

        # 5. 尺寸场衰减参数设置
        self.sizing_decay_spin.setValue(float(self.params.get("sizing_decay", 1.2)))

    def get_params(self):
        """获取用户设置的参数"""
        params = {}

        # 1. 自动输出网格设置
        params["auto_output"] = self.auto_output_check.isChecked()
        params["output_file"] = [self.output_path_edit.text()]

        # 2. 信息输出等级设置
        params["debug_level"] = self.verbosity_combo.currentIndex()

        # 3. 网格类型 / 算法设置
        combo_index = self.mesh_type_combo.currentIndex()
        selected_algo = self.mesh_algorithm_combo.currentData()
        if combo_index == 0:
            params["mesh_type"] = 4 if selected_algo == "delaunay" else 1
        else:
            params["mesh_type"] = 3

        # 三角形合并算法（从下拉框获取当前选择的值）
        params["triangle_to_quad_method"] = self.triangle_to_quad_combo.currentText()
        params["delaunay_backend"] = (
            "triangle" if self.triangle_radio.isChecked() else "bowyer_watson"
        )
        params["sizing_decay"] = float(self.sizing_decay_spin.value())

        # 4. 全局网格尺寸设置
        try:
            params["global_max_size"] = float(self.global_size_edit.text())
        except ValueError:
            params["global_max_size"] = 1.0

        return params

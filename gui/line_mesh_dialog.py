# -*- coding: utf-8 -*-
"""
线网格生成参数设置对话框
提供线网格生成的参数设置界面，支持多种离散化方法
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QRadioButton, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QLineEdit, QGridLayout, QMessageBox, QFrame, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal


class LineMeshGenerationDialog(QDialog):
    """线网格生成参数设置对话框"""
    
    # 定义离散化方法
    UNIFORM = "uniform"
    GEOMETRIC = "geometric"
    TANH = "tanh"
    
    dialog_closed = pyqtSignal()
    generation_requested = pyqtSignal(dict)
    pick_confirmed = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("线网格生成参数设置")
        self.setMinimumWidth(500)
        self.gui = parent
        self.selected_edges = []
        self._picking_enabled = False
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 几何线选择组
        self.create_line_selection_group(layout)
        
        # 离散化方法组
        self.create_discretization_method_group(layout)
        
        # 离散化参数组
        self.create_discretization_params_group(layout)
        
        # 边界条件组
        self.create_boundary_condition_group(layout)
        
        # 按钮组
        self.create_button_group(layout)
        
    def create_line_selection_group(self, parent_layout):
        """创建几何线选择组"""
        group = QGroupBox("几何线选择")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # 当前选中的线信息
        info_layout = QHBoxLayout()
        self.lbl_selected_lines = QLabel("未选择几何线")
        self.lbl_selected_lines.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self.lbl_selected_lines)
        layout.addLayout(info_layout)
        
        # 已选边列表
        self.edge_list_widget = QListWidget()
        self.edge_list_widget.setMaximumHeight(100)
        self.edge_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                background: white;
            }
            QListWidget::item {
                padding: 2px;
            }
        """)
        layout.addWidget(self.edge_list_widget)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.btn_pick_lines = QPushButton("拾取几何线")
        self.btn_pick_lines.setCheckable(True)
        self.btn_pick_lines.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
        """)
        self.btn_pick_lines.clicked.connect(self.on_pick_lines_clicked)
        button_layout.addWidget(self.btn_pick_lines)
        
        self.btn_clear_lines = QPushButton("清除选择")
        self.btn_clear_lines.clicked.connect(self.on_clear_lines_clicked)
        button_layout.addWidget(self.btn_clear_lines)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_discretization_method_group(self, parent_layout):
        """创建离散化方法选择组"""
        group = QGroupBox("离散化方法")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        self.radio_uniform = QRadioButton("均匀分布 - 各段长度相等")
        self.radio_uniform.setChecked(True)
        self.radio_uniform.toggled.connect(self.on_method_changed)
        layout.addWidget(self.radio_uniform)
        
        self.radio_geometric = QRadioButton("几何级数分布 - 段长按固定比率增长")
        self.radio_geometric.toggled.connect(self.on_method_changed)
        layout.addWidget(self.radio_geometric)
        
        self.radio_tanh = QRadioButton("Tanh函数分布 - 两端密中间疏")
        self.radio_tanh.toggled.connect(self.on_method_changed)
        layout.addWidget(self.radio_tanh)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_discretization_params_group(self, parent_layout):
        """创建离散化参数组"""
        group = QGroupBox("离散化参数")
        layout = QGridLayout()
        layout.setSpacing(10)
        
        layout.addWidget(QLabel("网格数量:"), 0, 0)
        self.spin_num_elements = QSpinBox()
        self.spin_num_elements.setRange(2, 1000)
        self.spin_num_elements.setValue(10)
        self.spin_num_elements.valueChanged.connect(self.on_params_changed)
        layout.addWidget(self.spin_num_elements, 0, 1)
        
        layout.addWidget(QLabel("起始尺寸:"), 1, 0)
        self.spin_start_size = QDoubleSpinBox()
        self.spin_start_size.setRange(1e-6, 1e6)
        self.spin_start_size.setValue(0.1)
        self.spin_start_size.setDecimals(6)
        self.spin_start_size.setSuffix(" (可选)")
        layout.addWidget(self.spin_start_size, 1, 1)
        
        layout.addWidget(QLabel("结束尺寸:"), 2, 0)
        self.spin_end_size = QDoubleSpinBox()
        self.spin_end_size.setRange(1e-6, 1e6)
        self.spin_end_size.setValue(0.2)
        self.spin_end_size.setDecimals(6)
        self.spin_end_size.setSuffix(" (可选)")
        layout.addWidget(self.spin_end_size, 2, 1)
        
        layout.addWidget(QLabel("增长比率:"), 3, 0)
        self.spin_growth_rate = QDoubleSpinBox()
        self.spin_growth_rate.setRange(0.1, 10.0)
        self.spin_growth_rate.setValue(1.2)
        self.spin_growth_rate.setDecimals(2)
        self.spin_growth_rate.setSingleStep(0.1)
        layout.addWidget(self.spin_growth_rate, 3, 1)
        
        layout.addWidget(QLabel("Tanh拉伸系数:"), 4, 0)
        self.spin_tanh_factor = QDoubleSpinBox()
        self.spin_tanh_factor.setRange(0.1, 10.0)
        self.spin_tanh_factor.setValue(2.0)
        self.spin_tanh_factor.setDecimals(2)
        self.spin_tanh_factor.setSingleStep(0.1)
        layout.addWidget(self.spin_tanh_factor, 4, 1)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
        self.update_params_group_state()
        
    def create_boundary_condition_group(self, parent_layout):
        """创建边界条件组"""
        group = QGroupBox("边界条件")
        layout = QGridLayout()
        layout.setSpacing(10)
        
        layout.addWidget(QLabel("边界类型:"), 0, 0)
        self.combo_bc_type = QComboBox()
        self.combo_bc_type.addItems([
            "无", "壁面(wall)", "进流(inflow)", "出流(outflow)",
            "对称(symmetry)", "周期(periodic)", "内部(interior)"
        ])
        layout.addWidget(self.combo_bc_type, 0, 1)
        
        layout.addWidget(QLabel("部件名称:"), 1, 0)
        self.edit_part_name = QLineEdit("default_line")
        layout.addWidget(self.edit_part_name, 1, 1)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def create_button_group(self, parent_layout):
        """创建按钮组"""
        layout = QHBoxLayout()
        layout.setSpacing(10)
        
        layout.addStretch()
        
        self.btn_preview = QPushButton("预览")
        self.btn_preview.clicked.connect(self.on_preview_clicked)
        layout.addWidget(self.btn_preview)
        
        self.btn_generate = QPushButton("生成线网格")
        self.btn_generate.clicked.connect(self.on_generate_clicked)
        layout.addWidget(self.btn_generate)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        layout.addWidget(self.btn_cancel)
        
        parent_layout.addLayout(layout)
        
    def get_discretization_method(self):
        """获取当前选择的离散化方法"""
        if self.radio_uniform.isChecked():
            return self.UNIFORM
        elif self.radio_geometric.isChecked():
            return self.GEOMETRIC
        elif self.radio_tanh.isChecked():
            return self.TANH
        return self.UNIFORM
        
    def get_parameters(self):
        """获取所有参数"""
        return {
            'method': self.get_discretization_method(),
            'num_elements': self.spin_num_elements.value(),
            'start_size': self.spin_start_size.value(),
            'end_size': self.spin_end_size.value(),
            'growth_rate': self.spin_growth_rate.value(),
            'tanh_factor': self.spin_tanh_factor.value(),
            'bc_type': self.combo_bc_type.currentText(),
            'part_name': self.edit_part_name.text(),
            'edges': list(self.selected_edges)
        }
        
    def set_selected_lines_count(self, count):
        """设置选中的几何线数量"""
        if count > 0:
            self.lbl_selected_lines.setText(f"已选择 {count} 条几何线")
            self.lbl_selected_lines.setStyleSheet("color: #006400; font-weight: bold;")
        else:
            self.lbl_selected_lines.setText("未选择几何线")
            self.lbl_selected_lines.setStyleSheet("color: #666; font-style: italic;")
            
    def on_method_changed(self):
        """离散化方法改变时的处理"""
        self.update_params_group_state()
        
    def update_params_group_state(self):
        """更新参数组的状态"""
        method = self.get_discretization_method()
        
        if method == self.UNIFORM:
            self.spin_start_size.setEnabled(False)
            self.spin_end_size.setEnabled(False)
            self.spin_growth_rate.setEnabled(False)
            self.spin_tanh_factor.setEnabled(False)
        elif method == self.GEOMETRIC:
            self.spin_start_size.setEnabled(True)
            self.spin_end_size.setEnabled(True)
            self.spin_growth_rate.setEnabled(True)
            self.spin_tanh_factor.setEnabled(False)
        elif method == self.TANH:
            self.spin_start_size.setEnabled(True)
            self.spin_end_size.setEnabled(True)
            self.spin_growth_rate.setEnabled(False)
            self.spin_tanh_factor.setEnabled(True)
            
    def on_params_changed(self):
        """参数改变时的处理"""
        pass
        
    def on_pick_lines_clicked(self):
        """点击拾取几何线按钮"""
        if self.btn_pick_lines.isChecked():
            self.start_edge_picking()
        else:
            self.stop_edge_picking()
            
    def start_edge_picking(self):
        """启动边拾取模式"""
        if not self.gui:
            QMessageBox.warning(self, "警告", "无法获取GUI对象")
            self.btn_pick_lines.setChecked(False)
            return
            
        # 检查是否有几何显示
        if not hasattr(self.gui, 'mesh_display') or not self.gui.mesh_display:
            QMessageBox.warning(self, "警告", "请先导入或创建几何")
            self.btn_pick_lines.setChecked(False)
            return
            
        # 检查是否有view_controller
        if not hasattr(self.gui, 'view_controller') or not self.gui.view_controller:
            QMessageBox.warning(self, "警告", "无法获取视图控制器")
            self.btn_pick_lines.setChecked(False)
            return
            
        # 启用边拾取
        self.btn_pick_lines.setText("取消拾取")
        
        # 通过view_controller启动拾取模式，它会自动显示提示信息
        self.gui.view_controller.start_geometry_pick(
            on_pick=self._on_edge_pick,
            on_confirm=self._on_pick_confirm,
            on_cancel=self._on_pick_cancel,
            on_delete=self._on_pick_delete
        )
        self._picking_enabled = True
            
    def stop_edge_picking(self):
        """停止边拾取模式"""
        self.btn_pick_lines.setText("拾取几何线")
        
        if self.gui and hasattr(self.gui, 'view_controller') and self.gui.view_controller:
            self.gui.view_controller.stop_geometry_pick(restore_display_mode=False)
        self._picking_enabled = False
            
    def _init_picking_helper(self):
        """初始化拾取助手 - 现在通过view_controller管理，不需要单独初始化"""
        pass
        
    def _on_edge_pick(self, element_type, element_obj, element_index):
        """边拾取回调"""
        if element_type not in ("edge", "edges"):
            return
            
        # 检查是否已经选择
        for edge in self.selected_edges:
            if edge['obj'] == element_obj:
                return
                
        # 添加到已选列表
        edge_info = {
            'type': element_type,
            'obj': element_obj,
            'index': element_index,
            'name': f"边 {len(self.selected_edges) + 1}"
        }
        self.selected_edges.append(edge_info)
        
        # 更新UI
        self._update_edge_list()
        self.set_selected_lines_count(len(self.selected_edges))
        
    def _on_pick_confirm(self):
        """确认选择"""
        # 先获取拾取结果，再停止拾取模式
        if self.gui and hasattr(self.gui, 'view_controller') and self.gui.view_controller:
            picking_helper = getattr(self.gui.view_controller, '_picking_helper', None)
            if picking_helper is not None:
                self.selected_edges = picking_helper.get_picked_elements()
        
        # 停止拾取模式
        self.stop_edge_picking()
        
        self.update_status(f"已确认选择 {len(self.selected_edges)} 条几何线")
        self.pick_confirmed.emit(self.selected_edges)
        
    def _on_pick_cancel(self):
        """取消选择"""
        self.stop_edge_picking()
        self.clear_selection()
        
    def _on_pick_delete(self):
        """删除最后选择的边"""
        if self.selected_edges:
            self.selected_edges.pop()
            self._update_edge_list()
            self.set_selected_lines_count(len(self.selected_edges))
            
    def _update_edge_list(self):
        """更新边列表UI"""
        self.edge_list_widget.clear()
        for i, edge in enumerate(self.selected_edges):
            item = QListWidgetItem(f"边 {i + 1}")
            self.edge_list_widget.addItem(item)
            
    def clear_selection(self):
        """清除所有选择"""
        self.selected_edges = []
        self.edge_list_widget.clear()
        self.set_selected_lines_count(0)
        
    def on_clear_lines_clicked(self):
        """清除选择的线"""
        self.clear_selection()
        if self._picking_enabled and self.gui and hasattr(self.gui, 'view_controller') and self.gui.view_controller:
            self.gui.view_controller.stop_geometry_pick(restore_display_mode=False)
            self._picking_enabled = False
            self.btn_pick_lines.setChecked(False)
        
    def on_preview_clicked(self):
        """预览按钮点击"""
        params = self.get_parameters()
        if not params['edges']:
            QMessageBox.warning(self, "警告", "请先选择几何线")
            return
        self.accept()
        self.generation_requested.emit(params)
        
    def on_generate_clicked(self):
        """生成按钮点击"""
        params = self.get_parameters()
        if not params['edges']:
            QMessageBox.warning(self, "警告", "请先选择几何线")
            return
        self.accept()
        self.generation_requested.emit(params)
        
    def update_status(self, message):
        """更新状态栏"""
        if self.gui and hasattr(self.gui, 'update_status'):
            self.gui.update_status(message)
            
    def closeEvent(self, event):
        """关闭事件"""
        if self._picking_enabled and self.gui and hasattr(self.gui, 'view_controller') and self.gui.view_controller:
            self.gui.view_controller.stop_geometry_pick(restore_display_mode=False)
        self.dialog_closed.emit()
        super().closeEvent(event)

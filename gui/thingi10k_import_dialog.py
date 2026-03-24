# -*- coding: utf-8 -*-
"""
Thingi10K模型导入对话框
用于从Thingi10K数据集中导入指定ID和类型的网格模型
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QRadioButton, QComboBox, QPushButton, QLineEdit,
    QGridLayout, QMessageBox, QFrame, QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread


class Thingi10kImportDialog(QDialog):
    """Thingi10K模型导入对话框"""

    importRequested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导入Thingi10K模型")
        self.setMinimumWidth(450)
        self.gui = parent
        self.selected_variant = "raw"
        self._create_widgets()
        self._connect_signals()

    def _create_widgets(self):
        """创建对话框部件"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info_group = QGroupBox("数据集信息")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "Thingi10K数据集包含超过10,000个来自Thingiverse的3D模型。\n"
            "请输入模型ID和选择数据类型进行导入。"
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        id_group = QGroupBox("模型ID")
        id_layout = QGridLayout()

        id_layout.addWidget(QLabel("文件ID:"), 0, 0)
        self.file_id_edit = QLineEdit()
        self.file_id_edit.setPlaceholderText("例如: 100010")
        self.file_id_edit.setStyleSheet("background-color: white;")
        id_layout.addWidget(self.file_id_edit, 0, 1)

        id_layout.addWidget(QLabel("常用ID示例:"), 1, 0)
        example_label = QLabel("100010, 101525, 105242, 109887, 111277, 113990, 117071, 118041")
        example_label.setStyleSheet("color: #666; font-size: 8pt;")
        id_layout.addWidget(example_label, 1, 1)

        id_group.setLayout(id_layout)
        layout.addWidget(id_group)

        variant_group = QGroupBox("数据类型")
        variant_layout = QVBoxLayout()

        self.radio_raw = QRadioButton("raw (原始网格数据)")
        self.radio_raw.setChecked(True)
        variant_layout.addWidget(self.radio_raw)

        self.radio_tetwild = QRadioButton("tetwild (四面体化网格数据)")
        variant_layout.addWidget(self.radio_tetwild)

        variant_note = QLabel(
            "提示: raw类型包含原始三角形网格，tetwild类型包含四面体化后的网格数据。"
        )
        variant_note.setStyleSheet("color: #666; font-size: 8pt; padding-left: 20px;")
        variant_layout.addWidget(variant_note)

        variant_group.setLayout(variant_layout)
        layout.addWidget(variant_group)

        button_group = QGroupBox()
        button_layout = QHBoxLayout()

        button_layout.addStretch()

        self.btn_import = QPushButton("导入")
        self.btn_import.setDefault(True)
        self.btn_import.setMinimumWidth(100)
        button_layout.addWidget(self.btn_import)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setMinimumWidth(100)
        button_layout.addWidget(self.btn_cancel)

        button_group.setLayout(button_layout)
        layout.addWidget(button_group)

    def _connect_signals(self):
        """连接信号"""
        self.radio_raw.toggled.connect(lambda: self._on_variant_changed("raw"))
        self.radio_tetwild.toggled.connect(lambda: self._on_variant_changed("tetwild"))
        self.btn_import.clicked.connect(self._on_import_clicked)
        self.btn_cancel.clicked.connect(self.reject)

    def _on_variant_changed(self, variant):
        """变体类型改变时的处理"""
        if variant == "raw" and self.radio_raw.isChecked():
            self.selected_variant = "raw"
        elif variant == "tetwild" and self.radio_tetwild.isChecked():
            self.selected_variant = "tetwild"

    def _on_import_clicked(self):
        """点击导入按钮"""
        file_id = self.file_id_edit.text().strip()

        if not file_id:
            QMessageBox.warning(self, "警告", "请输入文件ID")
            return

        if not file_id.isdigit():
            QMessageBox.warning(self, "警告", "文件ID必须是数字")
            return

        self.importRequested.emit(file_id, self.selected_variant)
        self.accept()

    def get_file_id(self):
        """获取文件ID"""
        return self.file_id_edit.text().strip()

    def get_variant(self):
        """获取选择的数据类型"""
        return self.selected_variant


class Thingi10kImportThread(QThread):
    """Thingi10K模型导入线程"""

    progress_updated = pyqtSignal(int, str)
    import_finished = pyqtSignal(bool, str, object)
    import_failed = pyqtSignal(str)

    def __init__(self, file_id, variant="raw", cache_dir=None):
        super().__init__()
        self.file_id = file_id
        self.variant = variant
        self.cache_dir = cache_dir

    def run(self):
        """执行导入"""
        try:
            from fileIO.thingi10k_loader import load_thingi10k_mesh, get_thingi10k_mesh_statistics

            self.progress_updated.emit(10, f"正在初始化Thingi10K数据集 ({self.variant})...")

            self.progress_updated.emit(30, f"正在加载模型 ID: {self.file_id}...")

            mesh_data = load_thingi10k_mesh(self.file_id, self.cache_dir, self.variant)

            if mesh_data is None:
                self.import_failed.emit(f"无法加载模型 ID: {self.file_id}，请检查ID是否正确或网络连接")
                return

            vertices, facets = mesh_data

            self.progress_updated.emit(70, "正在处理网格数据...")

            self.progress_updated.emit(90, "正在完成...")

            stats = {
                'num_vertices': len(vertices),
                'num_facets': len(facets),
                'num_triangles': sum(1 for f in facets if len(f) == 3),
                'num_quads': sum(1 for f in facets if len(f) == 4),
            }

            self.import_finished.emit(True, f"成功加载模型 ID: {self.file_id}", {
                'vertices': vertices,
                'facets': facets,
                'statistics': stats
            })

        except ImportError as e:
            self.import_failed.emit(f"thingi10k模块未安装: {str(e)}")
        except Exception as e:
            self.import_failed.emit(f"导入失败: {str(e)}")
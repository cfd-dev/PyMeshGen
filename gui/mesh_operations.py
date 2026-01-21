#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网格操作模块
处理网格生成、质量检查、平滑、优化等操作
"""

import os
import sys
import json
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QDialog, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QButtonGroup, QRadioButton
from PyQt5.QtCore import Qt
from gui.ui_utils import PARTS_INFO_RESERVED_KEYS


class MeshOperations:
    """网格操作类"""

    def __init__(self, gui_instance):
        self.gui = gui_instance

    def import_mesh(self):
        """导入网格（使用异步线程，避免GUI卡顿）"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        from gui.file_operations import FileOperations
        from gui.import_thread import MeshImportThread

        file_path, _ = QFileDialog.getOpenFileName(
            self.gui,
            "导入网格文件",
            os.path.join(self.gui.project_root, "meshes"),
            "网格文件 (*.vtk *.vtu *.stl *.obj *.cas *.msh *.ply *.xdmf *.xmf *.off *.med *.mesh *.meshb *.bdf *.fem *.nas *.inp *.e *.exo *.ex2 *.su2 *.cgns *.avs *.vol *.mdpa *.h5m *.f3grid *.dat *.tec *.ugrid *.ele *.node *.xml *.post *.wkt *.hmf);;所有文件 (*.*)"
        )

        if file_path:
            try:
                file_ops = FileOperations(self.gui.project_root, log_callback=self.gui.log_info)

                # 创建导入线程
                self.gui.import_thread = MeshImportThread(file_path, file_ops)

                # 连接信号
                self.gui.import_thread.progress_updated.connect(self.gui.on_import_progress)
                self.gui.import_thread.import_finished.connect(self.gui.on_import_finished)
                self.gui.import_thread.import_failed.connect(self.gui.on_import_failed)

                # 禁用导入按钮，防止重复导入
                if hasattr(self.gui, '_reset_progress_cache'):
                    self.gui._reset_progress_cache("mesh")
                if hasattr(self.gui, '_set_ribbon_button_enabled'):
                    self.gui._set_ribbon_button_enabled('file', 'import', False)

                # 启动线程
                self.gui.import_thread.start()
                self.gui.log_info(f"开始导入网格: {file_path}")
                self.gui.update_status("正在导入网格...")

            except Exception as e:
                QMessageBox.critical(self.gui, "错误", f"导入网格失败: {str(e)}")
                self.gui.log_error(f"导入网格失败: {str(e)}")

    def set_mesh_dimension(self):
        """设置网格维度"""
        dialog = QDialog(self.gui)
        dialog.setWindowTitle("设置网格维度")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)

        label = QLabel("请选择网格维度:")
        layout.addWidget(label)

        button_group = QButtonGroup(dialog)
        radio_2d = QRadioButton("2D")
        radio_3d = QRadioButton("3D")
        button_group.addButton(radio_2d, 2)
        button_group.addButton(radio_3d, 3)

        current_dim = getattr(self.gui, 'mesh_dimension', 2)
        if current_dim == 3:
            radio_3d.setChecked(True)
        else:
            radio_2d.setChecked(True)

        layout.addWidget(radio_2d)
        layout.addWidget(radio_3d)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        cancel_button = QPushButton("取消")
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        if dialog.exec_() != QDialog.Accepted:
            self.gui.log_info("已取消网格维度设置")
            self.gui.update_status("已取消网格维度设置")
            return

        selected_dim = button_group.checkedId()
        if selected_dim not in (2, 3):
            return

        if hasattr(self.gui, '_apply_mesh_dimension'):
            self.gui._apply_mesh_dimension(selected_dim)
        else:
            self.gui.mesh_dimension = selected_dim
            if hasattr(self.gui.current_mesh, 'dimension'):
                self.gui.current_mesh.dimension = selected_dim
            if hasattr(self.gui, 'status_bar'):
                self.gui.status_bar.update_mesh_dimension(selected_dim)
        self.gui.log_info(f"网格维度已设置为 {selected_dim}D")
        self.gui.update_status("网格维度已更新")

    def export_mesh(self):
        """导出网格"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        from gui.file_operations import FileOperations

        if not self.gui.current_mesh:
            QMessageBox.warning(self.gui, "警告", "没有可导出的网格")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.gui,
            "导出网格文件",
            os.path.join(self.gui.project_root, "meshes", "mesh.vtk"),
            "网格文件 (*.vtk *.stl *.obj *.msh *.ply)"
        )

        if file_path:
            try:
                file_ops = FileOperations(self.gui.project_root)
                vtk_poly_data = None
                if isinstance(self.gui.current_mesh, dict):
                    vtk_poly_data = self.gui.current_mesh.get('vtk_poly_data')

                if vtk_poly_data:
                    file_ops.export_mesh(vtk_poly_data, file_path)
                else:
                    if hasattr(self.gui.current_mesh, 'node_coords') and hasattr(self.gui.current_mesh, 'cell_container'):
                        if hasattr(self.gui.current_mesh, 'save_to_vtkfile'):
                            self.gui.current_mesh.save_to_vtkfile(file_path)
                        else:
                            QMessageBox.warning(self.gui, "警告", "当前网格格式不支持直接保存，请使用VTK格式")
                            return
                    else:
                        QMessageBox.warning(self.gui, "警告", "无法获取有效的VTK数据进行导出")
                        return

                self.gui.log_info(f"已导出网格文件: {file_path}")
                self.gui.update_status("已导出网格文件")
            except Exception as e:
                QMessageBox.critical(self.gui, "错误", f"导出网格失败: {str(e)}")
                self.gui.log_error(f"导出网格失败: {str(e)}")

    def _require_mesh(self, log_message, status_message="未找到网格数据", prompt_message="请先生成或导入网格"):
        """确保存在当前网格，否则提示并返回None"""
        if not hasattr(self.gui, 'current_mesh') or not self.gui.current_mesh:
            QMessageBox.warning(self.gui, "警告", prompt_message)
            self.gui.log_info(log_message)
            self.gui.update_status(status_message)
            return None
        return self.gui.current_mesh

    def _get_mesh_obj(self):
        """获取当前网格对象（优先使用unstr_grid）"""
        mesh_obj = self.gui.current_mesh
        if hasattr(self.gui.current_mesh, 'unstr_grid') and self.gui.current_mesh.unstr_grid:
            mesh_obj = self.gui.current_mesh.unstr_grid
        return mesh_obj

    def _require_cell_container(self, mesh_obj, warning_message, log_message, status_message="网格格式不支持"):
        """确保网格对象包含cell_container，否则提示并返回False"""
        if not hasattr(mesh_obj, 'cell_container'):
            QMessageBox.warning(self.gui, "警告", warning_message)
            self.gui.log_info(log_message)
            self.gui.update_status(status_message)
            return False
        return True

    def _collect_quality_metrics(self, mesh_obj):
        """收集网格质量与偏斜度统计值"""
        quality_values = []
        skewness_values = []
        if hasattr(mesh_obj, 'cell_container'):
            for cell in mesh_obj.cell_container:
                if hasattr(cell, 'init_metrics'):
                    cell.init_metrics()
                if cell.quality is not None:
                    quality_values.append(cell.quality)
                if cell.skewness is not None:
                    skewness_values.append(cell.skewness)
        return quality_values, skewness_values

    def generate_mesh(self):
        """生成网格 - 使用异步线程避免UI冻结"""
        try:
            if self.gui.mesh_generation_thread and self.gui.mesh_generation_thread.isRunning():
                QMessageBox.warning(self.gui, "警告", "网格生成任务正在进行中，请稍候...")
                return

            if not hasattr(self.gui, 'current_mesh') or not self.gui.current_mesh:
                QMessageBox.warning(self.gui, "警告", "请先导入网格文件")
                self.gui.log_info("未导入网格文件，无法生成网格")
                self.gui.update_status("未导入网格文件")
                return

            if not hasattr(self.gui, 'parts_params') or not self.gui.parts_params:
                QMessageBox.warning(self.gui, "警告", "请先配置部件参数")
                self.gui.log_info("未配置部件参数，无法生成网格")
                self.gui.update_status("未配置部件参数")
                return

            input_file = ""

            if isinstance(self.gui.current_mesh, dict):
                if 'file_path' in self.gui.current_mesh:
                    input_file = self.gui.current_mesh['file_path']
            elif hasattr(self.gui.current_mesh, 'file_path'):
                input_file = self.gui.current_mesh.file_path

            if input_file:
                input_file = os.path.abspath(input_file)

            has_mesh_data = False
            if isinstance(self.gui.current_mesh, dict):
                has_mesh_data = 'node_coords' in self.gui.current_mesh and 'cells' in self.gui.current_mesh
            elif hasattr(self.gui, 'current_mesh') and hasattr(self.gui.current_mesh, 'node_coords') and hasattr(self.gui.current_mesh, 'cells'):
                has_mesh_data = True

            has_parts_info = hasattr(self.gui, 'cas_parts_info') and self.gui.cas_parts_info

            if not has_mesh_data and not has_parts_info:
                QMessageBox.warning(self.gui, "警告", "无法获取有效的输入网格数据或部件信息")
                self.gui.log_info("无法获取有效的输入网格数据或部件信息")
                self.gui.update_status("无法获取有效的输入网格数据或部件信息")
                return

            config_data = {
                "debug_level": self.gui.params.debug_level if hasattr(self.gui, 'params') and self.gui.params else 0,
                "output_file": self.gui.params.output_file if hasattr(self.gui, 'params') and self.gui.params else ["./out/mesh.vtk"],
                "viz_enabled": False,
                "parts": self.gui.parts_params,
                "mesh_type": self.gui.params.mesh_type if hasattr(self.gui, 'params') and self.gui.params else 1,
                "auto_output": self.gui.params.auto_output if hasattr(self.gui, 'params') and self.gui.params else True
            }

            if hasattr(self.gui, 'cas_parts_info') and self.gui.cas_parts_info and not self.gui.parts_params:
                for part_name in self.gui.cas_parts_info.keys():
                    config_data["parts"].append({
                        "part_name": part_name,
                        "max_size": 1e6,
                        "PRISM_SWITCH": "off",
                        "first_height": 0.01,
                        "growth_rate": 1.2,
                        "max_layers": 5,
                        "full_layers": 5,
                        "multi_direction": False
                    })

            config_data["input_file"] = input_file if input_file else ""

            if config_data["output_file"]:
                if isinstance(config_data["output_file"], list):
                    config_data["output_file"] = [os.path.abspath(f) for f in config_data["output_file"]]
                else:
                    config_data["output_file"] = os.path.abspath(config_data["output_file"])

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(config_data, temp_file, indent=2)
                temp_file_path = temp_file.name

            try:
                self.gui.log_info("正在构建网格生成参数...")
                self.gui.update_status("正在构建网格生成参数...")

                from data_structure.parameters import Parameters
                params = Parameters("FROM_CASE_JSON", temp_file_path)

                self.gui.progress_dialog = QProgressDialog("正在生成网格...", "取消", 0, 100, self.gui)
                self.gui.progress_dialog.setWindowTitle("网格生成进度")
                self.gui.progress_dialog.setWindowModality(Qt.WindowModal)
                self.gui.progress_dialog.setMinimumDuration(0)
                self.gui.progress_dialog.setAutoClose(False)
                self.gui.progress_dialog.setAutoReset(False)
                self.gui.progress_dialog.show()

                from gui.mesh_generation_thread import MeshGenerationThread
                self.gui.mesh_generation_thread = MeshGenerationThread(params, self.gui.current_mesh, self.gui)

                if hasattr(self.gui, '_reset_progress_cache'):
                    self.gui._reset_progress_cache("mesh_generation")

                self.gui.mesh_generation_thread.signals.progress.connect(self._on_mesh_progress)
                self.gui.mesh_generation_thread.signals.finished.connect(self._on_mesh_finished)
                self.gui.mesh_generation_thread.signals.error.connect(self._on_mesh_error)
                self.gui.mesh_generation_thread.signals.log.connect(self._on_mesh_log)

                self.gui.progress_dialog.canceled.connect(self._cancel_mesh_generation)

                self.gui.mesh_generation_thread.start()

                self.gui.log_info("网格生成任务已启动...")
                self.gui.update_status("网格生成中...")

            except Exception as e:
                if self.gui.progress_dialog:
                    self.gui.progress_dialog.close()
                raise e

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"启动网格生成失败: {str(e)}")
            self.gui.log_error(f"启动网格生成失败: {str(e)}")
            self.gui.update_status("启动网格生成失败")
            if self.gui.progress_dialog:
                self.gui.progress_dialog.close()

    def _on_mesh_progress(self, progress, description):
        """处理网格生成进度更新"""
        if self.gui.progress_dialog:
            self.gui.progress_dialog.setValue(progress)
            self.gui.progress_dialog.setLabelText(description)
        if hasattr(self.gui, '_update_progress'):
            self.gui._update_progress(f"网格生成: {description}", progress, "mesh_generation")
        else:
            self.gui.update_status(f"网格生成: {description} ({progress}%)")
        if progress >= 100:
            if self.gui.progress_dialog:
                self.gui.progress_dialog.close()
                self.gui.progress_dialog = None
            if hasattr(self.gui, 'status_bar'):
                self.gui.status_bar.hide_progress()

    def _on_mesh_finished(self, result_mesh):
        """处理网格生成完成"""
        try:
            if self.gui.progress_dialog:
                self.gui.progress_dialog.close()
            if hasattr(self.gui, 'status_bar'):
                self.gui.status_bar.hide_progress()

            if result_mesh:
                self.gui.current_mesh = result_mesh

                self.gui.log_info("网格生成完成")
                self.gui.update_status("网格生成完成")

                try:
                    from utils.geom_toolkit import detect_mesh_dimension_by_metadata
                    resolved_dim = detect_mesh_dimension_by_metadata(result_mesh, default_dim=self.gui.mesh_dimension)
                    if hasattr(self.gui, '_apply_mesh_dimension'):
                        self.gui._apply_mesh_dimension(resolved_dim)
                    else:
                        self.gui.mesh_dimension = resolved_dim
                except Exception:
                    pass
                if hasattr(self.gui, 'status_bar') and not hasattr(self.gui, '_apply_mesh_dimension'):
                    self.gui.status_bar.update_mesh_dimension(self.gui.mesh_dimension)

                self.gui.display_mesh()

                # 更新模型树中的网格统计信息
                if hasattr(self.gui, 'model_tree_widget'):
                    self.gui.model_tree_widget.load_mesh(result_mesh)

                # 更新部件列表以显示新网格的部件信息
                self.gui.part_manager.update_parts_list_from_generated_mesh(result_mesh)

                # Refresh display to show all parts with different colors
                self.gui.part_manager.refresh_display_all_parts()
            else:
                QMessageBox.warning(self.gui, "警告", "网格生成失败，未返回有效结果")
                self.gui.log_info("网格生成失败，未返回有效结果")
                self.gui.update_status("网格生成失败")

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"处理网格生成结果失败: {str(e)}")
            self.gui.log_error(f"处理网格生成结果失败: {str(e)}")

    def _on_mesh_error(self, error_msg):
        """处理网格生成错误"""
        if self.gui.progress_dialog:
            self.gui.progress_dialog.close()
        if hasattr(self.gui, 'status_bar'):
            self.gui.status_bar.hide_progress()
        QMessageBox.critical(self.gui, "错误", f"网格生成失败:\n{error_msg}")
        self.gui.log_error(f"网格生成失败: {error_msg}")
        self.gui.update_status("网格生成失败")

    def _on_mesh_log(self, message):
        """处理网格生成日志"""
        if message.startswith('[INFO]') or message.startswith('[ERROR]') or message.startswith('[WARNING]'):
            self.gui.info_output.append_info_output(message)
        else:
            self.gui.log_info(message)

    def _cancel_mesh_generation(self):
        """取消网格生成"""
        if self.gui.mesh_generation_thread and self.gui.mesh_generation_thread.isRunning():
            self.gui.log_info("正在取消网格生成...")
            self.gui.mesh_generation_thread.stop()
            self.gui.update_status("网格生成已取消")
            if self.gui.progress_dialog:
                self.gui.progress_dialog.close()
            if hasattr(self.gui, 'status_bar'):
                self.gui.status_bar.hide_progress()

    def check_mesh_quality(self):
        """检查网格质量 - 显示网格质量skewness直方图"""
        try:
            if not self._require_mesh("未找到网格数据，无法检查质量"):
                return

            mesh_obj = self._get_mesh_obj()

            if hasattr(mesh_obj, 'quality_histogram'):
                quality_values, skewness_values = self._collect_quality_metrics(mesh_obj)

                if quality_values:
                    quality_min = min(quality_values)
                    quality_max = max(quality_values)
                    quality_avg = sum(quality_values) / len(quality_values)

                    quality_stats = f"质量统计信息:\n"
                    quality_stats += f"  最小质量值: {quality_min:.4f}\n"
                    quality_stats += f"  最大质量值: {quality_max:.4f}\n"
                    quality_stats += f"  平均质量值: {quality_avg:.4f}\n"
                    quality_stats += f"  总单元数: {len(quality_values)}"

                    if hasattr(self.gui, 'info_output'):
                        self.gui.info_output.append_info_output(quality_stats)
                    else:
                        self.gui.log_info(quality_stats)
                else:
                    if hasattr(self.gui, 'info_output'):
                        self.gui.info_output.append_info_output("质量统计信息: 无质量数据")
                    else:
                        self.gui.log_info("质量统计信息: 无质量数据")

                if skewness_values:
                    skewness_min = min(skewness_values)
                    skewness_max = max(skewness_values)
                    skewness_avg = sum(skewness_values) / len(skewness_values)

                    skewness_stats = f"偏斜度统计信息:\n"
                    skewness_stats += f"  最小偏斜度: {skewness_min:.4f}\n"
                    skewness_stats += f"  最大偏斜度: {skewness_max:.4f}\n"
                    skewness_stats += f"  平均偏斜度: {skewness_avg:.4f}\n"
                    skewness_stats += f"  总单元数: {len(skewness_values)}"

                    if hasattr(self.gui, 'info_output'):
                        self.gui.info_output.append_info_output(skewness_stats)
                    else:
                        self.gui.log_info(skewness_stats)
                else:
                    if hasattr(self.gui, 'info_output'):
                        self.gui.info_output.append_info_output("偏斜度统计信息: 无偏斜度数据")
                    else:
                        self.gui.log_info("偏斜度统计信息: 无偏斜度数据")

                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.figure import Figure
                import matplotlib.pyplot as plt

                dialog = QDialog(self.gui)
                dialog.setWindowTitle("网格质量分析")
                dialog.setGeometry(100, 100, 800, 600)

                layout = QVBoxLayout(dialog)

                fig = Figure(figsize=(10, 6), dpi=100)
                canvas = FigureCanvas(fig)

                ax = fig.add_subplot(111)
                mesh_obj.skewness_histogram(ax)

                layout.addWidget(canvas)

                close_btn = QPushButton("关闭")
                close_btn.clicked.connect(dialog.close)
                layout.addWidget(close_btn)

                dialog.exec_()

                self.gui.log_info("网格质量分析完成")
                self.gui.update_status("网格质量分析完成")
            else:
                QMessageBox.information(self.gui, "提示", "当前网格数据不支持质量分析功能")
                self.gui.log_info("当前网格数据不支持质量分析")
                self.gui.update_status("网格数据不支持质量分析")

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"网格质量分析失败：{str(e)}")
            self.gui.log_info(f"网格质量分析失败：{str(e)}")
            self.gui.update_status("网格质量分析失败")

    def smooth_mesh(self):
        """平滑网格 - 使用laplacian光滑算法"""
        try:
            if not self._require_mesh("未找到网格数据，无法进行光滑处理"):
                return

            mesh_obj = self._get_mesh_obj()

            if not self._require_cell_container(mesh_obj, "当前网格格式不支持光滑处理", "当前网格格式不支持光滑处理"):
                return

            dialog = QDialog(self.gui)
            dialog.setWindowTitle("网格平滑设置")
            dialog.setModal(True)

            layout = QVBoxLayout(dialog)

            method_layout = QHBoxLayout()
            method_label = QLabel("平滑算法:")
            method_combo = QComboBox()
            method_combo.addItem("Laplacian", "laplacian")
            method_combo.addItem("基于角度的平滑", "angle_based")
            method_combo.addItem("基于GetMe方法的平滑", "getme")
            method_combo.addItem("基于NN的平滑", "nn_based")
            method_combo.addItem("节点扰动", "perturbation")
            method_layout.addWidget(method_label)
            method_layout.addWidget(method_combo)

            lap_iter_layout = QHBoxLayout()
            lap_iter_label = QLabel("Laplacian迭代次数:")
            lap_iter_spin = QSpinBox()
            lap_iter_spin.setRange(1, 50)
            lap_iter_spin.setValue(3)
            lap_iter_layout.addWidget(lap_iter_label)
            lap_iter_layout.addWidget(lap_iter_spin)

            iter_layout = QHBoxLayout()
            iter_label = QLabel("迭代次数:")
            iter_spin = QSpinBox()
            iter_spin.setRange(1, 100)
            iter_spin.setValue(3)
            iter_layout.addWidget(iter_label)
            iter_layout.addWidget(iter_spin)

            ratio_layout = QHBoxLayout()
            ratio_label = QLabel("扰动比例:")
            ratio_spin = QDoubleSpinBox()
            ratio_spin.setRange(0.0, 10.0)
            ratio_spin.setSingleStep(0.05)
            ratio_spin.setValue(0.5)
            ratio_layout.addWidget(ratio_label)
            ratio_layout.addWidget(ratio_spin)

            button_layout = QHBoxLayout()
            ok_button = QPushButton("确定")
            cancel_button = QPushButton("取消")
            button_layout.addStretch()
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)

            layout.addLayout(method_layout)
            layout.addLayout(iter_layout)
            layout.addLayout(ratio_layout)
            layout.addLayout(button_layout)

            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)

            def _sync_controls():
                is_perturb = (method_combo.currentData() == "perturbation")
                iter_label.setVisible(not is_perturb)
                iter_spin.setVisible(not is_perturb)
                ratio_label.setVisible(is_perturb)
                ratio_spin.setVisible(is_perturb)

            method_combo.currentIndexChanged.connect(_sync_controls)
            _sync_controls()

            if dialog.exec_() != QDialog.Accepted:
                self.gui.log_info("已取消网格平滑")
                self.gui.update_status("已取消网格平滑")
                return

            method_key = method_combo.currentData() or "laplacian"
            iterations = int(iter_spin.value())
            ratio = float(ratio_spin.value())

            from optimize.optimize import (
                laplacian_smooth,
                smooth_mesh_angle_based,
                smooth_mesh_getme,
                smooth_mesh_nn,
                node_perturbation,
            )

            if method_key == "angle_based":
                method_name = "基于角度的平滑"
                smooth_func = lambda m: smooth_mesh_angle_based(m, iterations=iterations)
            elif method_key == "getme":
                method_name = "基于GetMe方法的平滑"
                smooth_func = lambda m: smooth_mesh_getme(m, iterations=iterations)
            elif method_key == "perturbation":
                method_name = "节点扰动"
                smooth_func = lambda m: node_perturbation(m, ratio=ratio)
            elif method_key == "nn_based":
                if smooth_mesh_nn is None:
                    QMessageBox.information(self.gui, "提示", "当前环境未安装Torch，无法使用NN平滑")
                    self.gui.log_info("NN平滑不可用：未安装Torch")
                    self.gui.update_status("NN平滑不可用")
                    return
                method_name = "基于NN的平滑"
                smooth_func = lambda m: smooth_mesh_nn(m, iterations=iterations)
            else:
                method_name = "Laplacian"
                smooth_func = lambda m: laplacian_smooth(m, num_iter=iterations)

            self.gui.log_info(f"开始进行{method_name}处理...")
            self.gui.update_status(f"正在进行{method_name}处理...")

            start_time = time.time()

            smoothed_mesh = smooth_func(mesh_obj)

            if hasattr(self.gui.current_mesh, 'unstr_grid'):
                self.gui.current_mesh.unstr_grid = smoothed_mesh
                if hasattr(smoothed_mesh, 'node_coords'):
                    self.gui.current_mesh.node_coords = smoothed_mesh.node_coords
            else:
                self.gui.current_mesh = smoothed_mesh

            end_time = time.time()

            if hasattr(self.gui, 'mesh_visualizer') and self.gui.mesh_visualizer:
                self.gui.mesh_visualizer.update_mesh(self.gui.current_mesh)
            if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display:
                self.gui.mesh_display.clear_mesh_actors()
                self.gui.mesh_display.set_mesh_data(self.gui.current_mesh)
                self.gui.mesh_display.display_mesh(render_immediately=False)

            self.gui.log_info(f"{method_name}平滑处理完成，耗时: {end_time - start_time:.3f}秒")
            if hasattr(smoothed_mesh, 'cell_container'):
                self.gui.log_info(f"光滑后网格包含 {len(smoothed_mesh.cell_container)} 个单元")
            self.gui.update_status("网格光滑处理完成")

            if hasattr(self.gui, 'canvas'):
                self.gui.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"网格光滑处理失败：{str(e)}")
            self.gui.log_info(f"网格光滑处理失败：{str(e)}")
            self.gui.update_status("网格光滑处理失败")

    def optimize_mesh(self):
        """优化网格 - 使用edge_swap和laplacian_smooth算法"""
        try:
            if not self._require_mesh("未找到网格数据，无法进行优化"):
                return

            mesh_obj = self._get_mesh_obj()

            if not self._require_cell_container(mesh_obj, "当前网格格式不支持优化处理", "当前网格格式不支持优化处理"):
                return

            dialog = QDialog(self.gui)
            dialog.setWindowTitle("网格优化设置")
            dialog.setModal(True)

            layout = QVBoxLayout(dialog)

            method_layout = QHBoxLayout()
            method_label = QLabel("优化算法:")
            method_combo = QComboBox()
            method_combo.addItem("边交换 + Laplacian", "classic")
            method_combo.addItem("Adam优化(神经网络)", "adam")
            method_layout.addWidget(method_label)
            method_layout.addWidget(method_combo)

            lap_iter_layout = QHBoxLayout()
            lap_iter_label = QLabel("Laplacian迭代次数:")
            lap_iter_spin = QSpinBox()
            lap_iter_spin.setRange(1, 50)
            lap_iter_spin.setValue(3)
            lap_iter_layout.addWidget(lap_iter_label)
            lap_iter_layout.addWidget(lap_iter_spin)

            iter_layout = QHBoxLayout()
            iter_label = QLabel("迭代次数:")
            iter_spin = QSpinBox()
            iter_spin.setRange(1, 500)
            iter_spin.setValue(100)
            iter_layout.addWidget(iter_label)
            iter_layout.addWidget(iter_spin)

            lr_layout = QHBoxLayout()
            lr_label = QLabel("学习率:")
            lr_spin = QDoubleSpinBox()
            lr_spin.setRange(0.0, 1.0)
            lr_spin.setSingleStep(0.05)
            lr_spin.setValue(0.2)
            lr_layout.addWidget(lr_label)
            lr_layout.addWidget(lr_spin)

            movement_layout = QHBoxLayout()
            movement_label = QLabel("位移限制:")
            movement_spin = QDoubleSpinBox()
            movement_spin.setRange(0.0, 1.0)
            movement_spin.setSingleStep(0.01)
            movement_spin.setValue(0.1)
            movement_layout.addWidget(movement_label)
            movement_layout.addWidget(movement_spin)

            tol_layout = QHBoxLayout()
            tol_label = QLabel("收敛容差:")
            tol_spin = QDoubleSpinBox()
            tol_spin.setRange(0.0, 1.0)
            tol_spin.setSingleStep(0.01)
            tol_spin.setValue(0.1)
            tol_layout.addWidget(tol_label)
            tol_layout.addWidget(tol_spin)

            obj_layout = QHBoxLayout()
            obj_label = QLabel("目标函数:")
            obj_combo = QComboBox()
            obj_combo.addItem("L1", "L1")
            obj_combo.addItem("L2", "L2")
            obj_combo.addItem("Loo", "Loo")
            obj_layout.addWidget(obj_label)
            obj_layout.addWidget(obj_combo)

            pre_layout = QHBoxLayout()
            pre_label = QLabel("预平滑次数:")
            pre_spin = QSpinBox()
            pre_spin.setRange(0, 10)
            pre_spin.setValue(2)
            pre_layout.addWidget(pre_label)
            pre_layout.addWidget(pre_spin)

            inner_layout = QHBoxLayout()
            inner_label = QLabel("内层步数:")
            inner_spin = QSpinBox()
            inner_spin.setRange(1, 10)
            inner_spin.setValue(1)
            inner_layout.addWidget(inner_label)
            inner_layout.addWidget(inner_spin)

            lr_step_layout = QHBoxLayout()
            lr_step_label = QLabel("LR步长:")
            lr_step_spin = QSpinBox()
            lr_step_spin.setRange(1, 20)
            lr_step_spin.setValue(1)
            lr_step_layout.addWidget(lr_step_label)
            lr_step_layout.addWidget(lr_step_spin)

            lr_gamma_layout = QHBoxLayout()
            lr_gamma_label = QLabel("LR衰减:")
            lr_gamma_spin = QDoubleSpinBox()
            lr_gamma_spin.setRange(0.1, 1.0)
            lr_gamma_spin.setSingleStep(0.05)
            lr_gamma_spin.setValue(0.9)
            lr_gamma_layout.addWidget(lr_gamma_label)
            lr_gamma_layout.addWidget(lr_gamma_spin)

            button_layout = QHBoxLayout()
            ok_button = QPushButton("确定")
            cancel_button = QPushButton("取消")
            button_layout.addStretch()
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)

            layout.addLayout(method_layout)
            layout.addLayout(lap_iter_layout)
            layout.addLayout(iter_layout)
            layout.addLayout(lr_layout)
            layout.addLayout(movement_layout)
            layout.addLayout(tol_layout)
            layout.addLayout(obj_layout)
            layout.addLayout(pre_layout)
            layout.addLayout(inner_layout)
            layout.addLayout(lr_step_layout)
            layout.addLayout(lr_gamma_layout)
            layout.addLayout(button_layout)

            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)

            def _sync_optimize_controls():
                is_adam = (method_combo.currentData() == "adam")
                lap_iter_label.setVisible(not is_adam)
                lap_iter_spin.setVisible(not is_adam)
                iter_label.setVisible(is_adam)
                iter_spin.setVisible(is_adam)
                lr_label.setVisible(is_adam)
                lr_spin.setVisible(is_adam)
                movement_label.setVisible(is_adam)
                movement_spin.setVisible(is_adam)
                tol_label.setVisible(is_adam)
                tol_spin.setVisible(is_adam)
                obj_label.setVisible(is_adam)
                obj_combo.setVisible(is_adam)
                pre_label.setVisible(is_adam)
                pre_spin.setVisible(is_adam)
                inner_label.setVisible(is_adam)
                inner_spin.setVisible(is_adam)
                lr_step_label.setVisible(is_adam)
                lr_step_spin.setVisible(is_adam)
                lr_gamma_label.setVisible(is_adam)
                lr_gamma_spin.setVisible(is_adam)
                dialog.adjustSize()
                dialog.setFixedSize(dialog.sizeHint())

            method_combo.currentIndexChanged.connect(_sync_optimize_controls)
            _sync_optimize_controls()

            if dialog.exec_() != QDialog.Accepted:
                self.gui.log_info("已取消网格优化")
                self.gui.update_status("已取消网格优化")
                return

            method_key = method_combo.currentData() or "classic"
            laplacian_iters = int(lap_iter_spin.value())
            iterations = int(iter_spin.value())
            learning_rate = float(lr_spin.value())
            movement_factor = float(movement_spin.value())
            convergence_tolerance = float(tol_spin.value())
            obj_func = obj_combo.currentData() or "L1"
            pre_smooth_iter = int(pre_spin.value())
            inner_steps = int(inner_spin.value())
            lr_step_size = int(lr_step_spin.value())
            lr_gamma = float(lr_gamma_spin.value())

            from optimize.optimize import edge_swap, laplacian_smooth, nn_smoothing_adam, edge_swap_delaunay
            from utils.message import set_gui_instance

            self.gui.log_info("开始进行网格优化...")
            self.gui.update_status("正在进行网格优化...")

            # 设置GUI实例到消息系统，以便优化函数可以输出到GUI
            set_gui_instance(self.gui)

            start_time = time.time()

            if method_key == "adam":
                if nn_smoothing_adam is None:
                    QMessageBox.information(self.gui, "提示", "当前环境未安装Torch，无法使用Adam优化")
                    self.gui.log_info("Adam优化不可用：未安装Torch")
                    self.gui.update_status("Adam优化不可用")
                    return

                method_name = "Adam优化"
                optimized_mesh = nn_smoothing_adam(
                    mesh_obj,
                    movement_factor=movement_factor,
                    iteration_limit=iterations,
                    learning_rate=learning_rate,
                    convergence_tolerance=convergence_tolerance,
                    obj_func=obj_func,
                    lr_step_size=lr_step_size,
                    lr_gamma=lr_gamma,
                    inner_steps=inner_steps,
                    pre_smooth_iter=pre_smooth_iter,
                )
            else:
                method_name = "边交换 + Laplacian"
                
                for _ in range(laplacian_iters):
                    self.gui.log_info(f"正在进行第{_}轮边交换优化...")
                    optimized_mesh = edge_swap(mesh_obj)

                    self.gui.log_info(f"正在进行第{_}轮laplacian光滑优化...")
                    optimized_mesh = laplacian_smooth(optimized_mesh, num_iter=1)


            if hasattr(self.gui.current_mesh, 'unstr_grid'):
                self.gui.current_mesh.unstr_grid = optimized_mesh
                if hasattr(optimized_mesh, 'node_coords'):
                    self.gui.current_mesh.node_coords = optimized_mesh.node_coords
                if hasattr(optimized_mesh, 'cell_container'):
                    self.gui.current_mesh.cells = [list(cell.node_ids) for cell in optimized_mesh.cell_container]
                    if hasattr(self.gui.current_mesh, 'update_counts'):
                        self.gui.current_mesh.update_counts()
            else:
                self.gui.current_mesh = optimized_mesh

            end_time = time.time()

            if hasattr(self.gui, 'mesh_visualizer') and self.gui.mesh_visualizer:
                self.gui.mesh_visualizer.update_mesh(self.gui.current_mesh)
            if hasattr(self.gui, 'mesh_display') and self.gui.mesh_display:
                self.gui.mesh_display.clear_mesh_actors()
                self.gui.mesh_display.set_mesh_data(self.gui.current_mesh)
                self.gui.mesh_display.display_mesh(render_immediately=False)

            self.gui.log_info(f"{method_name}完成，总耗时: {end_time - start_time:.3f}秒")
            self.gui.log_info(f"优化后网格包含 {len(optimized_mesh.cell_container)} 个单元")
            self.gui.update_status("网格优化完成")

            if hasattr(self.gui, 'canvas'):
                self.gui.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"网格优化失败：{str(e)}")
            self.gui.log_info(f"网格优化失败：{str(e)}")
            self.gui.update_status("网格优化失败")

        finally:
            # 重置GUI实例，避免影响其他操作
            set_gui_instance(None)

    def show_mesh_statistics(self):
        """显示网格统计信息 - 包括网格单元信息和质量统计"""
        try:
            if not self._require_mesh("未找到网格数据，无法显示统计信息"):
                return

            mesh_obj = self._get_mesh_obj()

            if not self._require_cell_container(mesh_obj, "当前网格格式不支持统计功能", "当前网格格式不支持统计功能"):
                return

            num_cells = len(mesh_obj.cell_container)
            num_nodes = len(mesh_obj.node_coords)
            num_boundary_nodes = len(mesh_obj.boundary_nodes)

            if num_nodes > 0:
                dimension = mesh_obj.dimension
            else:
                dimension = 0

            mesh_obj.calculate_edges()
            num_edges = len(mesh_obj.edges)

            quality_values, skewness_values = self._collect_quality_metrics(mesh_obj)
            triangle_count = 0
            quadrilateral_count = 0
            tetrahedron_count = 0
            pyramid_count = 0
            prism_count = 0
            hexahedron_count = 0

            for cell in mesh_obj.cell_container:
                if cell.quality is not None:
                    quality_values.append(cell.quality)
                if cell.skewness is not None:
                    skewness_values.append(cell.skewness)

                if hasattr(cell, 'p8'):
                    hexahedron_count += 1
                elif hasattr(cell, 'p6'):
                    prism_count += 1
                elif hasattr(cell, 'p5'):
                    pyramid_count += 1
                elif hasattr(cell, 'p4') and not hasattr(cell, 'p5'):
                    tetrahedron_count += 1
                elif hasattr(cell, 'p3') and not hasattr(cell, 'p4'):
                    triangle_count += 1
                elif hasattr(cell, 'p4') and hasattr(cell, 'p3'):
                    quadrilateral_count += 1

            stats_info = f"网格统计信息:\n"
            stats_info += f"  维度: {dimension}\n"
            stats_info += f"  总单元数: {num_cells}\n"
            stats_info += f"  节点数: {num_nodes}\n"
            stats_info += f"  边界节点数: {num_boundary_nodes}\n"
            stats_info += f"  边数: {num_edges}\n"
            stats_info += f"  三角形单元数: {triangle_count}\n"
            stats_info += f"  四边形单元数: {quadrilateral_count}\n"
            stats_info += f"  四面体单元数: {tetrahedron_count}\n"
            stats_info += f"  金字塔单元数: {pyramid_count}\n"
            stats_info += f"  三棱柱单元数: {prism_count}\n"
            stats_info += f"  六面体单元数: {hexahedron_count}\n"

            if quality_values:
                stats_info += f"\n质量统计:\n"
                stats_info += f"  最小质量值: {min(quality_values):.4f}\n"
                stats_info += f"  最大质量值: {max(quality_values):.4f}\n"
                stats_info += f"  平均质量值: {sum(quality_values)/len(quality_values):.4f}\n"
                stats_info += f"  质量值样本数: {len(quality_values)}\n"
            else:
                stats_info += f"\n质量统计: 无质量数据\n"

            if skewness_values:
                stats_info += f"\n偏斜度统计:\n"
                stats_info += f"  最小偏斜度: {min(skewness_values):.4f}\n"
                stats_info += f"  最大偏斜度: {max(skewness_values):.4f}\n"
                stats_info += f"  平均偏斜度: {sum(skewness_values)/len(skewness_values):.4f}\n"
                stats_info += f"  偏斜度样本数: {len(skewness_values)}\n"
            else:
                stats_info += f"\n偏斜度统计: 无偏斜度数据\n"

            self.gui.log_info(stats_info)
            self.gui.update_status("网格统计信息显示完成")

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"显示网格统计信息失败：{str(e)}")
            self.gui.log_info(f"显示网格统计信息失败：{str(e)}")
            self.gui.update_status("统计信息显示失败")

    def extract_boundary_mesh_info(self):
        """提取边界网格及部件信息"""
        try:
            from PyQt5.QtWidgets import QMessageBox
            from data_structure.basic_elements import NodeElement, Triangle, Quadrilateral

            if not self._require_mesh("未找到网格数据，无法提取边界信息", prompt_message="请先导入网格"):
                return

            has_boundary_info = False
            boundary_info = {}

            if hasattr(self.gui.current_mesh, 'boundary_nodes') and self.gui.current_mesh.boundary_nodes:
                has_boundary_info = True
                boundary_info['boundary_nodes'] = self.gui.current_mesh.boundary_nodes

            if hasattr(self.gui.current_mesh, 'parts_info') and self.gui.current_mesh.parts_info:
                has_boundary_info = True
                boundary_info['parts_info'] = self.gui.current_mesh.parts_info
            elif hasattr(self.gui, 'cas_parts_info') and self.gui.cas_parts_info:
                has_boundary_info = True
                boundary_info['parts_info'] = self.gui.cas_parts_info
            else:
                if hasattr(self.gui.current_mesh, 'boundary_nodes') and self.gui.current_mesh.boundary_nodes:
                    extracted_parts = self.gui._extract_parts_from_boundary_nodes(self.gui.current_mesh.boundary_nodes)
                    if extracted_parts:
                        has_boundary_info = True
                        boundary_info['parts_info'] = extracted_parts

            if 'parts_info' in boundary_info and boundary_info['parts_info']:
                parts_info = boundary_info['parts_info']
                
                boundary_part_names = set()
                for part_name in parts_info.keys():
                    if part_name not in PARTS_INFO_RESERVED_KEYS:
                        part_data = parts_info[part_name]
                        bc_type = part_data.get('bc_type', 'unspecified')
                        if bc_type != 'interior':
                            boundary_part_names.add(part_name)
                
                if boundary_part_names:
                    self.gui.log_info(f"检测到边界部件: {list(boundary_part_names)}")
                    
                    all_faces = []
                    kept_node_indices = set()
                    
                    for part_name in boundary_part_names:
                        if part_name in parts_info:
                            part_data = parts_info[part_name]
                            faces = part_data.get('faces', [])
                            for face in faces:
                                nodes = face.get('nodes', [])
                                if nodes:
                                    all_faces.append({
                                        'nodes': nodes,
                                        'part_name': part_name
                                    })
                                    kept_node_indices.update(nodes)
                    
                    if all_faces and kept_node_indices:
                        old_to_new_node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(kept_node_indices))}
                        
                        new_node_coords = []
                        if hasattr(self.gui.current_mesh, 'node_coords'):
                            for old_idx in sorted(kept_node_indices):
                                if old_idx < len(self.gui.current_mesh.node_coords):
                                    new_node_coords.append(self.gui.current_mesh.node_coords[old_idx])
                        
                        new_node_container = [NodeElement(new_node_coords[i], i) for i in range(len(new_node_coords))]
                        new_cell_container = []
                        
                        for face in all_faces:
                            nodes = face['nodes']
                            part_name = face['part_name']
                            
                            new_node_indices = [old_to_new_node_map[nid] for nid in nodes if nid in old_to_new_node_map]
                            
                            if len(new_node_indices) == 3:
                                cell = Triangle(
                                    new_node_container[new_node_indices[0]],
                                    new_node_container[new_node_indices[1]],
                                    new_node_container[new_node_indices[2]],
                                    "interior-triangle",
                                    idx=len(new_cell_container)
                                )
                                cell.part_name = part_name
                                new_cell_container.append(cell)
                            elif len(new_node_indices) == 4:
                                cell = Quadrilateral(
                                    new_node_container[new_node_indices[0]],
                                    new_node_container[new_node_indices[1]],
                                    new_node_container[new_node_indices[2]],
                                    new_node_container[new_node_indices[3]],
                                    "interior-quadrilateral",
                                    idx=len(new_cell_container)
                                )
                                cell.part_name = part_name
                                new_cell_container.append(cell)
                        
                        new_boundary_nodes = []
                        for node in new_node_container:
                            node.bc_type = 'boundary'
                            new_boundary_nodes.append(node)
                        
                        self.gui.current_mesh.cells = new_cell_container
                        self.gui.current_mesh.num_cells = len(new_cell_container)
                        self.gui.current_mesh.node_coords = new_node_coords
                        self.gui.current_mesh.num_points = len(new_node_coords)
                        self.gui.current_mesh.boundary_nodes = new_boundary_nodes
                        self.gui.current_mesh.boundary_nodes_list = [node.idx for node in new_boundary_nodes]
                        self.gui.current_mesh.num_boundary_nodes = len(new_boundary_nodes)
                        self.gui.current_mesh.unstr_grid = None
                        
                        mapped_parts_info = self.gui._map_parts_info_to_new_mesh(parts_info, old_to_new_node_map)
                        new_parts_info = {}
                        for part_name in boundary_part_names:
                            if part_name in mapped_parts_info:
                                part_data = mapped_parts_info[part_name]
                                if isinstance(part_data, dict):
                                    part_data = dict(part_data)
                                    part_data.setdefault('bc_type', 'boundary')
                                new_parts_info[part_name] = part_data
                        self.gui.current_mesh.parts_info = new_parts_info
                        self.gui.current_mesh.boundary_info = new_parts_info

                        self.gui.cas_parts_info = new_parts_info
                        
                        self.gui.original_node_coords = self.gui.current_mesh.node_coords

                        # 更新模型树要放在前，更新部件列表要放在后，按道理前后不影响 FIXME
                        if hasattr(self.gui, 'model_tree_widget'):
                            self.gui.model_tree_widget.load_mesh(self.gui.current_mesh)

                        self.gui.part_manager.update_parts_list_from_cas(parts_info=new_parts_info, update_status=False)
                        
                        self.gui.part_manager.refresh_display_all_parts()

                        self.gui.log_info(f"边界提取完成: 保留 {len(new_cell_container)} 个单元, {len(new_node_coords)} 个节点, {len(boundary_part_names)} 个边界部件, 原始共 {len(parts_info)} 个部件")
                        self.gui.update_status(f"边界提取完成: {len(boundary_part_names)} 个边界部件")
                    else:
                        self.gui.log_info("未找到边界faces或节点")
                        self.gui.update_status("未找到边界数据")
                else:
                    self.gui.log_info("未检测到边界部件")
                    self.gui.update_status("未检测到边界部件")
            else:
                self.gui.log_info("未找到部件信息")
                self.gui.update_status("未找到部件信息")

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self.gui, "错误", f"提取边界信息失败：{str(e)}")
            self.gui.log_info(f"提取边界信息失败：{str(e)}")
            self.gui.update_status("边界信息提取失败")

    def edit_boundary_conditions(self):
        """编辑边界条件"""
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QComboBox, QMessageBox, QHeaderView

            if not hasattr(self.gui, 'cas_parts_info') or not self.gui.cas_parts_info:
                QMessageBox.warning(self.gui, "警告", "请先导入网格文件以获取部件列表")
                self.gui.log_info("未检测到导入的网格数据，无法设置边界条件")
                self.gui.update_status("未检测到导入的网格数据")
                return

            dialog = QDialog(self.gui)
            dialog.setWindowTitle("编辑边界条件")
            dialog.setModal(True)
            dialog.resize(800, 600)

            layout = QVBoxLayout(dialog)

            table = QTableWidget()
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["部件名称", "边界类型", "边界值", "描述"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

            bc_types = ["wall", "inlet", "outlet", "symmetry", "interior", "pressure_inlet", "pressure_outlet", "velocity_inlet"]

            parts_list = []
            for part_name, part_data in self.gui.cas_parts_info.items():
                if part_name not in PARTS_INFO_RESERVED_KEYS:
                    bc_type = part_data.get('bc_type', 'wall')
                    bc_value = part_data.get('bc_value', '')
                    description = part_data.get('description', '')
                    parts_list.append({
                        'part_name': part_name,
                        'bc_type': bc_type,
                        'bc_value': bc_value,
                        'description': description
                    })

            table.setRowCount(len(parts_list))
            for row, part in enumerate(parts_list):
                table.setItem(row, 0, QTableWidgetItem(part['part_name']))

                bc_combo = QComboBox()
                bc_combo.addItems(bc_types)
                bc_combo.setCurrentText(part['bc_type'])
                table.setCellWidget(row, 1, bc_combo)

                table.setItem(row, 2, QTableWidgetItem(str(part['bc_value'])))
                table.setItem(row, 3, QTableWidgetItem(part['description']))

            button_layout = QHBoxLayout()
            ok_button = QPushButton("确定")
            cancel_button = QPushButton("取消")
            button_layout.addStretch()
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)

            layout.addWidget(QLabel("设置各部件的边界条件："))
            layout.addWidget(table)
            layout.addLayout(button_layout)

            def on_ok():
                for row in range(table.rowCount()):
                    part_name = table.item(row, 0).text()
                    bc_combo = table.cellWidget(row, 1)
                    bc_type = bc_combo.currentText()
                    bc_value = table.item(row, 2).text()
                    description = table.item(row, 3).text()

                    if part_name in self.gui.cas_parts_info:
                        self.gui.cas_parts_info[part_name]['bc_type'] = bc_type
                        self.gui.cas_parts_info[part_name]['bc_value'] = bc_value
                        self.gui.cas_parts_info[part_name]['description'] = description

                self.gui.log_info(f"已更新 {len(parts_list)} 个部件的边界条件")
                self.gui.update_status("边界条件已更新")
                dialog.accept()

            def on_cancel():
                self.gui.log_info("取消设置边界条件")
                self.gui.update_status("已取消边界条件设置")
                dialog.reject()

            ok_button.clicked.connect(on_ok)
            cancel_button.clicked.connect(on_cancel)

            dialog.exec_()

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self.gui, "错误", f"编辑边界条件失败：{str(e)}")
            self.gui.log_info(f"编辑边界条件失败：{str(e)}")
            self.gui.update_status("边界条件编辑失败")

    def export_mesh_report(self):
        """导出网格报告 - 将网格生成的主要参数、部件参数和生成结果写到md文档中"""
        try:
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            import os
            import time

            if not self._require_mesh("未找到网格数据，无法导出报告"):
                return

            mesh_obj = self._get_mesh_obj()

            has_params = hasattr(self.gui, 'params') and self.gui.params
            has_parts = hasattr(self.gui, 'parts_params') and self.gui.parts_params

            file_path, _ = QFileDialog.getSaveFileName(
                self.gui,
                "导出网格报告",
                os.path.join(self.gui.project_root, "reports", f"mesh_report_{int(time.time())}.md"),
                "Markdown文件 (*.md)"
            )

            if not file_path:
                self.gui.log_info("网格报告导出已取消")
                self.gui.update_status("报告导出已取消")
                return

            report_content = f"""# 网格生成报告\n\n"""
            report_content += f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            report_content += f"## 网格基本信息\n\n"
            if hasattr(mesh_obj, 'cell_container'):
                report_content += f"- **总单元数**: {len(mesh_obj.cell_container)}\n"
            if hasattr(mesh_obj, 'node_coords'):
                report_content += f"- **节点数**: {len(mesh_obj.node_coords)}\n"
            if hasattr(mesh_obj, 'boundary_nodes'):
                report_content += f"- **边界节点数**: {len(mesh_obj.boundary_nodes)}\n"

            report_content += f"\n## 主要参数配置\n\n"
            if has_params:
                report_content += f"- **调试级别**: {getattr(self.gui.params, 'debug_level', 'N/A')}\n"
                report_content += f"- **网格类型**: {getattr(self.gui.params, 'mesh_type', 'N/A')}\n"
                report_content += f"- **输入文件**: {getattr(self.gui.params, 'input_file', 'N/A')}\n"
                report_content += f"- **输出文件**: {getattr(self.gui.params, 'output_file', 'N/A')}\n"
                report_content += f"- **可视化启用**: {getattr(self.gui.params, 'viz_enabled', 'N/A')}\n"
            else:
                report_content += f"- **参数配置**: 未找到参数配置\n"

            report_content += f"\n## 部件参数配置\n\n"
            if has_parts and self.gui.parts_params:
                for i, part in enumerate(self.gui.parts_params):
                    report_content += f"### 部件 {i+1}: {part.get('part_name', 'Unknown')}\n"
                    report_content += f"- **最大尺寸**: {part.get('max_size', 'N/A')}\n"
                    report_content += f"- **PRISM开关**: {part.get('PRISM_SWITCH', 'N/A')}\n"
                    report_content += f"- **第一层高度**: {part.get('first_height', 'N/A')}\n"
                    report_content += f"- **增长比率**: {part.get('growth_rate', 'N/A')}\n"
                    report_content += f"- **增长方法**: {part.get('growth_method', 'N/A')}\n"
                    report_content += f"- **最大层数**: {part.get('max_layers', 'N/A')}\n"
                    report_content += f"- **完整层数**: {part.get('full_layers', 'N/A')}\n"
                    report_content += f"- **多方向**: {part.get('multi_direction', 'N/A')}\n\n"
            else:
                report_content += f"- **部件参数**: 未找到部件参数\n\n"

            report_content += f"## 网格质量统计\n\n"

            quality_values, skewness_values = self._collect_quality_metrics(mesh_obj)

            if quality_values:
                report_content += f"- **最小质量值**: {min(quality_values):.4f}\n"
                report_content += f"- **最大质量值**: {max(quality_values):.4f}\n"
                report_content += f"- **平均质量值**: {sum(quality_values)/len(quality_values):.4f}\n"
            else:
                report_content += f"- **质量统计**: 无质量数据\n"

            if skewness_values:
                report_content += f"- **最小偏斜度**: {min(skewness_values):.4f}\n"
                report_content += f"- **最大偏斜度**: {max(skewness_values):.4f}\n"
                report_content += f"- **平均偏斜度**: {sum(skewness_values)/len(skewness_values):.4f}\n"
            else:
                report_content += f"- **偏斜度统计**: 无偏斜度数据\n"

            report_content += f"\n## 生成结果\n\n"
            report_content += f"- **网格生成状态**: 成功完成\n"
            report_content += f"- **网格格式**: Unstructured Grid\n"
            if hasattr(mesh_obj, 'cell_container'):
                triangle_count = 0
                quadrilateral_count = 0
                for cell in mesh_obj.cell_container:
                    if hasattr(cell, 'p4'):
                        quadrilateral_count += 1
                    elif hasattr(cell, 'p3'):
                        triangle_count += 1

                report_content += f"- **三角形单元数**: {triangle_count}\n"
                report_content += f"- **四边形单元数**: {quadrilateral_count}\n"

            report_content += f"\n## 备注\n\n"
            report_content += f"- 本报告由 PyMeshGen 自动生成\n"
            report_content += f"- 如需更多信息，请查看控制台输出日志\n"

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            QMessageBox.information(self.gui, "成功", f"网格报告已成功导出到：\n{file_path}")
            self.gui.log_info(f"网格报告已导出到: {file_path}")
            self.gui.update_status("网格报告导出完成")

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self.gui, "错误", f"导出网格报告失败：{str(e)}")
            self.gui.log_info(f"网格报告导出失败：{str(e)}")
            self.gui.update_status("报告导出失败")

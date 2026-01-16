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

from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QDialog, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt


class MeshOperations:
    """网格操作类"""

    def __init__(self, gui_instance):
        self.gui = gui_instance

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

                self.gui.display_mesh()

                # 更新模型树中的网格统计信息
                if hasattr(self.gui, 'model_tree_widget'):
                    self.gui.model_tree_widget.load_mesh(result_mesh)

                # 更新部件列表以显示新网格的部件信息
                self.gui._update_parts_list_from_generated_mesh(result_mesh)

                # Refresh display to show all parts with different colors
                self.gui.refresh_display_all_parts()
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

            from optimize.optimize import laplacian_smooth

            self.gui.log_info("开始进行laplacian光滑处理...")
            self.gui.update_status("正在进行网格光滑处理...")

            start_time = time.time()

            smoothed_mesh = laplacian_smooth(mesh_obj, num_iter=3)

            if hasattr(self.gui.current_mesh, 'unstr_grid'):
                self.gui.current_mesh.unstr_grid = smoothed_mesh
            else:
                self.gui.current_mesh = smoothed_mesh

            end_time = time.time()

            if hasattr(self.gui, 'mesh_visualizer') and self.gui.mesh_visualizer:
                self.gui.mesh_visualizer.update_mesh(self.gui.current_mesh)

            self.gui.log_info(f"laplacian光滑处理完成，耗时: {end_time - start_time:.3f}秒")
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

            from optimize.optimize import edge_swap, laplacian_smooth

            self.gui.log_info("开始进行网格优化...")
            self.gui.update_status("正在进行网格优化...")

            self.gui.log_info("正在进行边交换优化...")
            start_time = time.time()

            optimized_mesh = edge_swap(mesh_obj)

            self.gui.log_info("正在进行laplacian光滑优化...")
            optimized_mesh = laplacian_smooth(optimized_mesh, num_iter=3)

            if hasattr(self.gui.current_mesh, 'unstr_grid'):
                self.gui.current_mesh.unstr_grid = optimized_mesh
            else:
                self.gui.current_mesh = optimized_mesh

            end_time = time.time()

            if hasattr(self.gui, 'mesh_visualizer') and self.gui.mesh_visualizer:
                self.gui.mesh_visualizer.update_mesh(self.gui.current_mesh)

            self.gui.log_info(f"网格优化完成，总耗时: {end_time - start_time:.3f}秒")
            self.gui.log_info(f"优化后网格包含 {len(optimized_mesh.cell_container)} 个单元")
            self.gui.update_status("网格优化完成")

            if hasattr(self.gui, 'canvas'):
                self.gui.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"网格优化失败：{str(e)}")
            self.gui.log_info(f"网格优化失败：{str(e)}")
            self.gui.update_status("网格优化失败")

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
                dim = len(mesh_obj.node_coords[0])
            else:
                dim = 0

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
            stats_info += f"  维度: {dim}\n"
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
                    if part_name not in ['type', 'node_coords', 'cells', 'num_points', 'num_cells', 'unstr_grid']:
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
                        
                        new_parts_info = {}
                        for part_name in boundary_part_names:
                            if part_name in parts_info:
                                new_parts_info[part_name] = parts_info[part_name]
                        self.gui.current_mesh.parts_info = new_parts_info
                        self.gui.current_mesh.boundary_info = new_parts_info

                        self.gui.cas_parts_info = parts_info
                        
                        self.gui.original_node_coords = self.gui.current_mesh.node_coords

                        self.gui.update_parts_list_from_cas(new_parts_info)
                        
                        if hasattr(self.gui, 'model_tree_widget'):
                            self.gui.model_tree_widget.load_mesh(self.gui.current_mesh)
                        
                        if hasattr(self.gui, 'mesh_display'):
                            self.gui.mesh_display.clear_mesh_actors()
                            self.gui.mesh_display.set_mesh_data(self.gui.current_mesh)
                            self.gui.mesh_display.display_mesh(render_immediately=False)
                        
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
                if part_name not in ['type', 'node_coords', 'cells', 'num_points', 'num_cells', 'unstr_grid']:
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

import json
import os
import pickle
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class ConfigManager:
    """配置管理器 - 负责管理工程配置的保存、加载、导入和导出"""

    def __init__(self, gui):
        self.gui = gui

    def save_config(self):
        """保存工程 - 将JSON配置文件和导入网格文件的路径保存到.pymg工程文件中"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self.gui,
                "保存工程",
                os.path.join(self.gui.project_root, "projects"),
                "PyMeshGen工程文件 (*.pymg)"
            )

            if not file_path:
                self.gui.log_info("保存工程操作已取消")
                self.gui.update_status("保存工程已取消")
                return

            if not file_path.endswith('.pymg'):
                file_path += '.pymg'

            project_data = {
                "version": "1.0",
                "created_at": __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
                "project_type": "PyMeshGen_Project"
            }

            if hasattr(self.gui, 'params') and self.gui.params:
                config_info = {
                    "debug_level": getattr(self.gui.params, 'debug_level', 0),
                    "input_file": getattr(self.gui.params, 'input_file', ''),
                    "output_file": getattr(self.gui.params, 'output_file', ''),
                    "mesh_type": getattr(self.gui.params, 'mesh_type', 1),
                    "viz_enabled": getattr(self.gui.params, 'viz_enabled', False)
                }

                if hasattr(self.gui.params, 'part_params'):
                    part_configs = []
                    for part_param in self.gui.params.part_params:
                        part_config = {
                            "part_name": part_param.part_name,
                            "max_size": part_param.part_params.max_size,
                            "PRISM_SWITCH": part_param.part_params.PRISM_SWITCH,
                            "first_height": part_param.part_params.first_height,
                            "growth_rate": part_param.part_params.growth_rate,
                            "growth_method": part_param.part_params.growth_method,
                            "max_layers": part_param.part_params.max_layers,
                            "full_layers": part_param.part_params.full_layers,
                            "multi_direction": part_param.part_params.multi_direction
                        }
                        part_configs.append(part_config)
                    config_info["parts"] = part_configs

                project_data["config"] = config_info

            mesh_file_path = ""
            if hasattr(self.gui, 'current_mesh'):
                if isinstance(self.gui.current_mesh, dict) and 'file_path' in self.gui.current_mesh:
                    mesh_file_path = self.gui.current_mesh['file_path']
                elif hasattr(self.gui.current_mesh, 'file_path'):
                    mesh_file_path = self.gui.current_mesh.file_path

            if mesh_file_path:
                project_data["mesh_file_path"] = mesh_file_path

            if hasattr(self.gui, 'current_mesh') and self.gui.current_mesh:
                try:
                    project_dir = os.path.dirname(file_path)
                    project_name = os.path.splitext(os.path.basename(file_path))[0]
                    mesh_file = os.path.join(project_dir, f"{project_name}_mesh.pymesh")

                    if hasattr(self.gui.current_mesh, 'save_to_vtkfile'):
                        with open(mesh_file, 'wb') as mf:
                            pickle.dump(self.gui.current_mesh, mf)
                        project_data["generated_mesh_file"] = mesh_file
                    elif hasattr(self.gui.current_mesh, 'save_to_file'):
                        self.gui.current_mesh.save_to_file(mesh_file)
                        project_data["generated_mesh_file"] = mesh_file
                    else:
                        with open(mesh_file, 'wb') as mf:
                            pickle.dump(self.gui.current_mesh, mf)
                        project_data["generated_mesh_file"] = mesh_file

                    self.gui.log_info(f"生成的网格已保存到: {project_data['generated_mesh_file']}")

                except Exception as e:
                    self.gui.log_info(f"保存生成的网格数据时出现错误: {str(e)}")

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)

            self.gui.log_info(f"工程已保存到: {file_path}")
            self.gui.update_status("工程保存完成")

            QMessageBox.information(self.gui, "成功", f"工程已成功保存到:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"保存工程失败：{str(e)}")
            self.gui.log_info(f"保存工程失败：{str(e)}")
            self.gui.update_status("工程保存失败")

    def import_config(self):
        """导入配置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.gui,
            "导入配置文件",
            os.path.join(self.gui.project_root, "config"),
            "JSON文件 (*.json)"
        )

        if file_path:
            try:
                if hasattr(self.gui, 'params') and self.gui.params:
                    self.gui.params.import_config(file_path)
                    self.gui.log_info(f"配置已从 {file_path} 导入")
                else:
                    from data_structure.parameters import Parameters
                    self.gui.params = Parameters('FROM_CASE_JSON', file_path)
                    self.gui.log_info(f"配置已从 {file_path} 导入")

                self.gui.parts_params = []
                for part_param in self.gui.params.part_params:
                    part_dict = {
                        "part_name": part_param.part_name,
                        "max_size": part_param.part_params.max_size,
                        "PRISM_SWITCH": part_param.part_params.PRISM_SWITCH,
                        "first_height": part_param.part_params.first_height,
                        "growth_rate": part_param.part_params.growth_rate,
                        "growth_method": part_param.part_params.growth_method,
                        "max_layers": part_param.part_params.max_layers,
                        "full_layers": part_param.part_params.full_layers,
                        "multi_direction": part_param.part_params.multi_direction
                    }
                    self.gui.parts_params.append(part_dict)

                if hasattr(self.gui, 'part_list_widget'):
                    self.gui.part_list_widget.clear()
                    for part_param in self.gui.params.part_params:
                        item_text = f"{part_param.part_name} - Max Size: {part_param.part_params.max_size}, Prism: {part_param.part_params.PRISM_SWITCH}"
                        self.gui.part_list_widget.addItem(item_text)

                reply = QMessageBox.question(
                    self.gui,
                    "配置导入成功",
                    f"配置已成功从 {file_path} 导入\n是否立即开始生成网格？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self.gui.mesh_operations.generate_mesh()
                    self.gui.update_status("配置导入并网格生成已启动")
                else:
                    QMessageBox.information(self.gui, "成功", f"配置已成功从 {file_path} 导入")
                    self.gui.update_status("配置导入成功")

            except Exception as e:
                QMessageBox.critical(self.gui, "错误", f"导入配置失败：{str(e)}")
                self.gui.log_info(f"配置导入失败：{str(e)}")
                self.gui.update_status("配置导入失败")
        else:
            self.gui.log_info("配置导入已取消")
            self.gui.update_status("配置导入已取消")

    def export_config(self):
        """导出配置"""
        if not hasattr(self.gui, 'parts_params') or not self.gui.parts_params:
            QMessageBox.warning(self.gui, "警告", "没有可导出的部件参数配置")
            self.gui.log_info("没有可导出的部件参数配置")
            self.gui.update_status("没有可导出的部件参数配置")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.gui,
            "导出配置文件",
            os.path.join(self.gui.project_root, "config", "exported_config.json"),
            "JSON文件 (*.json)"
        )

        if file_path:
            try:
                config_data = {
                    "debug_level": 0,
                    "input_file": "",
                    "output_file": "./out/mesh.vtk",
                    "viz_enabled": True,
                    "parts": self.gui.parts_params
                }

                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)

                self.gui.log_info(f"配置已成功导出到: {file_path}")
                self.gui.update_status(f"配置已成功导出到: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self.gui, "错误", f"导出配置失败: {str(e)}")
                self.gui.log_error(f"导出配置失败: {str(e)}")

    def reset_config(self):
        """重置配置"""
        self.gui.log_info("重置配置功能暂未实现")

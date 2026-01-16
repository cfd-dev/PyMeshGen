from PyQt5.QtWidgets import QDialog


class UIHelpers:
    """UI工具类 - 提供日志记录、状态更新和UI元素切换等辅助功能"""

    def __init__(self, gui):
        self.gui = gui

    def log_info(self, message):
        """记录信息日志"""
        if hasattr(self.gui, 'info_output'):
            self.gui.info_output.log_info(message)

    def log_error(self, message):
        """记录错误日志"""
        if hasattr(self.gui, 'info_output'):
            self.gui.info_output.log_error(message)

    def log_warning(self, message):
        """记录警告日志"""
        if hasattr(self.gui, 'info_output'):
            self.gui.info_output.log_warning(message)

    def update_status(self, message):
        """更新状态栏信息"""
        if hasattr(self.gui, 'status_bar'):
            self.gui.status_bar.update_status(message)

    def edit_params(self):
        """编辑全局参数"""
        from PyQt5.QtWidgets import QDialog
        from gui.global_params_dialog import GlobalParamsDialog
        
        # 准备当前参数
        current_params = {}
        if hasattr(self.gui, 'params') and self.gui.params:
            current_params["debug_level"] = self.gui.params.debug_level
            current_params["output_file"] = self.gui.params.output_file
            current_params["mesh_type"] = self.gui.params.mesh_type
            current_params["auto_output"] = getattr(self.gui.params, 'auto_output', True)
            
            # 从部件参数中获取全局最大尺寸（如果有）
            if hasattr(self.gui.params, 'part_params') and self.gui.params.part_params:
                # 假设第一个部件的max_size作为全局尺寸
                current_params["global_max_size"] = self.gui.params.part_params[0].param.max_size
        
        # 创建并显示对话框
        dialog = GlobalParamsDialog(self.gui, current_params)
        if dialog.exec_() == QDialog.Accepted:
            # 获取用户设置的参数
            new_params = dialog.get_params()
            
            # 如果params实例不存在，初始化它
            if not hasattr(self.gui, 'params') or not self.gui.params:
                from data_structure.parameters import Parameters
                # 创建临时配置文件来初始化Parameters实例
                import json
                import tempfile
                import os
                
                # 创建默认配置
                default_config = {
                    "debug_level": new_params["debug_level"],
                    "input_file": [],
                    "output_file": new_params["output_file"],
                    "viz_enabled": False,
                    "parts": []
                }
                
                # 写入临时文件
                temp_config_path = os.path.join(tempfile.gettempdir(), "temp_config.json")
                with open(temp_config_path, 'w') as f:
                    json.dump(default_config, f)
                
                # 初始化Parameters实例
                self.gui.params = Parameters("FROM_CASE_JSON", temp_config_path)
                
                # 删除临时文件
                os.remove(temp_config_path)
                
                self.gui.log_info("全局参数实例已初始化")
            
            # 更新全局参数实例
            self.gui.params.debug_level = new_params["debug_level"]
            self.gui.params.output_file = new_params["output_file"]
            self.gui.params.mesh_type = new_params["mesh_type"]
            self.gui.params.auto_output = new_params["auto_output"]
            
            # 更新所有部件的最大尺寸为全局尺寸
            if hasattr(self.gui.params, 'part_params') and self.gui.params.part_params:
                for part in self.gui.params.part_params:
                    part.param.max_size = new_params["global_max_size"]
            
            self.gui.log_info(f"全局参数已更新: 自动输出={new_params['auto_output']}, 网格类型={new_params['mesh_type']}, 输出路径={new_params['output_file'][0]}")
            self.gui.update_status("全局参数已更新")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt配置管理模块
处理网格配置的创建、保存和加载功能
"""

import os
import json
from parameters import Parameters


class ConfigManager:
    """配置管理器类"""

    def __init__(self, project_root):
        self.project_root = project_root
        self.config_dir = os.path.join(project_root, "configs")
        
        # 确保配置目录存在
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)

    def create_config_from_params(self, params):
        """从参数对象创建配置字典"""
        if not params:
            # 返回默认配置
            return {
                "debug_level": 0,
                "input_file": "",
                "output_file": "",
                "mesh_type": 1,
                "viz_enabled": True,
                "parts": []
            }

        # 创建配置字典
        config = {
            "debug_level": getattr(params, 'debug_level', 0),
            "input_file": getattr(params, 'input_file', ""),
            "output_file": getattr(params, 'output_file', ""),
            "mesh_type": getattr(params, 'mesh_type', 1),
            "viz_enabled": getattr(params, 'viz_enabled', True),
            "parts": getattr(params, 'parts', [])
        }
        
        return config

    def create_params_from_config(self, config):
        """从配置字典创建参数对象"""
        if not config:
            # 创建默认参数对象
            from parameters import Parameters
            default_config = {
                "debug_level": 0,
                "input_file": "",
                "output_file": "",
                "mesh_type": 1,
                "viz_enabled": True,
                "parts": []
            }
            # 创建临时JSON文件用于参数初始化
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(default_config, f)
                temp_file = f.name
            
            params = Parameters("FROM_CASE_JSON", temp_file)
            
            # 清理临时文件
            os.unlink(temp_file)
            return params

        # 创建临时JSON文件用于参数初始化
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_file = f.name

        # 创建参数对象
        params = Parameters("FROM_CASE_JSON", temp_file)

        # 清理临时文件
        os.unlink(temp_file)
        return params

    def save_config(self, config, file_path):
        """保存配置到文件"""
        try:
            # 确保目录存在
            config_dir = os.path.dirname(file_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)

            # 保存配置
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存配置失败: {str(e)}")
            return False

    def load_config(self, file_path):
        """从文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"加载配置失败: {str(e)}")
            return None

    def get_recent_configs(self, limit=10):
        """获取最近的配置文件列表"""
        config_files = []
        if os.path.exists(self.config_dir):
            for file_name in os.listdir(self.config_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.config_dir, file_name)
                    mod_time = os.path.getmtime(file_path)
                    config_files.append((file_path, mod_time))
            
            # 按修改时间排序，最新的在前
            config_files.sort(key=lambda x: x[1], reverse=True)
            return [f[0] for f in config_files[:limit]]
        return []
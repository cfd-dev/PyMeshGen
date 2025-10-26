import sys
import os
import json
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_structure'))
sys.path.insert(0, str(project_root / 'utils'))

from data_structure.parameters import Parameters

class TestConfigLoad(unittest.TestCase):
    """测试配置文件加载功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_config_path = Path(__file__).parent / 'test_config.json'
    
    def test_config_file_loading(self):
        """测试配置文件加载"""
        # 加载配置文件
        with open(self.test_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证配置文件加载成功
        self.assertIsNotNone(config)
        self.assertIn('debug_level', config)
        self.assertIn('input_file', config)
        self.assertIn('output_file', config)
        self.assertIn('mesh_type', config)
        self.assertIn('parts', config)
        
        # 验证配置项的类型
        self.assertIsInstance(config['debug_level'], int)
        self.assertIsInstance(config['input_file'], str)
        self.assertIsInstance(config['output_file'], str)
        self.assertIsInstance(config['mesh_type'], int)
        self.assertIsInstance(config['parts'], list)
    
    def test_parameters_object_creation(self):
        """测试Parameters对象创建"""
        # 创建Parameters对象
        params = Parameters("FROM_CASE_JSON", str(self.test_config_path))
        
        # 验证Parameters对象创建成功
        self.assertIsNotNone(params)
        self.assertIsInstance(params.debug_level, int)
        self.assertIsInstance(params.input_file, str)
        self.assertIsInstance(params.mesh_type, int)
        self.assertIsInstance(params.part_params, list)
        
        # 验证参数值
        with open(self.test_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.assertEqual(params.debug_level, config['debug_level'])
        self.assertEqual(params.input_file, config['input_file'])
        self.assertEqual(params.mesh_type, config['mesh_type'])
        
        # 如果有部件参数，验证部件参数
        if config['parts']:
            self.assertGreater(len(params.part_params), 0)
            self.assertEqual(params.part_params[0].part_name, config['parts'][0]['part_name'])

if __name__ == '__main__':
    unittest.main()
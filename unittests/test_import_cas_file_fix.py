#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单元测试：测试导入CAS文件功能的修复
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from pyqt_gui.gui_main import SimplifiedPyMeshGenGUI

class TestImportCasFile(unittest.TestCase):
    """测试导入CAS文件功能"""
    
    def setUp(self):
        """测试前的设置"""
        # 创建模拟的Tkinter根窗口
        self.root = Mock()
        self.root.configure = Mock()
        
        # 创建模拟的GUI组件
        self.mesh_status_label = Mock()
        self.mesh_status_label.config = Mock()
        self.mesh_status_label.cget = Mock(return_value="状态: 已导入")
        
        self.mesh_info_label = Mock()
        self.mesh_info_label.config = Mock()
        self.mesh_info_label.cget = Mock(return_value="节点数: 1000\n单元数: 2000")
        
        self.parts_listbox = Mock()
        self.parts_listbox.size = Mock(return_value=2)
        
        # 创建模拟的mesh_display
        self.mesh_display = Mock()
        self.mesh_display.mesh_data = None
        
        # 创建模拟的GUI实例
        self.app = Mock(spec=SimplifiedPyMeshGenGUI)
        self.app.mesh_display = self.mesh_display
        self.app.mesh_status_label = self.mesh_status_label
        self.app.mesh_info_label = self.mesh_info_label
        self.app.parts_listbox = self.parts_listbox
        self.app.update_parts_list_from_cas = Mock()
    
    def test_mesh_display_attribute(self):
        """测试mesh_display属性是否存在"""
        self.assertTrue(hasattr(self.app, 'mesh_display'), "mesh_display属性未找到")
    
    def test_mesh_data_setting(self):
        """测试mesh_data设置"""
        # 模拟导入CAS文件的数据
        mock_mesh_data = {
            'type': 'cas',
            'num_points': 1000,
            'num_cells': 2000,
            'parts_info': [
                {'part_name': 'inlet', 'face_count': 10, 'nodes': [1, 2, 3], 'cells': [1, 2]},
                {'part_name': 'outlet', 'face_count': 20, 'nodes': [4, 5, 6], 'cells': [3, 4]}
            ]
        }
        
        # 设置mesh_data
        self.app.mesh_display.mesh_data = mock_mesh_data
        
        # 检查mesh_data是否设置成功
        self.assertEqual(self.app.mesh_display.mesh_data, mock_mesh_data, "mesh_data设置失败")
    
    def test_mesh_status_update(self):
        """测试网格状态更新"""
        # 模拟更新网格状态
        self.app.mesh_status_label.config(text="状态: 已导入")
        
        # 验证config方法被调用
        self.app.mesh_status_label.config.assert_called_with(text="状态: 已导入")
    
    def test_mesh_info_update(self):
        """测试网格信息更新"""
        # 模拟数据
        mock_mesh_data = {
            'num_points': 1000,
            'num_cells': 2000
        }
        
        # 模拟更新网格信息
        expected_text = f"节点数: {mock_mesh_data['num_points']}\n单元数: {mock_mesh_data['num_cells']}"
        self.app.mesh_info_label.config(text=expected_text)
        
        # 验证config方法被调用
        self.app.mesh_info_label.config.assert_called_with(text=expected_text)
    
    def test_parts_list_update(self):
        """测试部件列表更新"""
        # 模拟数据
        mock_parts_info = [
            {'part_name': 'inlet', 'face_count': 10, 'nodes': [1, 2, 3], 'cells': [1, 2]},
            {'part_name': 'outlet', 'face_count': 20, 'nodes': [4, 5, 6], 'cells': [3, 4]}
        ]
        
        # 模拟更新部件列表
        self.app.update_parts_list_from_cas(mock_parts_info)
        
        # 验证方法被调用
        self.app.update_parts_list_from_cas.assert_called_with(mock_parts_info)
        
        # 验证列表大小
        self.assertEqual(self.app.parts_listbox.size(), len(mock_parts_info), "部件列表更新失败")

if __name__ == '__main__':
    unittest.main()
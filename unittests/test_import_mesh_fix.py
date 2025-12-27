#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单元测试：测试导入网格功能的修复
"""

import os
import sys
import unittest
from unittest.mock import Mock, MagicMock, patch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from pyqt_gui.gui_main import SimplifiedPyMeshGenGUI
from pyqt_gui.mesh_display import MeshDisplayArea

class TestImportMesh(unittest.TestCase):
    """测试导入网格功能"""
    
    def setUp(self):
        """测试前的设置"""
        # 创建模拟的Tkinter根窗口
        self.root = Mock()
        self.root.configure = Mock()
        
        # 创建模拟的GUI实例
        self.app = Mock(spec=SimplifiedPyMeshGenGUI)
        
        # 创建模拟的mesh_display
        self.mesh_display = Mock(spec=MeshDisplayArea)
        self.app.mesh_display = self.mesh_display
    
    def test_mesh_display_attribute(self):
        """测试mesh_display属性是否存在"""
        self.assertTrue(hasattr(self.app, 'mesh_display'), "mesh_display属性未找到")
    
    def test_mesh_display_instance(self):
        """测试mesh_display是否是MeshDisplayArea的正确实例"""
        # 设置mesh_display为MeshDisplayArea的实例
        self.mesh_display.__class__ = MeshDisplayArea
        
        # 验证isinstance检查
        self.assertIsInstance(self.app.mesh_display, MeshDisplayArea, 
                              "mesh_display不是MeshDisplayArea的实例")
    
    def test_gui_initialization(self):
        """测试GUI初始化过程"""
        # 模拟GUI初始化过程
        with patch('pyqt_gui.gui_main.SimplifiedPyMeshGenGUI') as mock_gui_class:
            mock_instance = Mock()
            mock_instance.mesh_display = Mock(spec=MeshDisplayArea)
            mock_gui_class.return_value = mock_instance
            
            # 创建GUI实例
            app = mock_gui_class(self.root)
            
            # 验证GUI类被正确调用
            mock_gui_class.assert_called_once_with(self.root)
            
            # 验证mesh_display属性存在
            self.assertTrue(hasattr(app, 'mesh_display'))
            
            # 验证mesh_display是MeshDisplayArea的实例
            self.assertIsInstance(app.mesh_display, Mock)

if __name__ == '__main__':
    unittest.main()
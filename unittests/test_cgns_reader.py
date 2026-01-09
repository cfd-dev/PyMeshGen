#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGNS读取器单元测试
测试UniversalCGNSReader的功能
"""

import sys
import os
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加 meshio 到 Python 路径
meshio_path = project_root / "3rd_party" / "meshio" / "src"
if meshio_path.exists():
    sys.path.insert(0, str(meshio_path))

# 导入UniversalCGNSReader
try:
    from fileIO.universal_cgns_reader import UniversalCGNSReader
except ImportError:
    UniversalCGNSReader = None


class TestUniversalCGNSReader(unittest.TestCase):
    """测试UniversalCGNSReader类"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.meshes_dir = project_root / "examples"
        
        # 检查examples目录是否存在
        if not cls.meshes_dir.exists():
            raise unittest.SkipTest(f"examples目录不存在: {cls.meshes_dir}")
        
        # 查找所有CGNS文件
        cls.cgns_files = sorted(cls.meshes_dir.rglob("*.cgns"))
        
        if not cls.cgns_files:
            raise unittest.SkipTest(f"examples目录中没有CGNS文件")

    def test_cgns_files_exist(self):
        """测试CGNS文件是否存在"""
        self.assertGreater(len(self.cgns_files), 0, "应该至少有一个CGNS文件")

    def test_universal_cgns_reader_available(self):
        """测试UniversalCGNSReader是否可用"""
        self.assertIsNotNone(UniversalCGNSReader, "UniversalCGNSReader应该可用")

    def test_read_all_cgns_files(self):
        """测试读取所有CGNS文件"""
        failed_files = []
        
        for cgns_file in self.cgns_files:
            try:
                reader = UniversalCGNSReader(str(cgns_file))
                success = reader.read()
                
                self.assertTrue(success, f"读取CGNS文件失败: {cgns_file.name}")
                
                # 验证节点数据
                self.assertIsNotNone(reader.points, f"节点数据不应为空: {cgns_file.name}")
                self.assertGreater(len(reader.points), 0, f"节点数应该大于0: {cgns_file.name}")
                
                # 验证单元数据
                self.assertIsNotNone(reader.cells, f"单元数据不应为空: {cgns_file.name}")
                self.assertGreater(len(reader.cells), 0, f"单元数应该大于0: {cgns_file.name}")
                
                # 验证节点坐标维度
                self.assertEqual(reader.points.shape[1], 3, f"节点坐标应该是3D: {cgns_file.name}")
                
            except Exception as e:
                failed_files.append((cgns_file.name, str(e)))
        
        # 如果有文件读取失败，打印错误信息
        if failed_files:
            error_msg = "\n".join([f"  - {name}: {error}" for name, error in failed_files])
            self.fail(f"以下CGNS文件读取失败:\n{error_msg}")

    def test_cgns_file_structure(self):
        """测试CGNS文件结构"""
        # 测试第一个CGNS文件
        cgns_file = self.cgns_files[0]
        
        reader = UniversalCGNSReader(str(cgns_file))
        success = reader.read()
        
        self.assertTrue(success, f"读取CGNS文件失败: {cgns_file.name}")
        
        # 验证元数据
        self.assertIsNotNone(reader.metadata, "元数据不应为空")
        
        # 验证单元信息
        for cell in reader.cells:
            self.assertIn('type', cell, "单元应该包含type字段")
            self.assertIn('data', cell, "单元应该包含data字段")
            self.assertIn('num_nodes', cell, "单元应该包含num_nodes字段")
            self.assertIn('num_cells', cell, "单元应该包含num_cells字段")

    def test_to_meshio_format(self):
        """测试转换为meshio格式"""
        # 测试第一个CGNS文件
        cgns_file = self.cgns_files[0]
        
        reader = UniversalCGNSReader(str(cgns_file))
        success = reader.read()
        
        self.assertTrue(success, f"读取CGNS文件失败: {cgns_file.name}")
        
        # 尝试转换为meshio格式
        mesh = reader.to_meshio_format()
        
        self.assertIsNotNone(mesh, "转换为meshio格式不应返回None")
        
        # 验证meshio对象
        self.assertIsNotNone(mesh.points, "meshio节点数据不应为空")
        self.assertIsNotNone(mesh.cells, "meshio单元数据不应为空")
        
        # 验证数据一致性
        self.assertEqual(len(mesh.points), len(reader.points), "节点数应该一致")
        
        mesh_cells_count = sum(len(cell_block.data) for cell_block in mesh.cells)
        reader_cells_count = sum(cell['num_cells'] for cell in reader.cells)
        self.assertEqual(mesh_cells_count, reader_cells_count, "单元数应该一致")

    def test_high_order_elements(self):
        """测试高阶元素处理"""
        # 查找包含高阶元素的CGNS文件
        high_order_files = []
        
        for cgns_file in self.cgns_files:
            try:
                reader = UniversalCGNSReader(str(cgns_file))
                success = reader.read()
                
                if success:
                    # 检查是否有高阶元素
                    for cell in reader.cells:
                        if cell['num_nodes'] > 4:  # 高阶元素通常有更多节点
                            high_order_files.append(cgns_file)
                            break
            except Exception:
                continue
        
        # 如果找到了高阶元素文件，验证它们能正确读取
        if high_order_files:
            for cgns_file in high_order_files:
                reader = UniversalCGNSReader(str(cgns_file))
                success = reader.read()
                
                self.assertTrue(success, f"高阶元素CGNS文件读取失败: {cgns_file.name}")
                
                # 验证高阶元素
                for cell in reader.cells:
                    if cell['num_nodes'] > 4:
                        self.assertGreater(cell['num_cells'], 0, 
                                         f"高阶元素应该有单元: {cell['type']}")

    def test_metadata_extraction(self):
        """测试元数据提取"""
        # 测试第一个CGNS文件
        cgns_file = self.cgns_files[0]
        
        reader = UniversalCGNSReader(str(cgns_file))
        success = reader.read()
        
        self.assertTrue(success, f"读取CGNS文件失败: {cgns_file.name}")
        
        # 验证元数据
        self.assertIsInstance(reader.metadata, dict, "元数据应该是字典类型")
        
        # 检查常见的元数据字段
        if reader.metadata:
            possible_keys = ['CGNSLibraryVersion', 'BaseName', 'ZoneName', 'GridCoordinates']
            has_metadata = any(key in reader.metadata for key in possible_keys)
            self.assertTrue(has_metadata, "应该包含一些元数据字段")

    def test_print_summary(self):
        """测试打印摘要功能"""
        # 测试第一个CGNS文件
        cgns_file = self.cgns_files[0]
        
        reader = UniversalCGNSReader(str(cgns_file))
        success = reader.read()
        
        self.assertTrue(success, f"读取CGNS文件失败: {cgns_file.name}")
        
        # 测试print_summary方法不会抛出异常
        try:
            reader.print_summary()
        except Exception as e:
            self.fail(f"print_summary方法抛出异常: {e}")


if __name__ == '__main__':
    unittest.main()

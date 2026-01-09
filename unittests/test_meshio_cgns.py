#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meshio CGNS 文件读取单元测试
测试 meshio 库读取 CGNS 文件的功能
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

import meshio


class TestMeshioCGNS(unittest.TestCase):
    """测试 meshio 读取 CGNS 文件的功能"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.examples_dir = project_root / "examples"
        
        # 检查examples目录是否存在
        if not cls.examples_dir.exists():
            cls.skipTest(f"examples目录不存在: {cls.examples_dir}")
        
        # 查找所有CGNS文件
        cls.cgns_files = sorted(cls.examples_dir.rglob("*.cgns"))
        
        if not cls.cgns_files:
            cls.skipTest(f"examples目录中没有CGNS文件")

    def test_cgns_files_exist(self):
        """测试CGNS文件是否存在"""
        self.assertGreater(len(self.cgns_files), 0, "应该至少有一个CGNS文件")

    def test_read_all_cgns_files(self):
        """测试读取所有CGNS文件"""
        failed_files = []
        
        for cgns_file in self.cgns_files:
            try:
                mesh = meshio.read(str(cgns_file))
                
                self.assertIsNotNone(mesh, f"读取CGNS文件返回None: {cgns_file.name}")
                
                # 验证节点数据
                self.assertIsNotNone(mesh.points, f"节点数据不应为空: {cgns_file.name}")
                self.assertGreater(len(mesh.points), 0, f"节点数应该大于0: {cgns_file.name}")
                
                # 验证单元数据
                self.assertIsNotNone(mesh.cells, f"单元数据不应为空: {cgns_file.name}")
                self.assertGreater(len(mesh.cells), 0, f"单元数应该大于0: {cgns_file.name}")
                
                # 验证节点坐标维度
                self.assertEqual(mesh.points.shape[1], 3, f"节点坐标应该是3D: {cgns_file.name}")
                
            except Exception as e:
                failed_files.append((cgns_file.name, str(e)))
        
        # 如果有文件读取失败，打印错误信息
        if failed_files:
            error_msg = "\n".join([f"  - {name}: {error}" for name, error in failed_files])
            self.fail(f"以下CGNS文件读取失败:\n{error_msg}")

    def test_mesh_structure(self):
        """测试mesh结构"""
        # 测试第一个CGNS文件
        cgns_file = self.cgns_files[0]
        
        mesh = meshio.read(str(cgns_file))
        
        self.assertIsNotNone(mesh, "mesh对象不应为None")
        
        # 验证节点数据
        self.assertIsNotNone(mesh.points, "节点数据不应为空")
        self.assertIsInstance(mesh.points, type(mesh.points), "节点数据应该是numpy数组")
        
        # 验证单元数据
        self.assertIsNotNone(mesh.cells, "单元数据不应为空")
        self.assertIsInstance(mesh.cells, list, "单元数据应该是列表")
        
        # 验证每个单元块
        for cell_block in mesh.cells:
            self.assertIsNotNone(cell_block.type, "单元块应该有类型")
            self.assertIsNotNone(cell_block.data, "单元块应该有数据")
            self.assertGreater(len(cell_block.data), 0, "单元块应该有单元")

    def test_cell_types(self):
        """测试单元类型"""
        # 收集所有单元类型
        all_cell_types = set()
        
        for cgns_file in self.cgns_files:
            try:
                mesh = meshio.read(str(cgns_file))
                for cell_block in mesh.cells:
                    all_cell_types.add(cell_block.type)
            except Exception:
                continue
        
        # 验证常见的单元类型
        common_types = {'tetra', 'triangle', 'quad', 'hexahedron', 'wedge', 'pyramid', 'line'}
        found_types = all_cell_types & common_types
        
        self.assertGreater(len(found_types), 0, f"应该找到至少一种常见单元类型，找到: {all_cell_types}")

    def test_high_order_elements(self):
        """测试高阶元素处理"""
        # 查找包含高阶元素的CGNS文件
        high_order_files = []
        
        for cgns_file in self.cgns_files:
            try:
                mesh = meshio.read(str(cgns_file))
                
                # 检查是否有高阶元素（节点数多于标准元素）
                for cell_block in mesh.cells:
                    if cell_block.type == 'triangle' and cell_block.data.shape[1] > 3:
                        high_order_files.append(cgns_file)
                        break
                    elif cell_block.type == 'quad' and cell_block.data.shape[1] > 4:
                        high_order_files.append(cgns_file)
                        break
                    elif cell_block.type == 'tetra' and cell_block.data.shape[1] > 4:
                        high_order_files.append(cgns_file)
                        break
                    elif cell_block.type == 'hexahedron' and cell_block.data.shape[1] > 8:
                        high_order_files.append(cgns_file)
                        break
            except Exception:
                continue
        
        # 如果找到了高阶元素文件，验证它们能正确读取
        if high_order_files:
            for cgns_file in high_order_files:
                mesh = meshio.read(str(cgns_file))
                
                self.assertIsNotNone(mesh, f"高阶元素CGNS文件读取失败: {cgns_file.name}")
                
                # 验证高阶元素
                for cell_block in mesh.cells:
                    if cell_block.type == 'triangle' and cell_block.data.shape[1] > 3:
                        self.assertGreater(len(cell_block.data), 0, 
                                         f"高阶三角形应该有单元: {cell_block.data.shape[1]} 节点")
                    elif cell_block.type == 'quad' and cell_block.data.shape[1] > 4:
                        self.assertGreater(len(cell_block.data), 0, 
                                         f"高阶四边形应该有单元: {cell_block.data.shape[1]} 节点")

    def test_large_mesh_files(self):
        """测试大规模网格文件"""
        # 查找大规模网格文件（节点数 > 10000）
        large_files = []
        
        for cgns_file in self.cgns_files:
            try:
                mesh = meshio.read(str(cgns_file))
                if len(mesh.points) > 10000:
                    large_files.append((cgns_file, len(mesh.points)))
            except Exception:
                continue
        
        # 如果找到了大规模文件，验证它们能正确读取
        if large_files:
            for cgns_file, num_points in large_files:
                mesh = meshio.read(str(cgns_file))
                
                self.assertIsNotNone(mesh, f"大规模网格文件读取失败: {cgns_file.name}")
                self.assertEqual(len(mesh.points), num_points, 
                               f"节点数应该一致: {cgns_file.name}")

    def test_mixed_mesh_files(self):
        """测试混合网格文件"""
        # 查找包含多种单元类型的文件
        mixed_files = []
        
        for cgns_file in self.cgns_files:
            try:
                mesh = meshio.read(str(cgns_file))
                cell_types = set(cell_block.type for cell_block in mesh.cells)
                
                if len(cell_types) > 1:
                    mixed_files.append((cgns_file, cell_types))
            except Exception:
                continue
        
        # 如果找到了混合网格文件，验证它们能正确读取
        if mixed_files:
            for cgns_file, cell_types in mixed_files:
                mesh = meshio.read(str(cgns_file))
                
                self.assertIsNotNone(mesh, f"混合网格文件读取失败: {cgns_file.name}")
                
                # 验证多种单元类型
                mesh_cell_types = set(cell_block.type for cell_block in mesh.cells)
                self.assertEqual(mesh_cell_types, cell_types, 
                               f"单元类型应该一致: {cgns_file.name}")

    def test_mesh_data_consistency(self):
        """测试网格数据一致性"""
        # 测试第一个CGNS文件
        cgns_file = self.cgns_files[0]
        
        mesh = meshio.read(str(cgns_file))
        
        # 验证节点索引范围
        max_node_index = 0
        for cell_block in mesh.cells:
            if len(cell_block.data) > 0:
                max_index = cell_block.data.max()
                if max_index > max_node_index:
                    max_node_index = max_index
        
        # 验证所有节点索引都在有效范围内
        self.assertLessEqual(max_node_index, len(mesh.points) - 1, 
                            f"节点索引超出范围: 最大索引 {max_node_index}, 节点数 {len(mesh.points)}")

    def test_boundary_elements(self):
        """测试边界元素"""
        # 查找包含边界元素的CGNS文件（通常有line类型的单元）
        boundary_files = []
        
        for cgns_file in self.cgns_files:
            try:
                mesh = meshio.read(str(cgns_file))
                for cell_block in mesh.cells:
                    if cell_block.type == 'line':
                        boundary_files.append(cgns_file)
                        break
            except Exception:
                continue
        
        # 如果找到了包含边界元素的文件，验证它们能正确读取
        if boundary_files:
            for cgns_file in boundary_files:
                mesh = meshio.read(str(cgns_file))
                
                self.assertIsNotNone(mesh, f"包含边界元素的CGNS文件读取失败: {cgns_file.name}")
                
                # 验证边界元素
                has_line_elements = any(cell_block.type == 'line' for cell_block in mesh.cells)
                self.assertTrue(has_line_elements, f"应该包含line类型单元: {cgns_file.name}")


if __name__ == '__main__':
    unittest.main()

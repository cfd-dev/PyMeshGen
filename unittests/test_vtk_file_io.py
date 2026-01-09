#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：VTK文件导入导出功能
整合了test_vtk_import.py, test_vtk_simple.py, test_vtk_display.py的测试用例
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from fileIO.vtk_io import read_vtk, reconstruct_mesh_from_vtk, write_vtk, parse_vtk_msh
from data_structure.unstructured_grid import Unstructured_Grid
from data_structure.basic_elements import Triangle, Quadrilateral, NodeElement, Tetrahedron
from data_structure.vtk_types import VTKCellType


class TestVTKFileIO(unittest.TestCase):
    """VTK文件导入导出功能测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 查找VTK文件，优先使用unittests/test_files目录下的文件
        self.vtk_files = []

        # 首先检查unittests/test_files目录下的VTK文件
        test_files_dir = os.path.join(current_dir, "unittests", "test_files")
        if os.path.exists(test_files_dir):
            for file in os.listdir(test_files_dir):
                if file.endswith('.vtk'):
                    full_path = os.path.join(test_files_dir, file)
                    if os.path.getsize(full_path) > 100:
                        self.vtk_files.append(full_path)

        # 如果没有找到，再搜索整个项目目录（排除out和tmp目录）
        if not self.vtk_files:
            for root, dirs, files in os.walk(current_dir):
                if 'out' not in root and 'tmp' not in root and 'test_' not in root:
                    for file in files:
                        if file.endswith('.vtk'):
                            if not any(exclude in file for exclude in ['_test_roundtrip', 'tmp_', 'temp_']):
                                full_path = os.path.join(root, file)
                                if os.path.getsize(full_path) > 100:
                                    self.vtk_files.append(full_path)

    def test_read_vtk(self):
        """测试read_vtk函数"""
        for vtk_file in self.vtk_files:
            with self.subTest(file=vtk_file):
                node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container, part_ids = read_vtk(vtk_file)

                self.assertIsInstance(node_coords, list)
                self.assertIsInstance(cell_idx_container, list)
                self.assertIsInstance(boundary_nodes_idx, list)
                self.assertIsInstance(cell_type_container, list)

                self.assertGreater(len(node_coords), 0)
                for i in range(min(3, len(node_coords))):
                    coords = node_coords[i]
                    self.assertGreaterEqual(len(coords), 2, f"节点{i}坐标至少应该是2D的")

                self.assertGreater(len(cell_idx_container), 0)

    def test_reconstruct_mesh_from_vtk(self):
        """测试reconstruct_mesh_from_vtk函数"""
        for vtk_file in self.vtk_files:
            with self.subTest(file=vtk_file):
                try:
                    node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container, part_ids = read_vtk(vtk_file)
                    mesh = reconstruct_mesh_from_vtk(node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container)

                    self.assertIsInstance(mesh, Unstructured_Grid)
                    self.assertEqual(len(mesh.node_coords), len(node_coords))

                    valid_cells = [cell for cell in mesh.cell_container if cell is not None]
                    self.assertGreater(len(valid_cells), 0)
                    self.assertGreater(mesh.dim, 0)
                except ValueError as e:
                    if "节点数量不足" in str(e):
                        self.skipTest(f"VTK文件 {vtk_file} 包含无效单元，跳过测试: {e}")
                    else:
                        raise

    def test_mesh_properties(self):
        """测试网格属性"""
        for vtk_file in self.vtk_files:
            with self.subTest(file=vtk_file):
                try:
                    node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container, part_ids = read_vtk(vtk_file)
                    mesh = reconstruct_mesh_from_vtk(node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container)

                    self.assertIn(mesh.dim, [2, 3])
                    self.assertIsNotNone(mesh.bbox)
                    self.assertIsInstance(mesh.bbox, list)

                    if mesh.dim == 2:
                        self.assertEqual(len(mesh.bbox), 4)
                    elif mesh.dim == 3:
                        self.assertEqual(len(mesh.bbox), 6)
                except ValueError as e:
                    if "节点数量不足" in str(e):
                        self.skipTest(f"VTK文件 {vtk_file} 包含无效单元，跳过测试: {e}")
                    else:
                        raise

    def test_vtk_roundtrip(self):
        """测试VTK文件的往返转换"""
        for vtk_file in self.vtk_files:
            with self.subTest(file=vtk_file):
                try:
                    node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container, part_ids = read_vtk(vtk_file)
                    mesh = reconstruct_mesh_from_vtk(node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container)

                    test_vtk_file = vtk_file.replace('.vtk', '_test_roundtrip.vtk')
                    mesh.save_to_vtkfile(test_vtk_file)

                    self.assertTrue(os.path.exists(test_vtk_file))

                    new_node_coords, new_cell_idx_container, new_boundary_nodes_idx, new_cell_type_container, new_part_ids = read_vtk(test_vtk_file)
                    new_mesh = reconstruct_mesh_from_vtk(new_node_coords, new_cell_idx_container, new_boundary_nodes_idx, new_cell_type_container)

                    self.assertEqual(len(mesh.node_coords), len(new_mesh.node_coords))
                    original_cells = [cell for cell in mesh.cell_container if cell is not None]
                    new_cells = [cell for cell in new_mesh.cell_container if cell is not None]
                    self.assertEqual(len(original_cells), len(new_cells))

                    if os.path.exists(test_vtk_file):
                        os.remove(test_vtk_file)
                except ValueError as e:
                    if "节点数量不足" in str(e):
                        self.skipTest(f"VTK文件 {vtk_file} 包含无效单元，跳过测试: {e}")
                    else:
                        raise


class TestVTKMeshStructure(unittest.TestCase):
    """VTK网格数据结构测试类"""

    def setUp(self):
        """在每个测试方法运行前执行"""
        nodes = [
            NodeElement([0.0, 0.0, 0.0], 0),
            NodeElement([1.0, 0.0, 0.0], 1),
            NodeElement([1.0, 1.0, 0.0], 2),
            NodeElement([0.0, 1.0, 0.0], 3),
            NodeElement([0.0, 0.0, 1.0], 4),
            NodeElement([1.0, 0.0, 1.0], 5),
            NodeElement([1.0, 1.0, 1.0], 6),
            NodeElement([0.0, 1.0, 1.0], 7),
        ]

        tri = Triangle(nodes[0], nodes[1], nodes[2])
        quad = Quadrilateral(nodes[4], nodes[5], nodes[6], nodes[7])

        cells = [tri, quad]
        node_coords = [node.coords for node in nodes]
        boundary_nodes = [nodes[0], nodes[1], nodes[2], nodes[3]]

        self.test_grid = Unstructured_Grid(cells, node_coords, boundary_nodes)

    def test_mesh_data_structure(self):
        """测试网格数据结构 - 验证网格对象的基本属性"""
        self.assertIsNotNone(self.test_grid)
        self.assertIsInstance(self.test_grid, Unstructured_Grid)

        self.assertTrue(hasattr(self.test_grid, 'cell_container'))
        self.assertTrue(hasattr(self.test_grid, 'node_coords'))
        self.assertTrue(hasattr(self.test_grid, 'boundary_nodes'))

        self.assertGreater(len(self.test_grid.node_coords), 0)
        self.assertGreater(len(self.test_grid.cell_container), 0)
        self.assertGreater(len(self.test_grid.boundary_nodes), 0)

    def test_basic_elements_creation(self):
        """测试基本元素创建 - 验证节点、三角形、四边形的创建"""
        node = NodeElement([1.0, 2.0, 3.0], 10)
        self.assertIsNotNone(node)
        self.assertEqual(node.coords, [1.0, 2.0, 3.0])
        self.assertEqual(node.idx, 10)

        node1 = NodeElement([0.0, 0.0, 0.0], 0)
        node2 = NodeElement([1.0, 0.0, 0.0], 1)
        node3 = NodeElement([0.0, 1.0, 0.0], 2)
        tri = Triangle(node1, node2, node3)
        self.assertIsNotNone(tri)

        node4 = NodeElement([1.0, 1.0, 0.0], 3)
        quad = Quadrilateral(node1, node2, node3, node4)
        self.assertIsNotNone(quad)

    def test_unstructured_grid_attributes(self):
        """测试Unstructured_Grid对象的属性 - 验证网格对象的属性和方法"""
        self.assertTrue(hasattr(self.test_grid, 'num_nodes'))
        self.assertTrue(hasattr(self.test_grid, 'num_cells'))
        self.assertTrue(hasattr(self.test_grid, 'num_boundary_nodes'))

        self.assertEqual(self.test_grid.num_nodes, len(self.test_grid.node_coords))
        self.assertGreater(self.test_grid.num_cells, 0)
        self.assertGreater(self.test_grid.num_boundary_nodes, 0)

        self.assertTrue(hasattr(self.test_grid, 'boundary_nodes_list'))
        self.assertGreater(len(self.test_grid.boundary_nodes_list), 0)

    def test_triangle_quadrilateral_properties(self):
        """测试三角形和四边形的属性 - 验证几何元素的基本属性"""
        nodes = [NodeElement([float(i), 0.0, 0.0], i) for i in range(4)]

        tri = Triangle(nodes[0], nodes[1], nodes[2])
        self.assertIsNotNone(tri)
        self.assertTrue(hasattr(tri, 'p1'))
        self.assertTrue(hasattr(tri, 'p2'))
        self.assertTrue(hasattr(tri, 'p3'))

        quad = Quadrilateral(nodes[0], nodes[1], nodes[2], nodes[3])
        self.assertIsNotNone(quad)
        self.assertTrue(hasattr(quad, 'p1'))
        self.assertTrue(hasattr(quad, 'p2'))
        self.assertTrue(hasattr(quad, 'p3'))
        self.assertTrue(hasattr(quad, 'p4'))

    def test_mesh_coordinates_validation(self):
        """测试网格坐标验证 - 验证坐标数据的正确性"""
        for coord in self.test_grid.node_coords:
            self.assertIsInstance(coord, list)
            self.assertGreaterEqual(len(coord), 2)
            for val in coord:
                self.assertIsInstance(val, (int, float))


class TestTetrahedronVTKIO(unittest.TestCase):
    """四面体VTK文件导入导出功能测试类"""

    def setUp(self):
        """在每个测试方法运行前执行"""
        self.test_dir = os.path.join(current_dir, "out")
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def test_tetrahedron_vtk_write_and_read(self):
        """测试四面体VTK文件的写入和读取"""
        nodes = [
            NodeElement((0, 0, 0), 0),
            NodeElement((1, 0, 0), 1),
            NodeElement((0, 1, 0), 2),
            NodeElement((0, 0, 1), 3),
            NodeElement((1, 1, 1), 4),
        ]
        
        tetra1 = Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3], part_name="tetra1", idx=0)
        tetra2 = Tetrahedron(nodes[1], nodes[2], nodes[3], nodes[4], part_name="tetra2", idx=1)
        
        tetra1.init_metrics()
        tetra2.init_metrics()
        
        node_coords = [node.coords for node in nodes]
        boundary_nodes = nodes
        mesh = Unstructured_Grid([tetra1, tetra2], node_coords, boundary_nodes)
        
        self.assertEqual(len(mesh.node_coords), 5)
        self.assertEqual(len(mesh.cell_container), 2)
        
        vtk_path = os.path.join(self.test_dir, "test_tetrahedron.vtk")
        
        cell_idx_container = [cell.node_ids for cell in mesh.cell_container]
        boundary_nodes_idx = [node.idx for node in mesh.boundary_nodes]
        cell_type_container = [VTKCellType.TETRA] * len(mesh.cell_container)
        cell_part_names = [cell.part_name for cell in mesh.cell_container]
        
        write_vtk(vtk_path, mesh.node_coords, cell_idx_container, boundary_nodes_idx, 
                 cell_type_container, cell_part_names)
        
        self.assertTrue(os.path.exists(vtk_path))
        
        read_mesh = parse_vtk_msh(vtk_path)
        
        self.assertEqual(len(read_mesh.node_coords), 5)
        self.assertEqual(len(read_mesh.cell_container), 2)
        
        for i, cell in enumerate(read_mesh.cell_container):
            self.assertIsInstance(cell, Tetrahedron, f"单元{i}应该是四面体类型")
        
        read_mesh.summary()
        
        if os.path.exists(vtk_path):
            os.remove(vtk_path)

    def test_tetrahedron_vtk_roundtrip(self):
        """测试四面体VTK文件的往返转换"""
        nodes = [
            NodeElement((0, 0, 0), 0),
            NodeElement((1, 0, 0), 1),
            NodeElement((0, 1, 0), 2),
            NodeElement((0, 0, 1), 3),
        ]
        
        tetra = Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3], part_name="test", idx=0)
        tetra.init_metrics()
        
        node_coords = [node.coords for node in nodes]
        boundary_nodes = nodes
        mesh = Unstructured_Grid([tetra], node_coords, boundary_nodes)
        
        vtk_path = os.path.join(self.test_dir, "test_tetrahedron_roundtrip.vtk")
        
        cell_idx_container = [cell.node_ids for cell in mesh.cell_container]
        boundary_nodes_idx = [node.idx for node in mesh.boundary_nodes]
        cell_type_container = [VTKCellType.TETRA] * len(mesh.cell_container)
        cell_part_names = [cell.part_name for cell in mesh.cell_container]
        
        write_vtk(vtk_path, mesh.node_coords, cell_idx_container, boundary_nodes_idx, 
                 cell_type_container, cell_part_names)
        
        read_mesh = parse_vtk_msh(vtk_path)
        
        self.assertEqual(len(mesh.node_coords), len(read_mesh.node_coords))
        self.assertEqual(len(mesh.cell_container), len(read_mesh.cell_container))
        
        if os.path.exists(vtk_path):
            os.remove(vtk_path)

    def test_tetrahedron_mesh_properties(self):
        """测试四面体网格属性"""
        nodes = [
            NodeElement((0, 0, 0), 0),
            NodeElement((1, 0, 0), 1),
            NodeElement((0, 1, 0), 2),
            NodeElement((0, 0, 1), 3),
        ]
        
        tetra = Tetrahedron(nodes[0], nodes[1], nodes[2], nodes[3], part_name="test", idx=0)
        tetra.init_metrics()
        
        node_coords = [node.coords for node in nodes]
        boundary_nodes = nodes
        mesh = Unstructured_Grid([tetra], node_coords, boundary_nodes)
        
        self.assertEqual(mesh.dim, 3)
        self.assertIsNotNone(mesh.bbox)
        self.assertEqual(len(mesh.bbox), 6)
        
        vtk_path = os.path.join(self.test_dir, "test_tetrahedron_props.vtk")
        
        cell_idx_container = [cell.node_ids for cell in mesh.cell_container]
        boundary_nodes_idx = [node.idx for node in mesh.boundary_nodes]
        cell_type_container = [VTKCellType.TETRA] * len(mesh.cell_container)
        cell_part_names = [cell.part_name for cell in mesh.cell_container]
        
        write_vtk(vtk_path, mesh.node_coords, cell_idx_container, boundary_nodes_idx, 
                 cell_type_container, cell_part_names)
        
        read_mesh = parse_vtk_msh(vtk_path)
        
        self.assertEqual(read_mesh.dim, 3)
        self.assertIsNotNone(read_mesh.bbox)
        self.assertEqual(len(read_mesh.bbox), 6)
        
        if os.path.exists(vtk_path):
            os.remove(vtk_path)


if __name__ == "__main__":
    unittest.main()

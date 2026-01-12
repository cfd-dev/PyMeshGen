#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：CAS文件部件信息解析
从test_cas_parts.py转换而来
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fileIO.read_cas import parse_fluent_msh, reconstruct_mesh_from_cas


class TestCASPartsInfo(unittest.TestCase):
    """测试CAS文件部件信息解析功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_files = [
            os.path.join(self.project_root, "config/input/quad.cas"),
            os.path.join(self.project_root, "config/input/concave.cas"),
            os.path.join(self.project_root, "config/input/convex.cas")
        ]
        self.existing_files = [f for f in self.test_files if os.path.exists(f)]

    def test_parse_fluent_msh(self):
        """测试parse_fluent_msh函数"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                raw_cas_data = parse_fluent_msh(test_file)

                self.assertIn('node_count', raw_cas_data)
                self.assertIn('face_count', raw_cas_data)
                self.assertIn('cell_count', raw_cas_data)
                self.assertIn('zones', raw_cas_data)

                self.assertGreater(raw_cas_data['node_count'], 0)
                self.assertGreater(raw_cas_data['face_count'], 0)
                self.assertGreater(raw_cas_data['cell_count'], 0)
                self.assertGreater(len(raw_cas_data['zones']), 0)

    def test_zones_info(self):
        """测试区域信息"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                raw_cas_data = parse_fluent_msh(test_file)

                for zone_id, zone in raw_cas_data['zones'].items():
                    self.assertIn('type', zone)
                    self.assertIn('zone_id', zone)

    def test_reconstruct_mesh_from_cas(self):
        """测试reconstruct_mesh_from_cas函数"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                raw_cas_data = parse_fluent_msh(test_file)
                mesh = reconstruct_mesh_from_cas(raw_cas_data)

                self.assertIsNotNone(mesh)
                self.assertTrue(hasattr(mesh, 'node_coords'))
                self.assertTrue(hasattr(mesh, 'cell_container'))

                self.assertGreater(len(mesh.node_coords), 0)
                self.assertGreater(len(mesh.cell_container), 0)

    def test_parts_info(self):
        """测试parts_info属性"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                raw_cas_data = parse_fluent_msh(test_file)
                mesh = reconstruct_mesh_from_cas(raw_cas_data)

                if hasattr(mesh, 'parts_info'):
                    self.assertIsInstance(mesh.parts_info, dict)

                    for part_name, part_data in mesh.parts_info.items():
                        self.assertIsInstance(part_data, dict)
                        self.assertIn('bc_type', part_data)
                        self.assertIn('faces', part_data)
                        self.assertIsInstance(part_data['faces'], list)

    def test_boundary_info(self):
        """测试boundary_info属性"""
        for test_file in self.existing_files:
            with self.subTest(file=test_file):
                raw_cas_data = parse_fluent_msh(test_file)
                mesh = reconstruct_mesh_from_cas(raw_cas_data)

                if hasattr(mesh, 'boundary_info'):
                    self.assertIsInstance(mesh.boundary_info, dict)

                    for bc_name, bc_data in mesh.boundary_info.items():
                        self.assertIsInstance(bc_data, dict)
                        self.assertIn('bc_type', bc_data)
                        self.assertIn('faces', bc_data)
                        self.assertIsInstance(bc_data['faces'], list)


if __name__ == "__main__":
    unittest.main()

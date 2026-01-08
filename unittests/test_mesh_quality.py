#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：网格质量评估功能
整合了test_mesh_quality.py的测试用例
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import importlib.util

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 直接加载mesh_quality模块
spec = importlib.util.spec_from_file_location(
    "mesh_quality_module",
    str(project_root / "optimize" / "mesh_quality.py")
)
mesh_quality_module = importlib.util.module_from_spec(spec)

# 在加载前设置sys.modules
sys.modules['mesh_quality_module'] = mesh_quality_module
spec.loader.exec_module(mesh_quality_module)

# 直接加载stl_io模块
spec2 = importlib.util.spec_from_file_location(
    "stl_io_module",
    str(project_root / "fileIO" / "stl_io.py")
)
stl_io_module = importlib.util.module_from_spec(spec2)
sys.modules['stl_io_module'] = stl_io_module
spec2.loader.exec_module(stl_io_module)

triangle_shape_quality = mesh_quality_module.triangle_shape_quality
triangle_skewness = mesh_quality_module.triangle_skewness
quadrilateral_skewness = mesh_quality_module.quadrilateral_skewness
quadrilateral_aspect_ratio = mesh_quality_module.quadrilateral_aspect_ratio
quadrilateral_shape_quality = mesh_quality_module.quadrilateral_shape_quality
tetrahedron_shape_quality = mesh_quality_module.tetrahedron_shape_quality
tetrahedron_skewness = mesh_quality_module.tetrahedron_skewness
parse_stl_msh = stl_io_module.parse_stl_msh

from data_structure.basic_elements import Tetrahedron, NodeElement
from utils.geom_toolkit import tetrahedron_volume


class TestMeshQuality(unittest.TestCase):
    """网格质量评估测试类"""

    def setUp(self):
        # 公共测试数据
        self.perfect_triangle = [
            (0, 0),
            (1, 0),
            (0.5, np.sqrt(3) / 2),
        ]
        self.degenerate_triangle = [(0, 0), (1, 0), (2, 0)]

        self.square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.non_convex_quad = [(0, 0), (2, 0), (1, 1), (0, 2)]
        self.isosceles_triangle = [(0, 0), (2, 0), (1, 1.5)]
        self.right_triangle = [(0, 0), (3, 0), (0, 4)]
        self.obtuse_triangle = [(0, 0), (2, 0), (0.5, 1)]
        self.trapezoid = [(0, 0), (2, 0), (1.5, 1), (0.5, 1)]
        self.parallelogram = [(0, 0), (2, 0), (3, 1), (1, 1)]

        self.test_dir = Path(__file__).parent / "test_files"

    def test_triangle_quality_valid(self):
        """测试理想三角形质量"""
        quality = triangle_shape_quality(*self.perfect_triangle)
        self.assertAlmostEqual(quality, 1.0, delta=0.01)

        quality = triangle_shape_quality(*self.degenerate_triangle)
        self.assertEqual(quality, 0.0)

    def test_triangle_skewness(self):
        """测试三角形偏斜度"""
        skew = triangle_skewness(*self.perfect_triangle)
        self.assertAlmostEqual(skew, 1.0, delta=0.01)

    def test_quadrilateral_skewness(self):
        """测试四边形偏斜度"""
        skew = quadrilateral_skewness(*self.square)
        self.assertAlmostEqual(skew, 1.0, delta=0.01)

        skew = quadrilateral_skewness(*self.square[:3], (0, 0))
        self.assertEqual(skew, 0.0)

    def test_quadrilateral_aspect_ratio(self):
        """测试四边形长宽比"""
        ratio = quadrilateral_aspect_ratio(*self.square)
        self.assertAlmostEqual(ratio, 1.0, delta=0.01)

        rect = [(0, 0), (10, 0), (10, 1), (0, 1)]
        ratio = quadrilateral_aspect_ratio(*rect)
        self.assertAlmostEqual(ratio, 10.0)

    def test_quadrilateral_quality(self):
        """测试四边形质量"""
        quality = quadrilateral_shape_quality(*self.square)
        self.assertGreater(quality, 0.9)

        quality = quadrilateral_shape_quality(*self.non_convex_quad)
        self.assertEqual(quality, 0.0)

    def test_additional_triangle_quality(self):
        """测试普通三角形质量"""
        quality = triangle_shape_quality(*self.isosceles_triangle)
        self.assertAlmostEqual(quality, 0.98974, delta=0.00001)

        expected = 4 * np.sqrt(3) * 6 / (3**2 + 4**2 + 5**2)
        self.assertAlmostEqual(
            triangle_shape_quality(*self.right_triangle), expected, delta=0.001
        )

        obtuse_quality = triangle_shape_quality(*self.obtuse_triangle)
        expected_obtuse = 4 * np.sqrt(3) * 1 / (2**2 + 1.118**2 + 1.802**2)
        self.assertAlmostEqual(obtuse_quality, expected_obtuse, delta=0.001)

    def test_additional_quadrilateral_quality(self):
        """测试普通四边形质量"""
        trapezoid_quality = quadrilateral_shape_quality(*self.trapezoid)
        self.assertAlmostEqual(trapezoid_quality, 0.78694, delta=0.001)

        para_quality = quadrilateral_shape_quality(*self.parallelogram)
        self.assertAlmostEqual(para_quality, 0.510204, delta=0.00001)

        bad_quad = [(0, 0), (1, 0), (1.1, 0.1), (0.1, 0.1)]
        self.assertAlmostEqual(
            quadrilateral_shape_quality(*bad_quad), 0.123196, delta=1e-3
        )

    def test_stl_quality(self):
        """测试STL文件质量评估"""
        stl_path = self.test_dir / "training_mesh.stl"
        if stl_path.exists():
            unstr_grid = parse_stl_msh(stl_path)
            unstr_grid.summary()
        else:
            self.skipTest("STL测试文件不存在")

    def test_tetrahedron_quality(self):
        """测试四面体质量计算"""
        p1 = (1, 1, 1)
        p2 = (1, -1, -1)
        p3 = (-1, 1, -1)
        p4 = (-1, -1, 1)
        
        quality = tetrahedron_shape_quality(p1, p2, p3, p4)
        self.assertGreater(quality, 0.9, f"正四面体质量应该接近1.0，实际为{quality}")
        
        p1 = (0, 0, 0)
        p2 = (1, 0, 0)
        p3 = (0, 1, 0)
        p4 = (0.5, 0.5, 0.001)
        
        quality = tetrahedron_shape_quality(p1, p2, p3, p4)
        self.assertLess(quality, 0.1, f"扁平四面体质量应该接近0，实际为{quality}")
        
        p1 = (0, 0, 0)
        p2 = (1, 0, 0)
        p3 = (0, 1, 0)
        p4 = (0, 0, 1)
        
        quality = tetrahedron_shape_quality(p1, p2, p3, p4)
        self.assertTrue(0 < quality < 1, f"普通四面体质量应该在0和1之间，实际为{quality}")

    def test_tetrahedron_skewness(self):
        """测试四面体偏斜度计算"""
        p1 = (1, 1, 1)
        p2 = (1, -1, -1)
        p3 = (-1, 1, -1)
        p4 = (-1, -1, 1)
        
        skewness = tetrahedron_skewness(p1, p2, p3, p4)
        self.assertGreater(skewness, 0.6, f"正四面体偏斜度应该较高，实际为{skewness}")
        
        p1 = (0, 0, 0)
        p2 = (1, 0, 0)
        p3 = (0, 1, 0)
        p4 = (0.5, 0.5, 0.001)
        
        skewness = tetrahedron_skewness(p1, p2, p3, p4)
        self.assertLess(skewness, 0.1, f"扁平四面体偏斜度应该接近0，实际为{skewness}")

    def test_tetrahedron_volume(self):
        """测试四面体体积计算"""
        p1 = (0, 0, 0)
        p2 = (1, 0, 0)
        p3 = (0, 1, 0)
        p4 = (0, 0, 1)
        
        volume = tetrahedron_volume(p1, p2, p3, p4)
        expected = 1.0 / 6.0
        self.assertAlmostEqual(volume, expected, delta=0.001, msg=f"体积应该为{expected}，实际为{volume}")

    def test_tetrahedron_class(self):
        """测试Tetrahedron类初始化和指标计算"""
        p1 = (0, 0, 0)
        p2 = (1, 0, 0)
        p3 = (0, 1, 0)
        p4 = (0, 0, 1)
        
        tetra = Tetrahedron(p1, p2, p3, p4, part_name="test", idx=0)
        
        self.assertIsNone(tetra.volume)
        self.assertIsNone(tetra.quality)
        self.assertIsNone(tetra.skewness)
        
        tetra.init_metrics()
        
        self.assertIsNotNone(tetra.volume, "体积应该被计算")
        self.assertIsNotNone(tetra.quality, "质量应该被计算")
        self.assertIsNotNone(tetra.skewness, "偏斜度应该被计算")
        
        tetra.init_metrics(force_update=True)

    def test_tetrahedron_with_nodes(self):
        """测试使用Node对象创建四面体"""
        node1 = NodeElement((0, 0, 0), 0)
        node2 = NodeElement((1, 0, 0), 1)
        node3 = NodeElement((0, 1, 0), 2)
        node4 = NodeElement((0, 0, 1), 3)
        
        tetra = Tetrahedron(node1, node2, node3, node4, part_name="test", idx=0)
        tetra.init_metrics()
        
        self.assertIsNotNone(tetra.volume)
        self.assertIsNotNone(tetra.quality)
        self.assertIsNotNone(tetra.skewness)
        self.assertEqual(tetra.node_ids, [0, 1, 2, 3], "节点ID应该正确")

    def test_tetrahedron_bbox(self):
        """测试四面体边界框"""
        p1 = (0, 0, 0)
        p2 = (2, 0, 0)
        p3 = (0, 3, 0)
        p4 = (0, 0, 4)
        
        tetra = Tetrahedron(p1, p2, p3, p4)
        
        expected_bbox = [0, 0, 0, 2, 3, 4]
        self.assertEqual(tetra.bbox, expected_bbox, f"边界框应该为{expected_bbox}，实际为{tetra.bbox}")

    def test_tetrahedron_hash(self):
        """测试四面体哈希"""
        p1 = (0, 0, 0)
        p2 = (1, 0, 0)
        p3 = (0, 1, 0)
        p4 = (0, 0, 1)
        
        tetra1 = Tetrahedron(p1, p2, p3, p4, idx=0)
        tetra2 = Tetrahedron(p1, p2, p3, p4, idx=1)
        
        self.assertEqual(tetra1, tetra2, "相同几何形状的四面体应该相等")
        self.assertEqual(hash(tetra1), hash(tetra2), "相同几何形状的四面体哈希应该相同")


if __name__ == "__main__":
    unittest.main()

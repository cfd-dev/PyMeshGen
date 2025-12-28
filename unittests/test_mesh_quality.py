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
parse_stl_msh = stl_io_module.parse_stl_msh


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


if __name__ == "__main__":
    unittest.main()

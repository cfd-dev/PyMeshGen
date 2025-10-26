import sys
from pathlib import Path
import unittest
import numpy as np
from optimize.mesh_quality import (
    triangle_shape_quality,
    triangle_skewness,
    quadrilateral_skewness,
    quadrilateral_aspect_ratio,
    quadrilateral_shape_quality,
)
from fileIO.stl_io import parse_stl_msh


class TestMeshQuality(unittest.TestCase):
    def setUp(self):
        # 公共测试数据
        self.perfect_triangle = [
            (0, 0),
            (1, 0),
            (0.5, np.sqrt(3) / 2),
        ]  # 理想等边三角形
        self.degenerate_triangle = [(0, 0), (1, 0), (2, 0)]  # 退化三角形

        self.square = [(0, 0), (1, 0), (1, 1), (0, 1)]  # 完美正方形
        self.non_convex_quad = [(0, 0), (2, 0), (1, 1), (0, 2)]  # 非凸四边形
        self.isosceles_triangle = [(0, 0), (2, 0), (1, 1.5)]  # 等腰三角形
        self.right_triangle = [(0, 0), (3, 0), (0, 4)]  # 直角三角形
        self.obtuse_triangle = [(0, 0), (2, 0), (0.5, 1)]  # 钝角三角形
        self.trapezoid = [(0, 0), (2, 0), (1.5, 1), (0.5, 1)]  # 梯形
        self.parallelogram = [(0, 0), (2, 0), (3, 1), (1, 1)]  # 平行四边形

        self.test_dir = Path(__file__).parent / "test_files"

    def test_additional_triangle_quality(self):
        """测试普通三角形质量"""
        quality = triangle_shape_quality(*self.isosceles_triangle)
        self.assertAlmostEqual(quality, 0.98974, delta=0.00001)

        # 直角三角形质量
        expected = 4 * np.sqrt(3) * 6 / (3**2 + 4**2 + 5**2)
        self.assertAlmostEqual(
            triangle_shape_quality(*self.right_triangle), expected, delta=0.001
        )

        # 钝角三角形质量应较低
        obtuse_quality = triangle_shape_quality(*self.obtuse_triangle)
        expected_obtuse = 4 * np.sqrt(3) * 1 / (2**2 + 1.118**2 + 1.802**2)
        self.assertAlmostEqual(obtuse_quality, expected_obtuse, delta=0.001)

    def test_additional_quadrilateral_quality(self):
        """测试普通四边形质量"""
        # 梯形质量
        trapezoid_quality = quadrilateral_shape_quality(*self.trapezoid)
        self.assertAlmostEqual(trapezoid_quality, 0.78694, delta=0.001)

        # 平行四边形质量
        para_quality = quadrilateral_shape_quality(*self.parallelogram)
        self.assertAlmostEqual(para_quality, 0.510204, delta=0.00001)

        # 接近退化的四边形
        bad_quad = [(0, 0), (1, 0), (1.1, 0.1), (0.1, 0.1)]
        self.assertAlmostEqual(
            quadrilateral_shape_quality(*bad_quad), 0.123196, delta=1e-3
        )

    def test_triangle_quality_valid(self):
        # 测试理想三角形质量
        quality = triangle_shape_quality(*self.perfect_triangle)
        self.assertAlmostEqual(quality, 1.0, delta=0.01)

        # 测试退化三角形
        self.assertEqual(triangle_shape_quality(*self.degenerate_triangle), 0.0)

    def test_triangle_skewness(self):
        # 等边三角形偏斜度应为1
        skew = triangle_skewness(*self.perfect_triangle)
        self.assertAlmostEqual(skew, 1.0, delta=0.01)

    def test_quadrilateral_skewness(self):
        # 完美正方形的偏斜度
        skew = quadrilateral_skewness(*self.square)
        self.assertAlmostEqual(skew, 1.0, delta=0.01)

        # 非四边形情况
        self.assertEqual(quadrilateral_skewness(*self.square[:3], (0, 0)), 0.0)

    def test_quadrilateral_aspect_ratio(self):
        # 正方形长宽比应为1
        ratio = quadrilateral_aspect_ratio(*self.square)
        self.assertAlmostEqual(ratio, 1.0, delta=0.01)

        # 极端长宽比情况
        rect = [(0, 0), (10, 0), (10, 1), (0, 1)]
        self.assertAlmostEqual(quadrilateral_aspect_ratio(*rect), 10.0)

    def test_quadrilateral_quality(self):
        # 完美四边形质量
        quality = quadrilateral_shape_quality(*self.square)
        self.assertGreater(quality, 0.9)

        # 非凸四边形质量应为0
        self.assertEqual(quadrilateral_shape_quality(*self.non_convex_quad), 0.0)

    def test_stl_quality(self):
        stl_path = self.test_dir / "training_mesh.stl"
        unstr_grid = parse_stl_msh(stl_path)
        unstr_grid.summary()


if __name__ == "__main__":
    unittest.main()

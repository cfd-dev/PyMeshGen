import unittest
import numpy as np
from mesh_quality import (
    triangle_quality,
    triangle_skewness,
    quadrilateral_skewness,
    quadrilateral_aspect_ratio,
    quadrilateral_quality
)

class TestMeshQuality(unittest.TestCase):
    def setUp(self):
        # 公共测试数据
        self.perfect_triangle = [(0,0), (1,0), (0.5, np.sqrt(3)/2)]  # 理想等边三角形
        self.degenerate_triangle = [(0,0), (1,0), (2,0)]  # 退化三角形
        
        self.square = [(0,0), (1,0), (1,1), (0,1)]  # 完美正方形
        self.non_convex_quad = [(0,0), (2,0), (1,1), (0,2)]  # 非凸四边形

    def test_triangle_quality_valid(self):
        # 测试理想三角形质量
        quality = triangle_quality(*self.perfect_triangle)
        self.assertAlmostEqual(quality, 1.0, delta=0.01)
        
        # 测试退化三角形
        self.assertEqual(triangle_quality(*self.degenerate_triangle), 0.0)

    def test_triangle_skewness(self):
        # 等边三角形偏斜度应为1
        skew = triangle_skewness(*self.perfect_triangle)
        self.assertAlmostEqual(skew, 1.0, delta=0.01)
        


    def test_quadrilateral_skewness(self):
        # 完美正方形的偏斜度
        skew = quadrilateral_skewness(*self.square)
        self.assertAlmostEqual(skew, 1.0, delta=0.01)
        
        # 非四边形情况
        self.assertEqual(quadrilateral_skewness(*self.square[:3], (0,0)), 0.0)

    def test_quadrilateral_aspect_ratio(self):
        # 正方形长宽比应为1
        ratio = quadrilateral_aspect_ratio(*self.square)
        self.assertAlmostEqual(ratio, 1.0, delta=0.01)
        
        # 极端长宽比情况
        rect = [(0,0), (10,0), (10,1), (0,1)]
        self.assertAlmostEqual(quadrilateral_aspect_ratio(*rect), 10.0)

    def test_quadrilateral_quality(self):
        # 完美四边形质量
        quality = quadrilateral_quality(*self.square)
        self.assertGreater(quality, 0.9)
        
        # 非凸四边形质量应为0
        self.assertEqual(quadrilateral_quality(*self.non_convex_quad), 0.0)

if __name__ == '__main__':
    unittest.main()
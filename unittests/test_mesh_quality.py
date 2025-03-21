import unittest
from unittest.mock import Mock, patch
import math
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "optimize"))
from optimize import calculate_triangle_twist, check_mesh_quality


class TestCalculateTriangleTwist(unittest.TestCase):
    def setUp(self):
        # 创建模拟三角形对象
        self.triangle = Mock()
        self.triangle.p1 = (0, 0)
        self.triangle.p2 = (1, 0)
        self.triangle.p3 = (0.5, math.sqrt(3) / 2)

        # 模拟geo_info的calculate_distance方法
        self.patcher = patch("geo_info.calculate_distance")
        self.mock_calculate = self.patcher.start()
        self.mock_calculate.side_effect = lambda p1, p2: math.dist(p1, p2)

    def test_equilateral_triangle(self):
        """测试等边三角形扭曲度计算"""
        twist = calculate_triangle_twist(self.triangle)
        self.assertAlmostEqual(twist, 1.0, delta=0.01)

    def test_degenerate_triangle(self):
        """测试退化三角形扭曲度接近0"""
        self.triangle.p3 = (1, 0)  # 使三角形退化
        twist = calculate_triangle_twist(self.triangle)
        self.assertAlmostEqual(twist, 0.0, delta=1e-6)

    def tearDown(self):
        self.patcher.stop()


class TestCheckMeshQuality(unittest.TestCase):
    @patch("matplotlib.pyplot.hist")
    @patch("matplotlib.pyplot.show")
    def test_quality_check_flow(self, mock_show, mock_hist):
        """测试质量检查流程是否完整执行"""
        # 创建模拟网格对象
        mock_grid = Mock()
        mock_grid.cell_nodes = [Mock() for _ in range(5)]  # 5个单元

        # 执行质量检查
        check_mesh_quality(mock_grid)

        # 验证执行流程
        self.assertEqual(len(mock_hist.call_args_list), 1)
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()

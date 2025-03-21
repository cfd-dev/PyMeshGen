import sys
from pathlib import Path
import unittest
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / "utils"))
import geometry_info as geo_info


class TestTriangle(unittest.TestCase):
    def plot_triangle(self, tri1, tri2):
        """绘制三角形"""
        fig, ax = plt.subplots(figsize=(8, 6))
        tri1_poly = Polygon(
            [tri1.p1, tri1.p2, tri1.p3],
            closed=True,
            fill=False,
            edgecolor="blue",
            linewidth=2,
        )
        ax.add_patch(tri1_poly)
        tri2_poly = Polygon(
            [tri2.p1, tri2.p2, tri2.p3],
            closed=True,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(tri2_poly)
        ax.autoscale_view()
        plt.show()

    def test_initialization(self):
        """测试三角形初始化及边界框计算"""
        tri = geo_info.Triangle((0, 0), (2, 0), (1, 2))
        self.assertEqual(tri.bbox, [0, 0, 2, 2])

    def test_area_calculation(self):
        """测试三角形面积计算"""
        tri = geo_info.Triangle((0, 0), (2, 0), (1, 3))
        tri.init_metrics()
        self.assertAlmostEqual(tri.area, 3.0, delta=1e-6)

    def test_non_intersecting_triangles(self):
        """测试不相交的三角形"""
        tri1 = geo_info.Triangle((0, 0), (2, 0), (1, 2))
        tri2 = geo_info.Triangle((3, 3), (5, 3), (4, 5))
        self.assertFalse(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)

    def test_edge_intersection(self):
        """测试边相交的三角形"""
        fig, ax = plt.subplots(figsize=(8, 6))
        tri1 = geo_info.Triangle((0, 0), (2, 0), (1, 2))
        tri2 = geo_info.Triangle((1, 1), (3, 1), (2, 3))
        self.assertTrue(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)

    def test_containment_intersection(self):
        """测试包含关系的三角形"""
        tri1 = geo_info.Triangle((0, 0), (4, 0), (2, 4))
        tri2 = geo_info.Triangle((1, 1), (3, 1), (2, 2))
        self.assertTrue(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)

    def test_shared_edge_no_intersection(self):
        """测试共享边但不相交的情况"""
        # 共享边 (0,0)-(2,0) 的情况
        # tri1 = geo_info.Triangle((0, 0), (2, 0), (1, 2))  # 原三角形
        # tri2 = geo_info.Triangle((0, 0), (2, 0), (1, -1))  # 共享底边，第三个顶点在下方
        tri1 = geo_info.Triangle(
            (-0.62349, 0.781831),
            (0.222521, 0.974928),
            (-0.3971206125805886, 1.7398963243961139),
        )
        tri2 = geo_info.Triangle(
            (0.6836290115743794, 1.9818825590168105),
            (-0.397121, 1.739896),
            (0.222521, 0.974928),
        )
        self.assertFalse(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)

    def test_vertex_touching(self):
        """测试顶点接触但不相交的情况"""
        tri1 = geo_info.Triangle((0, 0), (2, 0), (1, 2))
        tri2 = geo_info.Triangle((1, 2), (3, 2), (2, 4))
        self.assertFalse(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
import unittest
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from math import sqrt

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


def test_minimum_distance_between_segments(self):
    """测试线段间最小距离计算"""
    # 案例1：线段相交于内部点
    A, B = (0, 0), (2, 2)
    C, D = (1, 1), (3, 3)
    self.assertAlmostEqual(
        geo_info.min_distance_between_segments(A, B, C, D), 0.0, delta=1e-6
    )

    # 案例2：平行线段
    A, B = (0, 0), (1, 0)
    C, D = (2, 0), (3, 0)
    self.assertAlmostEqual(
        geo_info.min_distance_between_segments(A, B, C, D), 1.0, delta=1e-6
    )

    # 案例3：线段垂直相交
    A, B = (0, 0), (2, 0)
    C, D = (1, 1), (1, -1)
    self.assertAlmostEqual(
        geo_info.min_distance_between_segments(A, B, C, D), 0.0, delta=1e-6
    )

    # 案例4：浮点数精度测试
    A = [-4.99, -4.85]
    B = [-5.98, -4.85]
    C = [-5.14, -4.00]
    D = [-5.14, -5.00]
    self.assertAlmostEqual(
        geo_info.min_distance_between_segments(A, B, C, D), 0.0, delta=1e-6
    )


class TestPointToSegmentDistance(unittest.TestCase):
    """测试点到线段距离计算函数"""

    def test_point_at_segment_start(self):
        """测试点在线段起点的情况"""
        point = (0, 0)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geo_info.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 0.0, delta=1e-6)

    def test_point_at_segment_end(self):
        """测试点在线段终点的情况"""
        point = (2, 0)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geo_info.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 0.0, delta=1e-6)

    def test_point_projection_inside_segment(self):
        """测试投影点在线段内部的情况"""
        point = (1, 1)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geo_info.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_point_projection_beyond_start(self):
        """测试投影点在线段起点延长线的情况"""
        point = (-1, 0)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geo_info.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_point_projection_beyond_end(self):
        """测试投影点在线段终点延长线的情况"""
        point = (3, 0)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geo_info.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_horizontal_segment(self):
        """测试水平线段的情况"""
        point = (1, 1)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geo_info.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_vertical_segment(self):
        """测试垂直线段的情况"""
        point = (1, 1)
        seg_start = (0, 0)
        seg_end = (0, 2)
        distance = geo_info.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_diagonal_segment_with_external_point(self):
        """测试斜线段外部点的情况"""
        point = (0, 1)
        seg_start = (0, 0)
        seg_end = (1, 1)
        distance = geo_info.point_to_segment_distance(point, seg_start, seg_end)
        expected = sqrt(2) / 2
        print(f"Expected distance: {expected}")
        self.assertAlmostEqual(distance, expected, delta=1e-6)


class TestSegmentIntersection(unittest.TestCase):
    """测试线段相交判断函数"""

    def test_parallel_segments(self):
        """测试平行线段不相交"""
        a1, a2 = (0, 0), (2, 0)
        b1, b2 = (0, 1), (2, 1)
        self.assertFalse(geo_info.segments_intersect(a1, a2, b1, b2))

    def test_collinear_overlap(self):
        """测试共线重叠线段"""
        a1, a2 = (0, 0), (2, 0)
        b1, b2 = (1, 0), (3, 0)
        self.assertTrue(geo_info.segments_intersect(a1, a2, b1, b2))

    def test_intersecting_at_midpoint(self):
        """测试线段中点相交"""
        a1, a2 = (0, 0), (2, 2)
        b1, b2 = (0, 2), (2, 0)
        self.assertTrue(geo_info.segments_intersect(a1, a2, b1, b2))

    def test_touching_endpoints(self):
        """测试端点接触，共用端点，不相交"""
        a1, a2 = (0, 0), (1, 1)
        b1, b2 = (1, 1), (2, 2)
        self.assertFalse(geo_info.segments_intersect(a1, a2, b1, b2))

    def test_epsilon_precision(self):
        """测试浮点精度边界情况，共用端点，不相交"""
        a1, a2 = (0.000001, 0), (0, 0.000001)
        b1, b2 = (0, 0), (0.000001, 0)
        self.assertFalse(geo_info.segments_intersect(a1, a2, b1, b2))


class TestPointEquality(unittest.TestCase):
    """测试点相等判断函数"""

    def test_identical_points(self):
        """测试完全相同点"""
        p1 = (1.234567, 5.678901)
        p2 = (1.234567, 5.678901)
        self.assertTrue(geo_info.points_equal(p1, p2))

    def test_epsilon_range(self):
        """测试epsilon范围内视为相等"""
        p1 = (1.0, 2.0)
        p2 = (1.0 + 1e-7, 2.0 - 1e-7)
        self.assertTrue(geo_info.points_equal(p1, p2))

    def test_exceed_epsilon(self):
        """测试超出epsilon范围"""
        p1 = (1.0, 2.0)
        p2 = (1.0 + 2e-6, 2.0 - 2e-6)
        self.assertFalse(geo_info.points_equal(p1, p2))

    def test_different_coordinates(self):
        """测试不同坐标组合"""
        cases = [
            ((1.0, 2.0), (1.0, 2.000002)),
            ((1.0, 2.0), (1.000002, 2.0)),
            ((1.000002, 2.000001), (1.0, 2.0)),
        ]
        for p1, p2 in cases:
            with self.subTest(p1=p1, p2=p2):
                self.assertFalse(geo_info.points_equal(p1, p2))


if __name__ == "__main__":
    unittest.main()

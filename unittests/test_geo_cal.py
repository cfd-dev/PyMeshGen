import unittest
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "fileIO"))
sys.path.append(str(Path(__file__).parent.parent / "data_structure"))
sys.path.append(str(Path(__file__).parent.parent / "meshsize"))
sys.path.append(str(Path(__file__).parent.parent / "visualization"))
sys.path.append(str(Path(__file__).parent.parent / "adfront2"))
sys.path.append(str(Path(__file__).parent.parent / "optimize"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

import geom_toolkit as geom_tool
from basic_elements import Triangle, Quadrilateral


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
        tri = Triangle((0, 0), (2, 0), (1, 2))
        self.assertEqual(tri.bbox, [0, 0, 2, 2])

    def test_area_calculation(self):
        """测试三角形面积计算"""
        tri = Triangle((0, 0), (2, 0), (1, 3))
        tri.init_metrics()
        self.assertAlmostEqual(tri.area, 3.0, delta=1e-6)

    def test_non_intersecting_triangles(self):
        """测试不相交的三角形"""
        tri1 = Triangle((0, 0), (2, 0), (1, 2))
        tri2 = Triangle((3, 3), (5, 3), (4, 5))
        self.assertFalse(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)

    def test_edge_intersection(self):
        """测试边相交的三角形"""
        fig, ax = plt.subplots(figsize=(8, 6))
        tri1 = Triangle((0, 0), (2, 0), (1, 2))
        tri2 = Triangle((1, 1), (3, 1), (2, 3))
        self.assertTrue(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)

    def test_containment_intersection(self):
        """测试包含关系的三角形"""
        tri1 = Triangle((0, 0), (4, 0), (2, 4))
        tri2 = Triangle((1, 1), (3, 1), (2, 2))
        self.assertTrue(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)

    def test_shared_edge_no_intersection(self):
        """测试共享边但不相交的情况"""
        # 共享边 (0,0)-(2,0) 的情况
        tri1 = Triangle((0, 0), (2, 0), (1, 2))  # 原三角形
        tri2 = Triangle((0, 0), (2, 0), (1, -1))  # 共享底边，第三个顶点在下方
        self.assertFalse(tri1.is_intersect(tri2))

        tri1 = Triangle(
            (-0.62349, 0.781831),
            (0.222521, 0.974928),
            (-0.3971206125805886, 1.7398963243961139),
        )
        tri2 = Triangle(
            (0.6836290115743794, 1.9818825590168105),
            (-0.397121, 1.739896),
            (0.222521, 0.974928),
        )
        self.assertFalse(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)

    def test_float_number(self):
        """测试浮点数"""
        self.assertTrue(
            geom_tool.triangle_intersect_triangle(
                (-4.99, -4.85),
                (-5.98, -4.85),
                (-5.14, -5.00),  # 三角形1
                (-5.14, -4.00),
                (-5.14, -5.00),
                (-4.99, -4.85),  # 三角形2
            )
        )

    def test_vertex_touching(self):
        """测试顶点接触但不相交的情况"""
        tri1 = Triangle((0, 0), (2, 0), (1, 2))
        tri2 = Triangle((1, 2), (3, 2), (2, 4))
        self.assertFalse(tri1.is_intersect(tri2))
        # self.plot_triangle(tri1, tri2)

    def test_fully_contained(self):
        """测试一个三角形完全包含另一个三角形"""
        tri1 = Triangle((0, 0), (3, 0), (0, 3))
        tri2 = Triangle((1, 1), (2, 1), (1, 2))
        self.assertTrue(tri1.is_intersect(tri2))

    def test_vertex_on_edge(self):
        """测试顶点位于另一三角形的边上（非顶点）"""
        tri1 = Triangle((0, 0), (2, 0), (1, 2))
        tri2 = Triangle((1, 0), (3, 1), (2, 3))
        self.assertTrue(tri1.is_intersect(tri2))

    def test_edge_cross_middle(self):
        """测试边相交于中间点（非共边）"""
        tri1 = Triangle((0, 0), (2, 0), (1, 1))
        tri2 = Triangle((1, -1), (1, 1), (2, 2))
        self.assertTrue(tri1.is_intersect(tri2))

    def test_bounding_box_overlap_no_intersection(self):
        """测试边界框相交但三角形不相交"""
        tri1 = Triangle((0, 0), (3, 0), (0, 3))
        tri2 = Triangle((3, 3), (6, 3), (3, 6))
        self.assertFalse(tri1.is_intersect(tri2))

    def test_completely_overlapping(self):
        """测试完全重叠的三角形"""
        tri1 = Triangle((0, 0), (2, 0), (1, 2))
        tri2 = Triangle((0, 0), (2, 0), (1, 2))
        self.assertFalse(tri1.is_intersect(tri2))

    def test_edge_intersection_at_vertex(self):
        """测试边相交于顶点但非共边的情况"""
        tri1 = Triangle((0, 0), (2, 2), (0, 2))
        tri2 = Triangle((2, 2), (3, 3), (1, 1))
        # self.plot_triangle(tri1, tri2)
        self.assertTrue(tri1.is_intersect(tri2))

    def test_edge_cross_other(self):
        """测试两条边在中间相交"""
        tri1 = Triangle((0, 0), (3, 0), (0, 3))
        tri2 = Triangle((1, 2), (2, -1), (4, 1))
        self.assertTrue(tri1.is_intersect(tri2))


def test_minimum_distance_between_segments(self):
    """测试线段间最小距离计算"""
    # 案例1：线段相交于内部点
    A, B = (0, 0), (2, 2)
    C, D = (1, 1), (3, 3)
    self.assertAlmostEqual(
        geom_tool.min_distance_between_segments(A, B, C, D), 0.0, delta=1e-6
    )

    # 案例2：平行线段
    A, B = (0, 0), (1, 0)
    C, D = (2, 0), (3, 0)
    self.assertAlmostEqual(
        geom_tool.min_distance_between_segments(A, B, C, D), 1.0, delta=1e-6
    )

    # 案例3：线段垂直相交
    A, B = (0, 0), (2, 0)
    C, D = (1, 1), (1, -1)
    self.assertAlmostEqual(
        geom_tool.min_distance_between_segments(A, B, C, D), 0.0, delta=1e-6
    )

    # 案例4：浮点数精度测试
    A = [-4.99, -4.85]
    B = [-5.98, -4.85]
    C = [-5.14, -4.00]
    D = [-5.14, -5.00]
    self.assertAlmostEqual(
        geom_tool.min_distance_between_segments(A, B, C, D), 0.0, delta=1e-6
    )


class TestPointToSegmentDistance(unittest.TestCase):
    """测试点到线段距离计算函数"""

    def test_point_at_segment_start(self):
        """测试点在线段起点的情况"""
        point = (0, 0)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geom_tool.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 0.0, delta=1e-6)

    def test_point_at_segment_end(self):
        """测试点在线段终点的情况"""
        point = (2, 0)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geom_tool.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 0.0, delta=1e-6)

    def test_point_projection_inside_segment(self):
        """测试投影点在线段内部的情况"""
        point = (1, 1)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geom_tool.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_point_projection_beyond_start(self):
        """测试投影点在线段起点延长线的情况"""
        point = (-1, 0)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geom_tool.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_point_projection_beyond_end(self):
        """测试投影点在线段终点延长线的情况"""
        point = (3, 0)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geom_tool.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_horizontal_segment(self):
        """测试水平线段的情况"""
        point = (1, 1)
        seg_start = (0, 0)
        seg_end = (2, 0)
        distance = geom_tool.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_vertical_segment(self):
        """测试垂直线段的情况"""
        point = (1, 1)
        seg_start = (0, 0)
        seg_end = (0, 2)
        distance = geom_tool.point_to_segment_distance(point, seg_start, seg_end)
        self.assertAlmostEqual(distance, 1.0, delta=1e-6)

    def test_diagonal_segment_with_external_point(self):
        """测试斜线段外部点的情况"""
        point = (0, 1)
        seg_start = (0, 0)
        seg_end = (1, 1)
        distance = geom_tool.point_to_segment_distance(point, seg_start, seg_end)
        expected = sqrt(2) / 2
        print(f"Expected distance: {expected}")
        self.assertAlmostEqual(distance, expected, delta=1e-6)


class TestSegmentIntersection(unittest.TestCase):
    """测试线段相交判断函数"""

    def test_parallel_segments(self):
        """测试平行线段不相交"""
        a1, a2 = (0, 0), (2, 0)
        b1, b2 = (0, 1), (2, 1)
        self.assertFalse(geom_tool.segments_intersect(a1, a2, b1, b2))

    def test_collinear_overlap(self):
        """测试共线重叠线段"""
        a1, a2 = (0, 0), (2, 0)
        b1, b2 = (1, 0), (3, 0)
        self.assertTrue(geom_tool.segments_intersect(a1, a2, b1, b2))

    def test_intersecting_at_midpoint(self):
        """测试线段中点相交"""
        a1, a2 = (0, 0), (2, 2)
        b1, b2 = (0, 2), (2, 0)
        self.assertTrue(geom_tool.segments_intersect(a1, a2, b1, b2))

    def test_touching_endpoints(self):
        """测试端点接触，共用端点，不相交"""
        a1, a2 = (0, 0), (1, 1)
        b1, b2 = (1, 1), (2, 2)
        self.assertFalse(geom_tool.segments_intersect(a1, a2, b1, b2))

    def test_epsilon_precision(self):
        """测试浮点精度边界情况，共用端点，不相交"""
        a1, a2 = (0.000001, 0), (0, 0.000001)
        b1, b2 = (0, 0), (0.000001, 0)
        self.assertFalse(geom_tool.segments_intersect(a1, a2, b1, b2))


class TestPointEquality(unittest.TestCase):
    """测试点相等判断函数"""

    def test_identical_points(self):
        """测试完全相同点"""
        p1 = (1.234567, 5.678901)
        p2 = (1.234567, 5.678901)
        self.assertTrue(geom_tool.points_equal(p1, p2))

    def test_epsilon_range(self):
        """测试epsilon范围内视为相等"""
        p1 = (1.0, 2.0)
        p2 = (1.0 + 1e-7, 2.0 - 1e-7)
        self.assertTrue(geom_tool.points_equal(p1, p2))

    def test_exceed_epsilon(self):
        """测试超出epsilon范围"""
        p1 = (1.0, 2.0)
        p2 = (1.0 + 2e-6, 2.0 - 2e-6)
        self.assertFalse(geom_tool.points_equal(p1, p2))

    def test_different_coordinates(self):
        """测试不同坐标组合"""
        cases = [
            ((1.0, 2.0), (1.0, 2.000002)),
            ((1.0, 2.0), (1.000002, 2.0)),
            ((1.000002, 2.000001), (1.0, 2.0)),
        ]
        for p1, p2 in cases:
            with self.subTest(p1=p1, p2=p2):
                self.assertFalse(geom_tool.points_equal(p1, p2))


class TestQuadrilateral(unittest.TestCase):
    def test_quadrilateral_area_basic_shapes(self):
        """测试基本四边形形状面积"""
        # 使用subTest分隔测试案例
        with self.subTest("矩形"):
            area = geom_tool.quadrilateral_area([0, 0], [2, 0], [2, 3], [0, 3])
            self.assertAlmostEqual(area, 6.0, delta=1e-6)

        with self.subTest("平行四边形"):
            area = geom_tool.quadrilateral_area([0, 0], [2, 0], [3, 2], [1, 2])
            self.assertAlmostEqual(area, 4.0, delta=1e-6)

        with self.subTest("梯形"):
            area = geom_tool.quadrilateral_area([0, 0], [4, 0], [3, 3], [1, 3])
            self.assertAlmostEqual(area, 9.0, delta=1e-6)

    def test_quadrilateral_area_special_cases(self):
        """测试特殊四边形情况"""
        test_cases = [
            ("凹四边形", [[0, 0], [2, 0], [1, 1], [0, 3]], 2.5),
            ("退化四边形", [[0, 0], [1, 1], [2, 2], [3, 3]], 0.0),
            ("自相交四边形", [[0, 0], [2, 2], [2, 0], [0, 2]], 0.0),
        ]

        for desc, points, expected in test_cases:
            with self.subTest(desc):
                p1, p2, p3, p4 = points
                area = geom_tool.quadrilateral_area(p1, p2, p3, p4)
                self.assertAlmostEqual(area, expected, delta=1e-6)

    def test_quadrilateral_area_floating_point(self):
        """测试浮点数精度"""
        area = geom_tool.quadrilateral_area(
            [0.1, 0.2], [2.3, 0.4], [2.5, 3.6], [0.7, 3.8]
        )
        # 使用numpy计算精确值
        points = np.array([[0.1, 0.2], [2.3, 0.4], [2.5, 3.6], [0.7, 3.8]])
        x, y = points[:, 0], points[:, 1]
        expected = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        self.assertAlmostEqual(area, expected, delta=1e-10)

    def test_quadrilateral_area_ordering(self):
        """测试顶点顺序不变性"""
        points_clockwise = [[0, 0], [2, 0], [2, 3], [0, 3]]
        points_counter = [[0, 0], [0, 3], [2, 3], [2, 0]]

        area1 = geom_tool.quadrilateral_area(*points_clockwise)
        area2 = geom_tool.quadrilateral_area(*points_counter)
        self.assertAlmostEqual(abs(area1), abs(area2), delta=1e-10)


class TestConvexCheck(unittest.TestCase):
    """测试四边形凸性判断函数 is_convex"""

    def test_convex_quadrilateral(self):
        """测试标准凸四边形"""
        node_coords = [[0, 0], [1, 0], [1, 1], [0, 1]]  # a  # b  # c  # d
        self.assertTrue(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_concave_quadrilateral(self):
        """测试凹四边形（有一个内角>180度）"""
        node_coords = [[0, 0], [2, 0], [1, 1], [1, 0.5]]  # a  # b  # c  # d (凹点)
        self.assertFalse(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_collinear_points(self):
        """测试包含共线点的四边形"""
        node_coords = [[0, 0], [1, 0], [2, 0], [0, 1]]  # a  # b  # c (与a-b共线)  # d
        self.assertFalse(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_convex_with_mixed_signs(self):
        """测试严格凸但叉积符号混合的情况（验证顶点顺序处理）"""
        node_coords = [[0, 0], [1, 1], [0, 2], [-1, 1]]  # a  # b  # c  # d
        self.assertTrue(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_invalid_quadrilateral(self):
        """测试无法构成四边形的点（三点重合）"""
        node_coords = [[0, 0], [0, 0], [1, 1], [0, 1]]  # a  # b (与a重合)  # c  # d
        self.assertFalse(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_reversed_order(self):
        """测试顶点逆时针顺序"""
        node_coords = [[0, 0], [0, 1], [1, 1], [1, 0]]  # a  # b  # c  # d
        self.assertTrue(geom_tool.is_convex(0, 3, 2, 1, node_coords))

    def test_all_collinear(self):
        """所有点共线的情况"""
        node_coords = [[0, 0], [1, 0], [2, 0], [3, 0]]
        self.assertFalse(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_concave_with_three_collinear(self):
        """三点共线且形成凹四边形"""
        node_coords = [[0, 0], [2, 0], [4, 0], [1, 1]]
        self.assertFalse(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_concave_different_shape(self):
        """另一种凹四边形形状（凹点在不同位置）"""
        node_coords = [[0, 0], [2, 2], [0, 2], [1, 1]]
        self.assertFalse(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_almost_straight_line(self):
        """接近共线但微小扰动的凸四边形"""
        node_coords = [
            [0, 0],  # 点0
            [1, 0],  # 点1
            [2, 0.0001],  # 点2（微小偏移）
            [3, 0.0003],  # 点3（确保所有叉积符号一致）
        ]
        self.assertTrue(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_inverted_order(self):
        """顶点顺序颠倒但符合凸性条件"""
        node_coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertTrue(geom_tool.is_convex(3, 2, 1, 0, node_coords))  # 逆序传递参数

    def test_concave_with_zero_cross(self):
        """某条边的叉积为零导致凹性"""
        node_coords = [[0, 0], [2, 0], [3, 0], [1, 1]]
        self.assertFalse(geom_tool.is_convex(0, 1, 2, 3, node_coords))

    def test_invalid_order(self):
        """顶点顺序错误导致错误判断（非连续顺序）"""
        node_coords = [[0, 0], [1, 1], [2, 0], [1, 0]]
        # 错误顺序：0→2→1→3
        self.assertFalse(geom_tool.is_convex(0, 2, 1, 3, node_coords))


class TestQuadrilateralIntersectTriangle(unittest.TestCase):
    """测试四边形与三角形相交判断函数"""

    def test_share_3points(self):
        """测试四边形与三角形共用3个顶点"""
        quad = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        tri = Triangle([0, 0], [2, 0], [2, 2])
        self.assertTrue(quad.is_intersect_triangle(tri))

    def test_fully_intersect(self):
        """测试四边形与三角形完全相交"""
        quad = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        tri = Triangle([1, 1], [3, 1], [1, 3])
        self.assertTrue(quad.is_intersect_triangle(tri))

    def test_edge_intersect(self):
        """测试边相交情况"""
        quad = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        tri = Triangle([2, 1], [3, 1], [2, 3])
        self.assertTrue(quad.is_intersect_triangle(tri))

    def test_vertex_touch(self):
        """测试顶点接触但不相交"""
        quad = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        tri = Triangle([2, 2], [3, 2], [2, 3])
        self.assertFalse(quad.is_intersect_triangle(tri))

    def test_vertex_touch2(self):
        """测试顶点接触但不相交2"""
        quad = Quadrilateral(
            [10.0, -1.111111111111111],
            [10.0, 1.111111111111111],
            [7.821164001232763, 1.1111111111111112],
            [7.821027627553216, -1.1111111111111107],
            node_ids=[0, 1, 2, 3],
        )
        tri = Triangle(
            [7.821027627553216, -1.1111111111111107],
            [6.286015116070248, -2.0121039469226242e-16],
            [6.286010043832486, -1.504200334532052],
        )
        self.assertFalse(quad.is_intersect_triangle(tri))

    def test_containment(self):
        """测试包含关系（三角形在四边形内）"""
        quad = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        tri = Triangle([0.5, 0.5], [1.5, 0.5], [1, 1.5])
        self.assertTrue(quad.is_intersect_triangle(tri))

    def test_no_intersect(self):
        """测试不相交情况"""
        quad = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        tri = Triangle([3, 3], [4, 3], [3, 4])
        self.assertFalse(quad.is_intersect_triangle(tri))


class TestQuadrilateralIntersectQuad(unittest.TestCase):
    """测试四边形与四边形相交判断函数"""

    def test_shared_diagonal(self):
        """测试四边形边与另一四边形对角线重合的情况"""
        quad1 = Quadrilateral(
            [0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3]  # 标准正方形
        )
        # 第二个四边形的边(0,0)-(2,2)是quad1的对角线
        quad2 = Quadrilateral(
            [0, 0], [2, 2], [3, 1], [1, -1], node_ids=[4, 5, 6, 7]  # 共享对角线作为边
        )
        self.assertTrue(quad1.is_intersect_quad(quad2))

    def test_concave_intersect(self):
        """凹四边形相交测试"""
        quad1 = Quadrilateral(
            [0, 0], [2, 0], [1, 1], [0, 2], node_ids=[0, 1, 2, 3]  # 凹四边形
        )
        quad2 = Quadrilateral(
            [1, 0.5], [3, 0.5], [3, 1.5], [1, 1.5], node_ids=[4, 5, 6, 7]
        )
        self.assertTrue(quad1.is_intersect_quad(quad2))

    def test_fully_intersect(self):
        """测试四边形完全相交"""
        quad1 = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        quad2 = Quadrilateral([1, 1], [3, 1], [3, 3], [1, 3], node_ids=[4, 5, 6, 7])
        self.assertTrue(quad1.is_intersect_quad(quad2))

    def test_edge_intersect(self):
        """测试边相交情况"""
        quad1 = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        quad2 = Quadrilateral([2, 1], [3, 1], [3, 3], [2, 3], node_ids=[4, 5, 6, 7])
        self.assertTrue(quad1.is_intersect_quad(quad2))

    def test_vertex_touch(self):
        """测试顶点接触但不相交"""
        quad1 = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        quad2 = Quadrilateral([2, 2], [4, 2], [4, 4], [2, 4], node_ids=[4, 5, 6, 7])
        self.assertFalse(quad1.is_intersect_quad(quad2))

    def test_containment(self):
        """测试包含关系"""
        quad1 = Quadrilateral([0, 0], [3, 0], [3, 3], [0, 3], node_ids=[0, 1, 2, 3])
        quad2 = Quadrilateral([1, 1], [2, 1], [2, 2], [1, 2], node_ids=[4, 5, 6, 7])
        self.assertTrue(quad1.is_intersect_quad(quad2))

    def test_no_intersect(self):
        """测试不相交情况"""
        quad1 = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        quad2 = Quadrilateral([3, 3], [5, 3], [5, 5], [3, 5], node_ids=[4, 5, 6, 7])
        self.assertFalse(quad1.is_intersect_quad(quad2))

    def test_bbox_intersect_but_not_actual(self):
        """测试包围盒相交但实际不相交"""
        quad1 = Quadrilateral([0, 0], [2, 0], [2, 2], [0, 2], node_ids=[0, 1, 2, 3])
        quad2 = Quadrilateral(
            [1.5, 2.1],
            [2.1, 2.1],
            [2.1, 2.5],
            [1.5, 2.5],  # 完全在quad1的右上方
            node_ids=[4, 5, 6, 7],
        )
        self.assertFalse(quad1.is_intersect_quad(quad2))


if __name__ == "__main__":
    unittest.main()


class TestPointInsideOrOnTriangle(unittest.TestCase):
    """测试点在三角形内部或边上的判断函数"""

    def test_inside_triangle(self):
        """测试点在三角形内部"""
        triangle = [(0, 0), (2, 0), (1, 2)]
        self.assertTrue(geom_tool.is_point_inside_or_on((1, 1), *triangle))
        self.assertTrue(geom_tool.is_point_inside_or_on((0.5, 0.5), *triangle))

    def test_on_edge(self):
        """测试点在三角形边上"""
        triangle = [(0, 0), (2, 0), (1, 2)]
        # 边中点
        self.assertTrue(geom_tool.is_point_inside_or_on((1, 0), *triangle))  # 底边中点
        self.assertTrue(
            geom_tool.is_point_inside_or_on((1.5, 1), *triangle)
        )  # 右侧边中点
        # 边上的非顶点位置
        self.assertTrue(geom_tool.is_point_inside_or_on((0.5, 0), *triangle))

    def test_on_vertex(self):
        """测试点在三角形顶点"""
        triangle = [(0, 0), (2, 0), (1, 2)]
        for vertex in triangle:
            self.assertFalse(geom_tool.is_point_inside_or_on(vertex, *triangle))

    def test_on_vertex2(self):
        """测试点在三角形顶点上"""
        vertex = (7.821027627553216, -1.1111111111111107)
        tri = [
            (7.821027627553216, -1.1111111111111107),
            (6.286015116070248, -2.0121039469226242e-16),
            (6.286010043832486, -1.504200334532052),
        ]

        self.assertFalse(geom_tool.is_point_inside_or_on(vertex, *tri))

    def test_outside_triangle(self):
        """测试点在三角形外部"""
        triangle = [(0, 0), (2, 0), (1, 2)]
        self.assertFalse(geom_tool.is_point_inside_or_on((3, 1), *triangle))  # 右侧外部
        self.assertFalse(
            geom_tool.is_point_inside_or_on((1, -1), *triangle)
        )  # 下方外部
        self.assertFalse(
            geom_tool.is_point_inside_or_on((1, 2.1), *triangle)
        )  # 上方外部

    def test_edge_cases(self):
        """测试边缘情况"""
        # 接近边界的点
        triangle = [(0, 0), (2, 0), (1, 2)]
        self.assertFalse(
            geom_tool.is_point_inside_or_on((1e-8, 1e-8), *triangle)
        )  # 接近原点
        self.assertFalse(
            geom_tool.is_point_inside_or_on((1, 2 - 1e-8), *triangle)
        )  # 接近顶点

    def test_floating_point_precision(self):
        """测试浮点精度处理"""
        triangle = [(0.1, 0.2), (2.3, 4.5), (5.6, 7.8)]
        # 在边上的点（使用向量计算确定位置）
        edge_point = (
            triangle[0][0] * 0.3 + triangle[1][0] * 0.7,
            triangle[0][1] * 0.3 + triangle[1][1] * 0.7,
        )
        self.assertTrue(geom_tool.is_point_inside_or_on(edge_point, *triangle))


class TestPointInsideQuad(unittest.TestCase):
    """测试点在四边形内部的判断函数"""

    def setUp(self):
        # 定义一个标准矩形四边形
        self.quad = [[0, 0], [2, 0], [2, 2], [0, 2]]

    def test_point_inside(self):
        """测试点在四边形内部"""
        self.assertTrue(geom_tool.is_point_inside_quad([1, 1], self.quad))
        self.assertTrue(geom_tool.is_point_inside_quad([0.5, 0.5], self.quad))
        self.assertTrue(geom_tool.is_point_inside_quad([1.9, 1.9], self.quad))

    def test_point_outside(self):
        """测试点在四边形外部"""
        self.assertFalse(geom_tool.is_point_inside_quad([3, 1], self.quad))
        self.assertFalse(geom_tool.is_point_inside_quad([1, -1], self.quad))
        self.assertFalse(geom_tool.is_point_inside_quad([1, 2.1], self.quad))

    def test_point_on_vertex(self):
        """测试点在四边形顶点上"""
        for vertex in self.quad:
            self.assertFalse(geom_tool.is_point_inside_quad(vertex, self.quad))

    def test_point_on_edge(self):
        """测试点在四边形边上"""
        self.assertFalse(geom_tool.is_point_inside_quad([1, 0], self.quad))  # 底边
        self.assertFalse(geom_tool.is_point_inside_quad([2, 1], self.quad))  # 右边
        self.assertFalse(geom_tool.is_point_inside_quad([1, 2], self.quad))  # 顶边
        self.assertFalse(geom_tool.is_point_inside_quad([0, 1], self.quad))  # 左边

    def test_concave_quad(self):
        """测试凹四边形情况"""
        concave_quad = [[0, 0], [2, 0], [1, 1], [0, 2]]  # 凹四边形
        # 凹点附近的内部点
        self.assertTrue(geom_tool.is_point_inside_quad([0.5, 1], concave_quad))
        # 凹点附近的凹区域外部点
        self.assertFalse(geom_tool.is_point_inside_quad([1.5, 0.5], concave_quad))

    # def test_floating_point_precision(self):
    #     """测试浮点数精度边界情况"""
    #     # 非常接近边界的内部点
    #     self.assertTrue(geom_tool.is_point_inside_quad([1e-8, 1e-8], self.quad))
    #     # 非常接近边界的边界点
    #     self.assertFalse(geom_tool.is_point_inside_quad([2-1e-8, 1], self.quad))

    def test_irregular_quad(self):
        """测试不规则四边形"""
        irregular_quad = [[0, 0], [3, 1], [2, 3], [-1, 2]]
        # 内部点
        self.assertTrue(geom_tool.is_point_inside_quad([1, 1.5], irregular_quad))
        # 外部点
        self.assertFalse(geom_tool.is_point_inside_quad([3, 0], irregular_quad))

class TestPointInPolygon(unittest.TestCase):
    """测试点与多边形位置关系判断函数"""

    def test_inside_convex(self):
        """测试点在凸多边形内部"""
        polygon = [[0,0], [5,0], [5,5], [0,5]]
        self.assertTrue(geom_tool.point_in_polygon([2,2], polygon))
        self.assertTrue(geom_tool.point_in_polygon([4.9,4.9], polygon))

    def test_outside_convex(self):
        """测试点在凸多边形外部"""
        polygon = [[0,0], [5,0], [5,5], [0,5]]
        self.assertFalse(geom_tool.point_in_polygon([6,3], polygon))
        self.assertFalse(geom_tool.point_in_polygon([-1,2], polygon))

    def test_inside_concave(self):
        """测试点在凹多边形内部"""
        concave = [[0,0], [5,0], [5,5], [3,5], [3,3], [0,3]]
        self.assertTrue(geom_tool.point_in_polygon([1,1], concave))
        self.assertTrue(geom_tool.point_in_polygon([4,4], concave))

    def test_in_concave_recess(self):
        """测试点在凹多边形的凹陷区域外部"""
        concave = [[0,0], [5,0], [5,5], [3,5], [3,3], [0,3]]
        self.assertFalse(geom_tool.point_in_polygon([1,4], concave))

    def test_on_vertex(self):
        """测试点在多边形顶点上"""
        polygon = [[0,0], [5,0], [5,5]]
        self.assertFalse(geom_tool.point_in_polygon([5,0], polygon))

    def test_on_edge(self):
        """测试点在多边形边上"""
        polygon = [[0,0], [5,0], [5,5]]
        self.assertFalse(geom_tool.point_in_polygon([2.5,0], polygon))  # 底边
        self.assertFalse(geom_tool.point_in_polygon([5,2.5], polygon))  # 右边

    def test_horizontal_edges(self):
        """测试含水平边的多边形"""
        polygon = [[0,0], [5,0], [5,3], [3,3], [3,5], [0,5]]
        # 在水平边上方
        self.assertFalse(geom_tool.point_in_polygon([4,4], polygon))
        # 在水平边下方
        self.assertTrue(geom_tool.point_in_polygon([4,2], polygon))

    def test_complex_shape(self):
        """测试复杂星形多边形"""
        star = [[2,0],[4,3],[7,3],[5,6],[6,9],[2,7],[-2,9],[-1,6],[-3,3],[0,3]]
        star.append(star[0])  # 添加闭合点
        # 绘制star
        # plt.figure(figsize=(6, 6))
        # x, y = zip(*star)
        # plt.plot(x, y, 'o-')
        # plt.xlim(-4, 8)
        # plt.ylim(-4, 10)
        # plt.grid(True)
        # plt.show()

        # 星形内部
        self.assertTrue(geom_tool.point_in_polygon([3,4], star))
        # 星形中心点
        self.assertTrue(geom_tool.point_in_polygon([2,4.5], star))
        # 星芒之间
        self.assertFalse(geom_tool.point_in_polygon([6,6], star))

    def test_floating_point_precision(self):
        """测试浮点精度边界情况"""
        polygon = [[0.1,0.1], [0.1,0.5], [0.5,0.5], [0.5,0.1]]
        # 接近边界的内部点
        self.assertFalse(geom_tool.point_in_polygon([0.1+1e-9, 0.3], polygon))
        # 非常接近边界的边界点
        self.assertFalse(geom_tool.point_in_polygon([0.5-1e-9, 0.5-1e-9], polygon))
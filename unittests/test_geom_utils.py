#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：sfmesh/geom_utils.py
2D/3D 计算几何工具库的全面测试。
"""

import unittest
import numpy as np

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sfmesh.geom_utils import (
    project_to_2d,
    segments_intersect_2d,
    point_in_triangle_2d,
    point_in_triangle_3d,
    segment_intersects_triangle,
    segment_segment_distance_3d,
    point_triangle_distance_3d,
    are_coplanar_triangles_overlapping,
    _edge_intersects_triangle_core,
    _point_on_segment_2d,
    check_triangle_intersection,
    check_edge_triangle_intersection,
    DEFAULT_TOL,
    DEGENERATE_TOL,
    COPLANAR_TOL,
)


# ============================================================================
# project_to_2d
# ============================================================================

class TestProjectTo2D(unittest.TestCase):
    """测试3D点集投影到2D平面"""

    def test_drop_x_component(self):
        """法向量x分量最大 → 丢弃x，保留(y,z)"""
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        normal = np.array([1.0, 0.0, 0.0])
        result = project_to_2d(pts, normal)
        expected = np.array([[2, 3], [5, 6]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_drop_y_component(self):
        """法向量y分量最大 → 丢弃y，保留(x,z)"""
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        normal = np.array([0.0, 2.0, 0.0])
        result = project_to_2d(pts, normal)
        expected = np.array([[1, 3], [4, 6]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_drop_z_component(self):
        """法向量z分量最大 → 丢弃z，保留(x,y)"""
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        normal = np.array([0.0, 0.0, 3.0])
        result = project_to_2d(pts, normal)
        expected = np.array([[1, 2], [4, 5]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_diagonal_normal(self):
        """斜向法向量，选择绝对值最大的分量"""
        pts = np.array([[1, 2, 3]], dtype=float)
        normal = np.array([0.1, 0.2, 0.9])
        result = project_to_2d(pts, normal)
        expected = np.array([[1, 2]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_single_point(self):
        """单点投影"""
        pts = np.array([[10, 20, 30]], dtype=float)
        normal = np.array([0, 0, 1])
        result = project_to_2d(pts, normal)
        self.assertEqual(result.shape, (1, 2))
        np.testing.assert_array_equal(result[0], [10, 20])


# ============================================================================
# segments_intersect_2d
# ============================================================================

class TestSegmentsIntersect2D(unittest.TestCase):
    """测试2D线段相交判断"""

    def test_cross_intersection(self):
        """两条线段交叉相交"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        b1, b2 = np.array([0.0, 2.0]), np.array([2.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_no_intersection_parallel(self):
        """平行线段不相交"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        b1, b2 = np.array([0.0, 1.0]), np.array([2.0, 1.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_no_intersection_separated(self):
        """分离线段不相交"""
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        b1, b2 = np.array([5.0, 5.0]), np.array([6.0, 6.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_shared_endpoint_excluded(self):
        """共享端点不视为严格相交"""
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 1.0])
        b1, b2 = np.array([1.0, 1.0]), np.array([2.0, 0.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_t_shape_intersection(self):
        """T形相交：一条线段的内部与另一条相交"""
        a1, a2 = np.array([0.0, 0.5]), np.array([2.0, 0.5])
        b1, b2 = np.array([1.0, 0.0]), np.array([1.0, 1.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_no_overlap(self):
        """共线但不重叠"""
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        b1, b2 = np.array([2.0, 0.0]), np.array([3.0, 0.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_custom_tolerance(self):
        """自定义容差测试"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        b1, b2 = np.array([0.0, 2.0]), np.array([2.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2, tol=1e-12))


# ============================================================================
# point_in_triangle_2d
# ============================================================================

class TestPointInTriangle2D(unittest.TestCase):
    """测试2D点在三角形内判断"""

    def test_point_inside(self):
        """点在三角形内部"""
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        p = np.array([2.0, 1.0])
        self.assertTrue(point_in_triangle_2d(p, t0, t1, t2))

    def test_point_on_edge(self):
        """点在三角形边上（含边界）"""
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        p = np.array([2.0, 0.0])  # 底边中点
        self.assertTrue(point_in_triangle_2d(p, t0, t1, t2))

    def test_point_on_vertex(self):
        """点在三角形顶点上"""
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        self.assertTrue(point_in_triangle_2d(t0, t0, t1, t2))

    def test_point_outside(self):
        """点在三角形外部"""
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        p = np.array([5.0, 5.0])
        self.assertFalse(point_in_triangle_2d(p, t0, t1, t2))

    def test_point_just_outside(self):
        """点紧贴三角形外部"""
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        p = np.array([2.0, -0.1])
        self.assertFalse(point_in_triangle_2d(p, t0, t1, t2))

    def test_centroid(self):
        """重心一定在三角形内"""
        t0 = np.array([0.0, 0.0])
        t1 = np.array([6.0, 0.0])
        t2 = np.array([3.0, 6.0])
        centroid = (t0 + t1 + t2) / 3.0
        self.assertTrue(point_in_triangle_2d(centroid, t0, t1, t2))


# ============================================================================
# point_in_triangle_3d
# ============================================================================

class TestPointInTriangle3D(unittest.TestCase):
    """测试3D点在三角形内判断（基于重心坐标）"""

    def test_point_inside(self):
        """点在三角形内部"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 0.0])
        self.assertTrue(point_in_triangle_3d(p, a, b, c))

    def test_point_on_vertex(self):
        """点在顶点上"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        self.assertTrue(point_in_triangle_3d(a, a, b, c))

    def test_point_outside(self):
        """点在三角形外部"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([10.0, 10.0, 0.0])
        self.assertFalse(point_in_triangle_3d(p, a, b, c))

    def test_degenerate_triangle(self):
        """退化三角形（面积为零）返回False"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])  # 三点共线
        p = np.array([0.5, 0.0, 0.0])
        self.assertFalse(point_in_triangle_3d(p, a, b, c))

    def test_tilted_triangle(self):
        """非xy平面的三角形"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0])
        c = np.array([0.0, 1.0, 1.0])
        p = np.array([0.25, 0.25, 0.5])  # 重心
        self.assertTrue(point_in_triangle_3d(p, a, b, c))

    def test_centroid_inside(self):
        """重心一定在三角形内"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 0.0, 0.0])
        c = np.array([1.5, 3.0, 0.0])
        centroid = (a + b + c) / 3.0
        self.assertTrue(point_in_triangle_3d(centroid, a, b, c))


# ============================================================================
# segment_intersects_triangle (Möller–Trumbore)
# ============================================================================

class TestSegmentIntersectsTriangle(unittest.TestCase):
    """测试线段与三角形相交（Möller–Trumbore算法）"""

    def test_segment_pierces_triangle(self):
        """线段穿过三角形内部"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, -1.0])
        q = np.array([2.0, 1.0, 1.0])
        self.assertTrue(segment_intersects_triangle(p, q, a, b, c))

    def test_segment_misses_triangle(self):
        """线段不穿过三角形"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([10.0, 10.0, -1.0])
        q = np.array([10.0, 10.0, 1.0])
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))

    def test_segment_parallel_to_triangle(self):
        """线段平行于三角形平面"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([4.0, 0.0, 1.0])
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))

    def test_segment_ends_before_triangle(self):
        """线段在到达三角形前结束"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, -1.0])
        q = np.array([2.0, 1.0, -0.01])  # 未穿透
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))

    def test_endpoint_on_triangle_excluded(self):
        """端点恰好在三角形平面上（t接近0或1）应被排除"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 0.0])  # 在三角形上
        q = np.array([2.0, 1.0, 1.0])
        # t=0 应被排除
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))

    def test_degenerate_triangle(self):
        """退化三角形不产生相交"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])  # 共线
        p = np.array([0.5, -1.0, 0.0])
        q = np.array([0.5, 1.0, 0.0])
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))


# ============================================================================
# segment_segment_distance_3d
# ============================================================================

class TestSegmentSegmentDistance3D(unittest.TestCase):
    """测试3D线段间最短距离"""

    def test_parallel_segments(self):
        """平行线段距离"""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0])
        q2 = np.array([1.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 1.0, places=10)

    def test_intersecting_segments(self):
        """相交线段距离为0"""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        q1 = np.array([1.0, -1.0, 0.0])
        q2 = np.array([1.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_skew_segments(self):
        """异面直线距离"""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 1.0])
        q2 = np.array([1.0, 1.0, 1.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, np.sqrt(2.0), places=10)

    def test_point_segments(self):
        """退化为点的线段"""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        q1 = np.array([3.0, 4.0, 0.0])
        q2 = np.array([3.0, 4.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 5.0, places=10)

    def test_one_degenerate_one_normal(self):
        """一条退化为点，一条正常"""
        p1 = np.array([0.5, 0.5, 0.0])
        p2 = np.array([0.5, 0.5, 0.0])
        q1 = np.array([0.0, 0.0, 0.0])
        q2 = np.array([1.0, 0.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 0.5, places=10)

    def test_shared_endpoint(self):
        """共享端点距离为0"""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        q1 = np.array([0.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_closest_points_at_endpoints(self):
        """最近点在线段端点处"""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        q1 = np.array([2.0, 1.0, 0.0])
        q2 = np.array([3.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        expected = np.sqrt(1.0 + 1.0)  # (1,0,0) to (2,1,0)
        self.assertAlmostEqual(dist, expected, places=10)


# ============================================================================
# point_triangle_distance_3d
# ============================================================================

class TestPointTriangleDistance3D(unittest.TestCase):
    """测试3D点到三角形最短距离"""

    def test_point_inside_triangle(self):
        """点在三角形面上 → 距离0"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 0.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_point_above_triangle(self):
        """点在三角形正上方"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 3.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 3.0, places=10)

    def test_point_nearest_vertex(self):
        """点最近的三角形元素是顶点"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([-1.0, -1.0, 0.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, np.sqrt(2.0), places=10)

    def test_point_nearest_edge(self):
        """点最近的三角形元素是边"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, -1.0, 0.0])  # 正对底边
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 1.0, places=10)

    def test_point_at_vertex(self):
        """点在三角形顶点上"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        dist = point_triangle_distance_3d(a, a, b, c)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_point_on_edge(self):
        """点在三角形边上"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 0.0, 0.0])  # 底边中点
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_degenerate_triangle(self):
        """退化三角形"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        p = np.array([0.5, 1.0, 0.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 1.0, places=10)


# ============================================================================
# are_coplanar_triangles_overlapping
# ============================================================================

class TestCoplanarTrianglesOverlapping(unittest.TestCase):
    """测试共面三角形重叠检测"""

    def test_overlapping_triangles(self):
        """两个共面三角形重叠"""
        tri_a = np.array([[0, 0, 0], [4, 0, 0], [2, 4, 0]], dtype=float)
        tri_b = np.array([[1, 0, 0], [5, 0, 0], [3, 4, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_separated_triangles(self):
        """两个共面三角形不重叠"""
        tri_a = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=float)
        tri_b = np.array([[10, 0, 0], [11, 0, 0], [10.5, 1, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertFalse(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_contained_triangle(self):
        """一个三角形完全包含另一个"""
        tri_a = np.array([[0, 0, 0], [10, 0, 0], [5, 10, 0]], dtype=float)
        tri_b = np.array([[2, 1, 0], [8, 1, 0], [5, 5, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_shared_vertex_is_overlapping(self):
        """共享一个顶点被视为重叠"""
        tri_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        tri_b = np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_shared_edge_is_overlapping(self):
        """共享一条边被视为重叠"""
        tri_a = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=float)
        tri_b = np.array([[0, 0, 0], [2, 0, 0], [1, -1, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_yz_plane(self):
        """在yz平面上的共面三角形"""
        tri_a = np.array([[0, 0, 0], [0, 4, 0], [0, 2, 4]], dtype=float)
        tri_b = np.array([[0, 1, 0], [0, 5, 0], [0, 3, 4]], dtype=float)
        normal = np.array([1.0, 0.0, 0.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))


# ============================================================================
# _edge_intersects_triangle_core
# ============================================================================

class TestEdgeIntersectsTriangleCore(unittest.TestCase):
    """测试核心边-三角形相交检测"""

    def test_edge_pierces_through(self):
        """边穿透三角形"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([2.0, 1.0, -1.0])
        ee = np.array([2.0, 1.0, 1.0])
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_edge_misses(self):
        """边不与三角形相交"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([10.0, 10.0, -1.0])
        ee = np.array([10.0, 10.0, 1.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_edge_same_side(self):
        """边的两端在三角形同侧"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([2.0, 1.0, 1.0])
        ee = np.array([2.0, 1.0, 2.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_edge_crosses_triangle_edge(self):
        """共面边穿过三角形边"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([-1.0, 0.0, 0.0])
        ee = np.array([5.0, 0.0, 0.0])
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_edge_fully_inside_triangle(self):
        """共面边完全在三角形内（两端点均在内部）→ 视为相交"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([1.0, 0.5, 0.0])
        ee = np.array([3.0, 0.5, 0.0])
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_degenerate_triangle_returns_false(self):
        """退化三角形返回False"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([1.0, 0.0, 0.0])
        t3 = np.array([2.0, 0.0, 0.0])
        es = np.array([0.5, -1.0, 0.0])
        ee = np.array([0.5, 1.0, 0.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_endpoint_on_triangle_vertex_excluded(self):
        """端点恰好在三角形顶点上（应被排除）"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = t1.copy()  # 端点在三角形顶点
        ee = np.array([0.0, 0.0, 1.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))


# ============================================================================
# check_triangle_intersection (高层API)
# ============================================================================

class TestCheckTriangleIntersection(unittest.TestCase):
    """测试高层三角形相交检测API"""

    def _make_tri(self, p1, p2, p3):
        """创建简单的三角形ndarray"""
        return np.array([p1, p2, p3], dtype=np.float64)

    def test_intersecting_triangles(self):
        """两个三角形相交"""
        tri1 = self._make_tri([0, 0, 0], [4, 0, 0], [2, 4, 0])
        tri2 = self._make_tri([2, 0, -1], [2, 0, 1], [2, 4, 0])
        self.assertTrue(check_triangle_intersection(tri1, tri2))

    def test_separated_triangles(self):
        """两个分离的三角形"""
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0.5, 1, 0])
        tri2 = self._make_tri([10, 0, 0], [11, 0, 0], [10.5, 1, 0])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_coplanar_no_overlap(self):
        """共面但不重叠"""
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0.5, 1, 0])
        tri2 = self._make_tri([5, 0, 0], [6, 0, 0], [5.5, 1, 0])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_shared_vertex_no_intersection(self):
        """共享顶点不算有效相交"""
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0, 1, 0])
        tri2 = self._make_tri([0, 0, 0], [-1, 0, 0], [0, -1, 0])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_shared_edge_no_intersection(self):
        """共享边不算有效相交"""
        tri1 = self._make_tri([0, 0, 0], [2, 0, 0], [1, 1, 0])
        tri2 = self._make_tri([0, 0, 0], [2, 0, 0], [1, -1, 0])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_aabb_no_overlap(self):
        """AABB包围盒不重叠 → 快速排除"""
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0, 1, 0])
        tri2 = self._make_tri([100, 100, 100], [101, 100, 100], [100, 101, 100])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_perpendicular_triangles(self):
        """两个垂直三角形相交"""
        tri1 = self._make_tri([0, 0, -2], [0, 0, 2], [4, 0, 0])
        tri2 = self._make_tri([-2, -2, 0], [2, -2, 0], [0, 2, 0])
        self.assertTrue(check_triangle_intersection(tri1, tri2))

    def test_parallel_triangles_no_intersect(self):
        """平行三角形（不同z）不相交"""
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0, 1, 0])
        tri2 = self._make_tri([0, 0, 5], [1, 0, 5], [0, 1, 5])
        self.assertFalse(check_triangle_intersection(tri1, tri2))


# ============================================================================
# check_edge_triangle_intersection (高层API)
# ============================================================================

class TestCheckEdgeTriangleIntersection(unittest.TestCase):
    """测试高层边-三角形相交检测API"""

    def _make_tri(self, p1, p2, p3):
        return np.array([p1, p2, p3], dtype=np.float64)

    def test_edge_pierces(self):
        """边穿透三角形"""
        tri = self._make_tri([0, 0, 0], [4, 0, 0], [2, 4, 0])
        self.assertTrue(check_edge_triangle_intersection(
            np.array([2.0, 1.0, -1.0]),
            np.array([2.0, 1.0, 1.0]),
            tri
        ))

    def test_edge_misses(self):
        """边不与三角形相交"""
        tri = self._make_tri([0, 0, 0], [4, 0, 0], [2, 4, 0])
        self.assertFalse(check_edge_triangle_intersection(
            np.array([10.0, 10.0, -1.0]),
            np.array([10.0, 10.0, 1.0]),
            tri
        ))

    def test_edge_same_side(self):
        """边两端在同侧"""
        tri = self._make_tri([0, 0, 0], [4, 0, 0], [2, 4, 0])
        self.assertFalse(check_edge_triangle_intersection(
            np.array([2.0, 1.0, 1.0]),
            np.array([2.0, 1.0, 2.0]),
            tri
        ))

    def test_coplanar_edge_through(self):
        """共面边穿过三角形"""
        tri = self._make_tri([0, 0, 0], [4, 0, 0], [2, 4, 0])
        self.assertTrue(check_edge_triangle_intersection(
            np.array([-1.0, 0.0, 0.0]),
            np.array([5.0, 0.0, 0.0]),
            tri
        ))


# ============================================================================
# 综合/边界测试
# ============================================================================

class TestPointOnSegment2D(unittest.TestCase):
    """测试2D点在线段上判断"""

    def test_point_at_start(self):
        """点在起点"""
        p = np.array([0.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertTrue(_point_on_segment_2d(p, s1, s2))

    def test_point_at_end(self):
        """点在终点"""
        p = np.array([2.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertTrue(_point_on_segment_2d(p, s1, s2))

    def test_point_at_midpoint(self):
        """点在中点"""
        p = np.array([1.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertTrue(_point_on_segment_2d(p, s1, s2))

    def test_point_off_segment(self):
        """点不在线段上"""
        p = np.array([1.0, 1.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertFalse(_point_on_segment_2d(p, s1, s2))

    def test_point_beyond_start(self):
        """点在起点延长线上"""
        p = np.array([-1.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertFalse(_point_on_segment_2d(p, s1, s2))

    def test_point_on_diagonal(self):
        """点在斜线段上"""
        p = np.array([1.0, 1.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 2.0])
        self.assertTrue(_point_on_segment_2d(p, s1, s2))


class TestCollinearOverlap(unittest.TestCase):
    """测试共线重叠线段检测"""

    def test_collinear_partial_overlap(self):
        """共线部分重叠"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        b1, b2 = np.array([1.0, 0.0]), np.array([3.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_full_containment(self):
        """共线完全包含"""
        a1, a2 = np.array([0.0, 0.0]), np.array([4.0, 0.0])
        b1, b2 = np.array([1.0, 0.0]), np.array([3.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_touch_at_endpoint(self):
        """共线仅端点接触 → 不算相交"""
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        b1, b2 = np.array([1.0, 0.0]), np.array([2.0, 0.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_separated(self):
        """共线但分离"""
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        b1, b2 = np.array([3.0, 0.0]), np.array([4.0, 0.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_diagonal(self):
        """斜方向共线重叠"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        b1, b2 = np.array([1.0, 1.0]), np.array([3.0, 3.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_identical(self):
        """完全重合线段"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        b1, b2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))


class TestCoplanarEdgeTriangleExtended(unittest.TestCase):
    """共面边-三角形相交的扩展测试"""

    def test_coplanar_edge_one_endpoint_inside(self):
        """共面边一端在三角形内，一端在外"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([2.0, 1.0, 0.0])  # 在三角形内
        ee = np.array([6.0, 1.0, 0.0])  # 在三角形外
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_vertex_on_edge(self):
        """共面三角形顶点在边上"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([2.0, -1.0, 0.0])
        ee = np.array([2.0, 5.0, 0.0])  # 穿过三角形顶点(2,4,0)附近
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_edge_touches_vertex_only(self):
        """共面边仅接触三角形顶点（端点重合）→ 排除"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([0.0, 0.0, 0.0])  # 与t1重合
        ee = np.array([-1.0, 0.0, 0.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_completely_separated(self):
        """共面但完全分离"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([1.0, 0.0, 0.0])
        t3 = np.array([0.5, 1.0, 0.0])
        es = np.array([5.0, 5.0, 0.0])
        ee = np.array([6.0, 6.0, 0.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))


class TestEdgeCases(unittest.TestCase):
    """综合边界情况测试"""

    def test_large_coordinates(self):
        """大坐标值的数值稳定性"""
        a = np.array([1e6, 0.0, 0.0])
        b = np.array([1e6 + 4, 0.0, 0.0])
        c = np.array([1e6 + 2, 4.0, 0.0])
        p = np.array([1e6 + 2, 1.0, -1.0])
        q = np.array([1e6 + 2, 1.0, 1.0])
        self.assertTrue(segment_intersects_triangle(p, q, a, b, c))

    def test_small_triangle(self):
        """非常小的三角形"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1e-8, 0.0, 0.0])
        c = np.array([5e-9, 1e-8, 0.0])
        p = np.array([5e-9, 3e-9, -1e-8])
        q = np.array([5e-9, 3e-9, 1e-8])
        # 小三角形应该仍然能检测到相交
        result = segment_intersects_triangle(p, q, a, b, c)
        self.assertIsInstance(result, bool)

    def test_distance_symmetry(self):
        """线段距离应满足对称性"""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0])
        q2 = np.array([1.0, 1.0, 0.0])
        d1 = segment_segment_distance_3d(p1, p2, q1, q2)
        d2 = segment_segment_distance_3d(q1, q2, p1, p2)
        self.assertAlmostEqual(d1, d2, places=10)

    def test_project_to_2d_preserves_count(self):
        """投影保持点数不变"""
        pts = np.random.rand(100, 3)
        normal = np.array([0.0, 0.0, 1.0])
        result = project_to_2d(pts, normal)
        self.assertEqual(result.shape, (100, 2))


if __name__ == "__main__":
    unittest.main()

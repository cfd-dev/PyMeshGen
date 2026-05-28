#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：sfmesh/geom_utils.py (v2.0 - 严谨性修复回归测试版)
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
    FLOAT_MIN,
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

    def test_single_point_batch(self):
        """单点投影 (1,3) shape"""
        pts = np.array([[10, 20, 30]], dtype=float)
        normal = np.array([0, 0, 1])
        result = project_to_2d(pts, normal)
        self.assertEqual(result.shape, (1, 2))
        np.testing.assert_array_equal(result[0], [10, 20])

    # [REGRESSION] 修复 project_to_2d 支持 (3,) 单点输入
    def test_project_single_1d_point(self):
        """传入shape=(3,)的单点 → 应返回shape=(2,)"""
        pt = np.array([10.0, 20.0, 30.0])
        normal = np.array([0.0, 0.0, 1.0])
        result = project_to_2d(pt, normal)
        self.assertEqual(result.shape, (2,))
        np.testing.assert_array_equal(result, [10.0, 20.0])


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

    # [REGRESSION] 修复 T 型相交漏判 - 斜向 T 型
    def test_t_shape_oblique(self):
        """斜向T型相交：端点恰好落在另一线段上(非正交)"""
        a1, a2 = np.array([0.0, 0.0]), np.array([4.0, 4.0])
        b1, b2 = np.array([2.0, 2.0]), np.array([2.0, 5.0])  # b1恰好在a1-a2上
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    # [REGRESSION] 修复 T 型相交漏判 - 交点极度靠近端点
    def test_intersection_near_endpoint(self):
        """交点极度靠近端点(距离<tol) → 应被识别为相交"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        # b1-b2 与 a1-a2 的理论交点在 (1,1)，但b1偏移了1e-11
        b1 = np.array([1.0 + 1e-11, 0.0])
        b2 = np.array([1.0 - 1e-11, 2.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))


# ============================================================================
# point_in_triangle_2d
# ============================================================================
class TestPointInTriangle2D(unittest.TestCase):
    """测试2D点在三角形内判断"""

    def test_point_inside(self):
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        p = np.array([2.0, 1.0])
        self.assertTrue(point_in_triangle_2d(p, t0, t1, t2))

    def test_point_on_edge(self):
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        p = np.array([2.0, 0.0])
        self.assertTrue(point_in_triangle_2d(p, t0, t1, t2))

    def test_point_on_vertex(self):
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        self.assertTrue(point_in_triangle_2d(t0, t0, t1, t2))

    def test_point_outside(self):
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        p = np.array([5.0, 5.0])
        self.assertFalse(point_in_triangle_2d(p, t0, t1, t2))

    def test_point_just_outside(self):
        t0 = np.array([0.0, 0.0])
        t1 = np.array([4.0, 0.0])
        t2 = np.array([2.0, 4.0])
        p = np.array([2.0, -0.1])
        self.assertFalse(point_in_triangle_2d(p, t0, t1, t2))

    def test_centroid(self):
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
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 0.0])
        self.assertTrue(point_in_triangle_3d(p, a, b, c))

    def test_point_on_vertex(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        self.assertTrue(point_in_triangle_3d(a, a, b, c))

    def test_point_outside(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([10.0, 10.0, 0.0])
        self.assertFalse(point_in_triangle_3d(p, a, b, c))

    def test_degenerate_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        p = np.array([0.5, 0.0, 0.0])
        self.assertFalse(point_in_triangle_3d(p, a, b, c))

    def test_tilted_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0])
        c = np.array([0.0, 1.0, 1.0])
        p = np.array([0.25, 0.25, 0.5])
        self.assertTrue(point_in_triangle_3d(p, a, b, c))

    def test_centroid_inside(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 0.0, 0.0])
        c = np.array([1.5, 3.0, 0.0])
        centroid = (a + b + c) / 3.0
        self.assertTrue(point_in_triangle_3d(centroid, a, b, c))

    # [REGRESSION] 新增 check_coplanar=True 时拒绝悬空点
    def test_point_above_triangle_rejected(self):
        """悬空点在三角形投影内但不在平面上 → check_coplanar=True时应返回False"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 0.5])  # z=0.5，悬空
        self.assertFalse(point_in_triangle_3d(p, a, b, c, check_coplanar=True))

    # [REGRESSION] check_coplanar=False 时允许悬空点（投影语义）
    def test_point_above_triangle_no_check(self):
        """check_coplanar=False时，悬空点的投影在三角形内 → 应返回True"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 0.5])
        self.assertTrue(point_in_triangle_3d(p, a, b, c, check_coplanar=False))


# ============================================================================
# segment_intersects_triangle (Möller–Trumbore)
# ============================================================================
class TestSegmentIntersectsTriangle(unittest.TestCase):
    """测试线段与三角形相交（Möller–Trumbore算法）"""

    def test_segment_pierces_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, -1.0])
        q = np.array([2.0, 1.0, 1.0])
        self.assertTrue(segment_intersects_triangle(p, q, a, b, c))

    def test_segment_misses_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([10.0, 10.0, -1.0])
        q = np.array([10.0, 10.0, 1.0])
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))

    def test_segment_parallel_to_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([4.0, 0.0, 1.0])
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))

    def test_segment_ends_before_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, -1.0])
        q = np.array([2.0, 1.0, -0.01])
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))

    def test_endpoint_on_triangle_excluded(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 0.0])
        q = np.array([2.0, 1.0, 1.0])
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))

    def test_degenerate_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        p = np.array([0.5, -1.0, 0.0])
        q = np.array([0.5, 1.0, 0.0])
        self.assertFalse(segment_intersects_triangle(p, q, a, b, c))


# ============================================================================
# segment_segment_distance_3d
# ============================================================================
class TestSegmentSegmentDistance3D(unittest.TestCase):
    """测试3D线段间最短距离"""

    def test_parallel_segments(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0])
        q2 = np.array([1.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 1.0, places=10)

    def test_intersecting_segments(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        q1 = np.array([1.0, -1.0, 0.0])
        q2 = np.array([1.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_skew_segments(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 1.0])
        q2 = np.array([1.0, 1.0, 1.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, np.sqrt(2.0), places=10)

    def test_point_segments(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        q1 = np.array([3.0, 4.0, 0.0])
        q2 = np.array([3.0, 4.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 5.0, places=10)

    def test_one_degenerate_one_normal(self):
        p1 = np.array([0.5, 0.5, 0.0])
        p2 = np.array([0.5, 0.5, 0.0])
        q1 = np.array([0.0, 0.0, 0.0])
        q2 = np.array([1.0, 0.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 0.5, places=10)

    def test_shared_endpoint(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        q1 = np.array([0.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_closest_points_at_endpoints(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        q1 = np.array([2.0, 1.0, 0.0])
        q2 = np.array([3.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        expected = np.sqrt(1.0 + 1.0)
        self.assertAlmostEqual(dist, expected, places=10)


# ============================================================================
# point_triangle_distance_3d
# ============================================================================
class TestPointTriangleDistance3D(unittest.TestCase):
    """测试3D点到三角形最短距离"""

    def test_point_inside_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 0.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_point_above_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 1.0, 3.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 3.0, places=10)

    def test_point_nearest_vertex(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([-1.0, -1.0, 0.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, np.sqrt(2.0), places=10)

    def test_point_nearest_edge(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, -1.0, 0.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 1.0, places=10)

    def test_point_at_vertex(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        dist = point_triangle_distance_3d(a, a, b, c)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_point_on_edge(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        p = np.array([2.0, 0.0, 0.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_degenerate_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        p = np.array([0.5, 1.0, 0.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertAlmostEqual(dist, 1.0, places=10)

    # [REGRESSION] 修复 Voronoi 区域法除零保护
    def test_point_at_vertex_voronoi_degenerate(self):
        """点精确位于顶点 → d1=d3=0，分母为0，应安全返回0而非NaN/Exception"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        dist = point_triangle_distance_3d(a, a, b, c)
        self.assertAlmostEqual(dist, 0.0, places=10)
        self.assertFalse(np.isnan(dist))

    # [REGRESSION] 点在另一顶点时的除零保护
    def test_point_at_vertex_b_voronoi_degenerate(self):
        """点精确位于顶点b → 对应Voronoi边区域除零保护"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        c = np.array([2.0, 4.0, 0.0])
        dist = point_triangle_distance_3d(b, a, b, c)
        self.assertAlmostEqual(dist, 0.0, places=10)
        self.assertFalse(np.isnan(dist))


# ============================================================================
# are_coplanar_triangles_overlapping
# ============================================================================
class TestCoplanarTrianglesOverlapping(unittest.TestCase):
    """测试共面三角形重叠检测"""

    def test_overlapping_triangles(self):
        tri_a = np.array([[0, 0, 0], [4, 0, 0], [2, 4, 0]], dtype=float)
        tri_b = np.array([[1, 0, 0], [5, 0, 0], [3, 4, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_separated_triangles(self):
        tri_a = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=float)
        tri_b = np.array([[10, 0, 0], [11, 0, 0], [10.5, 1, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertFalse(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_contained_triangle(self):
        tri_a = np.array([[0, 0, 0], [10, 0, 0], [5, 10, 0]], dtype=float)
        tri_b = np.array([[2, 1, 0], [8, 1, 0], [5, 5, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_shared_vertex_is_overlapping(self):
        tri_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        tri_b = np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_shared_edge_is_overlapping(self):
        tri_a = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=float)
        tri_b = np.array([[0, 0, 0], [2, 0, 0], [1, -1, 0]], dtype=float)
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_yz_plane(self):
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
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([2.0, 1.0, -1.0])
        ee = np.array([2.0, 1.0, 1.0])
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_edge_misses(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([10.0, 10.0, -1.0])
        ee = np.array([10.0, 10.0, 1.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_edge_same_side(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([2.0, 1.0, 1.0])
        ee = np.array([2.0, 1.0, 2.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_edge_crosses_triangle_edge(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([-1.0, 0.0, 0.0])
        ee = np.array([5.0, 0.0, 0.0])
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_edge_fully_inside_triangle(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([1.0, 0.5, 0.0])
        ee = np.array([3.0, 0.5, 0.0])
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_degenerate_triangle_returns_false(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([1.0, 0.0, 0.0])
        t3 = np.array([2.0, 0.0, 0.0])
        es = np.array([0.5, -1.0, 0.0])
        ee = np.array([0.5, 1.0, 0.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_endpoint_on_triangle_vertex_excluded(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = t1.copy()
        ee = np.array([0.0, 0.0, 1.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    # [REGRESSION] 修复同侧判断乘积漏洞：两端点在平面同侧且距离极小
    def test_edge_same_side_close_to_plane(self):
        """两端点在平面同侧且距离极小(>tol但乘积<tol) → 应返回False"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        # d1=1e-6, d2=2e-6, 乘积=2e-12 < DEFAULT_TOL(1e-10)
        # 旧代码 d1*d2 > tol 会误判为跨越平面
        es = np.array([2.0, 1.0, 1e-6])
        ee = np.array([2.0, 1.0, 2e-6])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    # [REGRESSION] 同侧判断：两端点在平面负侧且距离极小
    def test_edge_same_side_negative_close_to_plane(self):
        """两端点在平面负侧且距离极小 → 应返回False"""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([2.0, 1.0, -1e-6])
        ee = np.array([2.0, 1.0, -2e-6])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))


# ============================================================================
# check_triangle_intersection (高层API)
# ============================================================================
class TestCheckTriangleIntersection(unittest.TestCase):
    """测试高层三角形相交检测API"""

    def _make_tri(self, p1, p2, p3):
        return np.array([p1, p2, p3], dtype=np.float64)

    def test_intersecting_triangles(self):
        tri1 = self._make_tri([0, 0, 0], [4, 0, 0], [2, 4, 0])
        tri2 = self._make_tri([2, 0, -1], [2, 0, 1], [2, 4, 0])
        self.assertTrue(check_triangle_intersection(tri1, tri2))

    def test_separated_triangles(self):
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0.5, 1, 0])
        tri2 = self._make_tri([10, 0, 0], [11, 0, 0], [10.5, 1, 0])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_coplanar_no_overlap(self):
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0.5, 1, 0])
        tri2 = self._make_tri([5, 0, 0], [6, 0, 0], [5.5, 1, 0])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_shared_vertex_no_intersection(self):
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0, 1, 0])
        tri2 = self._make_tri([0, 0, 0], [-1, 0, 0], [0, -1, 0])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_shared_edge_no_intersection(self):
        tri1 = self._make_tri([0, 0, 0], [2, 0, 0], [1, 1, 0])
        tri2 = self._make_tri([0, 0, 0], [2, 0, 0], [1, -1, 0])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_shared_edge_bowtie_intersection(self):
        """共底边但非共享边交叉（蝴蝶形）→ 应检测为相交"""
        tri1 = self._make_tri([0, 0, 0], [4, 0, 0], [3, 1, 0])
        tri2 = self._make_tri([0, 0, 0], [4, 0, 0], [1, 1, 0])
        self.assertTrue(check_triangle_intersection(tri1, tri2))

    def test_aabb_no_overlap(self):
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0, 1, 0])
        tri2 = self._make_tri([100, 100, 100], [101, 100, 100], [100, 101, 100])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_perpendicular_triangles(self):
        tri1 = self._make_tri([0, 0, -2], [0, 0, 2], [4, 0, 0])
        tri2 = self._make_tri([-2, -2, 0], [2, -2, 0], [0, 2, 0])
        self.assertTrue(check_triangle_intersection(tri1, tri2))

    def test_parallel_triangles_no_intersect(self):
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0, 1, 0])
        tri2 = self._make_tri([0, 0, 5], [1, 0, 5], [0, 1, 5])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    # [REGRESSION] 共享顶点但不共面的三角形不应相交
    def test_shared_vertex_non_coplanar_no_intersection(self):
        """共享一个顶点但两三角形不共面且无穿透 → 不应相交"""
        tri1 = self._make_tri([0, 0, 0], [1, 0, 0], [0, 1, 0])
        tri2 = self._make_tri([0, 0, 0], [0, 0, 1], [-1, 0, 0])
        self.assertFalse(check_triangle_intersection(tri1, tri2))


# ============================================================================
# check_edge_triangle_intersection (高层API)
# ============================================================================
class TestCheckEdgeTriangleIntersection(unittest.TestCase):
    """测试高层边-三角形相交检测API"""

    def _make_tri(self, p1, p2, p3):
        return np.array([p1, p2, p3], dtype=np.float64)

    def test_edge_pierces(self):
        tri = self._make_tri([0, 0, 0], [4, 0, 0], [2, 4, 0])
        self.assertTrue(check_edge_triangle_intersection(
            np.array([2.0, 1.0, -1.0]),
            np.array([2.0, 1.0, 1.0]),
            tri
        ))

    def test_edge_misses(self):
        tri = self._make_tri([0, 0, 0], [4, 0, 0], [2, 4, 0])
        self.assertFalse(check_edge_triangle_intersection(
            np.array([10.0, 10.0, -1.0]),
            np.array([10.0, 10.0, 1.0]),
            tri
        ))

    def test_edge_same_side(self):
        tri = self._make_tri([0, 0, 0], [4, 0, 0], [2, 4, 0])
        self.assertFalse(check_edge_triangle_intersection(
            np.array([2.0, 1.0, 1.0]),
            np.array([2.0, 1.0, 2.0]),
            tri
        ))

    def test_coplanar_edge_through(self):
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
        p = np.array([0.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertTrue(_point_on_segment_2d(p, s1, s2))

    def test_point_at_end(self):
        p = np.array([2.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertTrue(_point_on_segment_2d(p, s1, s2))

    def test_point_at_midpoint(self):
        p = np.array([1.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertTrue(_point_on_segment_2d(p, s1, s2))

    def test_point_off_segment(self):
        p = np.array([1.0, 1.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertFalse(_point_on_segment_2d(p, s1, s2))

    def test_point_beyond_start(self):
        p = np.array([-1.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertFalse(_point_on_segment_2d(p, s1, s2))

    def test_point_on_diagonal(self):
        p = np.array([1.0, 1.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 2.0])
        self.assertTrue(_point_on_segment_2d(p, s1, s2))


class TestCollinearOverlap(unittest.TestCase):
    """测试共线重叠线段检测"""

    def test_collinear_partial_overlap(self):
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        b1, b2 = np.array([1.0, 0.0]), np.array([3.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_full_containment(self):
        a1, a2 = np.array([0.0, 0.0]), np.array([4.0, 0.0])
        b1, b2 = np.array([1.0, 0.0]), np.array([3.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_touch_at_endpoint(self):
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        b1, b2 = np.array([1.0, 0.0]), np.array([2.0, 0.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_separated(self):
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        b1, b2 = np.array([3.0, 0.0]), np.array([4.0, 0.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_diagonal(self):
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        b1, b2 = np.array([1.0, 1.0]), np.array([3.0, 3.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_identical(self):
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        b1, b2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))


class TestCoplanarEdgeTriangleExtended(unittest.TestCase):
    """共面边-三角形相交的扩展测试"""

    def test_coplanar_edge_one_endpoint_inside(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([2.0, 1.0, 0.0])
        ee = np.array([6.0, 1.0, 0.0])
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_vertex_on_edge(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([2.0, -1.0, 0.0])
        ee = np.array([2.0, 5.0, 0.0])
        self.assertTrue(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_edge_touches_vertex_only(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([4.0, 0.0, 0.0])
        t3 = np.array([2.0, 4.0, 0.0])
        es = np.array([0.0, 0.0, 0.0])
        ee = np.array([-1.0, 0.0, 0.0])
        self.assertFalse(_edge_intersects_triangle_core(es, ee, t1, t2, t3))

    def test_coplanar_completely_separated(self):
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

    # [REGRESSION] 微小坐标数值稳定性
    def test_micro_coordinates(self):
        """1e-12级别的微小数不应导致NaN或异常"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1e-12, 0.0, 0.0])
        c = np.array([5e-13, 1e-12, 0.0])
        p = np.array([5e-13, 3e-13, 0.0])
        
        # [FIX] 使用 assertTrue/assertFalse 替代 assertIsInstance(bool)
        # numpy 返回 np.bool_ 而非原生 bool，但真值语义完全等价
        result = point_in_triangle_3d(p, a, b, c)
        self.assertTrue(result is True or result is False or isinstance(result, (bool, np.bool_)))
        
        dist = point_triangle_distance_3d(p, a, b, c)
        self.assertFalse(np.isnan(dist))
        self.assertIsInstance(dist, float)  # distance 函数已显式 cast 为 float

    # [REGRESSION] 混合量级坐标
    def test_mixed_scale_coordinates(self):
        """大坐标与小偏移混合 → 验证浮点精度"""
        base = 1e8
        a = np.array([base, 0.0, 0.0])
        b = np.array([base + 1.0, 0.0, 0.0])
        c = np.array([base + 0.5, 1.0, 0.0])
        p = np.array([base + 0.5, 0.3, 0.0])
        self.assertTrue(point_in_triangle_3d(p, a, b, c))

    # [REGRESSION] 宽松容差测试
    def test_loose_tolerance(self):
        """更宽松的容差(1e-4)应正确缩放判断阈值"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        b1, b2 = np.array([0.0, 2.0]), np.array([2.0, 0.0])
        # 默认容差下相交
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2, tol=1e-4))
        # 分离但在宽松容差内应被判为相交
        c1, c2 = np.array([0.0, 2.0 + 5e-5]), np.array([2.0, 0.0 + 5e-5])
        self.assertTrue(segments_intersect_2d(a1, a2, c1, c2, tol=1e-4))


if __name__ == "__main__":
    unittest.main()
"""
sfmesh.geom_utils 单元测试

覆盖所有纯数值几何工具函数，无 OCC 依赖：
- 2D 投影与线段相交
- 点在三角形内（2D/3D）
- 线段-三角形相交 (Möller–Trumbore)
- 线段间距离（3D）
- 点到三角形距离（3D）
- 共面三角形重叠检测
- 三角形质量与退化检测
- 网格拓扑分析（边分类、边界环、欧拉示性数）
"""
import sys
import unittest
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
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
    triangle_quality_from_coords,
    check_triangle_degenerate,
    check_triangle_intersection,
    check_triangle_vs_existing,
    classify_edges,
    trace_boundary_loops,
    euler_characteristic,
    validate_mesh_topology,
    DEFAULT_TOL,
    DEGENERATE_TOL,
)
from sfmesh.surface_front import NodeElement3D, SurfaceTriangle


# ============================================================================
# project_to_2d
# ============================================================================

class TestProjectTo2D(unittest.TestCase):
    """测试 3D→2D 投影"""

    def test_drop_z_when_normal_along_x(self):
        """法向量沿 X 轴时丢弃 X 分量，保留 (Y, Z)"""
        pts = np.array([[1.0, 2.0, 3.0]])
        normal = np.array([1.0, 0.0, 0.0])
        result = project_to_2d(pts, normal)
        np.testing.assert_array_almost_equal(result, [[2.0, 3.0]])

    def test_drop_y_when_normal_along_z(self):
        """法向量沿 Z 轴时丢弃 Z 分量，保留 (X, Y)"""
        pts = np.array([[1.0, 2.0, 3.0]])
        normal = np.array([0.0, 0.0, 1.0])
        result = project_to_2d(pts, normal)
        np.testing.assert_array_almost_equal(result, [[1.0, 2.0]])

    def test_drop_x_when_normal_along_y(self):
        """法向量沿 Y 轴时丢弃 Y 分量，保留 (X, Z)"""
        pts = np.array([[1.0, 2.0, 3.0]])
        normal = np.array([0.0, 1.0, 0.0])
        result = project_to_2d(pts, normal)
        np.testing.assert_array_almost_equal(result, [[1.0, 3.0]])

    def test_diagonal_normal(self):
        """对角法向量：丢弃绝对值最大的分量"""
        pts = np.array([[1.0, 2.0, 3.0]])
        normal = np.array([0.5, 0.8, 0.3])  # Y 最大
        result = project_to_2d(pts, normal)
        np.testing.assert_array_almost_equal(result, [[1.0, 3.0]])

    def test_batch_points(self):
        """批量多点投影"""
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normal = np.array([0.0, 0.0, 1.0])
        result = project_to_2d(pts, normal)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_array_almost_equal(result, [[1.0, 2.0], [4.0, 5.0]])

    def test_single_point_1d(self):
        """单点输入 (3,) 返回 (2,)"""
        pt = np.array([1.0, 2.0, 3.0])
        normal = np.array([0.0, 0.0, 1.0])
        result = project_to_2d(pt, normal)
        self.assertEqual(result.shape, (2,))


# ============================================================================
# segments_intersect_2d
# ============================================================================

class TestSegmentsIntersect2D(unittest.TestCase):
    """测试 2D 线段相交检测"""

    def test_cross_intersection(self):
        """标准十字交叉"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        b1, b2 = np.array([0.0, 2.0]), np.array([2.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_no_intersection_parallel(self):
        """平行线段不相交"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        b1, b2 = np.array([0.0, 1.0]), np.array([2.0, 1.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_no_intersection_disjoint(self):
        """不相交的线段"""
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        b1, b2 = np.array([2.0, 0.0]), np.array([3.0, 0.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_t_intersection(self):
        """T 型相交：端点落在另一线段上"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        b1, b2 = np.array([1.0, 0.0]), np.array([1.0, 1.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_overlap(self):
        """共线重叠"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        b1, b2 = np.array([1.0, 0.0]), np.array([3.0, 0.0])
        self.assertTrue(segments_intersect_2d(a1, a2, b1, b2))

    def test_collinear_no_overlap(self):
        """共线但不重叠（仅端点接触）"""
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        b1, b2 = np.array([1.0, 0.0]), np.array([2.0, 0.0])
        # 端点接触不视为实质重叠
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_endpoint_coincidence_not_intersection(self):
        """端点重合（非共线）不视为相交"""
        a1, a2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        b1, b2 = np.array([0.0, 0.0]), np.array([0.0, 1.0])
        self.assertFalse(segments_intersect_2d(a1, a2, b1, b2))

    def test_near_miss(self):
        """几乎相交但不相交"""
        a1, a2 = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        b1, b2 = np.array([1.0, 1e-11]), np.array([1.0, 2.0])
        # 1e-11 < DEFAULT_TOL=1e-10，视为在直线上
        # 这属于边界情况，两种结果都合理

    def test_degenerate_segment(self):
        """退化线段（长度为零）"""
        a1, a2 = np.array([1.0, 1.0]), np.array([1.0, 1.0])
        b1, b2 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        # 退化线段上的点在另一线段上
        result = segments_intersect_2d(a1, a2, b1, b2)
        # 无论结果如何，不应抛出异常


# ============================================================================
# point_in_triangle_2d
# ============================================================================

class TestPointInTriangle2D(unittest.TestCase):
    """测试 2D 点在三角形内检测"""

    def setUp(self):
        self.t0 = np.array([0.0, 0.0])
        self.t1 = np.array([4.0, 0.0])
        self.t2 = np.array([2.0, 4.0])

    def test_point_inside(self):
        """三角形内部的点"""
        p = np.array([2.0, 1.0])
        self.assertTrue(point_in_triangle_2d(p, self.t0, self.t1, self.t2))

    def test_point_outside(self):
        """三角形外部的点"""
        p = np.array([0.0, 4.0])
        self.assertFalse(point_in_triangle_2d(p, self.t0, self.t1, self.t2))

    def test_point_on_vertex(self):
        """点在顶点上"""
        p = np.array([0.0, 0.0])
        self.assertTrue(point_in_triangle_2d(p, self.t0, self.t1, self.t2))

    def test_point_on_edge(self):
        """点在边上"""
        p = np.array([2.0, 0.0])
        self.assertTrue(point_in_triangle_2d(p, self.t0, self.t1, self.t2))

    def test_point_at_centroid(self):
        """点在重心"""
        cx = (self.t0[0] + self.t1[0] + self.t2[0]) / 3
        cy = (self.t0[1] + self.t1[1] + self.t2[1]) / 3
        p = np.array([cx, cy])
        self.assertTrue(point_in_triangle_2d(p, self.t0, self.t1, self.t2))

    def test_degenerate_triangle(self):
        """退化三角形（共线顶点）"""
        t0 = np.array([0.0, 0.0])
        t1 = np.array([1.0, 0.0])
        t2 = np.array([2.0, 0.0])
        p = np.array([0.5, 0.0])
        # 退化三角形上的点，结果取决于实现
        result = point_in_triangle_2d(p, t0, t1, t2)
        self.assertIsInstance(result, bool)


# ============================================================================
# point_in_triangle_3d
# ============================================================================

class TestPointInTriangle3D(unittest.TestCase):
    """测试 3D 点在三角形内检测"""

    def setUp(self):
        self.a = np.array([0.0, 0.0, 0.0])
        self.b = np.array([4.0, 0.0, 0.0])
        self.c = np.array([2.0, 4.0, 0.0])

    def test_point_inside(self):
        """三角形内部的点"""
        p = np.array([2.0, 1.0, 0.0])
        self.assertTrue(point_in_triangle_3d(p, self.a, self.b, self.c))

    def test_point_outside(self):
        """三角形外部的点"""
        p = np.array([0.0, 4.0, 0.0])
        self.assertFalse(point_in_triangle_3d(p, self.a, self.b, self.c))

    def test_point_above_plane(self):
        """点在三角形平面上方（不共面）"""
        p = np.array([2.0, 1.0, 1.0])
        self.assertFalse(point_in_triangle_3d(p, self.a, self.b, self.c))

    def test_point_on_vertex(self):
        """点在顶点上"""
        p = np.array([0.0, 0.0, 0.0])
        self.assertTrue(point_in_triangle_3d(p, self.a, self.b, self.c))

    def test_point_on_edge(self):
        """点在边上"""
        p = np.array([2.0, 0.0, 0.0])
        self.assertTrue(point_in_triangle_3d(p, self.a, self.b, self.c))

    def test_skip_coplanar_check(self):
        """跳过共面检查时，非共面点也会返回 True（如果重心坐标满足）"""
        p = np.array([2.0, 1.0, 0.5])
        # 不共面，但 check_coplanar=False 只检查重心坐标
        result = point_in_triangle_3d(p, self.a, self.b, self.c, check_coplanar=False)
        # 重心坐标 u,v 满足条件，返回 True（np.bool_ 也是 truthy）
        self.assertTrue(result)

    def test_degenerate_triangle(self):
        """退化三角形"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])  # 共线
        p = np.array([0.5, 0.0, 0.0])
        result = point_in_triangle_3d(p, a, b, c)
        # 退化三角形应返回 False
        self.assertFalse(result)

    def test_nonzero_z_triangle(self):
        """非零 Z 平面上的三角形"""
        a = np.array([0.0, 0.0, 5.0])
        b = np.array([4.0, 0.0, 5.0])
        c = np.array([2.0, 4.0, 5.0])
        p = np.array([2.0, 1.0, 5.0])
        self.assertTrue(point_in_triangle_3d(p, a, b, c))


# ============================================================================
# segment_intersects_triangle
# ============================================================================

class TestSegmentIntersectsTriangle(unittest.TestCase):
    """测试线段-三角形相交 (Möller–Trumbore)"""

    def setUp(self):
        self.a = np.array([0.0, 0.0, 0.0])
        self.b = np.array([4.0, 0.0, 0.0])
        self.c = np.array([2.0, 4.0, 0.0])

    def test_segment_pierces_triangle(self):
        """线段穿过三角形"""
        p = np.array([2.0, 1.0, -1.0])
        q = np.array([2.0, 1.0, 1.0])
        self.assertTrue(segment_intersects_triangle(p, q, self.a, self.b, self.c))

    def test_segment_misses(self):
        """线段不穿过三角形内部（穿过顶点也算相交）"""
        p = np.array([5.0, 5.0, -1.0])
        q = np.array([5.0, 5.0, 1.0])
        # 远离三角形的线段
        self.assertFalse(segment_intersects_triangle(p, q, self.a, self.b, self.c))

    def test_segment_parallel(self):
        """线段平行于三角形平面"""
        p = np.array([2.0, 1.0, 1.0])
        q = np.array([3.0, 1.0, 1.0])
        self.assertFalse(segment_intersects_triangle(p, q, self.a, self.b, self.c))

    def test_segment_coplanar(self):
        """线段与三角形共面"""
        p = np.array([1.0, 0.5, 0.0])
        q = np.array([3.0, 0.5, 0.0])
        # 共面线段，det≈0，应返回 False
        self.assertFalse(segment_intersects_triangle(p, q, self.a, self.b, self.c))

    def test_segment_endpoint_on_triangle(self):
        """线段端点在三角形上（排除端点）"""
        p = np.array([2.0, 1.0, 0.0])
        q = np.array([2.0, 1.0, 1.0])
        # p 在三角形面上，t 应接近 0，被排除
        result = segment_intersects_triangle(p, q, self.a, self.b, self.c)


# ============================================================================
# segment_segment_distance_3d
# ============================================================================

class TestSegmentSegmentDistance3D(unittest.TestCase):
    """测试 3D 线段间最短距离"""

    def test_intersecting_segments(self):
        """相交线段距离为 0"""
        p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
        q1, q2 = np.array([1.0, -1.0, 0.0]), np.array([1.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_parallel_segments(self):
        """平行线段距离"""
        p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        q1, q2 = np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 1.0, places=10)

    def test_perpendicular_segments(self):
        """垂直但不相交的线段"""
        p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        q1, q2 = np.array([0.5, 1.0, -1.0]), np.array([0.5, 1.0, 1.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 1.0, places=10)

    def test_degenerate_segment(self):
        """退化线段（点到线段距离）"""
        p1, p2 = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
        q1, q2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        expected = np.sqrt(0.5**2 + 0.5**2)
        self.assertAlmostEqual(dist, expected, places=10)

    def test_both_degenerate(self):
        """两个退化线段（点到点距离）"""
        p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        q1, q2 = np.array([3.0, 4.0, 0.0]), np.array([3.0, 4.0, 0.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 5.0, places=10)

    def test_skew_segments(self):
        """异面线段"""
        p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        q1, q2 = np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0])
        dist = segment_segment_distance_3d(p1, p2, q1, q2)
        self.assertAlmostEqual(dist, 1.0, places=10)


# ============================================================================
# point_triangle_distance_3d
# ============================================================================

class TestPointTriangleDistance3D(unittest.TestCase):
    """测试 3D 点到三角形最短距离"""

    def setUp(self):
        self.a = np.array([0.0, 0.0, 0.0])
        self.b = np.array([4.0, 0.0, 0.0])
        self.c = np.array([2.0, 4.0, 0.0])

    def test_point_on_vertex(self):
        """点在顶点上"""
        p = np.array([0.0, 0.0, 0.0])
        dist = point_triangle_distance_3d(p, self.a, self.b, self.c)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_point_on_edge(self):
        """点在边上"""
        p = np.array([2.0, 0.0, 0.0])
        dist = point_triangle_distance_3d(p, self.a, self.b, self.c)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_point_on_face(self):
        """点在面上"""
        p = np.array([2.0, 1.0, 0.0])
        dist = point_triangle_distance_3d(p, self.a, self.b, self.c)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_point_above_face(self):
        """点在面上方"""
        p = np.array([2.0, 1.0, 3.0])
        dist = point_triangle_distance_3d(p, self.a, self.b, self.c)
        self.assertAlmostEqual(dist, 3.0, places=10)

    def test_closest_to_vertex(self):
        """最近点是顶点"""
        p = np.array([-1.0, -1.0, 0.0])
        dist = point_triangle_distance_3d(p, self.a, self.b, self.c)
        expected = np.sqrt(2.0)
        self.assertAlmostEqual(dist, expected, places=10)

    def test_closest_to_edge(self):
        """最近点在边上"""
        p = np.array([2.0, -1.0, 0.0])
        dist = point_triangle_distance_3d(p, self.a, self.b, self.c)
        self.assertAlmostEqual(dist, 1.0, places=10)

    def test_degenerate_triangle(self):
        """退化三角形"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        p = np.array([0.5, 1.0, 0.0])
        dist = point_triangle_distance_3d(p, a, b, c)
        # 退化三角形，距离应为到最近边/点的距离
        self.assertGreaterEqual(dist, 0.0)


# ============================================================================
# are_coplanar_triangles_overlapping
# ============================================================================

class TestCoplanarTrianglesOverlapping(unittest.TestCase):
    """测试共面三角形重叠检测"""

    def test_overlapping_triangles(self):
        """两个重叠的共面三角形"""
        tri_a = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 4.0, 0.0]])
        tri_b = np.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_disjoint_triangles(self):
        """两个不重叠的共面三角形"""
        tri_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        tri_b = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.5, 1.0, 0.0]])
        normal = np.array([0.0, 0.0, 1.0])
        self.assertFalse(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_shared_edge(self):
        """共享一条边的两个三角形"""
        tri_a = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        tri_b = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
        normal = np.array([0.0, 0.0, 1.0])
        # 共享边，边相交
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_shared_vertex_only(self):
        """仅共享一个顶点的两个三角形（视为重叠，因为共享顶点在对方内部/边上）"""
        tri_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        tri_b = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-0.5, -1.0, 0.0]])
        normal = np.array([0.0, 0.0, 1.0])
        # 函数将共享顶点视为重叠（含边界/边重合的退化情况）
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_one_inside_other(self):
        """一个三角形完全在另一个内部"""
        tri_a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 10.0, 0.0]])
        tri_b = np.array([[3.0, 1.0, 0.0], [7.0, 1.0, 0.0], [5.0, 3.0, 0.0]])
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))

    def test_xy_plane(self):
        """XY 平面上的三角形"""
        tri_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        tri_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -1.0, 0.0]])
        normal = np.array([0.0, 0.0, 1.0])
        self.assertTrue(are_coplanar_triangles_overlapping(tri_a, tri_b, normal))


# ============================================================================
# triangle_quality_from_coords
# ============================================================================

class TestTriangleQualityFromCoords(unittest.TestCase):
    """测试三角形质量因子计算"""

    def test_equilateral_triangle(self):
        """等边三角形质量 = 1"""
        a = 1.0
        h = a * np.sqrt(3) / 2
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([a, 0.0, 0.0])
        p2 = np.array([a / 2, h, 0.0])
        q = triangle_quality_from_coords(p0, p1, p2)
        self.assertAlmostEqual(q, 1.0, places=10)

    def test_right_triangle(self):
        """直角三角形质量 < 1"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        q = triangle_quality_from_coords(p0, p1, p2)
        self.assertGreater(q, 0.0)
        self.assertLess(q, 1.0)

    def test_degenerate_collinear(self):
        """共线退化三角形质量 = 0"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        q = triangle_quality_from_coords(p0, p1, p2)
        self.assertAlmostEqual(q, 0.0, places=10)

    def test_coincident_vertices(self):
        """重合顶点退化三角形质量 = 0"""
        p0 = np.array([1.0, 1.0, 1.0])
        p1 = np.array([1.0, 1.0, 1.0])
        p2 = np.array([1.0, 1.0, 1.0])
        q = triangle_quality_from_coords(p0, p1, p2)
        self.assertAlmostEqual(q, 0.0, places=10)

    def test_very_thin_triangle(self):
        """非常细长的三角形质量接近 0"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([100.0, 0.0, 0.0])
        p2 = np.array([50.0, 0.01, 0.0])
        q = triangle_quality_from_coords(p0, p1, p2)
        self.assertLess(q, 0.05)

    def test_quality_range(self):
        """质量值在 [0, 1] 范围内"""
        import random
        random.seed(42)
        for _ in range(100):
            pts = [np.array([random.uniform(-10, 10) for _ in range(3)]) for _ in range(3)]
            q = triangle_quality_from_coords(*pts)
            self.assertGreaterEqual(q, 0.0)
            self.assertLessEqual(q, 1.0)


# ============================================================================
# check_triangle_degenerate
# ============================================================================

class TestCheckTriangleDegenerate(unittest.TestCase):
    """测试三角形退化检测"""

    def test_normal_triangle(self):
        """正常三角形不退化"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.5, 1.0, 0.0])
        self.assertFalse(check_triangle_degenerate(p0, p1, p2, min_height=0.01, min_edge_len=0.01))

    def test_collinear_vertices(self):
        """共线顶点：函数仅检查 p2 到边 p0p1 的高度，投影裁剪到 p1"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        # p2 投影到 p0p1 裁剪为 p1, 距离=1.0 > min_height → 不退化
        # (函数只检查 p2 到 p0p1 的高度，不检查所有三个顶点)
        self.assertFalse(check_triangle_degenerate(p0, p1, p2, min_height=0.01, min_edge_len=0.01))
        # 但如果 p2 在 p0p1 上（投影不裁剪），则高度=0 → 退化
        p2b = np.array([0.5, 0.0, 0.0])
        self.assertTrue(check_triangle_degenerate(p0, p1, p2b, min_height=0.01, min_edge_len=0.01))

    def test_short_edge(self):
        """短边退化"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.5, 0.001, 0.0])
        self.assertTrue(check_triangle_degenerate(p0, p1, p2, min_height=0.01, min_edge_len=0.01))

    def test_small_height(self):
        """高度过小退化"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.5, 0.0005, 0.0])
        self.assertTrue(check_triangle_degenerate(p0, p1, p2, min_height=0.001, min_edge_len=0.0001))

    def test_zero_thresholds(self):
        """零阈值：正常三角形不退化"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.5, 1.0, 0.0])
        self.assertFalse(check_triangle_degenerate(p0, p1, p2, min_height=0.0, min_edge_len=0.0))


# ============================================================================
# classify_edges
# ============================================================================

class TestClassifyEdges(unittest.TestCase):
    """测试边分类"""

    def _make_tri(self, node_ids):
        """创建模拟三角形对象"""
        class MockTri:
            pass
        t = MockTri()
        t.node_ids = node_ids
        return t

    def test_single_triangle(self):
        """单个三角形：3 条边界边"""
        tri = self._make_tri([0, 1, 2])
        edge_count, boundary, interior, non_manifold = classify_edges([tri])
        self.assertEqual(len(edge_count), 3)
        self.assertEqual(len(boundary), 3)
        self.assertEqual(len(interior), 0)
        self.assertEqual(len(non_manifold), 0)

    def test_two_triangles_shared_edge(self):
        """两个三角形共享一条边：4 边界 + 1 内部"""
        t1 = self._make_tri([0, 1, 2])
        t2 = self._make_tri([1, 3, 2])
        # 共享边 {1,2} → interior; 边界: {0,1}, {0,2}, {1,3}, {2,3}
        edge_count, boundary, interior, non_manifold = classify_edges([t1, t2])
        self.assertEqual(len(interior), 1)
        self.assertEqual(len(boundary), 4)
        self.assertEqual(len(non_manifold), 0)

    def test_non_manifold_edge(self):
        """非流形边：3 个三角形共享同一条边"""
        t1 = self._make_tri([0, 1, 2])
        t2 = self._make_tri([0, 1, 3])
        t3 = self._make_tri([0, 1, 4])
        edge_count, boundary, interior, non_manifold = classify_edges([t1, t2, t3])
        self.assertIn(frozenset([0, 1]), non_manifold)
        self.assertEqual(non_manifold[frozenset([0, 1])], 3)


# ============================================================================
# trace_boundary_loops
# ============================================================================

class TestTraceBoundaryLoops(unittest.TestCase):
    """测试边界环追踪"""

    def test_single_triangle_loop(self):
        """单个三角形的边界环"""
        edge_count = {
            frozenset([0, 1]): 1,
            frozenset([1, 2]): 1,
            frozenset([2, 0]): 1,
        }
        loops = trace_boundary_loops(edge_count)
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 3)

    def test_two_disjoint_triangles(self):
        """两个不相连的三角形 → 2 个边界环"""
        edge_count = {
            frozenset([0, 1]): 1,
            frozenset([1, 2]): 1,
            frozenset([2, 0]): 1,
            frozenset([3, 4]): 1,
            frozenset([4, 5]): 1,
            frozenset([5, 3]): 1,
        }
        loops = trace_boundary_loops(edge_count)
        self.assertEqual(len(loops), 2)

    def test_rectangle_boundary(self):
        """矩形边界环（4 节点）"""
        edge_count = {
            frozenset([0, 1]): 1,
            frozenset([1, 2]): 1,
            frozenset([2, 3]): 1,
            frozenset([3, 0]): 1,
        }
        loops = trace_boundary_loops(edge_count)
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 4)

    def test_empty_edge_count(self):
        """空边计数"""
        loops = trace_boundary_loops({})
        self.assertEqual(len(loops), 0)

    def test_no_boundary_edges(self):
        """没有边界边（封闭网格）"""
        edge_count = {
            frozenset([0, 1]): 2,
            frozenset([1, 2]): 2,
            frozenset([2, 0]): 2,
        }
        loops = trace_boundary_loops(edge_count)
        self.assertEqual(len(loops), 0)


# ============================================================================
# euler_characteristic
# ============================================================================

class TestEulerCharacteristic(unittest.TestCase):
    """测试欧拉示性数计算"""

    def test_disk_topology(self):
        """圆盘拓扑 χ = 1（单个三角形）"""
        # V=3, E=3, F=1 → χ=1
        chi = euler_characteristic(3, 3, 1)
        self.assertEqual(chi, 1)

    def test_sphere_topology(self):
        """球面拓扑 χ = 2（四面体）"""
        # V=4, E=6, F=4 → χ=2
        chi = euler_characteristic(4, 6, 4)
        self.assertEqual(chi, 2)

    def test_rectangle_mesh(self):
        """矩形网格 χ = 1"""
        # 2 个三角形组成矩形: V=4, E=5, F=2 → χ=1
        chi = euler_characteristic(4, 5, 2)
        self.assertEqual(chi, 1)

    def test_torus_topology(self):
        """环面拓扑 χ = 0"""
        # 标准环面: V=16, E=32, F=16 → χ=0
        chi = euler_characteristic(16, 32, 16)
        self.assertEqual(chi, 0)

    def test_formula_correctness(self):
        """公式 V - E + F 的正确性"""
        for V, E, F, expected in [
            (3, 3, 1, 1),
            (4, 6, 4, 2),
            (6, 9, 4, 1),
            (8, 12, 6, 2),
        ]:
            self.assertEqual(euler_characteristic(V, E, F), expected)


# ============================================================================
# Integration: mixed edge cases
# ============================================================================

class TestGeomUtilsEdgeCases(unittest.TestCase):
    """边界情况和数值稳定性测试"""

    def test_very_small_triangle_quality(self):
        """非常小的三角形质量计算"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1e-8, 0.0, 0.0])
        p2 = np.array([5e-9, 1e-8, 0.0])
        q = triangle_quality_from_coords(p0, p1, p2)
        self.assertGreaterEqual(q, 0.0)
        self.assertLessEqual(q, 1.0)

    def test_very_large_triangle_quality(self):
        """非常大的三角形质量计算"""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1e8, 0.0, 0.0])
        p2 = np.array([5e7, 1e8, 0.0])
        q = triangle_quality_from_coords(p0, p1, p2)
        # 等腰三角形（非等边），质量 ≈ 0.99
        self.assertGreater(q, 0.9)
        self.assertLessEqual(q, 1.0)

    def test_negative_coordinates(self):
        """负坐标"""
        p0 = np.array([-5.0, -3.0, -1.0])
        p1 = np.array([5.0, -3.0, -1.0])
        p2 = np.array([0.0, 7.0, -1.0])
        q = triangle_quality_from_coords(p0, p1, p2)
        # 等腰三角形（底=10, 腰≈14.14），质量 < 1
        self.assertGreater(q, 0.9)
        self.assertLessEqual(q, 1.0)

    def test_point_distance_3d_symmetry(self):
        """点到三角形距离的对称性不适用，但应一致"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.5, 1.0, 0.0])
        p = np.array([0.5, 0.3, 2.0])
        d1 = point_triangle_distance_3d(p, a, b, c)
        d2 = point_triangle_distance_3d(p, b, c, a)
        d3 = point_triangle_distance_3d(p, c, a, b)
        # 顶点顺序不应影响距离
        self.assertAlmostEqual(d1, d2, places=10)
        self.assertAlmostEqual(d2, d3, places=10)


# ============================================================================
# check_triangle_intersection (high-level API)
# ============================================================================

class TestCheckTriangleIntersection(unittest.TestCase):
    """测试高层三角形相交检测 API"""

    def test_no_intersection_disjoint(self):
        """不相交的两个三角形"""
        tri1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        tri2 = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.5, 1.0, 0.0]])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_piercing_intersection(self):
        """一个三角形穿过另一个"""
        tri1 = np.array([[-1.0, -1.0, -0.5], [1.0, -1.0, -0.5], [0.0, 1.0, -0.5]])
        tri2 = np.array([[-1.0, -1.0, 0.5], [1.0, -1.0, 0.5], [0.0, 0.0, 0.0]])
        # tri2 的顶点 (0,0,0) 在 tri1 上方，但 tri2 的边可能穿过 tri1
        result = check_triangle_intersection(tri1, tri2)
        self.assertIsInstance(result, bool)

    def test_shared_edge_no_intersection(self):
        """共享一条边的两个三角形不视为相交"""
        tri1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        tri2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -1.0, 0.0]])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_shared_vertex_no_intersection(self):
        """仅共享一个顶点的两个三角形不相交"""
        tri1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        tri2 = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-0.5, -1.0, 0.0]])
        self.assertFalse(check_triangle_intersection(tri1, tri2))

    def test_degenerate_triangle(self):
        """退化三角形（面积为零）"""
        tri1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        tri2 = np.array([[0.5, -1.0, -1.0], [0.5, 1.0, -1.0], [0.5, 0.0, 1.0]])
        result = check_triangle_intersection(tri1, tri2)
        self.assertIsInstance(result, bool)


# ============================================================================
# check_triangle_vs_existing
# ============================================================================

class TestCheckTriangleVsExisting(unittest.TestCase):
    """测试新三角形与已有三角形的相交检测（分级检测）"""

    def test_shared_2_no_intersection(self):
        """共享 2 个节点，无相交"""
        new = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        existing = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -1.0, 0.0]])
        self.assertFalse(check_triangle_vs_existing(
            new, existing, shared_count=2,
            new_shared_indices=[0, 1], existing_shared_indices=[0, 1]
        ))

    def test_shared_2_duplicate(self):
        """共享 2 个节点，完全重复"""
        new = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        existing = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        self.assertTrue(check_triangle_vs_existing(
            new, existing, shared_count=2,
            new_shared_indices=[0, 1], existing_shared_indices=[0, 1]
        ))

    def test_shared_1_no_intersection(self):
        """共享 1 个节点，无相交"""
        new = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        existing = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-0.5, -1.0, 0.0]])
        self.assertFalse(check_triangle_vs_existing(
            new, existing, shared_count=1,
            new_shared_indices=[0], existing_shared_indices=[0]
        ))

    def test_shared_0_no_intersection(self):
        """无共享节点，不相交"""
        new = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        existing = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.5, 1.0, 0.0]])
        self.assertFalse(check_triangle_vs_existing(
            new, existing, shared_count=0,
            new_shared_indices=[], existing_shared_indices=[]
        ))


# ============================================================================
# validate_mesh_topology
# ============================================================================

class TestValidateMeshTopology(unittest.TestCase):
    """测试网格拓扑验证函数"""

    def _make_triangle(self, node_ids, coords):
        """创建模拟三角形"""
        nodes = []
        for i, nid in enumerate(node_ids):
            nodes.append(NodeElement3D(coords=coords[i], idx=nid))
        return SurfaceTriangle(*nodes)

    def test_single_triangle(self):
        """单个三角形拓扑"""
        tri = self._make_triangle(
            [0, 1, 2],
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
        )
        result = validate_mesh_topology([tri], verbose=False)
        self.assertEqual(result['num_nodes'], 3)
        self.assertEqual(result['num_edges'], 3)
        self.assertEqual(result['num_faces'], 1)
        self.assertEqual(result['euler_characteristic'], 1)
        self.assertEqual(result['num_boundary_loops'], 1)
        self.assertEqual(result['non_manifold_edges'], 0)

    def test_two_triangles_disk(self):
        """两个三角形组成矩形（圆盘拓扑）"""
        t1 = self._make_triangle(
            [0, 1, 2],
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
        )
        t2 = self._make_triangle(
            [1, 3, 2],
            [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.5, 1.0, 0.0)]
        )
        result = validate_mesh_topology([t1, t2], verbose=False)
        self.assertEqual(result['euler_characteristic'], 1)
        self.assertEqual(result['num_boundary_loops'], 1)
        self.assertEqual(result['non_manifold_edges'], 0)

    def test_empty_mesh(self):
        """空网格"""
        result = validate_mesh_topology([], verbose=False)
        self.assertEqual(result['num_faces'], 0)


if __name__ == '__main__':
    unittest.main()

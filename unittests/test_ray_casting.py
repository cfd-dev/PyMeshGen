"""utils/ray_casting.py 单元测试"""
import unittest
import numpy as np
from utils.ray_casting import (
    build_spatial_grid,
    ray_triangle_intersect,
    count_ray_intersections,
    classify_points_inside,
)


def make_cube_triangles(size=2.0):
    """创建立方体表面三角形（顶点数组格式）"""
    s = size
    # 6 个面，每个面 2 个三角形
    verts = [
        # 底面 (z=0)
        [[0, 0, 0], [s, 0, 0], [s, s, 0]],
        [[0, 0, 0], [s, s, 0], [0, s, 0]],
        # 顶面 (z=s)
        [[0, 0, s], [s, s, s], [s, 0, s]],
        [[0, 0, s], [0, s, s], [s, s, s]],
        # 前面 (y=0)
        [[0, 0, 0], [s, 0, s], [s, 0, 0]],
        [[0, 0, 0], [0, 0, s], [s, 0, s]],
        # 后面 (y=s)
        [[0, s, 0], [s, s, 0], [s, s, s]],
        [[0, s, 0], [s, s, s], [0, s, s]],
        # 左面 (x=0)
        [[0, 0, 0], [0, s, 0], [0, s, s]],
        [[0, 0, 0], [0, s, s], [0, 0, s]],
        # 右面 (x=s)
        [[s, 0, 0], [s, 0, s], [s, s, s]],
        [[s, 0, 0], [s, s, s], [s, s, 0]],
    ]
    return np.array(verts, dtype=np.float64)


def make_sphere_triangles(radius=1.0, n_lat=8, n_lon=16):
    """创建球面三角形（顶点数组格式）"""
    triangles = []
    for i in range(n_lat):
        theta0 = np.pi * i / n_lat
        theta1 = np.pi * (i + 1) / n_lat
        for j in range(n_lon):
            phi0 = 2 * np.pi * j / n_lon
            phi1 = 2 * np.pi * (j + 1) / n_lon

            p00 = [radius * np.sin(theta0) * np.cos(phi0),
                   radius * np.sin(theta0) * np.sin(phi0),
                   radius * np.cos(theta0)]
            p10 = [radius * np.sin(theta1) * np.cos(phi0),
                   radius * np.sin(theta1) * np.sin(phi0),
                   radius * np.cos(theta1)]
            p01 = [radius * np.sin(theta0) * np.cos(phi1),
                   radius * np.sin(theta0) * np.sin(phi1),
                   radius * np.cos(theta0)]
            p11 = [radius * np.sin(theta1) * np.cos(phi1),
                   radius * np.sin(theta1) * np.sin(phi1),
                   radius * np.cos(theta1)]

            triangles.append([p00, p10, p11])
            triangles.append([p00, p11, p01])

    return np.array(triangles, dtype=np.float64)


class TestRayTriangleIntersect(unittest.TestCase):
    """Möller-Trumbore 射线-三角形求交测试"""

    def test_hit_front_face(self):
        """射线从正面穿过三角形"""
        p0 = np.array([0.0, 0.0, 0.0])
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        origin = np.array([0.25, 0.25, -1.0])
        ray_dir = np.array([0.0, 0.0, 1.0])

        t = ray_triangle_intersect(origin, ray_dir, p0, e1, e2)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 1.0, places=6)

    def test_hit_back_face(self):
        """射线从背面穿过三角形（t < 0）"""
        p0 = np.array([0.0, 0.0, 0.0])
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        origin = np.array([0.25, 0.25, 1.0])
        ray_dir = np.array([0.0, 0.0, 1.0])

        t = ray_triangle_intersect(origin, ray_dir, p0, e1, e2)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, -1.0, places=6)

    def test_miss_parallel(self):
        """射线平行于三角形"""
        p0 = np.array([0.0, 0.0, 0.0])
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        origin = np.array([0.5, 0.5, 0.0])
        ray_dir = np.array([1.0, 0.0, 0.0])

        t = ray_triangle_intersect(origin, ray_dir, p0, e1, e2)
        self.assertIsNone(t)

    def test_miss_outside_edge(self):
        """射线穿过三角形外部"""
        p0 = np.array([0.0, 0.0, 0.0])
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        origin = np.array([2.0, 2.0, -1.0])
        ray_dir = np.array([0.0, 0.0, 1.0])

        t = ray_triangle_intersect(origin, ray_dir, p0, e1, e2)
        self.assertIsNone(t)

    def test_hit_vertex(self):
        """射线穿过三角形顶点"""
        p0 = np.array([0.0, 0.0, 0.0])
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        origin = np.array([0.0, 0.0, -1.0])
        ray_dir = np.array([0.0, 0.0, 1.0])

        t = ray_triangle_intersect(origin, ray_dir, p0, e1, e2)
        # 顶点命中可能因浮点精度返回 None 或 t
        # 只要不崩溃即可
        if t is not None:
            self.assertAlmostEqual(t, 1.0, places=4)

    def test_hit_center(self):
        """射线穿过三角形中心"""
        p0 = np.array([0.0, 0.0, 0.0])
        e1 = np.array([2.0, 0.0, 0.0])
        e2 = np.array([0.0, 2.0, 0.0])
        origin = np.array([0.5, 0.5, -2.0])
        ray_dir = np.array([0.0, 0.0, 1.0])

        t = ray_triangle_intersect(origin, ray_dir, p0, e1, e2)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 2.0, places=6)

    def test_non_unit_direction(self):
        """非单位方向向量"""
        p0 = np.array([0.0, 0.0, 0.0])
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        origin = np.array([0.25, 0.25, -2.0])
        ray_dir = np.array([0.0, 0.0, 3.0])

        t = ray_triangle_intersect(origin, ray_dir, p0, e1, e2)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 2.0 / 3.0, places=6)


class TestBuildSpatialGrid(unittest.TestCase):
    """空间网格构建测试"""

    def test_empty_triangles(self):
        """空三角形列表"""
        result = build_spatial_grid(np.empty((0, 3, 3)))
        self.assertIsNone(result)

    def test_single_triangle(self):
        """单个三角形"""
        tv = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
        result = build_spatial_grid(tv)
        self.assertIsNotNone(result)
        self.assertEqual(result['tri_verts'].shape, (1, 3, 3))
        self.assertEqual(result['edge1'].shape, (1, 3))
        self.assertEqual(result['edge2'].shape, (1, 3))
        self.assertGreater(len(result['grid']), 0)

    def test_cube_grid_coverage(self):
        """立方体表面的网格覆盖"""
        tv = make_cube_triangles(2.0)
        result = build_spatial_grid(tv)
        self.assertIsNotNone(result)
        # 所有三角形都应该被至少一个格子包含
        all_indices = set()
        for indices in result['grid'].values():
            all_indices.update(indices)
        self.assertEqual(len(all_indices), len(tv))

    def test_grid_resolution(self):
        """自定义网格分辨率"""
        tv = make_cube_triangles(2.0)
        result = build_spatial_grid(tv, grid_n=8)
        self.assertEqual(result['grid_n'], 8)


class TestCountRayIntersections(unittest.TestCase):
    """射线交点计数测试"""

    def test_inside_cube_x(self):
        """立方体内部点，沿 x 轴射线"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        origin = np.array([1.0, 1.0, 1.0])
        n = count_ray_intersections(origin, np.array([1.0, 0.0, 0.0]), gd)
        self.assertEqual(n, 1, "从内部沿+x应穿过1个面")

    def test_outside_cube_x(self):
        """立方体外部点，沿 x 轴射线"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        origin = np.array([3.0, 1.0, 1.0])
        n = count_ray_intersections(origin, np.array([1.0, 0.0, 0.0]), gd)
        self.assertEqual(n, 0, "从外部沿+x不应穿过任何面")

    def test_inside_cube_y(self):
        """立方体内部点，沿 y 轴射线"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        origin = np.array([1.0, 1.0, 1.0])
        n = count_ray_intersections(origin, np.array([0.0, 1.0, 0.0]), gd)
        self.assertEqual(n, 1)

    def test_inside_cube_z(self):
        """立方体内部点，沿 z 轴射线"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        origin = np.array([1.0, 1.0, 1.0])
        n = count_ray_intersections(origin, np.array([0.0, 0.0, 1.0]), gd)
        self.assertEqual(n, 1)

    def test_on_surface(self):
        """点在表面上（容差内）"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        origin = np.array([0.0, 1.0, 1.0])
        n = count_ray_intersections(origin, np.array([1.0, 0.0, 0.0]), gd)
        # 在表面上的行为可能因浮点精度而异
        self.assertIn(n, [0, 1, 2])


class TestClassifyPointsInside(unittest.TestCase):
    """批量点分类测试"""

    def test_cube_center_inside(self):
        """立方体中心点应在内部"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        pts = np.array([[1.0, 1.0, 1.0]])
        result = classify_points_inside(pts, gd)
        self.assertTrue(result[0], "立方体中心应在内部")

    def test_cube_corner_inside(self):
        """立方体角落内部点"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        pts = np.array([[0.5, 0.5, 0.5]])
        result = classify_points_inside(pts, gd)
        self.assertTrue(result[0], "立方体角落内部点应在内部")

    def test_cube_outside(self):
        """立方体外部点"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        pts = np.array([[3.0, 3.0, 3.0]])
        result = classify_points_inside(pts, gd)
        self.assertFalse(result[0], "立方体外部点应在外部")

    def test_cube_negative_outside(self):
        """立方体负方向外部点"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        pts = np.array([[-1.0, -1.0, -1.0]])
        result = classify_points_inside(pts, gd)
        self.assertFalse(result[0])

    def test_multiple_points(self):
        """批量测试多个点"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        pts = np.array([
            [1.0, 1.0, 1.0],   # 内部
            [3.0, 3.0, 3.0],   # 外部
            [0.5, 0.5, 0.5],   # 内部
            [-1.0, 0.0, 0.0],  # 外部
        ])
        result = classify_points_inside(pts, gd)
        self.assertTrue(result[0])
        self.assertFalse(result[1])
        self.assertTrue(result[2])
        self.assertFalse(result[3])

    def test_sphere_center_inside(self):
        """球体中心点应在内部"""
        tv = make_sphere_triangles(radius=1.0, n_lat=8, n_lon=16)
        gd = build_spatial_grid(tv)
        pts = np.array([[0.0, 0.0, 0.0]])
        result = classify_points_inside(pts, gd)
        self.assertTrue(result[0], "球心应在内部")

    def test_sphere_inside(self):
        """球体内部点"""
        tv = make_sphere_triangles(radius=1.0, n_lat=8, n_lon=16)
        gd = build_spatial_grid(tv)
        pts = np.array([[0.5, 0.0, 0.0]])
        result = classify_points_inside(pts, gd)
        self.assertTrue(result[0], "球内部点应在内部")

    def test_sphere_outside(self):
        """球体外部点"""
        tv = make_sphere_triangles(radius=1.0, n_lat=8, n_lon=16)
        gd = build_spatial_grid(tv)
        pts = np.array([[2.0, 0.0, 0.0]])
        result = classify_points_inside(pts, gd)
        self.assertFalse(result[0], "球外部点应在外部")

    def test_sphere_far_outside(self):
        """球体远外部点"""
        tv = make_sphere_triangles(radius=1.0, n_lat=8, n_lon=16)
        gd = build_spatial_grid(tv)
        pts = np.array([[5.0, 5.0, 5.0]])
        result = classify_points_inside(pts, gd)
        self.assertFalse(result[0])

    def test_none_grid_data(self):
        """空网格数据"""
        pts = np.array([[1.0, 1.0, 1.0]])
        result = classify_points_inside(pts, None)
        self.assertTrue(result[0], "无网格数据时默认返回内部")

    def test_single_point_1d(self):
        """单点（1D 输入）"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)
        result = classify_points_inside(np.array([1.0, 1.0, 1.0]), gd)
        self.assertTrue(result[0])


class TestCubeIntegration(unittest.TestCase):
    """立方体端到端集成测试"""

    def test_grid_of_points(self):
        """立方体内部网格点全在内部"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)

        # 在 [0.25, 1.75]^3 内生成网格
        xs = np.linspace(0.25, 1.75, 4)
        pts = np.array([[x, y, z] for x in xs for y in xs for z in xs])
        result = classify_points_inside(pts, gd)

        # 所有点都应在内部
        self.assertTrue(np.all(result),
                        f"内部网格点应全在内部，但有 {np.sum(~result)} 个被判定为外部")

    def test_outside_shell(self):
        """立方体外部壳层点全在外部"""
        tv = make_cube_triangles(2.0)
        gd = build_spatial_grid(tv)

        # 在 [2.5, 3.5]^3 内生成网格
        xs = np.linspace(2.5, 3.5, 3)
        pts = np.array([[x, y, z] for x in xs for y in xs for z in xs])
        result = classify_points_inside(pts, gd)

        self.assertTrue(not np.any(result),
                        f"外部壳层点应全在外部，但有 {np.sum(result)} 个被判定为内部")


if __name__ == '__main__':
    unittest.main()

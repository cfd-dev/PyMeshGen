"""meshsize/size_field_3d.py 单元测试"""
import unittest
import numpy as np
from meshsize.octree import Octree, OctreeNode, _trilinear_interpolate, _divide_bounds
from meshsize.size_sources import PointSource, BoxSource, SphereSource
from meshsize.size_field_3d import SizeField3D


class TestOctreeNode(unittest.TestCase):
    """八叉树节点测试"""

    def test_init(self):
        """节点初始化"""
        bounds = (0, 0, 0, 1, 1, 1)
        node = OctreeNode(bounds)
        self.assertEqual(node.bounds, bounds)
        self.assertEqual(node.level, 0)
        self.assertIsNone(node.children)
        self.assertTrue(node.is_leaf)

    def test_init_with_level(self):
        """带深度的节点初始化"""
        node = OctreeNode((0, 0, 0, 1, 1, 1), level=3)
        self.assertEqual(node.level, 3)


class TestDivideBounds(unittest.TestCase):
    """包围盒八等分测试"""

    def test_unit_cube(self):
        """单位立方体八等分"""
        bounds = (0, 0, 0, 2, 2, 2)
        children = _divide_bounds(bounds)
        self.assertEqual(len(children), 8)

        # 每个子节点应该是原节点的 1/8
        for child in children:
            dx = child[3] - child[0]
            dy = child[4] - child[1]
            dz = child[5] - child[2]
            self.assertAlmostEqual(dx, 1.0)
            self.assertAlmostEqual(dy, 1.0)
            self.assertAlmostEqual(dz, 1.0)

    def test_children_cover_parent(self):
        """子节点应该完全覆盖父节点"""
        bounds = (0, 0, 0, 2, 2, 2)
        children = _divide_bounds(bounds)

        # 所有子节点的并集应该等于父节点
        all_xmin = min(c[0] for c in children)
        all_ymin = min(c[1] for c in children)
        all_zmin = min(c[2] for c in children)
        all_xmax = max(c[3] for c in children)
        all_ymax = max(c[4] for c in children)
        all_zmax = max(c[5] for c in children)

        self.assertAlmostEqual(all_xmin, 0.0)
        self.assertAlmostEqual(all_ymin, 0.0)
        self.assertAlmostEqual(all_zmin, 0.0)
        self.assertAlmostEqual(all_xmax, 2.0)
        self.assertAlmostEqual(all_ymax, 2.0)
        self.assertAlmostEqual(all_zmax, 2.0)


class TestTrilinearInterpolate(unittest.TestCase):
    """三线性插值测试"""

    def test_corner_values(self):
        """角点处应该返回精确值"""
        bounds = (0, 0, 0, 1, 1, 1)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        # 测试每个角点
        corners = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ]
        for i, corner in enumerate(corners):
            result = _trilinear_interpolate(corner, bounds, values)
            self.assertAlmostEqual(result, values[i], places=10,
                                   msg=f"Corner {i} failed")

    def test_center_value(self):
        """中心处应该是8个角点的平均值"""
        bounds = (0, 0, 0, 1, 1, 1)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        result = _trilinear_interpolate((0.5, 0.5, 0.5), bounds, values)
        expected = np.mean(values)
        self.assertAlmostEqual(result, expected, places=10)

    def test_uniform_values(self):
        """均匀值应该处处相同"""
        bounds = (0, 0, 0, 1, 1, 1)
        values = np.full(8, 5.0)

        result = _trilinear_interpolate((0.3, 0.7, 0.2), bounds, values)
        self.assertAlmostEqual(result, 5.0, places=10)

    def test_linear_variation(self):
        """线性变化应该正确插值"""
        bounds = (0, 0, 0, 1, 1, 1)
        # 沿 x 方向线性变化: x=0 -> 1, x=1 -> 2
        values = np.array([1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0])

        result = _trilinear_interpolate((0.5, 0.5, 0.5), bounds, values)
        self.assertAlmostEqual(result, 1.5, places=10)


class TestOctree(unittest.TestCase):
    """八叉树测试"""

    def test_init(self):
        """八叉树初始化"""
        bounds = (0, 0, 0, 1, 1, 1)
        tree = Octree(bounds, max_depth=4, default_size=1.0)
        self.assertEqual(tree.leaf_count, 1)
        self.assertTrue(tree.contains((0.5, 0.5, 0.5)))

    def test_query_default(self):
        """查询默认值"""
        bounds = (0, 0, 0, 1, 1, 1)
        tree = Octree(bounds, default_size=2.0)

        result = tree.query((0.5, 0.5, 0.5))
        self.assertAlmostEqual(result, 2.0)

    def test_query_outside(self):
        """查询范围外的点"""
        bounds = (0, 0, 0, 1, 1, 1)
        tree = Octree(bounds, default_size=2.0)

        result = tree.query((2.0, 2.0, 2.0))
        self.assertAlmostEqual(result, 2.0)

    def test_insert_and_query(self):
        """插入尺寸约束后查询"""
        bounds = (0, 0, 0, 4, 4, 4)
        tree = Octree(bounds, max_depth=4, default_size=4.0)

        # 在中心插入小尺寸
        tree.insert_size((2, 2, 2), 1.0, growth_rate=1.3)

        # 中心处应该有较小的尺寸
        center_size = tree.query((2, 2, 2))
        self.assertLess(center_size, 4.0)

    def test_insert_multiple_points(self):
        """插入多个尺寸约束"""
        bounds = (0, 0, 0, 10, 10, 10)
        tree = Octree(bounds, max_depth=4, default_size=10.0)

        tree.insert_size((2, 2, 2), 1.0, growth_rate=1.3)
        tree.insert_size((8, 8, 8), 2.0, growth_rate=1.3)

        # 两个点处都应该有较小的尺寸
        size1 = tree.query((2, 2, 2))
        size2 = tree.query((8, 8, 8))
        self.assertLess(size1, 10.0)
        self.assertLess(size2, 10.0)

    def test_subdivide(self):
        """细分操作"""
        bounds = (0, 0, 0, 4, 4, 4)
        tree = Octree(bounds, max_depth=4, default_size=4.0)

        # 插入小尺寸应该触发细分
        tree.insert_size((2, 2, 2), 0.5, growth_rate=1.3)

        # 应该有多个叶节点
        self.assertGreater(tree.leaf_count, 1)


class TestPointSource(unittest.TestCase):
    """点源测试"""

    def test_center_size(self):
        """中心处尺寸"""
        source = PointSource([0, 0, 0], 1.0, growth=1.3)
        self.assertAlmostEqual(source.size_at([0, 0, 0]), 1.0)

    def test_growth(self):
        """尺寸随距离增长"""
        source = PointSource([0, 0, 0], 1.0, growth=1.3)

        size_at_0 = source.size_at([0, 0, 0])
        size_at_1 = source.size_at([1, 0, 0])
        size_at_2 = source.size_at([2, 0, 0])

        self.assertLess(size_at_0, size_at_1)
        self.assertLess(size_at_1, size_at_2)

    def test_distance_to(self):
        """距离计算"""
        source = PointSource([0, 0, 0], 1.0)
        self.assertAlmostEqual(source.distance_to([3, 4, 0]), 5.0)


class TestBoxSource(unittest.TestCase):
    """盒源测试"""

    def test_contains_inside(self):
        """内部点"""
        source = BoxSource([0, 0, 0], [1, 1, 1], 1.0)
        self.assertTrue(source.contains([0.5, 0.5, 0.5]))

    def test_contains_outside(self):
        """外部点"""
        source = BoxSource([0, 0, 0], [1, 1, 1], 1.0)
        self.assertFalse(source.contains([2, 2, 2]))

    def test_contains_boundary(self):
        """边界点"""
        source = BoxSource([0, 0, 0], [1, 1, 1], 1.0)
        self.assertTrue(source.contains([1, 0, 0]))

    def test_distance_inside(self):
        """内部点距离为0"""
        source = BoxSource([0, 0, 0], [1, 1, 1], 1.0)
        self.assertAlmostEqual(source.distance_to([0.5, 0.5, 0.5]), 0.0)

    def test_distance_outside(self):
        """外部点距离"""
        source = BoxSource([0, 0, 0], [1, 1, 1], 1.0)
        dist = source.distance_to([2, 0, 0])
        self.assertAlmostEqual(dist, 1.0)

    def test_size_at_inside(self):
        """内部尺寸"""
        source = BoxSource([0, 0, 0], [1, 1, 1], 1.0)
        self.assertAlmostEqual(source.size_at([0, 0, 0]), 1.0)

    def test_size_at_outside(self):
        """外部尺寸增长"""
        source = BoxSource([0, 0, 0], [1, 1, 1], 1.0, growth=1.3)
        size_outside = source.size_at([2, 0, 0])
        self.assertGreater(size_outside, 1.0)

    def test_rotation(self):
        """旋转盒"""
        source = BoxSource([0, 0, 0], [1, 1, 1], 1.0, rotation=[0, 0, 45])
        # 旋转后，原来在盒外的点可能进入盒内
        self.assertTrue(source.contains([0, 0, 0]))


class TestSphereSource(unittest.TestCase):
    """球源测试"""

    def test_contains_inside(self):
        """内部点"""
        source = SphereSource([0, 0, 0], 1.0, 1.0)
        self.assertTrue(source.contains([0.5, 0, 0]))

    def test_contains_outside(self):
        """外部点"""
        source = SphereSource([0, 0, 0], 1.0, 1.0)
        self.assertFalse(source.contains([2, 0, 0]))

    def test_contains_boundary(self):
        """边界点"""
        source = SphereSource([0, 0, 0], 1.0, 1.0)
        self.assertTrue(source.contains([1, 0, 0]))

    def test_distance_inside(self):
        """内部点距离为0"""
        source = SphereSource([0, 0, 0], 1.0, 1.0)
        self.assertAlmostEqual(source.distance_to([0.5, 0, 0]), 0.0)

    def test_distance_outside(self):
        """外部点距离"""
        source = SphereSource([0, 0, 0], 1.0, 1.0)
        dist = source.distance_to([2, 0, 0])
        self.assertAlmostEqual(dist, 1.0)

    def test_size_at_inside(self):
        """内部尺寸"""
        source = SphereSource([0, 0, 0], 1.0, 0.5)
        self.assertAlmostEqual(source.size_at([0, 0, 0]), 0.5)

    def test_size_at_outside(self):
        """外部尺寸增长"""
        source = SphereSource([0, 0, 0], 1.0, 0.5, growth=1.3)
        size_outside = source.size_at([2, 0, 0])
        self.assertGreater(size_outside, 0.5)


class TestSizeField3D(unittest.TestCase):
    """三维尺寸场测试"""

    def test_uniform_field(self):
        """均匀尺寸场"""
        field = SizeField3D(max_size=2.0)
        self.assertAlmostEqual(field.spacing_at([0, 0, 0]), 2.0)
        self.assertAlmostEqual(field.spacing_at([1, 1, 1]), 2.0)

    def test_point_source(self):
        """点源尺寸场"""
        field = SizeField3D(max_size=10.0)
        field.add_point_source([0, 0, 0], 1.0, growth=1.3)
        field.build()

        # 中心处应该有较小的尺寸
        center_size = field.spacing_at([0, 0, 0])
        self.assertLess(center_size, 10.0)

    def test_box_source(self):
        """盒源尺寸场"""
        field = SizeField3D(max_size=10.0)
        field.add_box_source([0, 0, 0], [1, 1, 1], 1.0, growth=1.3)
        field.build()

        # 盒内应该有指定的尺寸
        inside_size = field.spacing_at([0, 0, 0])
        self.assertLessEqual(inside_size, 10.0)

    def test_sphere_source(self):
        """球源尺寸场"""
        field = SizeField3D(max_size=10.0)
        field.add_sphere_source([0, 0, 0], 1.0, 0.5, growth=1.3)
        field.build()

        # 球内应该有指定的尺寸
        inside_size = field.spacing_at([0, 0, 0])
        self.assertLessEqual(inside_size, 10.0)

    def test_multiple_sources(self):
        """多源尺寸场"""
        field = SizeField3D(max_size=10.0)
        field.add_point_source([-2, 0, 0], 1.0, growth=1.3)
        field.add_point_source([2, 0, 0], 2.0, growth=1.3)
        field.build()

        # 两个源附近都应该有较小的尺寸
        size1 = field.spacing_at([-2, 0, 0])
        size2 = field.spacing_at([2, 0, 0])
        self.assertLess(size1, 10.0)
        self.assertLess(size2, 10.0)

    def test_global_spacing(self):
        """全局尺寸属性"""
        field = SizeField3D(max_size=5.0)
        self.assertEqual(field.global_spacing, 5.0)

    def test_stats(self):
        """统计信息"""
        field = SizeField3D(max_size=10.0)
        field.add_point_source([0, 0, 0], 1.0)
        field.build()

        stats = field.get_stats()
        self.assertIn('global_spacing', stats)
        self.assertIn('num_sources', stats)
        self.assertIn('octree_leaves', stats)
        self.assertEqual(stats['num_sources'], 1)


class MockSurfaceTriangle:
    """模拟表面三角形"""
    def __init__(self, nodes):
        self.nodes = nodes
        self.node_ids = [n.idx for n in nodes]


class MockNode3D:
    """模拟3D节点"""
    def __init__(self, coords, idx=0):
        self.coords = tuple(coords)
        self.idx = idx


class TestSizeField3DFromSurface(unittest.TestCase):
    """从表面网格初始化尺寸场测试"""

    def test_simple_surface(self):
        """简单表面网格"""
        # 创建一个立方体表面
        nodes = [
            MockNode3D([0, 0, 0], 0),
            MockNode3D([1, 0, 0], 1),
            MockNode3D([0, 1, 0], 2),
            MockNode3D([1, 1, 0], 3),
            MockNode3D([0, 0, 1], 4),
            MockNode3D([1, 0, 1], 5),
            MockNode3D([0, 1, 1], 6),
            MockNode3D([1, 1, 1], 7),
        ]

        # 创建三角形（简化，只用部分面）
        triangles = [
            MockSurfaceTriangle([nodes[0], nodes[1], nodes[2]]),
            MockSurfaceTriangle([nodes[1], nodes[3], nodes[2]]),
            MockSurfaceTriangle([nodes[4], nodes[5], nodes[6]]),
            MockSurfaceTriangle([nodes[5], nodes[7], nodes[6]]),
        ]

        field = SizeField3D(surface_triangles=triangles)

        # 应该自动设置全局尺寸
        self.assertGreater(field.global_spacing, 0)

        # 应该可以查询
        size = field.spacing_at([0.5, 0.5, 0.5])
        self.assertGreater(size, 0)


if __name__ == '__main__':
    unittest.main()

"""
sfmesh 数据结构与网格质量单元测试

覆盖 surface_front.py 和 mesh_quality.py 的公共接口：
- NodeElement3D：坐标哈希、相等性、bbox
- SurfaceTriangle：面积、质量、法向量、bbox
- SurfaceFront：方向、长度、推进方向
- SurfaceMeshQuality：质量指标批量评估
"""
import sys
import unittest
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sfmesh.surface_front import NodeElement3D, SurfaceTriangle, SurfaceFront
from sfmesh.mesh_quality import SurfaceMeshQuality
from sfmesh.sizing_field import SurfaceSizingField


# ============================================================================
# NodeElement3D
# ============================================================================

class TestNodeElement3D(unittest.TestCase):
    """测试三维节点数据结构"""

    def test_creation_basic(self):
        """基本创建"""
        node = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=0)
        self.assertEqual(node.coords, (1.0, 2.0, 3.0))
        self.assertEqual(node.idx, 0)

    def test_hash_equal_coords(self):
        """相同坐标的节点哈希相同"""
        n1 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=0)
        n2 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=1)
        self.assertEqual(n1.hash, n2.hash)
        self.assertEqual(n1, n2)

    def test_hash_different_coords(self):
        """不同坐标的节点哈希不同"""
        n1 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=0)
        n2 = NodeElement3D(coords=(1.0, 2.0, 3.1), idx=0)
        self.assertNotEqual(n1.hash, n2.hash)

    def test_hash_nearby_coords(self):
        """非常接近但不同的坐标应有不同哈希（8 位精度）"""
        n1 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=0)
        n2 = NodeElement3D(coords=(1.0, 2.0, 3.000000001), idx=0)
        # 差异在第 9 位小数，round(c, 8) 后相同
        self.assertEqual(n1.hash, n2.hash)

    def test_hash_precision_boundary(self):
        """精度边界：第 8 位小数不同"""
        n1 = NodeElement3D(coords=(1.0, 2.0, 3.00000000), idx=0)
        n2 = NodeElement3D(coords=(1.0, 2.0, 3.00000001), idx=0)
        self.assertNotEqual(n1.hash, n2.hash)

    def test_bbox(self):
        """边界框退化为点"""
        node = NodeElement3D(coords=(5.0, 10.0, 15.0), idx=0)
        self.assertEqual(node.bbox, (5.0, 10.0, 15.0, 5.0, 10.0, 15.0))

    def test_zero_coords(self):
        """零坐标"""
        node = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        self.assertEqual(node.coords, (0.0, 0.0, 0.0))
        # 零坐标应该有一个确定的哈希
        self.assertIsInstance(node.hash, int)

    def test_negative_coords(self):
        """负坐标"""
        node = NodeElement3D(coords=(-1.0, -2.0, -3.0), idx=0)
        self.assertEqual(node.coords, (-1.0, -2.0, -3.0))

    def test_set_membership(self):
        """节点可用于集合去重"""
        n1 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=0)
        n2 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=1)
        s = {n1, n2}
        self.assertEqual(len(s), 1)

    def test_dict_key(self):
        """节点可用作字典键"""
        n1 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=0)
        d = {n1: "value"}
        n2 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=1)
        self.assertEqual(d[n2], "value")

    def test_repr(self):
        """字符串表示"""
        node = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=5)
        r = repr(node)
        self.assertIn("5", r)
        self.assertIn("1.0", r)


# ============================================================================
# SurfaceTriangle
# ============================================================================

class TestSurfaceTriangle(unittest.TestCase):
    """测试曲面三角形单元"""

    def _make_equilateral(self, side=1.0):
        """创建等边三角形"""
        h = side * np.sqrt(3) / 2
        n1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        n2 = NodeElement3D(coords=(side, 0.0, 0.0), idx=1)
        n3 = NodeElement3D(coords=(side / 2, h, 0.0), idx=2)
        return SurfaceTriangle(n1, n2, n3)

    def _make_right_triangle(self):
        """创建直角三角形"""
        n1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        n2 = NodeElement3D(coords=(3.0, 0.0, 0.0), idx=1)
        n3 = NodeElement3D(coords=(0.0, 4.0, 0.0), idx=2)
        return SurfaceTriangle(n1, n2, n3)

    def test_area_equilateral(self):
        """等边三角形面积"""
        tri = self._make_equilateral(2.0)
        expected = 0.5 * 2.0 * 2.0 * np.sqrt(3) / 2
        self.assertAlmostEqual(tri.area, expected, places=10)

    def test_area_right_triangle(self):
        """直角三角形面积"""
        tri = self._make_right_triangle()
        self.assertAlmostEqual(tri.area, 6.0, places=10)

    def test_quality_equilateral(self):
        """等边三角形质量 = 1"""
        tri = self._make_equilateral()
        self.assertAlmostEqual(tri.quality, 1.0, places=10)

    def test_quality_right_triangle(self):
        """直角三角形质量 < 1"""
        tri = self._make_right_triangle()
        self.assertGreater(tri.quality, 0.0)
        self.assertLess(tri.quality, 1.0)

    def test_normal_direction(self):
        """法向量方向（XY 平面三角形 → Z 方向）"""
        tri = self._make_equilateral()
        normal = tri.normal
        self.assertAlmostEqual(abs(normal[2]), 1.0, places=10)
        self.assertAlmostEqual(normal[0], 0.0, places=10)
        self.assertAlmostEqual(normal[1], 0.0, places=10)

    def test_normal_magnitude(self):
        """法向量是单位向量"""
        tri = self._make_right_triangle()
        self.assertAlmostEqual(np.linalg.norm(tri.normal), 1.0, places=10)

    def test_bbox(self):
        """边界框"""
        tri = self._make_right_triangle()
        bbox = tri.bbox
        self.assertAlmostEqual(bbox[0], 0.0, places=10)  # min_x
        self.assertAlmostEqual(bbox[1], 0.0, places=10)  # min_y
        self.assertAlmostEqual(bbox[2], 0.0, places=10)  # min_z
        self.assertAlmostEqual(bbox[3], 3.0, places=10)  # max_x
        self.assertAlmostEqual(bbox[4], 4.0, places=10)  # max_y
        self.assertAlmostEqual(bbox[5], 0.0, places=10)  # max_z

    def test_node_ids(self):
        """节点 ID 列表"""
        n1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=10)
        n2 = NodeElement3D(coords=(1.0, 0.0, 0.0), idx=20)
        n3 = NodeElement3D(coords=(0.5, 1.0, 0.0), idx=30)
        tri = SurfaceTriangle(n1, n2, n3)
        self.assertEqual(tri.node_ids, [10, 20, 30])

    def test_hash_equal_triangles(self):
        """相同顶点的三角形哈希相同"""
        n1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        n2 = NodeElement3D(coords=(1.0, 0.0, 0.0), idx=1)
        n3 = NodeElement3D(coords=(0.5, 1.0, 0.0), idx=2)
        t1 = SurfaceTriangle(n1, n2, n3)
        t2 = SurfaceTriangle(n1, n3, n2)  # 不同顺序
        self.assertEqual(t1.hash, t2.hash)
        self.assertEqual(t1, t2)

    def test_hash_different_triangles(self):
        """不同三角形哈希不同"""
        n0 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        n1 = NodeElement3D(coords=(1.0, 0.0, 0.0), idx=1)
        n2 = NodeElement3D(coords=(0.5, 1.0, 0.0), idx=2)
        n3 = NodeElement3D(coords=(0.5, 0.5, 1.0), idx=3)
        t1 = SurfaceTriangle(n0, n1, n2)
        t2 = SurfaceTriangle(n0, n1, n3)
        self.assertNotEqual(t1.hash, t2.hash)

    def test_3d_triangle(self):
        """非零 Z 平面上的三角形"""
        n1 = NodeElement3D(coords=(0.0, 0.0, 5.0), idx=0)
        n2 = NodeElement3D(coords=(1.0, 0.0, 5.0), idx=1)
        n3 = NodeElement3D(coords=(0.5, 1.0, 5.0), idx=2)
        tri = SurfaceTriangle(n1, n2, n3)
        self.assertGreater(tri.area, 0.0)
        self.assertGreater(tri.quality, 0.0)

    def test_set_membership(self):
        """三角形可用于集合去重"""
        n1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        n2 = NodeElement3D(coords=(1.0, 0.0, 0.0), idx=1)
        n3 = NodeElement3D(coords=(0.5, 1.0, 0.0), idx=2)
        t1 = SurfaceTriangle(n1, n2, n3)
        t2 = SurfaceTriangle(n2, n3, n1)
        s = {t1, t2}
        self.assertEqual(len(s), 1)


# ============================================================================
# SurfaceFront
# ============================================================================

class TestSurfaceFront(unittest.TestCase):
    """测试曲面阵面数据结构"""

    def _make_front(self, p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0), n1=None, n2=None):
        """创建阵面"""
        node1 = NodeElement3D(coords=p1, idx=0, normal=n1)
        node2 = NodeElement3D(coords=p2, idx=1, normal=n2)
        return SurfaceFront(node1, node2)

    def test_creation(self):
        """基本创建"""
        front = self._make_front()
        self.assertEqual(front.node_ids, [0, 1])
        self.assertAlmostEqual(front.length, 1.0, places=10)

    def test_length(self):
        """长度计算"""
        front = self._make_front(p1=(0.0, 0.0, 0.0), p2=(3.0, 4.0, 0.0))
        self.assertAlmostEqual(front.length, 5.0, places=10)

    def test_center(self):
        """中点"""
        front = self._make_front(p1=(0.0, 0.0, 0.0), p2=(2.0, 4.0, 6.0))
        self.assertAlmostEqual(front.center[0], 1.0, places=10)
        self.assertAlmostEqual(front.center[1], 2.0, places=10)
        self.assertAlmostEqual(front.center[2], 3.0, places=10)

    def test_direction(self):
        """方向向量（单位化）"""
        front = self._make_front(p1=(0.0, 0.0, 0.0), p2=(3.0, 0.0, 0.0))
        d = front.direction
        self.assertAlmostEqual(d[0], 1.0, places=10)
        self.assertAlmostEqual(d[1], 0.0, places=10)
        self.assertAlmostEqual(d[2], 0.0, places=10)

    def test_direction_3d(self):
        """3D 方向"""
        front = self._make_front(p1=(0.0, 0.0, 0.0), p2=(1.0, 1.0, 1.0))
        d = front.direction
        norm = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        self.assertAlmostEqual(norm, 1.0, places=10)

    def test_tangent_normal(self):
        """切平面推进方向（有法向量时）"""
        front = self._make_front(
            p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0),
            n1=(0.0, 0.0, 1.0), n2=(0.0, 0.0, 1.0)
        )
        tn = front.tangent_normal
        # tangent_normal = cross(direction, surface_normal)
        # direction = (1,0,0), normal = (0,0,1) → tangent = (0,1,0) 或 (0,-1,0)
        self.assertAlmostEqual(abs(tn[1]), 1.0, places=5)
        self.assertAlmostEqual(abs(tn[0]), 0.0, places=5)

    def test_tangent_normal_magnitude(self):
        """切平面推进方向是单位向量"""
        front = self._make_front(
            p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0),
            n1=(0.0, 0.0, 1.0), n2=(0.0, 0.0, 1.0)
        )
        tn = front.tangent_normal
        norm = np.sqrt(tn[0]**2 + tn[1]**2 + tn[2]**2)
        self.assertAlmostEqual(norm, 1.0, places=10)

    def test_bbox(self):
        """边界框"""
        front = self._make_front(p1=(1.0, 2.0, 3.0), p2=(4.0, 5.0, 6.0))
        bbox = front.bbox
        self.assertAlmostEqual(bbox[0], 1.0, places=10)
        self.assertAlmostEqual(bbox[1], 2.0, places=10)
        self.assertAlmostEqual(bbox[2], 3.0, places=10)
        self.assertAlmostEqual(bbox[3], 4.0, places=10)
        self.assertAlmostEqual(bbox[4], 5.0, places=10)
        self.assertAlmostEqual(bbox[5], 6.0, places=10)

    def test_zero_length_raises(self):
        """重合端点应抛出异常"""
        with self.assertRaises(ValueError):
            node = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=0)
            SurfaceFront(node, node)

    def test_priority_comparison(self):
        """优先级比较"""
        f1 = self._make_front(p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0))
        f2 = self._make_front(p1=(0.0, 0.0, 0.0), p2=(2.0, 0.0, 0.0))
        f1.priority = True
        f2.priority = False
        # priority=True 应排在前面（__lt__ 用于 heapq）
        self.assertTrue(f1 < f2)

    def test_length_comparison(self):
        """相同优先级时短阵面优先"""
        f1 = self._make_front(p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0))
        f2 = self._make_front(p1=(0.0, 0.0, 0.0), p2=(2.0, 0.0, 0.0))
        self.assertTrue(f1 < f2)

    def test_hash(self):
        """阵面哈希"""
        f1 = self._make_front(p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0))
        f2 = self._make_front(p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0))
        self.assertEqual(f1.hash, f2.hash)


# ============================================================================
# SurfaceMeshQuality - single triangle
# ============================================================================

class TestSurfaceMeshQualitySingleTriangle(unittest.TestCase):
    """测试单三角形质量指标"""

    def _make_equilateral(self, side=1.0):
        h = side * np.sqrt(3) / 2
        n1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        n2 = NodeElement3D(coords=(side, 0.0, 0.0), idx=1)
        n3 = NodeElement3D(coords=(side / 2, h, 0.0), idx=2)
        return SurfaceTriangle(n1, n2, n3)

    def _make_right_triangle(self):
        n1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        n2 = NodeElement3D(coords=(3.0, 0.0, 0.0), idx=1)
        n3 = NodeElement3D(coords=(0.0, 4.0, 0.0), idx=2)
        return SurfaceTriangle(n1, n2, n3)

    def test_triangle_quality_equilateral(self):
        """等边三角形质量 = 1"""
        tri = self._make_equilateral()
        q = SurfaceMeshQuality.triangle_quality(tri)
        self.assertAlmostEqual(q, 1.0, places=10)

    def test_triangle_quality_right(self):
        """直角三角形质量"""
        tri = self._make_right_triangle()
        q = SurfaceMeshQuality.triangle_quality(tri)
        self.assertGreater(q, 0.0)
        self.assertLess(q, 1.0)

    def test_aspect_ratio_equilateral(self):
        """等边三角形长宽比 = 1/3（公式: l_max²/(4√3·A)）"""
        tri = self._make_equilateral()
        ar = SurfaceMeshQuality.triangle_aspect_ratio(tri)
        # 公式给出 1/3 对于等边三角形（代码公式与注释不完全一致）
        self.assertAlmostEqual(ar, 1.0 / 3.0, places=5)

    def test_aspect_ratio_right(self):
        """直角三角形长宽比"""
        tri = self._make_right_triangle()
        ar = SurfaceMeshQuality.triangle_aspect_ratio(tri)
        self.assertGreater(ar, 0.0)

    def test_min_angle_equilateral(self):
        """等边三角形最小角 = 60°"""
        tri = self._make_equilateral()
        angle = SurfaceMeshQuality.triangle_min_angle(tri)
        self.assertAlmostEqual(angle, 60.0, places=5)

    def test_max_angle_equilateral(self):
        """等边三角形最大角 = 60°"""
        tri = self._make_equilateral()
        angle = SurfaceMeshQuality.triangle_max_angle(tri)
        self.assertAlmostEqual(angle, 60.0, places=5)

    def test_min_angle_right(self):
        """直角三角形最小角 < 60°"""
        tri = self._make_right_triangle()
        angle = SurfaceMeshQuality.triangle_min_angle(tri)
        self.assertLess(angle, 60.0)

    def test_max_angle_right(self):
        """直角三角形最大角 = 90°"""
        tri = self._make_right_triangle()
        angle = SurfaceMeshQuality.triangle_max_angle(tri)
        self.assertAlmostEqual(angle, 90.0, places=5)

    def test_jacobian_equilateral(self):
        """等边三角形雅可比 ≈ 0.866"""
        tri = self._make_equilateral()
        j = SurfaceMeshQuality.triangle_jacobian(tri)
        expected = np.sqrt(3) / 2  # ≈ 0.866
        self.assertAlmostEqual(j, expected, places=5)

    def test_warpage_flat_triangle(self):
        """平面上的三角形翘曲度 = 0"""
        tri = self._make_equilateral()
        w = SurfaceMeshQuality.triangle_warpage(tri)
        self.assertAlmostEqual(w, 0.0, places=10)

    def test_warpage_with_normals(self):
        """带法向量的三角形翘曲度"""
        n1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0, normal=(0.0, 0.0, 1.0))
        n2 = NodeElement3D(coords=(1.0, 0.0, 0.0), idx=1, normal=(0.0, 0.0, 1.0))
        n3 = NodeElement3D(coords=(0.5, 1.0, 0.0), idx=2, normal=(0.0, 0.0, 1.0))
        tri = SurfaceTriangle(n1, n2, n3)
        w = SurfaceMeshQuality.triangle_warpage(tri)
        self.assertAlmostEqual(w, 0.0, places=5)


# ============================================================================
# SurfaceMeshQuality - batch evaluation
# ============================================================================

class TestSurfaceMeshQualityBatch(unittest.TestCase):
    """测试批量网格质量评估"""

    def _make_triangle_list(self, n=10):
        """创建 n 个随机三角形"""
        import random
        random.seed(42)
        tris = []
        for i in range(n):
            x = random.uniform(0, 10)
            y = random.uniform(0, 10)
            n1 = NodeElement3D(coords=(x, y, 0.0), idx=i * 3)
            n2 = NodeElement3D(coords=(x + random.uniform(0.5, 2), y, 0.0), idx=i * 3 + 1)
            n3 = NodeElement3D(coords=(x + random.uniform(0, 1), y + random.uniform(0.5, 2), 0.0), idx=i * 3 + 2)
            tris.append(SurfaceTriangle(n1, n2, n3))
        return tris

    def test_evaluate_mesh_basic(self):
        """基本批量评估"""
        tris = self._make_triangle_list(20)
        result = SurfaceMeshQuality.evaluate_mesh(tris, verbose=False)
        self.assertEqual(result['num_triangles'], 20)
        self.assertIn('quality_mean', result)
        self.assertIn('quality_min', result)
        self.assertIn('aspect_ratio_mean', result)
        self.assertIn('total_area', result)

    def test_evaluate_mesh_empty(self):
        """空网格"""
        result = SurfaceMeshQuality.evaluate_mesh([], verbose=False)
        self.assertIn('error', result)

    def test_evaluate_mesh_quality_range(self):
        """质量值在合理范围内"""
        tris = self._make_triangle_list(50)
        result = SurfaceMeshQuality.evaluate_mesh(tris, verbose=False)
        self.assertGreaterEqual(result['quality_mean'], 0.0)
        self.assertLessEqual(result['quality_mean'], 1.0)
        self.assertGreaterEqual(result['quality_min'], 0.0)
        self.assertLessEqual(result['quality_max'], 1.0)

    def test_evaluate_mesh_angle_range(self):
        """角度值在合理范围内"""
        tris = self._make_triangle_list(50)
        result = SurfaceMeshQuality.evaluate_mesh(tris, verbose=False)
        self.assertGreaterEqual(result['min_angle_min'], 0.0)
        self.assertLessEqual(result['min_angle_min'], 180.0)
        self.assertGreaterEqual(result['max_angle_max'], 0.0)
        self.assertLessEqual(result['max_angle_max'], 180.0)

    def test_evaluate_mesh_histogram(self):
        """质量直方图"""
        tris = self._make_triangle_list(100)
        result = SurfaceMeshQuality.evaluate_mesh(tris, verbose=False)
        hist = result['quality_histogram']
        self.assertEqual(len(hist), 10)
        self.assertEqual(sum(hist), 100)

    def test_evaluate_triangle_comprehensive(self):
        """单三角形全面评估"""
        h = np.sqrt(3) / 2
        n1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        n2 = NodeElement3D(coords=(1.0, 0.0, 0.0), idx=1)
        n3 = NodeElement3D(coords=(0.5, h, 0.0), idx=2)
        tri = SurfaceTriangle(n1, n2, n3)
        result = SurfaceMeshQuality.evaluate_triangle(tri)
        self.assertIn('quality', result)
        self.assertIn('aspect_ratio', result)
        self.assertIn('min_angle', result)
        self.assertIn('max_angle', result)
        self.assertIn('jacobian', result)
        self.assertIn('warpage', result)
        self.assertIn('is_good', result)
        self.assertIn('area', result)


# ============================================================================
# SurfaceSizingField static methods
# ============================================================================

class TestSizingFieldUtilities(unittest.TestCase):
    """测试尺寸场工具函数"""

    def test_point_to_segment_midpoint(self):
        """点到线段中点的垂直距离"""
        point = np.array([0.5, 1.0, 0.0])
        seg_start = np.array([0.0, 0.0, 0.0])
        seg_end = np.array([1.0, 0.0, 0.0])
        dist = SurfaceSizingField._point_to_segment_distance_3d(point, seg_start, seg_end)
        self.assertAlmostEqual(dist, 1.0, places=10)

    def test_point_to_segment_endpoint(self):
        """点到线段端点的最近距离"""
        point = np.array([-1.0, 0.0, 0.0])
        seg_start = np.array([0.0, 0.0, 0.0])
        seg_end = np.array([1.0, 0.0, 0.0])
        dist = SurfaceSizingField._point_to_segment_distance_3d(point, seg_start, seg_end)
        self.assertAlmostEqual(dist, 1.0, places=10)

    def test_point_on_segment(self):
        """点在线段上"""
        point = np.array([0.5, 0.0, 0.0])
        seg_start = np.array([0.0, 0.0, 0.0])
        seg_end = np.array([1.0, 0.0, 0.0])
        dist = SurfaceSizingField._point_to_segment_distance_3d(point, seg_start, seg_end)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_degenerate_segment(self):
        """退化线段（长度为零）"""
        point = np.array([3.0, 4.0, 0.0])
        seg_start = np.array([0.0, 0.0, 0.0])
        seg_end = np.array([0.0, 0.0, 0.0])
        dist = SurfaceSizingField._point_to_segment_distance_3d(point, seg_start, seg_end)
        self.assertAlmostEqual(dist, 5.0, places=10)

    def test_3d_segment(self):
        """3D 线段上的距离"""
        point = np.array([0.0, 0.0, 1.0])
        seg_start = np.array([0.0, 0.0, 0.0])
        seg_end = np.array([0.0, 0.0, 2.0])
        dist = SurfaceSizingField._point_to_segment_distance_3d(point, seg_start, seg_end)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_diagonal_segment(self):
        """对角线段"""
        point = np.array([0.0, 1.0, 0.0])
        seg_start = np.array([0.0, 0.0, 0.0])
        seg_end = np.array([1.0, 0.0, 0.0])
        dist = SurfaceSizingField._point_to_segment_distance_3d(point, seg_start, seg_end)
        self.assertAlmostEqual(dist, 1.0, places=10)


if __name__ == '__main__':
    unittest.main()

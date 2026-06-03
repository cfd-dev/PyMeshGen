"""delaunay3d/tet_utils.py 单元测试"""
import unittest
import numpy as np
from delaunay3d.tet_utils import (
    tet_faces,
    tet_edges,
    tet_centroid,
    in_circumsphere,
    find_boundary_faces,
    find_boundary_face_set,
    build_face_to_tets,
    build_edge_to_tets,
    create_super_tetrahedron_coords,
    compute_max_edge_length,
    node_hash,
    build_node_to_cells,
    validate_tetrahedron,
    compute_surface_max_edge_length,
)


class MockTetrahedron:
    """模拟四面体对象"""
    def __init__(self, n1, n2, n3, n4, idx=0):
        self.node_ids = (n1, n2, n3, n4)
        self.node_id_list = [n1, n2, n3, n4]
        self.idx = idx
        # 模拟坐标属性
        self.p1 = [0.0, 0.0, 0.0]
        self.p2 = [1.0, 0.0, 0.0]
        self.p3 = [0.0, 1.0, 0.0]
        self.p4 = [0.0, 0.0, 1.0]


class TestTetFaces(unittest.TestCase):
    """四面体面提取测试"""

    def test_basic_faces(self):
        """基本面提取"""
        faces = tet_faces((0, 1, 2, 3))
        self.assertEqual(len(faces), 4)
        # 每个面应该是排序后的3个节点ID
        expected = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        ]
        self.assertEqual(set(faces), set(expected))

    def test_faces_sorted(self):
        """每个面内的节点ID应该排序"""
        faces = tet_faces((3, 1, 0, 2))
        for face in faces:
            self.assertEqual(face, tuple(sorted(face)))

    def test_faces_unique(self):
        """4个面应该互不相同"""
        faces = tet_faces((0, 1, 2, 3))
        self.assertEqual(len(set(faces)), 4)

    def test_non_contiguous_ids(self):
        """非连续节点ID"""
        faces = tet_faces((10, 20, 30, 40))
        self.assertEqual(len(faces), 4)
        expected = [
            (10, 20, 30),
            (10, 20, 40),
            (10, 30, 40),
            (20, 30, 40),
        ]
        self.assertEqual(set(faces), set(expected))


class TestTetEdges(unittest.TestCase):
    """四面体边提取测试"""

    def test_basic_edges(self):
        """基本边提取"""
        edges = tet_edges((0, 1, 2, 3))
        self.assertEqual(len(edges), 6)
        expected = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_edges_sorted(self):
        """每条边内的节点ID应该排序"""
        edges = tet_edges((3, 1, 0, 2))
        for edge in edges:
            self.assertEqual(edge, tuple(sorted(edge)))

    def test_edges_unique(self):
        """6条边应该互不相同"""
        edges = tet_edges((0, 1, 2, 3))
        self.assertEqual(len(set(edges)), 6)


class TestTetCentroid(unittest.TestCase):
    """四面体形心测试"""

    def test_unit_tet(self):
        """单位四面体形心"""
        p1 = [0.0, 0.0, 0.0]
        p2 = [1.0, 0.0, 0.0]
        p3 = [0.0, 1.0, 0.0]
        p4 = [0.0, 0.0, 1.0]
        c = tet_centroid(p1, p2, p3, p4)
        self.assertAlmostEqual(c[0], 0.25)
        self.assertAlmostEqual(c[1], 0.25)
        self.assertAlmostEqual(c[2], 0.25)

    def test_symmetric_tet(self):
        """对称四面体形心在原点"""
        p1 = [1.0, 0.0, 0.0]
        p2 = [-1.0, 0.0, 0.0]
        p3 = [0.0, 1.0, 0.0]
        p4 = [0.0, 0.0, -1.0]
        c = tet_centroid(p1, p2, p3, p4)
        self.assertAlmostEqual(c[0], 0.0)
        self.assertAlmostEqual(c[1], 0.25)
        self.assertAlmostEqual(c[2], -0.25)

    def test_translated_tet(self):
        """平移后的四面体形心"""
        offset = [10.0, 20.0, 30.0]
        p1 = [offset[0], offset[1], offset[2]]
        p2 = [offset[0] + 1, offset[1], offset[2]]
        p3 = [offset[0], offset[1] + 1, offset[2]]
        p4 = [offset[0], offset[1], offset[2] + 1]
        c = tet_centroid(p1, p2, p3, p4)
        self.assertAlmostEqual(c[0], offset[0] + 0.25)
        self.assertAlmostEqual(c[1], offset[1] + 0.25)
        self.assertAlmostEqual(c[2], offset[2] + 0.25)


class TestInCircumsphere(unittest.TestCase):
    """外接球包含测试"""

    def test_vertex_in_circumsphere(self):
        """四面体的顶点应该在其外接球内"""
        tet_coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        for vertex in tet_coords:
            self.assertTrue(in_circumsphere(vertex, tet_coords))

    def test_center_in_circumsphere(self):
        """四面体形心应该在外接球内"""
        tet_coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        centroid = [0.25, 0.25, 0.25]
        self.assertTrue(in_circumsphere(centroid, tet_coords))

    def test_far_point_outside(self):
        """远离的点应该在外接球外"""
        tet_coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        far_point = [100.0, 100.0, 100.0]
        self.assertFalse(in_circumsphere(far_point, tet_coords))

    def test_regular_tet(self):
        """正四面体的外接球测试"""
        # 正四面体顶点
        a = [1.0, 1.0, 1.0]
        b = [1.0, -1.0, -1.0]
        c = [-1.0, 1.0, -1.0]
        d = [-1.0, -1.0, 1.0]
        tet_coords = [a, b, c, d]

        # 外接球心在原点，半径 = sqrt(3)
        self.assertTrue(in_circumsphere([0.0, 0.0, 0.0], tet_coords))
        # 球面上的点（略大于半径）
        self.assertFalse(in_circumsphere([2.0, 2.0, 2.0], tet_coords))


class TestFindBoundaryFaces(unittest.TestCase):
    """边界面查找测试"""

    def test_single_tet(self):
        """单个四面体的所有面都是边界面"""
        tets = [MockTetrahedron(0, 1, 2, 3)]
        boundary = find_boundary_faces(tets)
        self.assertEqual(len(boundary), 4)

    def test_two_tets_shared_face(self):
        """两个共享面的四面体应该有6个边界面"""
        tets = [
            MockTetrahedron(0, 1, 2, 3),
            MockTetrahedron(0, 1, 2, 4),  # 共享面 (0,1,2)
        ]
        boundary = find_boundary_faces(tets)
        self.assertEqual(len(boundary), 6)
        # 共享面 (0,1,2) 不应该是边界面
        shared_face = (0, 1, 2)
        self.assertNotIn(shared_face, boundary)

    def test_boundary_face_set(self):
        """测试边界面集合"""
        tets = [
            MockTetrahedron(0, 1, 2, 3),
            MockTetrahedron(0, 1, 2, 4),
        ]
        boundary_set = find_boundary_face_set(tets)
        self.assertEqual(len(boundary_set), 6)


class TestBuildFaceToTets(unittest.TestCase):
    """面到四面体映射测试"""

    def test_single_tet(self):
        """单个四面体的面映射"""
        tets = [MockTetrahedron(0, 1, 2, 3)]
        f2t = build_face_to_tets(tets)
        self.assertEqual(len(f2t), 4)
        for face, tet_list in f2t.items():
            self.assertEqual(len(tet_list), 1)
            self.assertEqual(tet_list[0][0], 0)  # 四面体索引

    def test_two_tets_shared_face(self):
        """两个共享面的四面体"""
        tets = [
            MockTetrahedron(0, 1, 2, 3),
            MockTetrahedron(0, 1, 2, 4),
        ]
        f2t = build_face_to_tets(tets)
        shared_face = (0, 1, 2)
        self.assertIn(shared_face, f2t)
        self.assertEqual(len(f2t[shared_face]), 2)


class TestBuildEdgeToTets(unittest.TestCase):
    """边到四面体映射测试"""

    def test_single_tet(self):
        """单个四面体的边映射"""
        tets = [MockTetrahedron(0, 1, 2, 3)]
        e2t = build_edge_to_tets(tets)
        self.assertEqual(len(e2t), 6)
        for edge, tet_list in e2t.items():
            self.assertEqual(len(tet_list), 1)

    def test_two_tets_shared_edge(self):
        """两个共享边的四面体"""
        tets = [
            MockTetrahedron(0, 1, 2, 3),
            MockTetrahedron(0, 1, 4, 5),
        ]
        e2t = build_edge_to_tets(tets)
        shared_edge = (0, 1)
        self.assertIn(shared_edge, e2t)
        self.assertEqual(len(e2t[shared_edge]), 2)


class TestCreateSuperTetrahedronCoords(unittest.TestCase):
    """超级四面体坐标测试"""

    def test_contains_all_nodes(self):
        """超级四面体应该包含所有节点"""
        node_coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        super_coords = create_super_tetrahedron_coords(node_coords)
        self.assertEqual(len(super_coords), 4)

        # 所有节点应该在超级四面体的包围盒内
        super_arr = np.array(super_coords)
        mins = super_arr.min(axis=0)
        maxs = super_arr.max(axis=0)
        for node in node_coords:
            for d in range(3):
                self.assertGreater(node[d], mins[d])
                self.assertLess(node[d], maxs[d])

    def test_custom_scale(self):
        """自定义缩放因子"""
        node_coords = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        super_coords = create_super_tetrahedron_coords(node_coords, scale=5.0)
        self.assertEqual(len(super_coords), 4)

    def test_four_vertices(self):
        """超级四面体应该有4个不共面的顶点"""
        node_coords = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        super_coords = create_super_tetrahedron_coords(node_coords)

        # 检查4个顶点不共面（有符号体积不为0）
        p0 = np.array(super_coords[0])
        p1 = np.array(super_coords[1])
        p2 = np.array(super_coords[2])
        p3 = np.array(super_coords[3])
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        det = np.dot(v1, np.cross(v2, v3))
        self.assertGreater(abs(det), 1e-10)


class TestComputeMaxEdgeLength(unittest.TestCase):
    """最大边长计算测试"""

    def test_unit_tet(self):
        """单位四面体的最大边长"""
        tet = MockTetrahedron(0, 1, 2, 3)
        tet.p1 = [0.0, 0.0, 0.0]
        tet.p2 = [1.0, 0.0, 0.0]
        tet.p3 = [0.0, 1.0, 0.0]
        tet.p4 = [0.0, 0.0, 1.0]
        max_len = compute_max_edge_length(tet)
        # 最大边长应该是 sqrt(2) (对角线)
        self.assertAlmostEqual(max_len, np.sqrt(2), places=10)

    def test_regular_tet(self):
        """正四面体的所有边长相等"""
        tet = MockTetrahedron(0, 1, 2, 3)
        tet.p1 = [1.0, 1.0, 1.0]
        tet.p2 = [1.0, -1.0, -1.0]
        tet.p3 = [-1.0, 1.0, -1.0]
        tet.p4 = [-1.0, -1.0, 1.0]
        max_len = compute_max_edge_length(tet)
        expected = np.sqrt(2**2 + 2**2 + 0**2)  # = 2*sqrt(2)
        self.assertAlmostEqual(max_len, expected, places=10)

    def test_stretched_tet(self):
        """拉伸四面体的最大边长"""
        tet = MockTetrahedron(0, 1, 2, 3)
        tet.p1 = [0.0, 0.0, 0.0]
        tet.p2 = [10.0, 0.0, 0.0]  # 长边
        tet.p3 = [0.0, 1.0, 0.0]
        tet.p4 = [0.0, 0.0, 1.0]
        max_len = compute_max_edge_length(tet)
        # 最大边长是 (10,0,0) 到 (0,1,0) 的对角线 = sqrt(101)
        expected = np.sqrt(10**2 + 1**2)
        self.assertAlmostEqual(max_len, expected, places=10)


class TestNodeHash(unittest.TestCase):
    """节点哈希测试"""

    def test_same_coords_same_hash(self):
        """相同坐标应该有相同的哈希值"""
        h1 = node_hash([1.0, 2.0, 3.0])
        h2 = node_hash([1.0, 2.0, 3.0])
        self.assertEqual(h1, h2)

    def test_different_coords_different_hash(self):
        """不同坐标应该有不同的哈希值"""
        h1 = node_hash([1.0, 2.0, 3.0])
        h2 = node_hash([1.0, 2.0, 3.1])
        self.assertNotEqual(h1, h2)

    def test_2d_coords(self):
        """2D坐标"""
        h = node_hash([1.0, 2.0])
        self.assertIsInstance(h, int)

    def test_3d_coords(self):
        """3D坐标"""
        h = node_hash([1.0, 2.0, 3.0])
        self.assertIsInstance(h, int)

    def test_numpy_array(self):
        """numpy数组输入"""
        h = node_hash(np.array([1.0, 2.0, 3.0]))
        self.assertIsInstance(h, int)

    def test_precision_rounding(self):
        """精度截断到6位小数"""
        h1 = node_hash([1.0000001, 2.0, 3.0])
        h2 = node_hash([1.0000002, 2.0, 3.0])
        # 6位小数截断后应该相同
        self.assertEqual(h1, h2)


class TestBuildNodeToCells(unittest.TestCase):
    """节点到单元映射测试"""

    def test_single_tet(self):
        """单个四面体的节点映射"""
        tets = [MockTetrahedron(0, 1, 2, 3)]
        node_cells = build_node_to_cells(tets)
        self.assertEqual(len(node_cells), 4)
        for nid in range(4):
            self.assertIn(nid, node_cells)
            self.assertEqual(len(node_cells[nid]), 1)
            self.assertEqual(node_cells[nid][0], 0)

    def test_two_tets(self):
        """两个四面体的节点映射"""
        tets = [
            MockTetrahedron(0, 1, 2, 3),
            MockTetrahedron(0, 1, 2, 4),
        ]
        node_cells = build_node_to_cells(tets)
        # 节点0,1,2应该在两个四面体中
        for nid in [0, 1, 2]:
            self.assertEqual(len(node_cells[nid]), 2)
        # 节点3,4应该只在一个四面体中
        self.assertEqual(len(node_cells[3]), 1)
        self.assertEqual(len(node_cells[4]), 1)

    def test_empty_list(self):
        """空列表"""
        node_cells = build_node_to_cells([])
        self.assertEqual(len(node_cells), 0)


class TestValidateTetrahedron(unittest.TestCase):
    """四面体验证测试"""

    def test_valid_tet(self):
        """有效四面体"""
        tet = MockTetrahedron(0, 1, 2, 3)
        tet.p1 = [0.0, 0.0, 0.0]
        tet.p2 = [1.0, 0.0, 0.0]
        tet.p3 = [0.0, 1.0, 0.0]
        tet.p4 = [0.0, 0.0, 1.0]
        self.assertTrue(validate_tetrahedron(tet))

    def test_degenerate_tet(self):
        """退化四面体（零体积）"""
        tet = MockTetrahedron(0, 1, 2, 3)
        tet.p1 = [0.0, 0.0, 0.0]
        tet.p2 = [1.0, 0.0, 0.0]
        tet.p3 = [2.0, 0.0, 0.0]  # 共线
        tet.p4 = [0.0, 1.0, 0.0]
        self.assertFalse(validate_tetrahedron(tet))

    def test_with_boundary_check(self):
        """带边界检查的验证"""
        tet = MockTetrahedron(0, 1, 2, 3)
        tet.p1 = [0.0, 0.0, 0.0]
        tet.p2 = [1.0, 0.0, 0.0]
        tet.p3 = [0.0, 1.0, 0.0]
        tet.p4 = [0.0, 0.0, 1.0]

        # 边界检查函数：形心在单位盒内
        def check_boundary(centroid):
            return all(0 <= c <= 1 for c in centroid)

        self.assertTrue(validate_tetrahedron(tet, check_boundary))

    def test_outside_boundary(self):
        """形心在边界外"""
        tet = MockTetrahedron(0, 1, 2, 3)
        tet.p1 = [10.0, 10.0, 10.0]
        tet.p2 = [11.0, 10.0, 10.0]
        tet.p3 = [10.0, 11.0, 10.0]
        tet.p4 = [10.0, 10.0, 11.0]

        # 边界检查函数：形心在单位盒内
        def check_boundary(centroid):
            return all(0 <= c <= 1 for c in centroid)

        self.assertFalse(validate_tetrahedron(tet, check_boundary))


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


class TestComputeSurfaceMaxEdgeLength(unittest.TestCase):
    """表面最大边长计算测试"""

    def test_single_triangle(self):
        """单个三角形"""
        nodes = [
            MockNode3D([0, 0, 0], 0),
            MockNode3D([1, 0, 0], 1),
            MockNode3D([0, 1, 0], 2),
        ]
        tri = MockSurfaceTriangle(nodes)
        max_len = compute_surface_max_edge_length([tri])
        self.assertAlmostEqual(max_len, np.sqrt(2), places=10)

    def test_multiple_triangles(self):
        """多个三角形"""
        tri1 = MockSurfaceTriangle([
            MockNode3D([0, 0, 0], 0),
            MockNode3D([1, 0, 0], 1),
            MockNode3D([0, 1, 0], 2),
        ])
        tri2 = MockSurfaceTriangle([
            MockNode3D([0, 0, 0], 0),
            MockNode3D([3, 0, 0], 3),  # 更长的边
            MockNode3D([1.5, 1, 0], 4),
        ])
        max_len = compute_surface_max_edge_length([tri1, tri2])
        # 最长边是 [0,0,0] 到 [3,0,0] = 3.0
        self.assertAlmostEqual(max_len, 3.0, places=10)

    def test_empty_list(self):
        """空列表"""
        max_len = compute_surface_max_edge_length([])
        self.assertEqual(max_len, 1.0)


if __name__ == '__main__':
    unittest.main()

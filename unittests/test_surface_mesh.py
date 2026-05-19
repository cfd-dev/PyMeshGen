"""
曲面网格生成单元测试

测试 sfmesh 模块从 IGES/STEP 文件生成曲面网格的功能
"""
import sys
import os
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fileIO.occ_loader import ensure_occ_loaded
ensure_occ_loaded()

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE

from sfmesh.surface_mesh import SurfaceMeshGenerator, generate_surface_mesh_from_file, _export_combined_mesh
from sfmesh.mesh_quality import SurfaceMeshQuality
from sfmesh.surface_front import NodeElement3D, SurfaceTriangle, SurfaceFront
from sfmesh.surface_geometry import SurfaceGeometry
from sfmesh.sizing_field import SurfaceSizingField


class TestSurfaceGeometry(unittest.TestCase):
    """测试曲面几何操作类"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        cls.geometry = SurfaceGeometry()
        cls.sphere_path = Path(project_root) / "examples" / "cad" / "sphere.iges"
        cls.cylinder_path = Path(project_root) / "examples" / "cad" / "cylinder.stp"
    
    def test_geometry_files_exist(self):
        """测试几何文件是否存在"""
        self.assertTrue(self.sphere_path.exists(), 
                       f"球体文件不存在：{self.sphere_path}")
        self.assertTrue(self.cylinder_path.exists(), 
                       f"圆柱文件不存在：{self.cylinder_path}")


class TestNodeElement3D(unittest.TestCase):
    """测试三维节点元素"""
    
    def test_node_creation(self):
        """测试节点创建"""
        node = NodeElement3D(
            coords=(1.0, 2.0, 3.0),
            idx=0,
            normal=(0.0, 0.0, 1.0)
        )
        
        self.assertEqual(node.coords, (1.0, 2.0, 3.0))
        self.assertEqual(node.idx, 0)
        self.assertEqual(node.normal, (0.0, 0.0, 1.0))
        self.assertIsNotNone(node.hash)
        self.assertEqual(len(node.bbox), 6)
    
    def test_node_equality(self):
        """测试节点相等性"""
        node1 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=0)
        node2 = NodeElement3D(coords=(1.0, 2.0, 3.0), idx=1)
        node3 = NodeElement3D(coords=(1.0, 2.0, 4.0), idx=2)
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
    
    def test_node_bbox(self):
        """测试节点边界框"""
        node = NodeElement3D(coords=(5.0, -3.0, 2.0))
        
        self.assertEqual(node.bbox[0], 5.0)
        self.assertEqual(node.bbox[1], -3.0)
        self.assertEqual(node.bbox[2], 2.0)
        self.assertEqual(node.bbox[3], 5.0)
        self.assertEqual(node.bbox[4], -3.0)
        self.assertEqual(node.bbox[5], 2.0)


class TestSurfaceTriangle(unittest.TestCase):
    """测试曲面三角形单元"""
    
    def setUp(self):
        """设置测试"""
        self.node1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        self.node2 = NodeElement3D(coords=(1.0, 0.0, 0.0), idx=1)
        self.node3 = NodeElement3D(coords=(0.5, 0.866, 0.0), idx=2)
    
    def test_triangle_creation(self):
        """测试三角形创建"""
        triangle = SurfaceTriangle(self.node1, self.node2, self.node3)
        
        self.assertEqual(len(triangle.nodes), 3)
        self.assertEqual(triangle.node_ids, [0, 1, 2])
        self.assertIsNotNone(triangle.normal)
        self.assertGreater(triangle.area, 0)
        self.assertGreater(triangle.quality, 0)
    
    def test_triangle_area(self):
        """测试三角形面积计算"""
        triangle = SurfaceTriangle(self.node1, self.node2, self.node3)
        
        expected_area = 0.5 * 1.0 * 0.866
        self.assertAlmostEqual(triangle.area, expected_area, places=3)
    
    def test_triangle_quality(self):
        """测试三角形质量计算"""
        equilateral = SurfaceTriangle(self.node1, self.node2, self.node3)
        
        self.assertAlmostEqual(equilateral.quality, 1.0, places=2)
        
        node4 = NodeElement3D(coords=(10.0, 0.0, 0.0), idx=3)
        poor = SurfaceTriangle(self.node1, self.node2, node4)
        
        self.assertLess(poor.quality, 0.5)
    
    def test_triangle_bbox(self):
        """测试三角形边界框"""
        triangle = SurfaceTriangle(self.node1, self.node2, self.node3)
        
        self.assertAlmostEqual(triangle.bbox[0], 0.0, places=3)
        self.assertAlmostEqual(triangle.bbox[1], 0.0, places=3)
        self.assertAlmostEqual(triangle.bbox[2], 0.0, places=3)
        self.assertAlmostEqual(triangle.bbox[3], 1.0, places=3)
        self.assertAlmostEqual(triangle.bbox[4], 0.866, places=3)
        self.assertAlmostEqual(triangle.bbox[5], 0.0, places=3)


class TestSurfaceFront(unittest.TestCase):
    """测试曲面阵面"""
    
    def setUp(self):
        """设置测试"""
        self.node1 = NodeElement3D(
            coords=(0.0, 0.0, 0.0),
            idx=0,
            normal=(0.0, 0.0, 1.0)
        )
        self.node2 = NodeElement3D(
            coords=(1.0, 0.0, 0.0),
            idx=1,
            normal=(0.0, 0.0, 1.0)
        )
    
    def test_front_creation(self):
        """测试阵面创建"""
        front = SurfaceFront(self.node1, self.node2)
        
        self.assertEqual(len(front.node_elems), 2)
        self.assertEqual(front.length, 1.0)
        self.assertEqual(front.center, (0.5, 0.0, 0.0))
        self.assertIsNotNone(front.tangent_normal)
    
    def test_front_comparison(self):
        """测试阵面比较（优先队列）"""
        node3 = NodeElement3D(coords=(2.0, 0.0, 0.0), idx=2)
        front1 = SurfaceFront(self.node1, self.node2)
        front2 = SurfaceFront(self.node1, node3)
        
        self.assertLess(front1, front2)
    
    def test_front_bbox(self):
        """测试阵面边界框"""
        front = SurfaceFront(self.node1, self.node2)
        
        self.assertEqual(front.bbox[0], 0.0)
        self.assertEqual(front.bbox[3], 1.0)


class TestSizingField(unittest.TestCase):
    """测试尺寸场"""
    
    def test_uniform_sizing_field(self):
        """测试均匀尺寸场"""
        sizing = SurfaceSizingField(
            global_spacing=1.0,
            curvature_adaptation=False
        )
        
        spacing = sizing.spacing_at((0.0, 0.0, 0.0))
        self.assertEqual(spacing, 1.0)
    
    def test_sizing_field_bounds(self):
        """测试尺寸场边界"""
        sizing = SurfaceSizingField(
            global_spacing=0.5,
            min_spacing=0.1,
            max_spacing=2.0
        )
        
        spacing = sizing.spacing_at((0.0, 0.0, 0.0))
        self.assertGreaterEqual(spacing, 0.1)
        self.assertLessEqual(spacing, 2.0)


class TestSurfaceMeshGenerator(unittest.TestCase):
    """测试曲面网格生成器"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        cls.sphere_path = Path(project_root) / "examples" / "cad" / "sphere.iges"
        cls.cylinder_path = Path(project_root) / "examples" / "cad" / "cylinder.stp"
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "sfmesh"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_sphere_file_exists(self):
        """测试球体文件存在"""
        self.assertTrue(self.sphere_path.exists(), 
                       f"球体文件不存在：{self.sphere_path}")
    
    def test_cylinder_file_exists(self):
        """测试圆柱文件存在"""
        self.assertTrue(self.cylinder_path.exists(), 
                       f"圆柱文件不存在：{self.cylinder_path}")
    
    def test_generate_mesh_from_sphere(self):
        """测试从球体 IGES 文件生成网格"""
        if not self.sphere_path.exists():
            self.skipTest(f"球体文件不存在：{self.sphere_path}")

        from fileIO.geometry_io import import_geometry_file

        shape = import_geometry_file(str(self.sphere_path))

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        faces = []
        while explorer.More():
            faces.append(explorer.Current())
            explorer.Next()

        self.assertGreater(len(faces), 0, "球体模型中没有找到曲面")

        face = faces[0]

        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=0.3,
            max_iterations=5000
        )

        triangles = generator.generate()

        self.assertGreater(len(triangles), 50, "三角形数量不足")

        quality_result = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.3,
                          "平均网格质量过低")

        output_file = self.output_dir / "sphere_mesh.vtk"
        generator.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未生成")
    
    def test_generate_mesh_from_cylinder(self):
        """测试从圆柱 STEP 文件生成网格（统一网格生成，端面与柱面共享边界节点）"""
        if not self.cylinder_path.exists():
            self.skipTest(f"圆柱文件不存在：{self.cylinder_path}")

        from fileIO.geometry_io import import_geometry_file
        from sfmesh.surface_mesh import generate_surface_mesh_from_shape

        shape = import_geometry_file(str(self.cylinder_path))
        all_triangles = generate_surface_mesh_from_shape(shape, global_spacing=0.5)

        self.assertGreater(len(all_triangles), 0, "没有生成任何三角形")

        quality_result = SurfaceMeshQuality.evaluate_mesh(all_triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.3,
                          "平均网格质量过低")

        output_file = self.output_dir / "cylinder_mesh.vtk"
        _export_combined_mesh(all_triangles, str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未生成")
    
    def test_generate_mesh_with_curvature_adaptation(self):
        """测试曲率自适应网格生成（从球体 IGES 文件）"""
        if not self.sphere_path.exists():
            self.skipTest(f"球体文件不存在：{self.sphere_path}")

        from fileIO.geometry_io import import_geometry_file

        shape = import_geometry_file(str(self.sphere_path))

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face = explorer.Current()

        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=0.3,
            curvature_adaptation=True,
            max_iterations=5000
        )

        triangles = generator.generate()

        self.assertGreater(len(triangles), 50, "三角形数量不足")

        quality_result = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.3,
                          "平均网格质量过低")

        output_file = self.output_dir / "sphere_curvature_adaptation.vtk"
        generator.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未生成")


class TestMeshQuality(unittest.TestCase):
    """测试网格质量评估"""
    
    def test_quality_evaluation(self):
        """测试质量评估"""
        node1 = NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0)
        node2 = NodeElement3D(coords=(1.0, 0.0, 0.0), idx=1)
        node3 = NodeElement3D(coords=(0.5, 0.866, 0.0), idx=2)
        
        triangle = SurfaceTriangle(node1, node2, node3)
        
        result = SurfaceMeshQuality.evaluate_triangle(triangle)
        
        self.assertIn('quality', result)
        self.assertIn('aspect_ratio', result)
        self.assertIn('min_angle', result)
        self.assertIn('max_angle', result)
        self.assertTrue(result['is_good'])
    
    def test_mesh_statistics(self):
        """测试网格统计"""
        nodes = [
            NodeElement3D(coords=(0.0, 0.0, 0.0), idx=0),
            NodeElement3D(coords=(1.0, 0.0, 0.0), idx=1),
            NodeElement3D(coords=(0.5, 0.866, 0.0), idx=2),
            NodeElement3D(coords=(1.5, 0.866, 0.0), idx=3),
        ]
        
        triangles = [
            SurfaceTriangle(nodes[0], nodes[1], nodes[2]),
            SurfaceTriangle(nodes[1], nodes[3], nodes[2]),
        ]
        
        result = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        
        self.assertEqual(result['num_triangles'], 2)
        self.assertIn('quality_mean', result)
        self.assertIn('total_area', result)


def run_tests():
    """运行测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSurfaceGeometry))
    suite.addTests(loader.loadTestsFromTestCase(TestNodeElement3D))
    suite.addTests(loader.loadTestsFromTestCase(TestSurfaceTriangle))
    suite.addTests(loader.loadTestsFromTestCase(TestSurfaceFront))
    suite.addTests(loader.loadTestsFromTestCase(TestSizingField))
    suite.addTests(loader.loadTestsFromTestCase(TestSurfaceMeshGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestMeshQuality))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    run_tests()

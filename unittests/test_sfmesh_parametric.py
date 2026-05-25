"""参数化间接法网格生成单元测试

测试 sfmesh 模块使用参数化方法生成结构化曲面网格的功能：
- 球面（generate_sphere_mesh）
- 椭球面（generate_ellipsoid_mesh）
"""
import sys
import os
import math
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fileIO.occ_loader import ensure_occ_loaded
ensure_occ_loaded()

from sfmesh.shape_generators import PrimitiveMeshResult, generate_surface_mesh_from_shape, _export_combined_mesh
from sfmesh.mesh_parametric import (
    generate_sphere_mesh,
    generate_ellipsoid_mesh,
    _mesh_face_closed_surface,
)
from sfmesh.mesh_quality import SurfaceMeshQuality


class TestPrimitiveMeshResult(unittest.TestCase):
    """测试 PrimitiveMeshResult 数据结构"""

    def test_result_creation(self):
        """测试结果对象创建"""
        result = PrimitiveMeshResult()
        self.assertEqual(len(result.triangles), 0)
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(result.num_faces, 0)
        self.assertEqual(len(result.face_map), 0)
        self.assertEqual(len(result.face_types), 0)


class TestSphereMeshGeneration(unittest.TestCase):
    """测试球面网格生成"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_sphere_basic(self):
        """基本球面网格生成"""
        result = generate_sphere_mesh(center=(0, 0, 0), radius=1.0, spacing=0.3)
        self.assertGreater(len(result.triangles), 50, "三角形数量不足")
        self.assertEqual(result.num_faces, 1)
        self.assertEqual(result.face_types[0], "sphere")

    def test_sphere_node_count(self):
        """球面节点数量"""
        result = generate_sphere_mesh(center=(0, 0, 0), radius=1.0, spacing=0.3)
        n_theta = max(6, int(2 * math.pi * 1.0 / 0.3))
        n_phi = max(4, int(math.pi * 1.0 / 0.3))
        expected_nodes = 1 + n_theta * (n_phi - 1) + 1
        self.assertEqual(len(result.nodes), expected_nodes)

    def test_sphere_quality(self):
        """球面网格质量"""
        result = generate_sphere_mesh(center=(0, 0, 0), radius=1.0, spacing=0.3)
        quality_result = SurfaceMeshQuality.evaluate_mesh(result.triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.5, "平均网格质量过低")

    def test_sphere_coverage(self):
        """球面网格覆盖整个球面"""
        result = generate_sphere_mesh(center=(0, 0, 0), radius=1.0, spacing=0.3)
        coords = [n.coords for n in result.nodes]
        for i in range(3):
            self.assertLess(min(c[i] for c in coords), -0.9)
            self.assertGreater(max(c[i] for c in coords), 0.9)

    def test_sphere_different_params(self):
        """不同参数球面"""
        test_cases = [
            {"center": (0, 0, 0), "radius": 1.0, "spacing": 0.5},
            {"center": (0, 0, 0), "radius": 2.0, "spacing": 0.5},
            {"center": (1, 2, 3), "radius": 0.5, "spacing": 0.1},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                result = generate_sphere_mesh(**case)
                self.assertGreater(len(result.triangles), 0)
                self.assertEqual(result.num_faces, 1)

    def test_sphere_invalid_radius(self):
        """无效半径"""
        with self.assertRaises(ValueError):
            generate_sphere_mesh(center=(0, 0, 0), radius=-1.0)

    def test_sphere_vtk_export(self):
        """VTK导出"""
        output_file = str(self.output_dir / "parametric_sphere.vtk")
        result = generate_sphere_mesh(
            center=(0, 0, 0), radius=1.0, spacing=0.3, output_vtk=output_file,
        )
        self.assertGreater(len(result.triangles), 0)
        self.assertTrue(os.path.exists(output_file), "VTK文件未生成")
        self.assertGreater(os.path.getsize(output_file), 0)


class TestEllipsoidMeshGeneration(unittest.TestCase):
    """测试椭球面网格生成（参数化方法）"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_ellipsoid_basic(self):
        """基本椭球面网格生成"""
        result = generate_ellipsoid_mesh(
            center=(0, 0, 0), semi_axes=(1.0, 0.75, 0.5), spacing=0.2,
        )
        self.assertGreater(len(result.triangles), 50, "三角形数量不足")
        self.assertEqual(result.num_faces, 1)
        self.assertEqual(result.face_types[0], "ellipsoid")

    def test_ellipsoid_quality(self):
        """椭球面网格质量"""
        result = generate_ellipsoid_mesh(
            center=(0, 0, 0), semi_axes=(1.0, 0.75, 0.5), spacing=0.2,
        )
        quality_result = SurfaceMeshQuality.evaluate_mesh(result.triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.4, "平均网格质量过低")

    def test_ellipsoid_coverage(self):
        """椭球面网格覆盖范围"""
        result = generate_ellipsoid_mesh(
            center=(0, 0, 0), semi_axes=(1.0, 0.75, 0.5), spacing=0.2,
        )
        coords = [n.coords for n in result.nodes]
        self.assertLess(min(c[0] for c in coords), -0.9)
        self.assertGreater(max(c[0] for c in coords), 0.9)
        self.assertLess(min(c[2] for c in coords), -0.4)
        self.assertGreater(max(c[2] for c in coords), 0.4)

    def test_ellipsoid_different_params(self):
        """不同参数椭球面"""
        test_cases = [
            {"center": (0, 0, 0), "semi_axes": (1.0, 1.0, 1.0), "spacing": 0.3},
            {"center": (0, 0, 0), "semi_axes": (2.0, 1.0, 0.5), "spacing": 0.3},
            {"center": (1, 1, 1), "semi_axes": (0.5, 0.5, 0.5), "spacing": 0.1},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                result = generate_ellipsoid_mesh(**case)
                self.assertGreater(len(result.triangles), 0)

    def test_ellipsoid_invalid_axes(self):
        """无效半轴"""
        with self.assertRaises(ValueError):
            generate_ellipsoid_mesh(center=(0, 0, 0), semi_axes=(1.0, -1.0, 0.5))

    def test_ellipsoid_vtk_export(self):
        """VTK导出"""
        output_file = str(self.output_dir / "parametric_ellipsoid.vtk")
        result = generate_ellipsoid_mesh(
            center=(0, 0, 0), semi_axes=(1.0, 0.75, 0.5), spacing=0.2,
            output_vtk=output_file,
        )
        self.assertGreater(len(result.triangles), 0)
        self.assertTrue(os.path.exists(output_file), "VTK文件未生成")
        self.assertGreater(os.path.getsize(output_file), 0)


class TestCADFileMeshGeneration(unittest.TestCase):
    """从 CAD 文件生成网格测试（自动分发到参数化路径）"""

    @classmethod
    def setUpClass(cls):
        cls.sphere_path = Path(project_root) / "examples" / "cad" / "sphere.iges"
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_generate_mesh_from_sphere(self):
        """从球体 IGES 文件生成网格（自动分发到参数化路径）"""
        if not self.sphere_path.exists():
            self.skipTest(f"球体文件不存在：{self.sphere_path}")

        from fileIO.geometry_io import import_geometry_file

        shape = import_geometry_file(str(self.sphere_path))
        triangles = generate_surface_mesh_from_shape(shape, global_spacing=0.3)

        self.assertGreater(len(triangles), 50, "三角形数量不足")

        quality_result = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.3,
                          "平均网格质量过低")

        output_file = self.output_dir / "parametric_sphere_cad.vtk"
        _export_combined_mesh(triangles, str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未生成")

    def test_generate_mesh_with_curvature_adaptation(self):
        """曲率自适应网格生成（从球体 IGES 文件，参数化路径）"""
        if not self.sphere_path.exists():
            self.skipTest(f"球体文件不存在：{self.sphere_path}")

        from fileIO.geometry_io import import_geometry_file

        shape = import_geometry_file(str(self.sphere_path))
        triangles = generate_surface_mesh_from_shape(shape, global_spacing=0.3)

        self.assertGreater(len(triangles), 50, "三角形数量不足")

        quality_result = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.3,
                          "平均网格质量过低")

        output_file = self.output_dir / "parametric_sphere_curvature.vtk"
        _export_combined_mesh(triangles, str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未生成")


class TestClosedSurfaceParametric(unittest.TestCase):
    """闭合曲面参数化路径测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_torus_face():
        """完整环面（主半径 3，管半径 1）"""
        from OCC.Core.Geom import Geom_ToroidalSurface
        from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        torus = Geom_ToroidalSurface(ax3, 3.0, 1.0)
        return BRepBuilderAPI_MakeFace(torus, 0, 2 * math.pi, 0, 2 * math.pi, 1e-6).Face()

    def test_torus_parametric(self):
        """完整环面（闭合曲面，参数化路径 _mesh_face_closed_surface）"""
        from collections import Counter

        face = self._make_torus_face()
        triangles, nodes = _mesh_face_closed_surface(face, 1.0, 0)

        self.assertGreater(len(triangles), 30, "三角形数量不足")

        tri_sets = [tuple(sorted([t.nodes[j].idx for j in range(3)])) for t in triangles]
        dup_count = sum(1 for v in Counter(tri_sets).values() if v > 1)
        self.assertEqual(dup_count, 0, f"存在 {dup_count} 个重复三角形")

        quality_result = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.3,
                          f"平均质量 {quality_result['quality_mean']:.3f} < 0.3")


if __name__ == "__main__":
    unittest.main()

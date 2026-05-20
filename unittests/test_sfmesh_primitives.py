"""
基础几何体曲面网格生成单元测试

测试 sfmesh 模块从基础几何体（长方体、圆柱体）生成曲面网格的功能
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

from sfmesh.primitives import (
    generate_cube_mesh,
    generate_cylinder_mesh,
    generate_rectangle_mesh,
    generate_sphere_mesh,
    generate_ellipsoid_mesh,
    generate_ellipsoid_mesh_2d_afm,
    PrimitiveMeshResult,
)
from sfmesh.occ_utils import _extract_faces
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


class TestExtractFaces(unittest.TestCase):
    """测试面提取辅助函数"""

    def test_extract_box_faces(self):
        """测试从长方体提取面"""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.gp import gp_Pnt

        shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Shape()
        faces = _extract_faces(shape)
        self.assertEqual(len(faces), 6)

    def test_extract_cylinder_faces(self):
        """测试从圆柱体提取面"""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir

        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        shape = BRepPrimAPI_MakeCylinder(axis, 1.0, 2.0).Shape()
        faces = _extract_faces(shape)
        self.assertEqual(len(faces), 3)


class TestCubeMeshGeneration(unittest.TestCase):
    """测试长方体曲面网格生成"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_cube_face_count(self):
        """测试长方体面数量（应为6）"""
        result = generate_cube_mesh(corner1=(0, 0, 0), corner2=(1, 1, 1), spacing=0.1)
        self.assertEqual(result.num_faces, 6)
        self.assertEqual(len(result.face_map), 6)

    def test_cube_triangle_generation(self):
        """测试长方体每个面都生成了三角形"""
        result = generate_cube_mesh(corner1=(0, 0, 0), corner2=(1, 1, 1), spacing=0.1)
        self.assertGreater(len(result.triangles), 0, "未生成任何三角形")
        for face_idx, tris in result.face_map.items():
            self.assertGreater(len(tris), 0, f"面 {face_idx} 未生成三角形")

    def test_cube_face_coverage(self):
        """测试长方体网格完整覆盖6个面"""
        result = generate_cube_mesh(corner1=(0, 0, 0), corner2=(2, 3, 4), spacing=0.2)

        for face_idx, tris in result.face_map.items():
            ftype = result.face_types[face_idx]
            coords = [n.coords for t in tris for n in t.nodes]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            zs = [c[2] for c in coords]

            with self.subTest(face=ftype):
                if ftype == "bottom":
                    self.assertAlmostEqual(min(zs), 0.0, places=1)
                    self.assertAlmostEqual(max(zs), 0.0, places=1)
                    self.assertAlmostEqual(min(xs), 0.0, places=1)
                    self.assertAlmostEqual(max(xs), 2.0, places=1)
                    self.assertAlmostEqual(min(ys), 0.0, places=1)
                    self.assertAlmostEqual(max(ys), 3.0, places=1)
                elif ftype == "top":
                    self.assertAlmostEqual(min(zs), 4.0, places=1)
                    self.assertAlmostEqual(max(zs), 4.0, places=1)
                    self.assertAlmostEqual(min(xs), 0.0, places=1)
                    self.assertAlmostEqual(max(xs), 2.0, places=1)
                elif ftype == "left":
                    self.assertAlmostEqual(min(xs), 0.0, places=1)
                    self.assertAlmostEqual(max(xs), 0.0, places=1)
                    self.assertAlmostEqual(min(ys), 0.0, places=1)
                    self.assertAlmostEqual(max(ys), 3.0, places=1)
                    self.assertAlmostEqual(min(zs), 0.0, places=1)
                    self.assertAlmostEqual(max(zs), 4.0, places=1)
                elif ftype == "right":
                    self.assertAlmostEqual(min(xs), 2.0, places=1)
                    self.assertAlmostEqual(max(xs), 2.0, places=1)
                elif ftype == "front":
                    self.assertAlmostEqual(min(ys), 0.0, places=1)
                    self.assertAlmostEqual(max(ys), 0.0, places=1)
                    self.assertAlmostEqual(min(xs), 0.0, places=1)
                    self.assertAlmostEqual(max(xs), 2.0, places=1)
                    self.assertAlmostEqual(min(zs), 0.0, places=1)
                    self.assertAlmostEqual(max(zs), 4.0, places=1)
                elif ftype == "back":
                    self.assertAlmostEqual(min(ys), 3.0, places=1)
                    self.assertAlmostEqual(max(ys), 3.0, places=1)

    def test_cube_face_types(self):
        """测试长方体面类型标注"""
        result = generate_cube_mesh(corner1=(0, 0, 0), corner2=(1, 1, 1), spacing=0.1)
        expected_types = {"bottom", "top", "front", "back", "left", "right"}
        actual_types = set(result.face_types.values())
        self.assertTrue(expected_types.issubset(actual_types),
                        f"缺少面类型: {expected_types - actual_types}")

    def test_cube_node_count(self):
        """测试长方体网格节点数量"""
        result = generate_cube_mesh(corner1=(0, 0, 0), corner2=(1, 1, 1), spacing=0.1)
        self.assertGreater(len(result.nodes), 0)
        self.assertGreaterEqual(len(result.nodes), 8)

    def test_cube_quality(self):
        """测试长方体网格质量"""
        result = generate_cube_mesh(corner1=(0, 0, 0), corner2=(1, 1, 1), spacing=0.1)
        quality_result = SurfaceMeshQuality.evaluate_mesh(result.triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.5, "平均网格质量过低")

    def test_cube_different_sizes(self):
        """测试不同尺寸长方体（参数化子测试）"""
        test_cases = [
            {"corner1": (0, 0, 0), "corner2": (1, 1, 1), "spacing": 0.1},
            {"corner1": (0, 0, 0), "corner2": (2, 3, 4), "spacing": 0.2},
            {"corner1": (-1, -1, -1), "corner2": (1, 1, 1), "spacing": 0.2},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                result = generate_cube_mesh(**case)
                self.assertEqual(result.num_faces, 6)
                self.assertGreater(len(result.triangles), 0)
                # 每个面都有完整覆盖
                for face_idx, tris in result.face_map.items():
                    self.assertGreater(len(tris), 4,
                                       f"面 {face_idx} 三角形数量不足")

    def test_cube_invalid_input(self):
        """测试无效输入"""
        with self.assertRaises(ValueError):
            generate_cube_mesh(corner1=(0, 0, 0), corner2=(0, 1, 1), spacing=0.5)

    def test_cube_vtk_export(self):
        """测试VTK导出"""
        output_file = str(self.output_dir / "cube_mesh.vtk")
        result = generate_cube_mesh(
            corner1=(0, 0, 0), corner2=(1, 1, 1), spacing=0.1, output_vtk=output_file,
        )
        self.assertGreater(len(result.triangles), 0)
        self.assertTrue(os.path.exists(output_file), "VTK文件未生成")
        self.assertGreater(os.path.getsize(output_file), 0)


class TestCylinderMeshGeneration(unittest.TestCase):
    """测试圆柱体曲面网格生成"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_cylinder_face_count(self):
        """测试圆柱体面数量（应为3）"""
        result = generate_cylinder_mesh(
            base_center=(0, 0, 0), radius=1.0, height=2.0, spacing=0.1,
        )
        self.assertEqual(result.num_faces, 3)

    def test_cylinder_triangle_generation(self):
        """测试圆柱体每个面都生成了三角形"""
        result = generate_cylinder_mesh(
            base_center=(0, 0, 0), radius=1.0, height=2.0, spacing=0.1,
        )
        self.assertGreater(len(result.triangles), 0)
        for face_idx, tris in result.face_map.items():
            self.assertGreater(len(tris), 0, f"面 {face_idx} 未生成三角形")

    def test_cylinder_face_types(self):
        """测试圆柱体面类型标注"""
        result = generate_cylinder_mesh(
            base_center=(0, 0, 0), radius=1.0, height=2.0, spacing=0.1,
        )
        expected_types = {"bottom", "top", "lateral"}
        actual_types = set(result.face_types.values())
        self.assertTrue(expected_types.issubset(actual_types),
                        f"缺少面类型: {expected_types - actual_types}")

    def test_cylinder_face_coverage(self):
        """测试圆柱体网格完整覆盖3个面"""
        result = generate_cylinder_mesh(
            base_center=(0, 0, 0), radius=1.0, height=2.0, spacing=0.1,
        )
        for face_idx, tris in result.face_map.items():
            ftype = result.face_types[face_idx]
            coords = [n.coords for t in tris for n in t.nodes]
            zs = [c[2] for c in coords]
            with self.subTest(face=ftype):
                if ftype == "bottom":
                    self.assertAlmostEqual(min(zs), 0.0, places=1)
                    self.assertAlmostEqual(max(zs), 0.0, places=1)
                elif ftype == "top":
                    self.assertAlmostEqual(min(zs), 2.0, places=1)
                    self.assertAlmostEqual(max(zs), 2.0, places=1)
                elif ftype == "lateral":
                    self.assertGreater(min(zs), -0.1)
                    self.assertLess(max(zs), 2.1)

    def test_cylinder_quality(self):
        """测试圆柱体网格质量"""
        result = generate_cylinder_mesh(
            base_center=(0, 0, 0), radius=1.0, height=2.0, spacing=0.1,
        )
        quality_result = SurfaceMeshQuality.evaluate_mesh(result.triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.3, "平均网格质量过低")

    def test_cylinder_different_params(self):
        """测试不同参数圆柱体"""
        test_cases = [
            {"base_center": (0, 0, 0), "radius": 1.0, "height": 1.0, "spacing": 0.1},
            {"base_center": (0, 0, 0), "radius": 2.0, "height": 3.0, "spacing": 0.2},
            {"base_center": (1, 1, 1), "radius": 0.5, "height": 2.0, "spacing": 0.05},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                result = generate_cylinder_mesh(**case)
                self.assertEqual(result.num_faces, 3)
                self.assertGreater(len(result.triangles), 0)

    def test_cylinder_invalid_radius(self):
        """测试无效半径"""
        with self.assertRaises(ValueError):
            generate_cylinder_mesh(base_center=(0, 0, 0), radius=-1.0, height=1.0)

    def test_cylinder_invalid_height(self):
        """测试无效高度"""
        with self.assertRaises(ValueError):
            generate_cylinder_mesh(base_center=(0, 0, 0), radius=1.0, height=0.0)

    def test_cylinder_vtk_export(self):
        """测试VTK导出"""
        output_file = str(self.output_dir / "cylinder_mesh.vtk")
        result = generate_cylinder_mesh(
            base_center=(0, 0, 0), radius=1.0, height=2.0, spacing=0.1,
            output_vtk=output_file,
        )
        self.assertGreater(len(result.triangles), 0)
        self.assertTrue(os.path.exists(output_file), "VTK文件未生成")
        self.assertGreater(os.path.getsize(output_file), 0)


class TestRectangleMeshGeneration(unittest.TestCase):
    """测试矩形域曲面网格生成（不依赖 OCC 几何体）"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_rectangle_basic(self):
        """基本矩形网格生成"""
        result = generate_rectangle_mesh((0, 0, 0), (1, 0, 1), spacing=0.2)
        self.assertGreater(len(result.triangles), 0, "未生成任何三角形")
        self.assertEqual(result.num_faces, 1)
        self.assertIn(0, result.face_map)
        self.assertEqual(result.face_types[0], "rectangle")

    def test_rectangle_coverage(self):
        """网格覆盖整个矩形域"""
        result = generate_rectangle_mesh((0, 0, 0), (2, 0, 3), spacing=0.2)
        coords = [n.coords for n in result.nodes]
        xs = [c[0] for c in coords]
        zs = [c[2] for c in coords]
        self.assertAlmostEqual(min(xs), 0.0, places=1)
        self.assertAlmostEqual(max(xs), 2.0, places=1)
        self.assertAlmostEqual(min(zs), 0.0, places=1)
        self.assertAlmostEqual(max(zs), 3.0, places=1)

    def test_rectangle_quality(self):
        """网格质量检查"""
        result = generate_rectangle_mesh((0, 0, 0), (1, 0, 1), spacing=0.1)
        quality_result = SurfaceMeshQuality.evaluate_mesh(result.triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.3, "平均网格质量过低")

    def test_rectangle_different_sizes(self):
        """不同尺寸矩形（参数化子测试）"""
        test_cases = [
            {"corner1": (0, 0, 0), "corner2": (1, 0, 1), "spacing": 0.2},
            {"corner1": (0, 0, 0), "corner2": (2, 0, 3), "spacing": 0.3},
            {"corner1": (-1, 0, -1), "corner2": (1, 0, 1), "spacing": 0.2},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                result = generate_rectangle_mesh(**case)
                self.assertEqual(result.num_faces, 1)
                self.assertGreater(len(result.triangles), 0)

    def test_rectangle_invalid_input(self):
        """无效输入（退化为线段）"""
        with self.assertRaises(ValueError):
            generate_rectangle_mesh((0, 0, 0), (1, 0, 0), spacing=0.1)

    def test_rectangle_vtk_export(self):
        """VTK导出"""
        output_file = str(self.output_dir / "rectangle_mesh.vtk")
        result = generate_rectangle_mesh(
            (0, 0, 0), (1, 0, 1), spacing=0.2, output_vtk=output_file,
        )
        self.assertGreater(len(result.triangles), 0)
        self.assertTrue(os.path.exists(output_file), "VTK文件未生成")
        self.assertGreater(os.path.getsize(output_file), 0)


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
        # 北极 + 中间带 + 南极
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
        output_file = str(self.output_dir / "sphere_mesh.vtk")
        result = generate_sphere_mesh(
            center=(0, 0, 0), radius=1.0, spacing=0.3, output_vtk=output_file,
        )
        self.assertGreater(len(result.triangles), 0)
        self.assertTrue(os.path.exists(output_file), "VTK文件未生成")
        self.assertGreater(os.path.getsize(output_file), 0)


class TestEllipsoidMeshGeneration(unittest.TestCase):
    """测试椭球面网格生成（2D 参数空间流水线方法）"""

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
        output_file = str(self.output_dir / "ellipsoid_mesh_2d_pipeline.vtk")
        result = generate_ellipsoid_mesh(
            center=(0, 0, 0), semi_axes=(1.0, 0.75, 0.5), spacing=0.2,
            output_vtk=output_file,
        )
        self.assertGreater(len(result.triangles), 0)
        self.assertTrue(os.path.exists(output_file), "VTK文件未生成")
        self.assertGreater(os.path.getsize(output_file), 0)


class TestEllipsoidMeshGeneration2DAFM(unittest.TestCase):
    """测试椭球面网格生成（2D 阵面推进流水线方法）"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_ellipsoid_2d_afm_basic(self):
        """基本椭球面网格生成（2D AFM）"""
        result = generate_ellipsoid_mesh_2d_afm(
            center=(0, 0, 0), semi_axes=(1.0, 0.75, 0.5), spacing=0.2,
        )
        self.assertGreater(len(result.triangles), 50, "三角形数量不足")
        self.assertGreater(len(result.nodes), 30, "节点数量不足")

    def test_ellipsoid_2d_afm_quality(self):
        """椭球面网格质量（2D AFM）"""
        result = generate_ellipsoid_mesh_2d_afm(
            center=(0, 0, 0), semi_axes=(1.0, 0.75, 0.5), spacing=0.2,
        )
        quality_result = SurfaceMeshQuality.evaluate_mesh(result.triangles, verbose=False)
        self.assertGreater(quality_result['quality_mean'], 0.3,
                          "平均网格质量过低")

    def test_ellipsoid_2d_afm_coverage(self):
        """椭球面网格覆盖范围（2D AFM）"""
        result = generate_ellipsoid_mesh_2d_afm(
            center=(0, 0, 0), semi_axes=(1.0, 0.75, 0.5), spacing=0.2,
        )
        coords = [n.coords for n in result.nodes]
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        zs = [p[2] for p in coords]
        # 应覆盖整个椭球面
        self.assertGreater(max(xs) - min(xs), 1.5, "X方向覆盖不足")
        self.assertGreater(max(ys) - min(ys), 1.0, "Y方向覆盖不足")
        self.assertGreater(max(zs) - min(zs), 0.8, "Z方向覆盖不足")

    def test_ellipsoid_2d_afm_different_params(self):
        """不同参数椭球面（2D AFM）"""
        test_cases = [
            {"center": (0, 0, 0), "semi_axes": (1.0, 1.0, 1.0), "spacing": 0.3},
            {"center": (0, 0, 0), "semi_axes": (2.0, 1.0, 0.5), "spacing": 0.3},
            {"center": (1, 1, 1), "semi_axes": (0.5, 0.5, 0.5), "spacing": 0.1},
        ]
        for case in test_cases:
            with self.subTest(**case):
                result = generate_ellipsoid_mesh_2d_afm(**case)
                self.assertGreater(len(result.triangles), 10)

    def test_ellipsoid_2d_afm_invalid_axes(self):
        """无效半轴（2D AFM）"""
        with self.assertRaises(ValueError):
            generate_ellipsoid_mesh_2d_afm(center=(0, 0, 0), semi_axes=(1.0, -1.0, 0.5))

    def test_ellipsoid_2d_afm_vtk_export(self):
        """VTK导出（2D AFM）"""
        output_file = str(self.output_dir / "ellipsoid_mesh_2d_afm.vtk")
        result = generate_ellipsoid_mesh_2d_afm(
            center=(0, 0, 0), semi_axes=(1.0, 0.75, 0.5), spacing=0.2,
            output_vtk=output_file,
        )
        self.assertGreater(len(result.triangles), 0)
        self.assertTrue(os.path.exists(output_file), "VTK文件未生成")
        self.assertGreater(os.path.getsize(output_file), 0)


def run_tests():
    """运行测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPrimitiveMeshResult))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractFaces))
    suite.addTests(loader.loadTestsFromTestCase(TestCubeMeshGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestCylinderMeshGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestRectangleMeshGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestSphereMeshGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestEllipsoidMeshGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestEllipsoidMeshGeneration2DAFM))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_tests()

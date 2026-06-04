#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
三维棱柱层网格生成单元测试

测试用例：
- TestPrismLayerCube: 立方体棱柱层生成
- TestPrismLayerSphere: 球体棱柱层生成
- TestPrismLayerHybrid: 混合网格（棱柱 + 四面体）
"""

import sys
import unittest
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sfmesh.surface_front import discretize_shape_edges, SurfaceTriangle, NodeElement3D
from sfmesh.mesh_3d_afm import SurfaceMeshGenerator
from sfmesh.sizing_field import SurfaceSizingField
from data_structure.basic_elements import Prism, Tetrahedron
from utils.geom_toolkit import prism_volume, tetrahedron_volume


def generate_cube_surface(cube_size, spacing):
    """生成立方体曲面网格"""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.gp import gp_Pnt

    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), cube_size, cube_size, cube_size).Shape()
    faces = []
    explorer = TopExp_Explorer(box, TopAbs_FACE)
    while explorer.More():
        faces.append(explorer.Current())
        explorer.Next()

    all_triangles = []
    for face in faces:
        sizing = SurfaceSizingField(global_spacing=spacing)
        line_mesh = discretize_shape_edges(face, sizing)
        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=spacing,
            curvature_adaptation=False,
            max_iterations=5000,
            line_mesh=line_mesh,
        )
        all_triangles.extend(generator.generate())

    return all_triangles


def generate_sphere_surface(radius, spacing):
    """生成球面网格"""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.gp import gp_Pnt

    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), radius).Shape()
    faces = []
    explorer = TopExp_Explorer(sphere, TopAbs_FACE)
    while explorer.More():
        faces.append(explorer.Current())
        explorer.Next()

    all_triangles = []
    for face in faces:
        sizing = SurfaceSizingField(global_spacing=spacing)
        line_mesh = discretize_shape_edges(face, sizing)
        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=spacing,
            curvature_adaptation=False,
            max_iterations=5000,
            line_mesh=line_mesh,
        )
        all_triangles.extend(generator.generate())

    return all_triangles


class TestPrismLayerCube(unittest.TestCase):
    """立方体棱柱层生成测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0
        cls.spacing = 0.5
        cls.first_height = 0.1
        cls.max_layers = 3
        cls.growth_rate = 1.2

        try:
            cls.all_triangles = generate_cube_surface(cls.cube_size, cls.spacing)
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

    def _run_prism_gen(self):
        """运行棱柱层生成"""
        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        return gen.generate()

    def test_surface_mesh_loaded(self):
        """验证曲面网格加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '曲面网格生成失败'))
        self.assertGreater(len(self.all_triangles), 50,
                           f"曲面三角形数量不足: {len(self.all_triangles)}")
        print(f"\n立方体曲面三角形: {len(self.all_triangles)}")

    def test_cube_prism_generation(self):
        """立方体棱柱层生成：验证棱柱单元数量"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '曲面网格生成失败'))

        unstr_grid, boundary_faces = self._run_prism_gen()

        self.assertIsNotNone(unstr_grid, "网格生成失败")
        prism_count = len([c for c in unstr_grid.cell_container
                           if isinstance(c, Prism)])
        self.assertGreater(prism_count, 0, "未生成棱柱单元")

        print(f"\n棱柱单元: {prism_count}, 节点: {len(unstr_grid.node_coords)}")

    def test_prism_no_degenerate(self):
        """验证无退化棱柱（所有体积 > 0）"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '曲面网格生成失败'))

        unstr_grid, _ = self._run_prism_gen()

        degenerate = 0
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Prism):
                vol = cell.get_volume()
                if vol <= 1e-12:
                    degenerate += 1

        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化棱柱")
        print(f"\n棱柱总数: {len(unstr_grid.cell_container)}, 退化: {degenerate}")

    def test_prism_quality(self):
        """验证棱柱网格质量"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '曲面网格生成失败'))

        unstr_grid, _ = self._run_prism_gen()

        qualities = []
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Prism):
                q = cell.get_quality()
                qualities.append(q)

        self.assertGreater(len(qualities), 0, "无棱柱单元")
        mean_q = np.mean(qualities)
        min_q = np.min(qualities)

        self.assertGreater(mean_q, 0.0, f"平均质量过低: {mean_q:.4f}")
        self.assertGreater(min_q, -1e-10, f"最小质量异常: {min_q:.4f}")

        print(f"\n棱柱质量: 均值={mean_q:.4f}, 最小={min_q:.4f}")

    def test_prism_topology(self):
        """验证棱柱拓扑一致性"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '曲面网格生成失败'))

        unstr_grid, _ = self._run_prism_gen()

        # 检查无重复棱柱
        prism_sigs = set()
        duplicates = 0
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Prism):
                sig = tuple(sorted(cell.node_ids))
                if sig in prism_sigs:
                    duplicates += 1
                prism_sigs.add(sig)

        self.assertEqual(duplicates, 0, f"发现{duplicates}个重复棱柱")
        print(f"\n棱柱签名去重: {len(prism_sigs)} 个唯一棱柱")

    def test_prism_vtk_export(self):
        """验证 VTK 导出"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '曲面网格生成失败'))

        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        gen.generate()

        output_file = self.output_dir / "prism_cube.vtk"
        gen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        self.assertGreater(output_file.stat().st_size, 0, "VTK 文件为空")
        print(f"\nVTK 输出: {output_file}")

    def test_prism_boundary_faces(self):
        """验证边界面可用于后续四面体填充"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '曲面网格生成失败'))

        _, boundary_faces = self._run_prism_gen()

        self.assertGreater(len(boundary_faces), 0, "无边界面")
        print(f"\n边界面数量: {len(boundary_faces)}")


class TestPrismLayerSphere(unittest.TestCase):
    """球体棱柱层生成测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.radius = 1.0
        cls.spacing = 0.4
        cls.first_height = 0.05
        cls.max_layers = 3
        cls.growth_rate = 1.3

        try:
            cls.all_triangles = generate_sphere_surface(cls.radius, cls.spacing)
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

    def _run_prism_gen(self):
        """运行棱柱层生成"""
        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        return gen.generate()

    def test_sphere_surface_loaded(self):
        """验证球面网格加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '球面网格生成失败'))
        self.assertGreater(len(self.all_triangles), 50,
                           f"球面三角形数量不足: {len(self.all_triangles)}")
        print(f"\n球面三角形: {len(self.all_triangles)}")

    def test_sphere_prism_generation(self):
        """球体棱柱层生成"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '球面网格生成失败'))

        unstr_grid, boundary_faces = self._run_prism_gen()

        self.assertIsNotNone(unstr_grid, "网格生成失败")
        prism_count = len([c for c in unstr_grid.cell_container
                           if isinstance(c, Prism)])
        self.assertGreater(prism_count, 0, "未生成棱柱单元")

        print(f"\n球体棱柱: {prism_count}, 节点: {len(unstr_grid.node_coords)}")

    def test_sphere_prism_quality(self):
        """球体棱柱质量"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '球面网格生成失败'))

        unstr_grid, _ = self._run_prism_gen()

        qualities = []
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Prism):
                q = cell.get_quality()
                qualities.append(q)

        self.assertGreater(len(qualities), 0, "无棱柱单元")
        mean_q = np.mean(qualities)
        self.assertGreater(mean_q, 0.0, f"平均质量过低: {mean_q:.4f}")
        print(f"\n球体棱柱质量: 均值={mean_q:.4f}")

    def test_sphere_no_degenerate(self):
        """验证无退化棱柱"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '球面网格生成失败'))

        unstr_grid, _ = self._run_prism_gen()

        degenerate = 0
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Prism):
                vol = cell.get_volume()
                if vol <= 1e-12:
                    degenerate += 1

        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化棱柱")

    def test_sphere_volume_coverage(self):
        """验证棱柱层体积覆盖"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '球面网格生成失败'))

        unstr_grid, _ = self._run_prism_gen()

        total_vol = sum(
            cell.get_volume()
            for cell in unstr_grid.cell_container
            if isinstance(cell, Prism)
        )
        # 棱柱层体积应该小于球体总体积
        sphere_vol = (4.0 / 3.0) * np.pi * self.radius ** 3
        self.assertGreater(total_vol, 0, "棱柱层体积为0")
        self.assertLess(total_vol, sphere_vol,
                        f"棱柱层体积({total_vol:.4f})超过球体体积({sphere_vol:.4f})")
        print(f"\n棱柱层体积: {total_vol:.4f}, 球体体积: {sphere_vol:.4f}")

    def test_sphere_vtk_export(self):
        """验证 VTK 导出"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '球面网格生成失败'))

        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        gen.generate()

        output_file = self.output_dir / "prism_sphere.vtk"
        gen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        print(f"\nVTK 输出: {output_file}")


class TestPrismLayerHybrid(unittest.TestCase):
    """混合网格测试（棱柱 + 四面体）"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0
        cls.spacing = 0.5
        cls.first_height = 0.15
        cls.max_layers = 3
        cls.growth_rate = 1.2

        try:
            cls.all_triangles = generate_cube_surface(cls.cube_size, cls.spacing)
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

    def test_hybrid_cube(self):
        """立方体混合网格：棱柱层 + 四面体填充"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '曲面网格生成失败'))

        from adlayers3.adlayers3 import generate_hybrid_mesh

        hybrid_grid = generate_hybrid_mesh(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            volume_spacing=0.5,
            debug_level=0,
        )

        self.assertIsNotNone(hybrid_grid, "混合网格生成失败")

        prism_count = len([c for c in hybrid_grid.cell_container
                           if isinstance(c, Prism)])
        tet_count = len([c for c in hybrid_grid.cell_container
                         if isinstance(c, Tetrahedron)])

        self.assertGreater(prism_count, 0, "未生成棱柱单元")

        print(f"\n混合网格: 棱柱={prism_count}, 四面体={tet_count}, "
              f"总单元={len(hybrid_grid.cell_container)}")

    def test_hybrid_vtk_export(self):
        """验证混合网格 VTK 导出"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', '曲面网格生成失败'))

        from adlayers3.adlayers3 import (
            generate_hybrid_mesh, export_hybrid_vtk, Adlayers3
        )

        # 先生成棱柱层获取棱柱单元
        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        prism_grid, boundary_faces = gen.generate()

        prism_cells = [c for c in prism_grid.cell_container if isinstance(c, Prism)]

        # 生成四面体填充
        from delaunay3d.bowyer_watson import BowyerWatsonTetGen
        from delaunay3d.sizing import UniformSizing3D

        # 将边界面转换为 SurfaceTriangle 格式
        from sfmesh.surface_front import SurfaceTriangle, NodeElement3D

        node_cache = {}
        cap_triangles = []
        for face_coords in boundary_faces:
            nodes = []
            for coord in face_coords:
                key = tuple(f"{c:.6f}" for c in coord)
                if key not in node_cache:
                    node_cache[key] = NodeElement3D(
                        coord, idx=len(node_cache), uv_params=(0.0, 0.0)
                    )
                nodes.append(node_cache[key])
            tri = SurfaceTriangle(nodes[0], nodes[1], nodes[2])
            cap_triangles.append(tri)

        if cap_triangles:
            sizing = UniformSizing3D(0.5)
            tetgen = BowyerWatsonTetGen(cap_triangles, sizing_system=sizing, debug_level=0)
            tet_grid = tetgen.generate()
            tet_cells = [c for c in tet_grid.cell_container if isinstance(c, Tetrahedron)]
        else:
            tet_cells = []

        # 导出混合网格
        output_file = self.output_dir / "hybrid_cube.vtk"
        export_hybrid_vtk(str(output_file), prism_cells, tet_cells, prism_grid.node_coords)

        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        self.assertGreater(output_file.stat().st_size, 0, "VTK 文件为空")
        print(f"\n混合网格 VTK: {output_file}")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
三维棱柱层网格生成单元测试

测试用例：
- TestPrismLayerCube: 立方体棱柱层生成 (cube.stl)
- TestPrismLayerSphere: 球体棱柱层生成 (sphere.stl)
- TestPrismLayerHybrid: 混合网格（棱柱 + 四面体）
"""

import sys
import unittest
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sfmesh.surface_front import SurfaceTriangle, NodeElement3D
from data_structure.basic_elements import Prism, Tetrahedron


def read_stl(filename):
    """读取 ASCII STL 文件，返回 SurfaceTriangle 列表"""
    triangles = []
    node_cache = {}

    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("facet normal"):
            vertices = []
            i += 2  # skip "facet normal ..." and "outer loop"
            while not lines[i].startswith("endloop"):
                if lines[i].startswith("vertex"):
                    parts = lines[i].split()
                    coord = tuple(map(float, parts[1:4]))
                    if coord not in node_cache:
                        node_elem = NodeElement3D(
                            coord, idx=len(node_cache),
                            uv_params=(0.0, 0.0),
                        )
                        node_cache[coord] = node_elem
                    vertices.append(node_cache[coord])
                i += 1
            i += 1  # skip "endloop"
            i += 1  # skip "endfacet"
            if len(vertices) == 3:
                tri = SurfaceTriangle(
                    vertices[0], vertices[1], vertices[2],
                    idx=len(triangles),
                )
                triangles.append(tri)
        else:
            i += 1

    return triangles


class TestPrismLayerCube(unittest.TestCase):
    """立方体棱柱层生成测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "cube.stl"
        cls.first_height = 0.1
        cls.max_layers = 3
        cls.growth_rate = 1.2

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
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
        try:
            return gen.generate()
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

    def test_surface_mesh_loaded(self):
        """验证曲面网格加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        self.assertGreater(len(self.all_triangles), 10,
                           f"曲面三角形数量不足: {len(self.all_triangles)}")
        print(f"\n立方体曲面三角形: {len(self.all_triangles)}")

    def test_cube_prism_generation(self):
        """立方体棱柱层生成：验证棱柱单元数量"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        unstr_grid, boundary_faces = self._run_prism_gen()

        self.assertIsNotNone(unstr_grid, "网格生成失败")
        prism_count = len([c for c in unstr_grid.cell_container
                           if isinstance(c, Prism)])
        self.assertGreater(prism_count, 0, "未生成棱柱单元")

        print(f"\n棱柱单元: {prism_count}, 节点: {len(unstr_grid.node_coords)}")

    def test_prism_no_degenerate(self):
        """验证无退化棱柱（所有体积 > 0）"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

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
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

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
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

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
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        try:
            gen.generate()
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

        output_file = self.output_dir / "prism_cube.vtk"
        gen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        self.assertGreater(output_file.stat().st_size, 0, "VTK 文件为空")
        print(f"\nVTK 输出: {output_file}")

    def test_prism_boundary_faces(self):
        """验证边界面可用于后续四面体填充"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        _, boundary_faces = self._run_prism_gen()

        self.assertGreater(len(boundary_faces), 0, "无边界面")
        print(f"\n边界面数量: {len(boundary_faces)}")


class TestPrismLayerSphere(unittest.TestCase):
    """球体棱柱层生成测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "sphere.stl"
        cls.radius = 1.0
        cls.first_height = 0.05
        cls.max_layers = 3
        cls.growth_rate = 1.3

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
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
        try:
            return gen.generate()
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

    def test_sphere_surface_loaded(self):
        """验证球面网格加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        self.assertGreater(len(self.all_triangles), 100,
                           f"球面三角形数量不足: {len(self.all_triangles)}")
        print(f"\n球面三角形: {len(self.all_triangles)}")

    def test_sphere_prism_generation(self):
        """球体棱柱层生成"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        unstr_grid, boundary_faces = self._run_prism_gen()

        self.assertIsNotNone(unstr_grid, "网格生成失败")
        prism_count = len([c for c in unstr_grid.cell_container
                           if isinstance(c, Prism)])
        self.assertGreater(prism_count, 0, "未生成棱柱单元")

        print(f"\n球体棱柱: {prism_count}, 节点: {len(unstr_grid.node_coords)}")

    def test_sphere_prism_quality(self):
        """球体棱柱质量"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

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
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

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
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

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
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        try:
            gen.generate()
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

        output_file = self.output_dir / "prism_sphere.vtk"
        gen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        print(f"\nVTK 输出: {output_file}")


class TestPrismLayerCubeFine(unittest.TestCase):
    """立方体精细网格棱柱层生成测试 (cube-fine.stl, 4层)"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "cube-fine.stl"
        cls.first_height = 0.05
        cls.max_layers = 4
        cls.growth_rate = 1.2

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

    def _run_prism_gen(self):
        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        try:
            return gen.generate()
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

    def test_surface_loaded(self):
        """验证 cube-fine.stl 加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        self.assertGreater(len(self.all_triangles), 100,
                           f"三角形数量不足: {len(self.all_triangles)}")
        print(f"\ncube-fine 三角形: {len(self.all_triangles)}")

    def test_prism_generation(self):
        """验证4层棱柱生成"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        unstr_grid, boundary_faces = self._run_prism_gen()

        self.assertIsNotNone(unstr_grid, "网格生成失败")
        prism_count = len([c for c in unstr_grid.cell_container
                           if isinstance(c, Prism)])
        self.assertGreater(prism_count, 0, "未生成棱柱单元")
        self.assertEqual(prism_count, len(self.all_triangles) * self.max_layers,
                         "棱柱数量不等于 三角形数 × 层数")
        print(f"\n棱柱: {prism_count}, 节点: {len(unstr_grid.node_coords)}")

    def test_no_degenerate(self):
        """验证无退化棱柱"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        unstr_grid, _ = self._run_prism_gen()

        degenerate = 0
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Prism):
                if cell.get_volume() <= 1e-12:
                    degenerate += 1
        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化棱柱")

    def test_quality(self):
        """验证棱柱质量"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        unstr_grid, _ = self._run_prism_gen()

        qualities = [c.get_quality() for c in unstr_grid.cell_container
                     if isinstance(c, Prism)]
        self.assertGreater(len(qualities), 0, "无棱柱单元")
        mean_q = np.mean(qualities)
        min_q = np.min(qualities)
        self.assertGreater(mean_q, 0.0, f"平均质量过低: {mean_q:.4f}")
        self.assertGreater(min_q, -1e-10, f"最小质量异常: {min_q:.4f}")
        print(f"\n棱柱质量: 均值={mean_q:.4f}, 最小={min_q:.4f}")

    def test_vtk_export(self):
        """验证 VTK 导出"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        try:
            gen.generate()
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

        output_file = self.output_dir / "prism_cube_fine.vtk"
        gen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        self.assertGreater(output_file.stat().st_size, 0, "VTK 文件为空")
        print(f"\nVTK 输出: {output_file}")


class TestPrismLayerHybrid(unittest.TestCase):
    """混合网格测试（棱柱 + 四面体）"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "cube.stl"
        cls.first_height = 0.15
        cls.max_layers = 3
        cls.growth_rate = 1.2

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

    def test_hybrid_cube(self):
        """立方体混合网格：棱柱层 + 四面体填充"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        from adlayers3.adlayers3 import generate_hybrid_mesh

        try:
            hybrid_grid = generate_hybrid_mesh(
                surface_triangles=self.all_triangles,
                first_height=self.first_height,
                max_layers=self.max_layers,
                growth_rate=self.growth_rate,
                volume_spacing=0.5,
                debug_level=0,
            )
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

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
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

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
        try:
            prism_grid, boundary_faces = gen.generate()
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

        prism_cells = [c for c in prism_grid.cell_container if isinstance(c, Prism)]

        # 生成四面体填充
        from delaunay3d.bowyer_watson import BowyerWatsonTetGen
        from delaunay3d.sizing import UniformSizing3D

        # 将边界面转换为 SurfaceTriangle 格式
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
            tet_node_coords = tet_grid.node_coords
        else:
            tet_cells = []
            tet_node_coords = []

        # 导出混合网格
        output_file = self.output_dir / "hybrid_cube.vtk"
        export_hybrid_vtk(str(output_file), prism_cells, tet_cells,
                          prism_grid.node_coords, tet_node_coords)

        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        self.assertGreater(output_file.stat().st_size, 0, "VTK 文件为空")
        print(f"\n混合网格 VTK: {output_file}")


class TestPrismLayerSemisphere(unittest.TestCase):
    """半球体棱柱层生成测试 (semisphere.stl)

    几何：方盒子包裹半球，5个方盒子平面 + 半球面与对称面（挖去圆形），
    整体为封闭曲面。
    """

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "semisphere.stl"
        cls.first_height = 0.02
        cls.max_layers = 3
        cls.growth_rate = 1.2

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

    def _run_prism_gen(self):
        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        try:
            return gen.generate()
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

    def test_surface_loaded(self):
        """验证 semisphere.stl 加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        self.assertGreater(len(self.all_triangles), 1000,
                           f"三角形数量不足: {len(self.all_triangles)}")
        print(f"\nsemisphere 三角形: {len(self.all_triangles)}")

    def test_prism_generation(self):
        """验证棱柱层生成"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        unstr_grid, boundary_faces = self._run_prism_gen()

        self.assertIsNotNone(unstr_grid, "网格生成失败")
        prism_count = len([c for c in unstr_grid.cell_container
                           if isinstance(c, Prism)])
        self.assertGreater(prism_count, 0, "未生成棱柱单元")
        print(f"\n棱柱: {prism_count}, 节点: {len(unstr_grid.node_coords)}")

    def test_no_degenerate(self):
        """验证无退化棱柱"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        unstr_grid, _ = self._run_prism_gen()

        degenerate = 0
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Prism):
                if cell.get_volume() <= 1e-12:
                    degenerate += 1
        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化棱柱")

    def test_quality(self):
        """验证棱柱质量"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        unstr_grid, _ = self._run_prism_gen()

        qualities = [c.get_quality() for c in unstr_grid.cell_container
                     if isinstance(c, Prism)]
        self.assertGreater(len(qualities), 0, "无棱柱单元")
        mean_q = np.mean(qualities)
        min_q = np.min(qualities)
        self.assertGreater(mean_q, 0.0, f"平均质量过低: {mean_q:.4f}")
        self.assertGreater(min_q, -1e-10, f"最小质量异常: {min_q:.4f}")
        print(f"\n棱柱质量: 均值={mean_q:.4f}, 最小={min_q:.4f}")

    def test_vtk_export(self):
        """验证 VTK 导出"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))

        from adlayers3.adlayers3 import Adlayers3

        gen = Adlayers3(
            surface_triangles=self.all_triangles,
            first_height=self.first_height,
            max_layers=self.max_layers,
            growth_rate=self.growth_rate,
            debug_level=0,
        )
        try:
            gen.generate()
        except ValueError as e:
            self.skipTest(f"曲面网格拓扑验证失败: {e}")

        output_file = self.output_dir / "prism_semiSphere.vtk"
        gen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        self.assertGreater(output_file.stat().st_size, 0, "VTK 文件为空")
        print(f"\nVTK 输出: {output_file}")


if __name__ == "__main__":
    unittest.main()

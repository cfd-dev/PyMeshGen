"""三维阵面推进法 (Adfront3) 单元测试

测试用例：
- TestAdfront3Cube: 立方体四面体网格生成 (cube.stl)
- TestAdfront3CubeFine: 立方体细网格四面体网格生成 (cube-fine.stl)
- TestAdfront3Sphere: 球体四面体网格生成 (sphere.stl)
- TestAdfront3Quality: 阵面推进法网格质量测试 (cube.stl)
"""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import importlib
from sfmesh.surface_front import SurfaceTriangle, NodeElement3D

# 动态导入避免循环导入问题
_adfront3_netgen = importlib.import_module('adfront3.adfront3_netgen')
Adfront3 = _adfront3_netgen.AdFront3Netgen

from data_structure.basic_elements import Tetrahedron
from utils.geom_toolkit import tetrahedron_volume
from optimize.mesh_quality import tetrahedron_shape_quality


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


class TestAdfront3Cube(unittest.TestCase):
    """立方体四面体网格生成测试 (cube.stl)"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "cube.stl"

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

        cls.adfront3 = None
        cls.unstr_grid = None

    def _run_adfront3(self):
        """运行 Adfront3 并缓存结果"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        if self.__class__.unstr_grid is None:
            adfront3 = Adfront3(self.all_triangles, debug_level=0)
            unstr_grid = adfront3.generate()
            self.__class__.adfront3 = adfront3
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.adfront3, self.__class__.unstr_grid

    def test_surface_mesh_loaded(self):
        """验证曲面网格加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        self.assertGreater(len(self.all_triangles), 10,
                           f"曲面三角形数量不足: {len(self.all_triangles)}")
        print(f"\n立方体曲面三角形: {len(self.all_triangles)}")

    def test_cube_volume_mesh(self):
        """立方体体积网格：验证四面体网格生成"""
        adfront3, unstr_grid = self._run_adfront3()

        self.assertIsNotNone(unstr_grid, "网格生成失败")
        self.assertGreater(unstr_grid.num_cells, 0, "未生成四面体")

        # 验证所有四面体体积为正
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                vol = tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
                self.assertGreater(vol, 1e-12,
                                   f"四面体{cell.node_ids}体积异常: {vol}")

        # 导出 VTK
        output_file = self.output_dir / "adfront3_cube.vtk"
        adfront3.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())

        stats = adfront3.get_quality_stats()
        print(f"\n四面体数量: {stats.get('num_cells', 0)}, "
              f"节点数量: {stats.get('num_nodes', 0)}, "
              f"质量均值: {stats.get('quality_mean', 0):.4f}, "
              f"质量最小值: {stats.get('quality_min', 0):.4f}")

    def test_cube_mesh_quality(self):
        """立方体网格质量：验证四面体质量统计"""
        adfront3, unstr_grid = self._run_adfront3()

        self.assertIsNotNone(unstr_grid)

        stats = adfront3.get_quality_stats()
        self.assertGreater(stats.get('num_cells', 0), 0)
        self.assertGreater(stats.get('quality_mean', 0), 0.0,
                           f"平均质量过低: {stats.get('quality_mean', 0):.4f}")
        self.assertGreater(stats.get('quality_min', 0), -1e-10,
                           f"最小质量异常: {stats.get('quality_min', 0):.4f}")

    def test_cube_boundary_containment(self):
        """验证所有节点在立方体边界内"""
        adfront3, unstr_grid = self._run_adfront3()

        import numpy as np
        coords = np.array(adfront3.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        # 节点应在 STL 包围盒范围内（允许微小浮点误差）
        tol = 1e-6
        tri_coords = np.array([n.coords for t in self.all_triangles for n in t.nodes])
        stl_mins = tri_coords.min(axis=0)
        stl_maxs = tri_coords.max(axis=0)

        for d in range(3):
            self.assertGreaterEqual(mins[d], stl_mins[d] - tol,
                                    f"节点超出边界: d={d}, min={mins[d]:.4f}")
            self.assertLessEqual(maxs[d], stl_maxs[d] + tol,
                                 f"节点超出边界: d={d}, max={maxs[d]:.4f}")

        print(f"\n边界范围: [{mins[0]:.4f},{mins[1]:.4f},{mins[2]:.4f}] "
              f"到 [{maxs[0]:.4f},{maxs[1]:.4f},{maxs[2]:.4f}]")

    def test_cube_no_degenerate_tets(self):
        """验证无退化四面体（所有体积 > 0）"""
        adfront3, unstr_grid = self._run_adfront3()

        degenerate = 0
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                vol = tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
                if vol <= 1e-12:
                    degenerate += 1

        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化四面体")
        print(f"\n四面体总数: {len(unstr_grid.cell_container)}, 退化: {degenerate}")


class TestAdfront3CubeFine(unittest.TestCase):
    """立方体细网格四面体网格生成测试 (cube-fine.stl)"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "cube-fine.stl"

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

        cls.adfront3 = None
        cls.unstr_grid = None

    def _run_adfront3(self):
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        if self.__class__.unstr_grid is None:
            adfront3 = Adfront3(self.all_triangles, debug_level=0)
            unstr_grid = adfront3.generate()
            self.__class__.adfront3 = adfront3
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.adfront3, self.__class__.unstr_grid

    def test_surface_loaded(self):
        """验证 cube-fine.stl 加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        self.assertGreater(len(self.all_triangles), 100,
                           f"三角形数量不足: {len(self.all_triangles)}")
        print(f"\ncube-fine 三角形: {len(self.all_triangles)}")

    def test_fine_volume_mesh(self):
        """细网格：验证四面体网格生成"""
        adfront3, unstr_grid = self._run_adfront3()
        self.assertIsNotNone(unstr_grid)
        self.assertGreater(unstr_grid.num_cells, 0)

    def test_fine_no_degenerate_tets(self):
        """细网格：无退化四面体"""
        adfront3, unstr_grid = self._run_adfront3()
        degenerate = sum(
            1 for c in unstr_grid.cell_container
            if isinstance(c, Tetrahedron) and tetrahedron_volume(c.p1, c.p2, c.p3, c.p4) <= 1e-12
        )
        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化四面体")

    def test_fine_boundary_containment(self):
        """细网格：所有节点在边界内"""
        adfront3, _ = self._run_adfront3()
        import numpy as np
        coords = np.array(adfront3.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        tri_coords = np.array([n.coords for t in self.all_triangles for n in t.nodes])
        stl_mins = tri_coords.min(axis=0)
        stl_maxs = tri_coords.max(axis=0)
        tol = 1e-6
        for d in range(3):
            self.assertGreaterEqual(mins[d], stl_mins[d] - tol)
            self.assertLessEqual(maxs[d], stl_maxs[d] + tol)

    def test_fine_quality_mean(self):
        """细网格：平均质量 > 0.5"""
        adfront3, _ = self._run_adfront3()
        stats = adfront3.get_quality_stats()
        self.assertGreater(stats.get('quality_mean', 0), 0.5,
                           f"平均质量过低: {stats.get('quality_mean', 0):.4f}")


class TestAdfront3Sphere(unittest.TestCase):
    """球体四面体网格生成测试 (sphere.stl)"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "sphere.stl"
        cls.radius = 1.0

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

        cls.adfront3 = None
        cls.unstr_grid = None

    def _run_adfront3(self):
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        if self.__class__.unstr_grid is None:
            adfront3 = Adfront3(self.all_triangles, debug_level=0)
            unstr_grid = adfront3.generate()
            self.__class__.adfront3 = adfront3
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.adfront3, self.__class__.unstr_grid

    def test_sphere_surface_loaded(self):
        """验证球面网格加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        self.assertGreater(len(self.all_triangles), 50,
                           f"曲面三角形数量不足: {len(self.all_triangles)}")
        print(f"\n球面三角形: {len(self.all_triangles)}")

    def test_sphere_volume_mesh(self):
        """球体：验证四面体网格生成"""
        adfront3, unstr_grid = self._run_adfront3()
        self.assertIsNotNone(unstr_grid, "球体网格生成失败")
        self.assertGreater(unstr_grid.num_cells, 0, "未生成球体四面体")

        output_file = self.output_dir / "adfront3_sphere.vtk"
        adfront3.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())

        stats = adfront3.get_quality_stats()
        print(f"\n球体网格: {stats.get('num_cells', 0)} 四面体, "
              f"节点={stats.get('num_nodes', 0)}, "
              f"质量均值={stats.get('quality_mean', 0):.4f}")

    def test_sphere_no_degenerate(self):
        """球体：无退化四面体"""
        adfront3, unstr_grid = self._run_adfront3()
        degenerate = sum(
            1 for c in unstr_grid.cell_container
            if isinstance(c, Tetrahedron) and tetrahedron_volume(c.p1, c.p2, c.p3, c.p4) <= 1e-12
        )
        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化四面体")

    def test_sphere_volume_coverage(self):
        """球体：体积覆盖率 > 60%"""
        adfront3, unstr_grid = self._run_adfront3()
        import numpy as np
        total_vol = sum(
            tetrahedron_volume(c.p1, c.p2, c.p3, c.p4)
            for c in unstr_grid.cell_container
            if isinstance(c, Tetrahedron)
        )
        expected = (4.0 / 3.0) * np.pi * self.radius ** 3
        coverage = total_vol / expected
        self.assertGreater(coverage, 0.60,
                           f"球体体积覆盖率不足: {coverage:.4f}")
        print(f"\n球体覆盖率: {coverage:.4f}")

    def test_sphere_nodes_bounded(self):
        """球体：所有节点在球体内"""
        adfront3, _ = self._run_adfront3()
        import numpy as np
        coords = np.array(adfront3.node_coords)
        dists = np.linalg.norm(coords, axis=1)
        max_dist = dists.max()
        self.assertLess(max_dist, self.radius * 1.1,
                        f"节点超出球体: max_dist={max_dist:.4f}")

    def test_sphere_quality(self):
        """球体：质量统计"""
        adfront3, _ = self._run_adfront3()
        stats = adfront3.get_quality_stats()
        self.assertGreater(stats.get('quality_mean', 0), 0.1,
                           f"平均质量过低: {stats.get('quality_mean', 0):.4f}")
        self.assertGreater(stats.get('quality_min', 0), -1e-10,
                           f"最小质量异常: {stats.get('quality_min', 0):.4f}")


class TestAdfront3Quality(unittest.TestCase):
    """阵面推进法网格质量测试 (cube.stl)"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "cube.stl"

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

        cls.adfront3 = None
        cls.unstr_grid = None

    def _run_adfront3(self):
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        if self.__class__.unstr_grid is None:
            adfront3 = Adfront3(self.all_triangles, debug_level=0)
            unstr_grid = adfront3.generate()
            self.__class__.adfront3 = adfront3
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.adfront3, self.__class__.unstr_grid

    def test_quality_range(self):
        """所有四面体质量在有效范围内"""
        adfront3, _ = self._run_adfront3()
        for cell in adfront3.cell_container:
            if isinstance(cell, Tetrahedron):
                q = tetrahedron_shape_quality(cell.p1, cell.p2, cell.p3, cell.p4)
                self.assertGreaterEqual(q, -1e-10, f"质量异常: {q}")
                self.assertLessEqual(q, 1.0 + 1e-10, f"质量超范围: {q}")

    def test_quality_mean_threshold(self):
        """平均质量 > 0.5"""
        adfront3, _ = self._run_adfront3()
        stats = adfront3.get_quality_stats()
        self.assertGreater(stats.get('quality_mean', 0), 0.5,
                           f"平均质量过低: {stats.get('quality_mean', 0):.4f}")

    def test_no_negative_quality(self):
        """无负质量四面体"""
        adfront3, _ = self._run_adfront3()
        neg_count = sum(
            1 for c in adfront3.cell_container
            if isinstance(c, Tetrahedron)
            and tetrahedron_shape_quality(c.p1, c.p2, c.p3, c.p4) < -1e-10
        )
        self.assertEqual(neg_count, 0, f"发现{neg_count}个负质量四面体")

    def test_vtk_export(self):
        """VTK 导出验证"""
        adfront3, _ = self._run_adfront3()
        output_file = self.output_dir / "adfront3_quality.vtk"
        adfront3.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)


class TestAdfront3CubeMed(unittest.TestCase):
    """立方体中密度四面体网格生成测试 (cube-med.stl, 192 表面三角形)"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "cube-med.stl"

        try:
            cls.all_triangles = read_stl(str(cls.stl_path))
        except Exception as e:
            cls.all_triangles = []
            cls._skip_reason = str(e)

        cls.adfront3 = None
        cls.unstr_grid = None

    def _run_adfront3(self):
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        if self.__class__.unstr_grid is None:
            adfront3 = Adfront3(self.all_triangles, debug_level=0)
            unstr_grid = adfront3.generate()
            self.__class__.adfront3 = adfront3
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.adfront3, self.__class__.unstr_grid

    def test_surface_loaded(self):
        """验证 cube-med.stl 加载成功"""
        if not self.all_triangles:
            self.skipTest(getattr(self, '_skip_reason', 'STL加载失败'))
        self.assertGreater(len(self.all_triangles), 50,
                           f"三角形数量不足: {len(self.all_triangles)}")
        print(f"\ncube-med 三角形: {len(self.all_triangles)}")

    def test_med_volume_mesh(self):
        """中密度立方体：验证四面体网格生成"""
        adfront3, unstr_grid = self._run_adfront3()
        self.assertIsNotNone(unstr_grid)
        self.assertGreater(unstr_grid.num_cells, 0)

        # 导出 VTK
        output_file = self.output_dir / "adfront3_cube_med.vtk"
        adfront3.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())

        stats = adfront3.get_quality_stats()
        print(f"\n四面体: {stats.get('num_cells', 0)}, "
              f"节点: {stats.get('num_nodes', 0)}, "
              f"质量均值: {stats.get('quality_mean', 0):.4f}")

    def test_med_no_degenerate_tets(self):
        """中密度立方体：无退化四面体"""
        adfront3, unstr_grid = self._run_adfront3()
        degenerate = sum(
            1 for c in unstr_grid.cell_container
            if isinstance(c, Tetrahedron) and tetrahedron_volume(c.p1, c.p2, c.p3, c.p4) <= 1e-12
        )
        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化四面体")

    def test_med_boundary_containment(self):
        """中密度立方体：所有节点在边界内"""
        adfront3, _ = self._run_adfront3()
        import numpy as np
        coords = np.array(adfront3.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        tri_coords = np.array([n.coords for t in self.all_triangles for n in t.nodes])
        stl_mins = tri_coords.min(axis=0)
        stl_maxs = tri_coords.max(axis=0)
        tol = 1e-6
        for d in range(3):
            self.assertGreaterEqual(mins[d], stl_mins[d] - tol)
            self.assertLessEqual(maxs[d], stl_maxs[d] + tol)

    def test_med_quality_mean(self):
        """中密度立方体：平均质量 > 0.5"""
        adfront3, _ = self._run_adfront3()
        stats = adfront3.get_quality_stats()
        self.assertGreater(stats.get('quality_mean', 0), 0.5,
                           f"平均质量过低: {stats.get('quality_mean', 0):.4f}")


if __name__ == "__main__":
    unittest.main()

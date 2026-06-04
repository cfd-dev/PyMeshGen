import sys
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sfmesh.surface_front import discretize_shape_edges
from sfmesh.mesh_3d_afm import SurfaceMeshGenerator
from sfmesh.sizing_field import SurfaceSizingField
from adfront3.adfront3 import Adfront3
from adfront3.sizing3d import UniformSizing3D
from data_structure.basic_elements import Tetrahedron
from utils.geom_toolkit import tetrahedron_volume
from optimize.mesh_quality import tetrahedron_shape_quality


class TestAdfront3Cube(unittest.TestCase):
    """立方体四面体网格生成测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0
        cls.spacing = 0.5
        cls.all_triangles = cls._generate_cube_surface(cls.cube_size, cls.spacing)
        cls.adfront3 = None
        cls.unstr_grid = None

    @staticmethod
    def _generate_cube_surface(cube_size, spacing):
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

    def _run_adfront3(self):
        """运行 Adfront3 并缓存结果，间距自动根据边界面网格确定"""
        if self.__class__.unstr_grid is None:
            adfront3 = Adfront3(
                self.all_triangles,
                debug_level=0,
            )
            unstr_grid = adfront3.generate()
            self.__class__.adfront3 = adfront3
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.adfront3, self.__class__.unstr_grid

    def test_cube_volume_mesh(self):
        """立方体体积网格：验证四面体网格生成"""
        self.assertGreater(len(self.all_triangles), 50,
                           f"曲面三角形数量不足: {len(self.all_triangles)}")

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

        # 节点应在 [0, cube_size] 范围内（允许微小浮点误差）
        tol = 1e-6
        self.assertGreaterEqual(mins[0], -tol, f"节点超出左边界: x_min={mins[0]}")
        self.assertGreaterEqual(mins[1], -tol, f"节点超出下边界: y_min={mins[1]}")
        self.assertGreaterEqual(mins[2], -tol, f"节点超出前边界: z_min={mins[2]}")
        self.assertLessEqual(maxs[0], self.cube_size + tol, f"节点超出右边界: x_max={maxs[0]}")
        self.assertLessEqual(maxs[1], self.cube_size + tol, f"节点超出上边界: y_max={maxs[1]}")
        self.assertLessEqual(maxs[2], self.cube_size + tol, f"节点超出后边界: z_max={maxs[2]}")

        print(f"\n边界范围: [{mins[0]:.4f},{mins[1]:.4f},{mins[2]:.4f}] "
              f"到 [{maxs[0]:.4f},{maxs[1]:.4f},{maxs[2]:.4f}]")

    def test_cube_euler_characteristic(self):
        """立方体：欧拉示性数应为 2（拓扑球面）"""
        adfront3, unstr_grid = self._run_adfront3()

        import numpy as np
        nodes = set()
        edges = set()
        faces = set()
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                nids = cell.node_ids
                nodes.update(nids)
                for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                    edges.add(tuple(sorted((nids[i], nids[j]))))
                for tri in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
                    faces.add(tuple(sorted((nids[tri[0]], nids[tri[1]], nids[tri[2]]))))

        V, E, F = len(nodes), len(edges), len(faces)
        chi = V - E + F
        self.assertEqual(chi, 2, f"欧拉示性数异常: χ={chi} (V={V}, E={E}, F={F})")
        print(f"\n欧拉示性数: χ={chi} (V={V}, E={E}, F={F})")

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


class TestAdfront3CubeFineMesh(unittest.TestCase):
    """立方体细网格四面体网格生成测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0
        cls.spacing = 0.25
        cls.all_triangles = TestAdfront3Cube._generate_cube_surface(cls.cube_size, cls.spacing)
        cls.adfront3 = None
        cls.unstr_grid = None

    def _run_adfront3(self):
        if self.__class__.unstr_grid is None:
            adfront3 = Adfront3(self.all_triangles, debug_level=0)
            unstr_grid = adfront3.generate()
            self.__class__.adfront3 = adfront3
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.adfront3, self.__class__.unstr_grid

    def test_fine_volume_mesh(self):
        """细网格：验证四面体网格生成"""
        adfront3, unstr_grid = self._run_adfront3()
        self.assertIsNotNone(unstr_grid)
        self.assertGreater(unstr_grid.num_cells, 0)

    def test_fine_euler_characteristic(self):
        """细网格：欧拉示性数应为 2（拓扑球面）"""
        adfront3, unstr_grid = self._run_adfront3()

        nodes = set()
        edges = set()
        faces = set()
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                nids = cell.node_ids
                nodes.update(nids)
                for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                    edges.add(tuple(sorted((nids[i], nids[j]))))
                for tri in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
                    faces.add(tuple(sorted((nids[tri[0]], nids[tri[1]], nids[tri[2]]))))

        V, E, F = len(nodes), len(edges), len(faces)
        chi = V - E + F
        self.assertEqual(chi, 2, f"欧拉示性数异常: χ={chi} (V={V}, E={E}, F={F})")
        print(f"\n欧拉示性数: χ={chi} (V={V}, E={E}, F={F})")

    def test_fine_no_degenerate_tets(self):
        """细网格：无退化四面体"""
        adfront3, unstr_grid = self._run_adfront3()
        degenerate = sum(
            1 for c in unstr_grid.cell_container
            if isinstance(c, Tetrahedron) and tetrahedron_volume(c.p1, c.p2, c.p3, c.p4) <= 1e-12
        )
        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化四面体")

    def test_fine_volume_coverage(self):
        """细网格：体积覆盖率 > 25%"""
        adfront3, unstr_grid = self._run_adfront3()
        total_vol = sum(
            tetrahedron_volume(c.p1, c.p2, c.p3, c.p4)
            for c in unstr_grid.cell_container
            if isinstance(c, Tetrahedron)
        )
        expected = self.cube_size ** 3
        coverage = total_vol / expected
        self.assertGreater(coverage, 0.25, f"覆盖率不足: {coverage:.4f}")

    def test_fine_boundary_containment(self):
        """细网格：所有节点在边界内"""
        adfront3, _ = self._run_adfront3()
        import numpy as np
        coords = np.array(adfront3.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        tol = 1e-6
        for d in range(3):
            self.assertGreaterEqual(mins[d], -tol)
            self.assertLessEqual(maxs[d], self.cube_size + tol)

    def test_fine_quality_mean(self):
        """细网格：平均质量 > 0.5"""
        adfront3, _ = self._run_adfront3()
        stats = adfront3.get_quality_stats()
        self.assertGreater(stats.get('quality_mean', 0), 0.5,
                           f"平均质量过低: {stats.get('quality_mean', 0):.4f}")


class TestAdfront3Sphere(unittest.TestCase):
    """球体四面体网格生成测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.radius = 1.0
        cls.all_triangles = cls._load_sphere_stl()
        cls.adfront3 = None
        cls.unstr_grid = None

    @staticmethod
    def _read_stl(filename):
        """读取 ASCII STL 文件，返回 SurfaceTriangle 列表"""
        from sfmesh.surface_front import SurfaceTriangle, NodeElement3D

        triangles = []
        node_cache = {}

        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        while i < len(lines):
            if lines[i].startswith("facet normal"):
                vertices = []
                i += 2
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
                i += 1
                i += 1
                if len(vertices) == 3:
                    tri = SurfaceTriangle(
                        vertices[0], vertices[1], vertices[2],
                        idx=len(triangles),
                    )
                    triangles.append(tri)
            else:
                i += 1

        return triangles

    @classmethod
    def _load_sphere_stl(cls):
        """加载球体 STL 文件"""
        stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "sphere.stl"
        return cls._read_stl(str(stl_path))

    def _run_adfront3(self):
        if self.__class__.unstr_grid is None:
            adfront3 = Adfront3(self.all_triangles, debug_level=0)
            unstr_grid = adfront3.generate()
            self.__class__.adfront3 = adfront3
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.adfront3, self.__class__.unstr_grid

    def test_sphere_volume_mesh(self):
        """球体：验证四面体网格生成"""
        self.assertGreater(len(self.all_triangles), 50,
                           f"曲面三角形数量不足: {len(self.all_triangles)}")
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

    def test_sphere_euler_characteristic(self):
        """球体：欧拉示性数应为 2（拓扑球面）"""
        adfront3, unstr_grid = self._run_adfront3()

        nodes = set()
        edges = set()
        faces = set()
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                nids = cell.node_ids
                nodes.update(nids)
                for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                    edges.add(tuple(sorted((nids[i], nids[j]))))
                for tri in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
                    faces.add(tuple(sorted((nids[tri[0]], nids[tri[1]], nids[tri[2]]))))

        V, E, F = len(nodes), len(edges), len(faces)
        chi = V - E + F
        self.assertEqual(chi, 2, f"欧拉示性数异常: χ={chi} (V={V}, E={E}, F={F})")
        print(f"\n球体欧拉示性数: χ={chi} (V={V}, E={E}, F={F})")

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
    """阵面推进法网格质量测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0
        cls.spacing = 0.5
        cls.all_triangles = TestAdfront3Cube._generate_cube_surface(cls.cube_size, cls.spacing)
        cls.adfront3 = None
        cls.unstr_grid = None

    def _run_adfront3(self):
        if self.__class__.unstr_grid is None:
            adfront3 = Adfront3(self.all_triangles, debug_level=0)
            unstr_grid = adfront3.generate()
            self.__class__.adfront3 = adfront3
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.adfront3, self.__class__.unstr_grid

    def test_euler_characteristic(self):
        """欧拉示性数应为 2（拓扑球面）"""
        adfront3, _ = self._run_adfront3()

        nodes = set()
        edges = set()
        faces = set()
        for cell in adfront3.cell_container:
            if isinstance(cell, Tetrahedron):
                nids = cell.node_ids
                nodes.update(nids)
                for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                    edges.add(tuple(sorted((nids[i], nids[j]))))
                for tri in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
                    faces.add(tuple(sorted((nids[tri[0]], nids[tri[1]], nids[tri[2]]))))

        V, E, F = len(nodes), len(edges), len(faces)
        chi = V - E + F
        self.assertEqual(chi, 2, f"欧拉示性数异常: χ={chi} (V={V}, E={E}, F={F})")
        print(f"\n欧拉示性数: χ={chi} (V={V}, E={E}, F={F})")

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


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path
from collections import Counter

import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sfmesh.surface_front import discretize_shape_edges, SurfaceTriangle, NodeElement3D
from sfmesh.mesh_3d_afm import SurfaceMeshGenerator
from sfmesh.sizing_field import SurfaceSizingField
from delaunay3d.bowyer_watson import BowyerWatsonTetGen
from delaunay3d.sizing import UniformSizing3D
from data_structure.basic_elements import Tetrahedron
from utils.geom_toolkit import tetrahedron_volume, tetrahedron_signed_volume
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


def generate_cube_surface_fine(cube_size, spacing):
    """生成立方体细密曲面网格（逐面生成，max_iterations=10000）"""
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
            max_iterations=10000,
            line_mesh=line_mesh,
        )
        all_triangles.extend(generator.generate())

    return all_triangles


def run_bowyer_watson(surface_triangles, volume_spacing):
    """运行 BowyerWatsonTetGen 并返回结果"""
    sizing = UniformSizing3D(volume_spacing)
    tetgen = BowyerWatsonTetGen(surface_triangles, sizing_system=sizing, debug_level=0)
    unstr_grid = tetgen.generate()
    return tetgen, unstr_grid


class TestBowyerWatsonCube(unittest.TestCase):
    """Bowyer-Watson 立方体四面体网格生成测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0
        cls.spacing = 0.5
        cls.volume_spacing = 1.0
        cls.all_triangles = generate_cube_surface(cls.cube_size, cls.spacing)
        cls.tetgen = None
        cls.unstr_grid = None

    def _run_tetgen(self):
        """运行 BowyerWatsonTetGen 并缓存结果"""
        if self.__class__.unstr_grid is None:
            tetgen, unstr_grid = run_bowyer_watson(self.all_triangles, self.volume_spacing)
            self.__class__.tetgen = tetgen
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.tetgen, self.__class__.unstr_grid

    def test_cube_volume_mesh(self):
        """立方体体积网格：验证四面体网格生成"""
        self.assertGreater(len(self.all_triangles), 50,
                           f"曲面三角形数量不足: {len(self.all_triangles)}")

        tetgen, unstr_grid = self._run_tetgen()

        self.assertIsNotNone(unstr_grid, "网格生成失败")
        self.assertGreater(unstr_grid.num_cells, 0, "未生成四面体")

        for cell in unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                vol = tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
                self.assertGreater(vol, 1e-12,
                                   f"四面体{cell.node_ids}体积异常: {vol}")

        output_file = self.output_dir / "delaunay_cube.vtk"
        tetgen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())

        stats = tetgen.get_quality_stats()
        print(f"\n四面体数量: {stats.get('num_cells', 0)}, "
              f"节点数量: {stats.get('num_nodes', 0)}, "
              f"质量均值: {stats.get('quality_mean', 0):.4f}, "
              f"质量最小值: {stats.get('quality_min', 0):.4f}")

    def test_cube_mesh_quality(self):
        """立方体网格质量：验证四面体质量统计"""
        tetgen, unstr_grid = self._run_tetgen()
        self.assertIsNotNone(unstr_grid)

        stats = tetgen.get_quality_stats()
        self.assertGreater(stats.get('num_cells', 0), 0)
        self.assertGreater(stats.get('quality_mean', 0), 0.0,
                           f"平均质量过低: {stats.get('quality_mean', 0):.4f}")
        self.assertGreater(stats.get('quality_min', 0), -1e-10,
                           f"最小质量异常: {stats.get('quality_min', 0):.4f}")

    def test_cube_boundary_containment(self):
        """验证所有节点在立方体边界内"""
        tetgen, unstr_grid = self._run_tetgen()

        coords = np.array(tetgen.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        tol = 1e-6
        self.assertGreaterEqual(mins[0], -tol, f"节点超出左边界: x_min={mins[0]}")
        self.assertGreaterEqual(mins[1], -tol, f"节点超出下边界: y_min={mins[1]}")
        self.assertGreaterEqual(mins[2], -tol, f"节点超出前边界: z_min={mins[2]}")
        self.assertLessEqual(maxs[0], self.cube_size + tol, f"节点超出右边界: x_max={maxs[0]}")
        self.assertLessEqual(maxs[1], self.cube_size + tol, f"节点超出上边界: y_max={maxs[1]}")
        self.assertLessEqual(maxs[2], self.cube_size + tol, f"节点超出后边界: z_max={maxs[2]}")

        print(f"\n边界范围: [{mins[0]:.4f},{mins[1]:.4f},{mins[2]:.4f}] "
              f"到 [{maxs[0]:.4f},{maxs[1]:.4f},{maxs[2]:.4f}]")

    def test_cube_no_degenerate_tets(self):
        """验证无退化四面体（所有体积 > 0）"""
        tetgen, unstr_grid = self._run_tetgen()

        degenerate = 0
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                vol = tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
                if vol <= 1e-12:
                    degenerate += 1

        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化四面体")
        print(f"\n四面体总数: {len(unstr_grid.cell_container)}, 退化: {degenerate}")


class TestBowyerWatsonCubeFineMesh(unittest.TestCase):
    """Bowyer-Watson 立方体细密面网格四面体生成测试

    使用 spacing=0.25 的细密面网格作为输入，验证 Delaunay 四面体生成。
    对应 test_sfmesh_3d_afm.TestCubeAFM.test_cube_fine_mesh 的体网格版本。
    """

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0
        cls.surface_spacing = 0.25
        cls.volume_spacing = 0.5
        cls.all_triangles = generate_cube_surface_fine(cls.cube_size, cls.surface_spacing)
        cls.tetgen = None
        cls.unstr_grid = None

    def _run_tetgen(self):
        if self.__class__.unstr_grid is None:
            tetgen, unstr_grid = run_bowyer_watson(self.all_triangles, self.volume_spacing)
            self.__class__.tetgen = tetgen
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.tetgen, self.__class__.unstr_grid

    def test_fine_surface_mesh_count(self):
        """验证细密面网格三角形数量 > 150"""
        self.assertGreater(len(self.all_triangles), 150,
                           f"细密面网格三角形不足: {len(self.all_triangles)}")
        print(f"\n细密面网格三角形: {len(self.all_triangles)}")

    def test_fine_volume_mesh_generation(self):
        """验证细密面网格能生成四面体网格"""
        tetgen, unstr_grid = self._run_tetgen()
        self.assertIsNotNone(unstr_grid, "细密面网格四面体生成失败")
        num_cells = len([c for c in unstr_grid.cell_container
                         if isinstance(c, Tetrahedron)])
        self.assertGreater(num_cells, 0, "未生成四面体")
        print(f"\n细密网格四面体: {num_cells}, 节点: {tetgen.num_nodes}")

    def test_fine_no_degenerate_tets(self):
        """验证无退化四面体"""
        tetgen, unstr_grid = self._run_tetgen()
        degenerate = 0
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                vol = tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
                if vol <= 1e-12:
                    degenerate += 1
        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化四面体")

    def test_fine_volume_coverage(self):
        """验证体积覆盖率接近 100%"""
        tetgen, unstr_grid = self._run_tetgen()
        total_vol = sum(
            tetrahedron_volume(c.p1, c.p2, c.p3, c.p4)
            for c in unstr_grid.cell_container
            if isinstance(c, Tetrahedron)
        )
        expected = self.cube_size ** 3
        coverage = total_vol / expected
        self.assertGreater(coverage, 0.95,
                           f"体积覆盖率不足: {coverage:.4f}")
        self.assertLess(coverage, 1.05,
                        f"体积覆盖率过高: {coverage:.4f}")
        print(f"\n体积覆盖率: {coverage:.4f}")

    def test_fine_boundary_containment(self):
        """验证所有节点在立方体边界内"""
        tetgen, _ = self._run_tetgen()
        coords = np.array(tetgen.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        tol = 1e-6
        for i, name in enumerate(['x', 'y', 'z']):
            self.assertGreaterEqual(mins[i], -tol,
                                    f"节点超出{name}下界: {mins[i]}")
            self.assertLessEqual(maxs[i], self.cube_size + tol,
                                 f"节点超出{name}上界: {maxs[i]}")

    def test_fine_quality_mean(self):
        """验证平均质量 > 0.5"""
        tetgen, _ = self._run_tetgen()
        stats = tetgen.get_quality_stats()
        mean_q = stats.get('quality_mean', 0)
        self.assertGreater(mean_q, 0.5,
                           f"平均质量过低: {mean_q:.4f}")
        print(f"\n质量均值: {mean_q:.4f}, 最小: {stats.get('quality_min', 0):.4f}")

    def test_fine_vtk_export(self):
        """验证 VTK 导出"""
        tetgen, _ = self._run_tetgen()
        output_file = self.output_dir / "delaunay_cube_fine.vtk"
        tetgen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        self.assertGreater(output_file.stat().st_size, 0, "VTK 文件为空")
        print(f"\nVTK 输出: {output_file}")


class TestBowyerWatsonVolumeCoverage(unittest.TestCase):
    """Bowyer-Watson 体积覆盖率测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0
        cls.spacing = 0.5
        cls.volume_spacing = 1.0
        cls.all_triangles = generate_cube_surface(cls.cube_size, cls.spacing)
        cls.tetgen, cls.unstr_grid = run_bowyer_watson(cls.all_triangles, cls.volume_spacing)

    def test_volume_coverage(self):
        """验证体积覆盖率接近 100%"""
        total_tet_volume = sum(
            tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
            for cell in self.unstr_grid.cell_container
            if isinstance(cell, Tetrahedron)
        )
        expected_volume = self.cube_size ** 3
        coverage = total_tet_volume / expected_volume

        self.assertGreater(coverage, 0.95,
                           f"体积覆盖率不足: {coverage:.4f} (应 > 0.95)")
        self.assertLess(coverage, 1.05,
                        f"体积覆盖率过高: {coverage:.4f} (应 < 1.05)")
        print(f"\n体积覆盖率: {coverage:.4f}, "
              f"计算体积: {total_tet_volume:.4f}, "
              f"理论体积: {expected_volume:.4f}")

    def test_volume_positive_winding(self):
        """验证所有四面体有正确的绕向（正有符号体积）"""
        negative_count = 0
        for cell in self.unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                sv = tetrahedron_signed_volume(cell.p1, cell.p2, cell.p3, cell.p4)
                if sv < -1e-15:
                    negative_count += 1

        self.assertEqual(negative_count, 0,
                         f"发现{negative_count}个负绕向四面体")
        print(f"\n四面体总数: {len(self.unstr_grid.cell_container)}, 负绕向: {negative_count}")

    def test_volume_statistics(self):
        """验证体积统计合理性"""
        volumes = [
            tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
            for cell in self.unstr_grid.cell_container
            if isinstance(cell, Tetrahedron)
        ]

        self.assertGreater(len(volumes), 0)
        vol_mean = sum(volumes) / len(volumes)
        vol_min = min(volumes)
        vol_max = max(volumes)

        self.assertGreater(vol_min, 1e-15, f"最小体积异常: {vol_min}")
        self.assertGreater(vol_max, vol_min, "最大体积应大于最小体积")

        expected_mean = (self.cube_size ** 3) / len(volumes)
        ratio = vol_mean / expected_mean
        self.assertGreater(ratio, 0.5, f"平均体积偏小: {vol_mean:.6f}")
        self.assertLess(ratio, 2.0, f"平均体积偏大: {vol_mean:.6f}")
        print(f"\n体积统计: 均值={vol_mean:.6f}, 最小={vol_min:.6f}, 最大={vol_max:.6f}")


class TestBowyerWatsonTopology(unittest.TestCase):
    """Bowyer-Watson 拓扑一致性测试"""

    @classmethod
    def setUpClass(cls):
        cls.cube_size = 2.0
        cls.spacing = 0.5
        cls.volume_spacing = 1.0
        cls.all_triangles = generate_cube_surface(cls.cube_size, cls.spacing)
        cls.tetgen, cls.unstr_grid = run_bowyer_watson(cls.all_triangles, cls.volume_spacing)

    def _get_all_faces(self):
        """提取所有四面体的面"""
        faces = []
        for cell in self.unstr_grid.cell_container:
            if not isinstance(cell, Tetrahedron):
                continue
            ids = cell.node_ids
            tet_faces = [
                tuple(sorted([ids[0], ids[1], ids[2]])),
                tuple(sorted([ids[0], ids[1], ids[3]])),
                tuple(sorted([ids[0], ids[2], ids[3]])),
                tuple(sorted([ids[1], ids[2], ids[3]])),
            ]
            faces.extend(tet_faces)
        return faces

    def test_face_consistency(self):
        """验证面一致性：每个内部面被恰好2个四面体共享"""
        faces = self._get_all_faces()
        face_count = Counter(faces)

        boundary_faces = {f for f, c in face_count.items() if c == 1}
        interior_faces = {f for f, c in face_count.items() if c == 2}
        invalid_faces = {f for f, c in face_count.items() if c > 2}

        self.assertEqual(len(invalid_faces), 0,
                         f"发现{len(invalid_faces)}个被3个以上四面体共享的面")

        num_cells = len([c for c in self.unstr_grid.cell_container
                         if isinstance(c, Tetrahedron)])
        num_nodes = self.tetgen.num_nodes

        # Euler 公式: V - E + F = 2 (对于单连通体)
        # 对于四面体网格: 4*T = 2*F_interior + F_boundary
        expected_boundary = 4 * num_cells - 2 * len(interior_faces)
        self.assertEqual(len(boundary_faces), expected_boundary,
                         f"边界面数不一致: 实际={len(boundary_faces)}, 期望={expected_boundary}")

        print(f"\n面统计: 总面={len(face_count)}, "
              f"边界={len(boundary_faces)}, 内部={len(interior_faces)}, "
              f"异常={len(invalid_faces)}")

    def test_no_duplicate_tets(self):
        """验证无重复四面体"""
        tet_signatures = set()
        duplicates = 0

        for cell in self.unstr_grid.cell_container:
            if not isinstance(cell, Tetrahedron):
                continue
            sig = tuple(sorted(cell.node_ids))
            if sig in tet_signatures:
                duplicates += 1
            tet_signatures.add(sig)

        self.assertEqual(duplicates, 0, f"发现{duplicates}个重复四面体")
        print(f"\n四面体签名去重: {len(tet_signatures)} 个唯一四面体")

    def test_euler_formula(self):
        """验证欧拉公式: V - E + F - T = 1 (对于有边界的四面体网格)"""
        faces = self._get_all_faces()
        face_count = Counter(faces)
        all_edges = set()
        for face in face_count:
            for i in range(3):
                for j in range(i + 1, 3):
                    all_edges.add(tuple(sorted([face[i], face[j]])))

        V = self.tetgen.num_nodes
        E = len(all_edges)
        F = len(face_count)
        T = len([c for c in self.unstr_grid.cell_container
                 if isinstance(c, Tetrahedron)])

        euler = V - E + F - T
        # 对于有边界的单连通四面体网格，欧拉示性数为1
        self.assertEqual(euler, 1, f"欧拉公式不满足: V-E+F-T = {euler} (应为1)")
        print(f"\n欧拉公式: V={V}, E={E}, F={F}, T={T}, V-E+F-T={euler}")

    def test_node_connectivity(self):
        """验证每个节点至少被一个四面体使用"""
        used_nodes = set()
        for cell in self.unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                used_nodes.update(cell.node_ids)

        total_nodes = self.tetgen.num_nodes
        unused = total_nodes - len(used_nodes)
        self.assertEqual(unused, 0, f"发现{unused}个未使用节点")
        print(f"\n节点使用: {len(used_nodes)}/{total_nodes}")


class TestBowyerWatsonSpacing(unittest.TestCase):
    """Bowyer-Watson 不同间距参数测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0

    def test_coarse_mesh(self):
        """粗网格测试：大间距应产生较少单元"""
        spacing = 1.0
        triangles = generate_cube_surface(self.cube_size, spacing)
        tetgen, unstr_grid = run_bowyer_watson(triangles, volume_spacing=1.5)

        num_cells = len([c for c in unstr_grid.cell_container
                         if isinstance(c, Tetrahedron)])
        self.assertGreater(num_cells, 0, "未生成四面体")
        self.assertLess(num_cells, 200, f"粗网格单元过多: {num_cells}")

        output_file = self.output_dir / "delaunay_coarse.vtk"
        tetgen.export_to_vtk(str(output_file))
        print(f"\n粗网格: {num_cells} 个四面体")

    def test_fine_mesh(self):
        """细网格测试：小间距应产生较多单元"""
        spacing = 0.3
        triangles = generate_cube_surface(self.cube_size, spacing)
        tetgen, unstr_grid = run_bowyer_watson(triangles, volume_spacing=0.5)

        num_cells = len([c for c in unstr_grid.cell_container
                         if isinstance(c, Tetrahedron)])
        self.assertGreater(num_cells, 100, f"细网格单元过少: {num_cells}")

        output_file = self.output_dir / "delaunay_fine.vtk"
        tetgen.export_to_vtk(str(output_file))
        print(f"\n细网格: {num_cells} 个四面体")

    def test_spacing_affects_density(self):
        """验证间距越小，单元密度越高"""
        coarse_triangles = generate_cube_surface(self.cube_size, 0.8)
        _, coarse_grid = run_bowyer_watson(coarse_triangles, volume_spacing=1.2)
        coarse_count = len([c for c in coarse_grid.cell_container
                            if isinstance(c, Tetrahedron)])

        fine_triangles = generate_cube_surface(self.cube_size, 0.4)
        _, fine_grid = run_bowyer_watson(fine_triangles, volume_spacing=0.6)
        fine_count = len([c for c in fine_grid.cell_container
                          if isinstance(c, Tetrahedron)])

        self.assertGreater(fine_count, coarse_count,
                           f"细网格({fine_count})应比粗网格({coarse_count})产生更多单元")
        print(f"\n间距对比: 粗={coarse_count} 单元, 细={fine_count} 单元")


class TestBowyerWatsonGeometry(unittest.TestCase):
    """Bowyer-Watson 不同几何体测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_sphere_surface(self, radius, spacing):
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

    def test_sphere_mesh(self):
        """球体四面体网格生成"""
        radius = 1.0
        spacing = 0.4
        try:
            triangles = self._generate_sphere_surface(radius, spacing)
        except Exception as e:
            self.skipTest(f"球面网格生成失败: {e}")

        if len(triangles) < 50:
            self.skipTest(f"球面三角形数量不足: {len(triangles)}")

        tetgen, unstr_grid = run_bowyer_watson(triangles, volume_spacing=0.6)

        self.assertIsNotNone(unstr_grid, "球体网格生成失败")
        num_cells = len([c for c in unstr_grid.cell_container
                         if isinstance(c, Tetrahedron)])
        self.assertGreater(num_cells, 0, "未生成球体四面体")

        coords = np.array(tetgen.node_coords)
        dists = np.linalg.norm(coords, axis=1)
        max_dist = dists.max()
        self.assertLess(max_dist, radius * 1.1,
                        f"节点超出球体: max_dist={max_dist:.4f}")

        total_vol = sum(
            tetrahedron_volume(c.p1, c.p2, c.p3, c.p4)
            for c in unstr_grid.cell_container
            if isinstance(c, Tetrahedron)
        )
        expected_vol = (4.0 / 3.0) * np.pi * radius ** 3
        coverage = total_vol / expected_vol
        self.assertGreater(coverage, 0.85,
                           f"球体体积覆盖率不足: {coverage:.4f}")

        output_file = self.output_dir / "delaunay_sphere.vtk"
        tetgen.export_to_vtk(str(output_file))

        stats = tetgen.get_quality_stats()
        print(f"\n球体网格: {num_cells} 四面体, "
              f"覆盖率={coverage:.4f}, "
              f"质量均值={stats.get('quality_mean', 0):.4f}")

    def test_cylinder_mesh(self):
        """圆柱体四面体网格生成"""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir

        radius = 1.0
        height = 2.0
        spacing = 0.5

        cylinder = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
            radius, height
        ).Shape()

        faces = []
        explorer = TopExp_Explorer(cylinder, TopAbs_FACE)
        while explorer.More():
            faces.append(explorer.Current())
            explorer.Next()

        all_triangles = []
        for face in faces:
            try:
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
            except Exception:
                continue

        if len(all_triangles) < 50:
            self.skipTest(f"圆柱面三角形数量不足: {len(all_triangles)}")

        tetgen, unstr_grid = run_bowyer_watson(all_triangles, volume_spacing=0.6)

        self.assertIsNotNone(unstr_grid, "圆柱体网格生成失败")
        num_cells = len([c for c in unstr_grid.cell_container
                         if isinstance(c, Tetrahedron)])
        self.assertGreater(num_cells, 0, "未生成圆柱体四面体")

        output_file = self.output_dir / "delaunay_cylinder.vtk"
        tetgen.export_to_vtk(str(output_file))

        stats = tetgen.get_quality_stats()
        print(f"\n圆柱体网格: {num_cells} 四面体, "
              f"质量均值={stats.get('quality_mean', 0):.4f}")


class TestBowyerWatsonSphereSTL(unittest.TestCase):
    """Bowyer-Watson 球体 STL 四面体网格生成测试

    从 sphere.stl 读取球面网格，在球体内部生成四面体网格。
    球心约在原点，半径约 1.0。
    """

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.stl_path = Path(project_root) / "unittests" / "test_files" / "3d_cases" / "sphere.stl"
        cls.sphere_radius = 1.0
        cls.volume_spacing = 0.3
        cls.triangles = read_stl(str(cls.stl_path))
        cls.tetgen = None
        cls.unstr_grid = None

    def _run_tetgen(self):
        if self.__class__.unstr_grid is None:
            tetgen, unstr_grid = run_bowyer_watson(self.triangles, self.volume_spacing)
            self.__class__.tetgen = tetgen
            self.__class__.unstr_grid = unstr_grid
        return self.__class__.tetgen, self.__class__.unstr_grid

    def test_stl_load(self):
        """验证 STL 文件加载成功"""
        self.assertGreater(len(self.triangles), 100,
                           f"球面三角形不足: {len(self.triangles)}")
        print(f"\n球面三角形: {len(self.triangles)}")

    def test_sphere_volume_mesh(self):
        """验证球体内部生成四面体网格"""
        tetgen, unstr_grid = self._run_tetgen()
        self.assertIsNotNone(unstr_grid, "球体网格生成失败")
        num_cells = len([c for c in unstr_grid.cell_container
                         if isinstance(c, Tetrahedron)])
        self.assertGreater(num_cells, 0, "未生成四面体")
        print(f"\n球体四面体: {num_cells}, 节点: {tetgen.num_nodes}")

    def test_sphere_no_degenerate(self):
        """验证无退化四面体"""
        tetgen, unstr_grid = self._run_tetgen()
        degenerate = 0
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                vol = tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
                if vol <= 1e-12:
                    degenerate += 1
        self.assertEqual(degenerate, 0, f"发现{degenerate}个退化四面体")

    def test_sphere_volume_coverage(self):
        """验证体积覆盖率接近 100%"""
        tetgen, unstr_grid = self._run_tetgen()
        total_vol = sum(
            tetrahedron_volume(c.p1, c.p2, c.p3, c.p4)
            for c in unstr_grid.cell_container
            if isinstance(c, Tetrahedron)
        )
        expected = (4.0 / 3.0) * np.pi * self.sphere_radius ** 3
        coverage = total_vol / expected
        self.assertGreater(coverage, 0.90,
                           f"体积覆盖率不足: {coverage:.4f}")
        self.assertLess(coverage, 1.10,
                        f"体积覆盖率过高: {coverage:.4f}")
        print(f"\n球体体积覆盖率: {coverage:.4f}, "
              f"计算={total_vol:.4f}, 理论={expected:.4f}")

    def test_sphere_nodes_bounded(self):
        """验证所有节点在球体内部"""
        tetgen, _ = self._run_tetgen()
        coords = np.array(tetgen.node_coords)
        dists = np.linalg.norm(coords, axis=1)
        max_dist = dists.max()
        self.assertLess(max_dist, self.sphere_radius * 1.05,
                        f"节点超出球体: max_dist={max_dist:.4f}")
        print(f"\n节点最大距离: {max_dist:.4f} (半径={self.sphere_radius})")

    def test_sphere_quality(self):
        """验证网格质量"""
        tetgen, _ = self._run_tetgen()
        stats = tetgen.get_quality_stats()
        mean_q = stats.get('quality_mean', 0)
        self.assertGreater(mean_q, 0.3,
                           f"平均质量过低: {mean_q:.4f}")
        self.assertGreater(stats.get('quality_min', 0), -1e-10,
                           f"最小质量异常: {stats.get('quality_min', 0):.4f}")
        print(f"\n球体质量: 均值={mean_q:.4f}, "
              f"最小={stats.get('quality_min', 0):.4f}")

    def test_sphere_vtk_export(self):
        """验证 VTK 导出"""
        tetgen, _ = self._run_tetgen()
        output_file = self.output_dir / "delaunay_sphere_stl.vtk"
        tetgen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        self.assertGreater(output_file.stat().st_size, 0, "VTK 文件为空")
        print(f"\nVTK 输出: {output_file}")


class TestBowyerWatsonVTKExport(unittest.TestCase):
    """Bowyer-Watson VTK 导出测试"""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.cube_size = 2.0
        cls.spacing = 0.5
        cls.volume_spacing = 1.0
        cls.all_triangles = generate_cube_surface(cls.cube_size, cls.spacing)
        cls.tetgen, cls.unstr_grid = run_bowyer_watson(cls.all_triangles, cls.volume_spacing)

    def test_vtk_file_created(self):
        """验证 VTK 文件成功创建"""
        output_file = self.output_dir / "delaunay_export_test.vtk"
        self.tetgen.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), "VTK 文件未创建")
        self.assertGreater(output_file.stat().st_size, 0, "VTK 文件为空")

    def test_vtk_file_format(self):
        """验证 VTK 文件格式正确"""
        output_file = self.output_dir / "delaunay_format_test.vtk"
        self.tetgen.export_to_vtk(str(output_file))

        with open(output_file, 'r') as f:
            lines = f.readlines()

        self.assertTrue(lines[0].startswith("# vtk"), "缺少 VTK 头部")
        self.assertIn("DATASET UNSTRUCTURED_GRID", lines[3], "缺少数据集类型")

        points_line = None
        cells_line = None
        for line in lines:
            if line.startswith("POINTS"):
                points_line = line
            if line.startswith("CELLS"):
                cells_line = line

        self.assertIsNotNone(points_line, "缺少 POINTS 声明")
        self.assertIsNotNone(cells_line, "缺少 CELLS 声明")

        num_points = int(points_line.split()[1])
        self.assertEqual(num_points, self.tetgen.num_nodes,
                         f"点数不匹配: {num_points} vs {self.tetgen.num_nodes}")

    def test_vtk_cell_types(self):
        """验证 VTK 单元类型全部为四面体 (type=10)"""
        output_file = self.output_dir / "delaunay_types_test.vtk"
        self.tetgen.export_to_vtk(str(output_file))

        with open(output_file, 'r') as f:
            content = f.read()

        self.assertIn("CELL_TYPES", content, "缺少 CELL_TYPES 声明")

        lines = content.split('\n')
        in_cell_types = False
        type_count = 0
        for line in lines:
            if line.startswith("CELL_TYPES"):
                in_cell_types = True
                continue
            if in_cell_types and line.strip():
                cell_type = int(line.strip())
                self.assertEqual(cell_type, 10, f"非四面体类型: {cell_type}")
                type_count += 1

        num_cells = len([c for c in self.unstr_grid.cell_container
                         if isinstance(c, Tetrahedron)])
        self.assertEqual(type_count, num_cells,
                         f"单元类型数不匹配: {type_count} vs {num_cells}")


class TestBowyerWatsonQuality(unittest.TestCase):
    """Bowyer-Watson 网格质量详细测试"""

    @classmethod
    def setUpClass(cls):
        cls.cube_size = 2.0
        cls.spacing = 0.5
        cls.volume_spacing = 1.0
        all_triangles = generate_cube_surface(cls.cube_size, cls.spacing)
        cls.tetgen, cls.unstr_grid = run_bowyer_watson(all_triangles, cls.volume_spacing)

    def test_quality_range(self):
        """验证质量值在 [0, 1] 范围内"""
        for cell in self.unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                q = tetrahedron_shape_quality(cell.p1, cell.p2, cell.p3, cell.p4)
                self.assertGreaterEqual(q, -1e-10,
                                        f"质量值为负: {q:.6f}, 单元={cell.node_ids}")
                self.assertLessEqual(q, 1.0 + 1e-10,
                                     f"质量值超过1: {q:.6f}, 单元={cell.node_ids}")

    def test_quality_mean_threshold(self):
        """验证平均质量超过阈值"""
        stats = self.tetgen.get_quality_stats()
        mean_q = stats.get('quality_mean', 0)
        self.assertGreater(mean_q, 0.5,
                           f"平均质量过低: {mean_q:.4f} (应 > 0.5)")
        print(f"\n平均质量: {mean_q:.4f}")

    def test_quality_distribution(self):
        """验证质量分布合理性"""
        qualities = []
        for cell in self.unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                q = tetrahedron_shape_quality(cell.p1, cell.p2, cell.p3, cell.p4)
                qualities.append(q)

        excellent = sum(1 for q in qualities if q > 0.8)
        good = sum(1 for q in qualities if 0.5 < q <= 0.8)
        poor = sum(1 for q in qualities if q <= 0.5)

        total = len(qualities)
        self.assertGreater(total, 0, "无四面体单元")

        excellent_ratio = excellent / total
        poor_ratio = poor / total

        self.assertGreater(excellent_ratio, 0.15,
                           f"优秀单元比例过低: {excellent_ratio:.2%}")
        self.assertLess(poor_ratio, 0.5,
                        f"差质量单元比例过高: {poor_ratio:.2%}")

        print(f"\n质量分布: 优秀(>0.8)={excellent}({excellent_ratio:.1%}), "
              f"良好(0.5-0.8)={good}, "
              f"差(<=0.5)={poor}({poor_ratio:.1%})")

    def test_no_negative_quality(self):
        """验证无负质量单元"""
        negative_count = 0
        for cell in self.unstr_grid.cell_container:
            if isinstance(cell, Tetrahedron):
                q = tetrahedron_shape_quality(cell.p1, cell.p2, cell.p3, cell.p4)
                if q < -1e-10:
                    negative_count += 1

        self.assertEqual(negative_count, 0, f"发现{negative_count}个负质量单元")


if __name__ == "__main__":
    unittest.main()

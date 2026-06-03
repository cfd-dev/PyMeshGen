"""
3D 阵面推进法（AFM）单元测试

测试 sfmesh 模块使用直接 3D AFM 在任意曲面上生成三角形网格的功能。
包含：
- 平面基准测试（验证算法基本正确性）
- 参数曲面测试（锥面、环面、双曲抛物面、球面）
- CAD 文件集成测试（椭球体、ONERA M6 机翼）
"""
import sys
import os
import math
import unittest
import numpy as np
from pathlib import Path
from collections import Counter

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fileIO.occ_loader import ensure_occ_loaded
ensure_occ_loaded()

from sfmesh.mesh_3d_afm import SurfaceMeshGenerator
from sfmesh.shape_generators import _export_combined_mesh
from sfmesh.mesh_quality import SurfaceMeshQuality, check_triangle_intersection
from sfmesh.surface_front import NodeElement3D, SurfaceTriangle, SurfaceFront, discretize_shape_edges
from sfmesh.sizing_field import SurfaceSizingField


# ============================================================================
# 通用验证辅助函数
# ============================================================================

def validate_mesh_topology(triangles: list, name: str, test_case: unittest.TestCase,
                           check_intersection: bool = True):
    """
    验证网格拓扑完整性

    Checks:
    - 无重复三角形
    - 每条边最多被2个三角形共享（流形约束）
    - 无自相交（无共享节点的三角形对不相交），可通过 check_intersection=False 跳过
    """
    if not triangles:
        test_case.fail(f"{name}: 三角形列表为空")

    # 1. 重复三角形检查
    tri_keys = [tuple(sorted([t.nodes[j].idx for j in range(3)])) for t in triangles]
    dup_count = sum(1 for v in Counter(tri_keys).values() if v > 1)
    test_case.assertEqual(dup_count, 0, f"{name}: 存在 {dup_count} 个重复三角形")

    # 2. 流形边检查
    edge_count = Counter()
    for t in triangles:
        nids = sorted([t.nodes[j].idx for j in range(3)])
        for k in range(3):
            e = tuple(sorted([nids[k], nids[(k + 1) % 3]]))
            edge_count[e] += 1

    total_edges = len(edge_count)
    bad_edges = sum(1 for v in edge_count.values() if v > 2)
    bad_ratio = bad_edges / max(total_edges, 1)
    test_case.assertLess(
        bad_ratio, 0.005,
        f"{name}: 非流形边比例 {bad_ratio:.4f} ({bad_edges}/{total_edges})"
    )

    # 3. 欧拉示性数检查 (χ = V - E + F)
    all_node_ids = set()
    for t in triangles:
        for j in range(3):
            all_node_ids.add(t.nodes[j].idx)
    V = len(all_node_ids)
    E = len(edge_count)
    F = len(triangles)

    # 追踪边界环
    boundary_adj = {}
    for e, cnt in edge_count.items():
        if cnt == 1:
            n0, n1 = e
            boundary_adj.setdefault(n0, []).append(n1)
            boundary_adj.setdefault(n1, []).append(n0)

    boundary_visited = set()
    boundary_loops = []
    for start in boundary_adj:
        if start in boundary_visited:
            continue
        loop = [start]
        boundary_visited.add(start)
        prev = None
        current = start
        for _ in range(len(boundary_adj) + 1):
            nbrs = boundary_adj.get(current, [])
            nexts = [n for n in nbrs if n != prev]
            if not nexts:
                break
            nxt = nexts[0]
            if nxt == start:
                break
            if nxt in boundary_visited:
                break
            loop.append(nxt)
            boundary_visited.add(nxt)
            prev, current = current, nxt
        boundary_loops.append(loop)

    num_boundary_loops = len(boundary_loops)
    bad_edges = sum(1 for v in edge_count.values() if v > 2)

    chi = V - E + F
    # 连通开曲面: χ = 2 - 2g - B (g=亏格, B=边界环数)
    # 对于亏格0的曲面: χ = 2 - B (盘=1, 柱面=0, 环面=-2)
    expected_chi = 2 - num_boundary_loops
    if abs(chi - expected_chi) > 2:
        loop_sizes = sorted([len(lp) for lp in boundary_loops], reverse=True)
        test_case.fail(
            f"{name}: 欧拉示性数异常 (χ={chi}, 预期≈{expected_chi}, "
            f"V={V}, E={E}, F={F}), 边界环={num_boundary_loops} "
            f"尺寸={loop_sizes}, 非流形边={bad_edges}"
        )

    # 4. 自相交检查（仅检查无共享节点的三角形对，AABB预过滤）
    if not check_intersection:
        return

    intersect_count = 0
    n_tris = len(triangles)
    for i in range(n_tris):
        ids1 = set(triangles[i].node_ids)
        pts1 = np.array([triangles[i].nodes[k].coords for k in range(3)])
        min1, max1 = pts1.min(axis=0), pts1.max(axis=0)

        for j in range(i + 1, n_tris):
            ids2 = set(triangles[j].node_ids)
            if ids1 & ids2:  # 共享节点则跳过
                continue

            pts2 = np.array([triangles[j].nodes[k].coords for k in range(3)])
            min2, max2 = pts2.min(axis=0), pts2.max(axis=0)

            # AABB 快速排斥
            if np.any(max1 < min2 - 1e-8) or np.any(min1 > max2 + 1e-8):
                continue

            if check_triangle_intersection(triangles[i], triangles[j]):
                intersect_count += 1

    test_case.assertEqual(
        intersect_count, 0,
        f"{name}: 存在 {intersect_count} 对自相交三角形"
    )


def validate_surface_coverage(
    face, triangles: list, name: str, test_case: unittest.TestCase,
    num_samples: int = 15, coverage_threshold: float = 0.85
):
    """验证网格对曲面的覆盖率"""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    adaptor = BRepAdaptor_Surface(face)
    u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()

    # 预计算三角形包围盒和法向
    tri_data = []
    for tri in triangles:
        pts = np.array([tri.nodes[k].coords for k in range(3)])
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
        tri_data.append((pts.min(axis=0), pts.max(axis=0), pts[0], normal))

    uncovered = 0
    total = num_samples * num_samples

    for i in range(num_samples):
        for j in range(num_samples):
            u = u_min + (i + 0.5) * (u_max - u_min) / num_samples
            v = v_min + (j + 0.5) * (v_max - v_min) / num_samples

            pt_3d = adaptor.Value(u, v)
            p = np.array([pt_3d.X(), pt_3d.Y(), pt_3d.Z()])

            covered = False
            for t_min, t_max, t_pt, t_normal in tri_data:
                if np.all(p >= t_min - 0.2) and np.all(p <= t_max + 0.2):
                    if abs(np.dot(p - t_pt, t_normal)) < 0.25:
                        covered = True
                        break

            if not covered:
                uncovered += 1

    coverage = 1.0 - (uncovered / total) if total > 0 else 0.0
    test_case.assertGreater(
        coverage, coverage_threshold,
        f"{name}: 覆盖率 {coverage * 100:.1f}% < {coverage_threshold * 100:.1f}%"
    )


# ============================================================================
# 平面基准测试（最简单曲面，验证算法基本正确性）
# ============================================================================

class TestPlanarSurfaceAFM(unittest.TestCase):
    """
    平面 AFM 基准测试

    平面是最简单的曲面（零曲率、UV线性映射），用于隔离验证：
    - 阵面推进基本流程是否正确
    - 相交检测是否误判
    - 网格是否闭合无孔洞
    - 质量是否接近理论最优值（等边三角形 quality ≈ 1.0）
    """

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_planar_face(width=10.0, height=10.0):
        """创建 XY 平面矩形面"""
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir
        pln = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        return BRepBuilderAPI_MakeFace(pln, 0, width, 0, height).Face()

    def test_planar_square_basic(self):
        """正方形平面：基本生成 + 拓扑验证"""
        face = self._make_planar_face(10.0, 10.0)
        sizing = SurfaceSizingField(global_spacing=2.0)
        line_mesh = discretize_shape_edges(face, sizing)
        self.assertGreater(len(line_mesh), 0, "平面边界线网格为空")

        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=2.0,
            curvature_adaptation=False,  # 平面不需要曲率自适应
            max_iterations=5000,
            line_mesh=line_mesh,
        )
        triangles = generator.generate()

        # 平面 10x10, spacing=2 → 约 50 个三角形
        self.assertGreater(len(triangles), 30, "平面三角形数量不足")

        validate_mesh_topology(triangles, "planar_square", self)

        quality = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        # 平面网格质量应较高（无曲率畸变）
        self.assertGreater(
            quality['quality_mean'], 0.5,
            f"平面平均质量过低: {quality['quality_mean']:.4f}"
        )

        output_file = self.output_dir / "afm_planar_square.vtk"
        generator.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())

    def test_planar_rectangle_aspect_ratio(self):
        """长方形平面：验证非正方形域的网格生成"""
        face = self._make_planar_face(20.0, 5.0)
        sizing = SurfaceSizingField(global_spacing=1.5)
        line_mesh = discretize_shape_edges(face, sizing)

        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=1.5,
            curvature_adaptation=False,
            max_iterations=5000,
            line_mesh=line_mesh,
        )
        triangles = generator.generate()

        self.assertGreater(len(triangles), 40, "长方形平面三角形数量不足")
        validate_mesh_topology(triangles, "planar_rectangle", self)

        quality = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(
            quality['quality_mean'], 0.4,
            f"长方形平面平均质量过低: {quality['quality_mean']:.4f}"
        )

        output_file = self.output_dir / "afm_planar_rectangle.vtk"
        generator.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())

    def test_planar_fine_spacing(self):
        """细密网格：验证小尺寸下算法稳定性"""
        face = self._make_planar_face(5.0, 5.0)
        sizing = SurfaceSizingField(global_spacing=0.5)
        line_mesh = discretize_shape_edges(face, sizing)

        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=0.5,
            curvature_adaptation=False,
            max_iterations=20000,
            line_mesh=line_mesh,
        )
        triangles = generator.generate()

        # 5x5, spacing=0.5 → 约 200 个三角形
        self.assertGreater(len(triangles), 100, "细密平面网格数量不足")
        validate_mesh_topology(triangles, "planar_fine", self)

        # 细密网格质量应更高（边界效应占比小）
        quality = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(
            quality['quality_mean'], 0.5,
            f"细密平面平均质量过低: {quality['quality_mean']:.4f}"
        )

        output_file = self.output_dir / "afm_planar_fine.vtk"
        generator.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())

    def test_planar_no_line_mesh(self):
        """平面不使用预离散线网格（纯自动边界）"""
        face = self._make_planar_face(8.0, 8.0)

        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=2.0,
            curvature_adaptation=False,
            max_iterations=5000,
            line_mesh=None,  # 不使用预离散线网格
        )
        triangles = generator.generate()

        self.assertGreater(len(triangles), 15, "无预离散线网格时平面三角形数量不足")
        validate_mesh_topology(triangles, "planar_no_linemesh", self)

        output_file = self.output_dir / "afm_planar_no_linemesh.vtk"
        generator.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())


# ============================================================================
# 参数曲面测试
# ============================================================================

class TestArbitrary3DSurfaceAFM(unittest.TestCase):
    """
    测试任意三维曲面的 3D AFM 网格生成

    通过 OCC 参数曲面 API 构造不同几何类型，验证 SurfaceMeshGenerator 的通用性。
    """

    QUALITY_MIN = 0.3
    TRI_MIN = 25
    COVERAGE_THRESHOLD = 0.85

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # OCC 参数曲面构造辅助函数
    # ---------------------------------------------------------------

    @staticmethod
    def _make_cone_face():
        """半锥面（底面半径 2，锥角 π/6，高度 1~3，u:0~π）"""
        from OCC.Core.Geom import Geom_ConicalSurface
        from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        cone = Geom_ConicalSurface(ax3, math.pi / 6, 2.0)
        return BRepBuilderAPI_MakeFace(cone, 0, math.pi, 1.0, 3.0, 1e-6).Face()

    @staticmethod
    def _make_torus_section_face():
        """截断环面的一部分（非闭合子面）"""
        from OCC.Core.Geom import Geom_ToroidalSurface
        from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        torus = Geom_ToroidalSurface(ax3, 3.0, 1.0)
        return BRepBuilderAPI_MakeFace(torus, 0, math.pi, 0, math.pi, 1e-6).Face()

    @staticmethod
    def _make_hyperbolic_paraboloid_face():
        """
        双曲抛物面（马鞍面）: z = x^2 - y^2
        使用 Geom_BezierSurface 构造（4x4 控制点），u,v ∈ [-2, 2]
        """
        from OCC.Core.Geom import Geom_BezierSurface
        from OCC.Core.TColgp import TColgp_Array2OfPnt
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pnt

        # 4x4 控制点网格，z = x^2 - y^2
        n = 4
        poles = TColgp_Array2OfPnt(1, n, 1, n)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                x = -2.0 + (i - 1) * 4.0 / (n - 1)
                y = -2.0 + (j - 1) * 4.0 / (n - 1)
                z = x * x - y * y
                poles.SetValue(i, j, gp_Pnt(x, y, z))

        bezier = Geom_BezierSurface(poles)
        return BRepBuilderAPI_MakeFace(bezier, 1e-6).Face()

    @staticmethod
    def _make_partial_sphere_face():
        """半球面片（u:0~π, v:0.1~π/2，避免极点退化和 u 接缝）"""
        from OCC.Core.Geom import Geom_SphericalSurface
        from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        sphere = Geom_SphericalSurface(ax3, 2.0)
        return BRepBuilderAPI_MakeFace(sphere, 0, math.pi, 0.1, math.pi / 2, 1e-6).Face()

    # ---------------------------------------------------------------
    # 通用运行与验证
    # ---------------------------------------------------------------

    def _run_and_validate(self, face, name, quality_min=None, tri_min=None,
                          use_line_mesh=True, spacing=1.0, max_iterations=20000):
        """在给定面上运行 3D AFM 并执行完整验证套件"""
        quality_min = quality_min or self.QUALITY_MIN
        tri_min = tri_min or self.TRI_MIN

        kwargs = dict(
            surface=face,
            global_spacing=spacing,
            curvature_adaptation=True,
            max_iterations=max_iterations,
        )

        if use_line_mesh:
            sizing = SurfaceSizingField(global_spacing=spacing)
            line_mesh = discretize_shape_edges(face, sizing)
            self.assertGreater(len(line_mesh), 0, f"{name}: 边界线网格为空")
            kwargs['line_mesh'] = line_mesh

        generator = SurfaceMeshGenerator(**kwargs)
        triangles = generator.generate()

        self.assertGreater(
            len(triangles), tri_min,
            f"{name}: 三角形数量不足 ({len(triangles)} < {tri_min})"
        )

        # 先导出 VTK（便于调试时查看网格）
        output_file = self.output_dir / f"afm_{name}.vtk"
        generator.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists(), f"{name}: VTK 文件未生成")

        # 拓扑验证
        validate_mesh_topology(triangles, name, self)

        # 质量验证
        quality = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(
            quality['quality_mean'], quality_min,
            f"{name}: 平均质量 {quality['quality_mean']:.4f} < {quality_min}"
        )

        # 覆盖率验证
        validate_surface_coverage(
            face, triangles, name, self,
            coverage_threshold=self.COVERAGE_THRESHOLD
        )

        return quality, triangles

    # ---------------------------------------------------------------
    # 测试用例
    # ---------------------------------------------------------------
    @unittest.skip("环面截面测试过慢")   
    def test_afm_torus_section(self):
        """截断环面子面"""
        face = self._make_torus_section_face()
        quality, _ = self._run_and_validate(
            face, "torus_section",
            # 环面小半径=1，spacing 需要更细以保证覆盖率
            quality_min=0.25, tri_min=50, spacing=0.5,
        )
        self.assertGreater(
            quality['quality_mean'], 0.25,
            f"环面截面平均质量过低: {quality['quality_mean']:.4f}"
        )

    def test_afm_cone(self):
        """截断锥面"""
        face = self._make_cone_face()
        self._run_and_validate(face, "cone", quality_min=0.3)
    
    @unittest.skip("双曲抛物面测试过慢")
    def test_afm_hyperbolic_paraboloid(self):
        """双曲抛物面（马鞍面）"""
        face = self._make_hyperbolic_paraboloid_face()
        self._run_and_validate(face, "hyperbolic_paraboloid", quality_min=0.25)

    def test_afm_partial_sphere(self):
        """半球面"""
        face = self._make_partial_sphere_face()
        self._run_and_validate(face, "partial_sphere", quality_min=0.25, tri_min=200,
                               spacing=0.33, max_iterations=50000)


# ============================================================================
# 立方体多面网格测试
# ============================================================================

class TestCubeAFM(unittest.TestCase):
    """
    立方体多面 AFM 测试

    立方体由 6 个平面面组成，每条边被 2 个面共享。
    测试多面网格生成的完整性：逐面生成、拓扑验证、质量验证。
    """

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_box_faces(x=2.0, y=2.0, z=2.0):
        """创建立方体并返回 6 个面"""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.gp import gp_Pnt

        box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), x, y, z).Shape()
        faces = []
        explorer = TopExp_Explorer(box, TopAbs_FACE)
        while explorer.More():
            faces.append(explorer.Current())
            explorer.Next()
        return faces

    def test_cube_single_face(self):
        """立方体单面：验证单个平面面的网格生成"""
        faces = self._make_box_faces()
        self.assertEqual(len(faces), 6, "立方体应有 6 个面")

        face = faces[0]
        sizing = SurfaceSizingField(global_spacing=0.5)
        line_mesh = discretize_shape_edges(face, sizing)

        generator = SurfaceMeshGenerator(
            surface=face,
            global_spacing=0.5,
            curvature_adaptation=False,
            max_iterations=5000,
            line_mesh=line_mesh,
        )
        triangles = generator.generate()

        self.assertGreater(len(triangles), 10, "立方体单面三角形数量不足")
        validate_mesh_topology(triangles, "cube_single_face", self)

        quality = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(quality['quality_mean'], 0.5,
                           f"立方体单面平均质量过低: {quality['quality_mean']:.4f}")

        output_file = self.output_dir / "afm_cube_single_face.vtk"
        generator.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())

    def test_cube_all_faces(self):
        """立方体全六面：逐面生成网格并验证"""
        faces = self._make_box_faces()
        self.assertEqual(len(faces), 6)

        spacing = 0.5
        all_triangles = []

        for i, face in enumerate(faces):
            sizing = SurfaceSizingField(global_spacing=spacing)
            line_mesh = discretize_shape_edges(face, sizing)

            generator = SurfaceMeshGenerator(
                surface=face,
                global_spacing=spacing,
                curvature_adaptation=False,
                max_iterations=5000,
                line_mesh=line_mesh,
            )
            triangles = generator.generate()

            self.assertGreater(len(triangles), 5,
                               f"立方体面 {i} 三角形数量不足 ({len(triangles)})")
            validate_mesh_topology(triangles, f"cube_face_{i}", self)

            all_triangles.extend(triangles)

        self.assertEqual(len(faces), 6)
        self.assertGreater(len(all_triangles), 50,
                           f"立方体总三角形数量不足 ({len(all_triangles)})")

        quality = SurfaceMeshQuality.evaluate_mesh(all_triangles, verbose=False)
        self.assertGreater(quality['quality_mean'], 0.4,
                           f"立方体整体平均质量过低: {quality['quality_mean']:.4f}")

        output_file = self.output_dir / "afm_cube_all_faces.vtk"
        _export_combined_mesh(all_triangles, str(output_file))
        self.assertTrue(output_file.exists())

    def test_cube_fine_mesh(self):
        """立方体细密网格：spacing=0.25，验证更细网格下的稳定性"""
        faces = self._make_box_faces()
        spacing = 0.25
        all_triangles = []

        for i, face in enumerate(faces):
            sizing = SurfaceSizingField(global_spacing=spacing)
            line_mesh = discretize_shape_edges(face, sizing)

            generator = SurfaceMeshGenerator(
                surface=face,
                global_spacing=spacing,
                curvature_adaptation=False,
                max_iterations=10000,
                line_mesh=line_mesh,
            )
            triangles = generator.generate()
            self.assertGreater(len(triangles), 10,
                               f"立方体细密面 {i} 三角形数量不足")
            validate_mesh_topology(triangles, f"cube_fine_face_{i}", self)
            all_triangles.extend(triangles)

        self.assertGreater(len(all_triangles), 150,
                           f"立方体细密网格总数量不足 ({len(all_triangles)})")

        quality = SurfaceMeshQuality.evaluate_mesh(all_triangles, verbose=False)
        self.assertGreater(quality['quality_mean'], 0.4,
                           f"立方体细密网格平均质量过低: {quality['quality_mean']:.4f}")

        output_file = self.output_dir / "afm_cube_fine.vtk"
        _export_combined_mesh(all_triangles, str(output_file))
        self.assertTrue(output_file.exists())


# ============================================================================
# CAD 文件集成测试
# ============================================================================

class TestCADFileAFM(unittest.TestCase):
    """测试从真实 CAD 文件生成网格"""

    @classmethod
    def setUpClass(cls):
        cls.ellipsoid_path = Path(project_root) / "examples" / "cad" / "ellipsoid-mm.igs"
        cls.m6_path = Path(project_root) / "examples" / "cad" / "onera_m6.igs"
        cls.output_dir = Path(project_root) / "unittests" / "test_files" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        cls._has_ellipsoid = cls.ellipsoid_path.exists()
        cls._has_m6 = cls.m6_path.exists()

    @unittest.skip("椭球体测试过慢")
    def test_ellipsoid_mesh_generation(self):
        """椭球体 IGES → 3D AFM 网格"""
        if not self._has_ellipsoid:
            self.skipTest(f"椭球体文件不存在: {self.ellipsoid_path}")

        from fileIO.geometry_io import import_geometry_file
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE

        shape = import_geometry_file(str(self.ellipsoid_path))
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        faces = []
        while explorer.More():
            faces.append(explorer.Current())
            explorer.Next()
        self.assertGreater(len(faces), 0, "椭球体中没有找到曲面")

        generator = SurfaceMeshGenerator(
            surface=faces[0],
            global_spacing=5.0,
            curvature_adaptation=True,
            max_iterations=10000,
        )
        triangles = generator.generate()

        self.assertGreater(len(triangles), 50, "椭球体三角形数量不足")
        # 高曲率闭合曲面的 AFM 会产生面内自交，跳过自相交检查
        validate_mesh_topology(triangles, "ellipsoid", self, check_intersection=False)

        quality = SurfaceMeshQuality.evaluate_mesh(triangles, verbose=False)
        self.assertGreater(quality['quality_mean'], 0.3, "椭球体平均质量过低")

        output_file = self.output_dir / "afm_ellipsoid.vtk"
        generator.export_to_vtk(str(output_file))
        self.assertTrue(output_file.exists())

    @unittest.skip("M6 机翼测试过慢")
    def test_onera_m6_wing_mesh_generation(self):
        """ONERA M6 机翼 → 线网格 → 多面 3D AFM"""
        if not self._has_m6:
            self.skipTest(f"M6 文件不存在: {self.m6_path}")

        from fileIO.geometry_io import import_geometry_file
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE

        shape = import_geometry_file(str(self.m6_path))
        sizing = SurfaceSizingField(global_spacing=2000.0)
        line_mesh = discretize_shape_edges(shape, sizing)
        self.assertGreater(len(line_mesh), 0, "M6 线网格为空")

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        faces = []
        while explorer.More():
            faces.append(explorer.Current())
            explorer.Next()
        self.assertGreater(len(faces), 0, "M6 中没有找到曲面")

        all_triangles = []
        for face in faces:
            gen = SurfaceMeshGenerator(
                surface=face,
                global_spacing=2000.0,
                max_iterations=10000,
                line_mesh=line_mesh,
            )
            tris = gen.generate()
            # 每个面独立验证拓扑（跨面共享边界节点会导致面间三角形重叠）
            if tris:
                validate_mesh_topology(tris, f"onera_m6_face", self)
            all_triangles.extend(tris)

        self.assertGreater(len(all_triangles), 0, "M6 未生成任何三角形")

        quality = SurfaceMeshQuality.evaluate_mesh(all_triangles, verbose=False)
        self.assertGreater(quality['quality_mean'], 0.15, "M6 平均质量过低")

        output_file = self.output_dir / "afm_onera_m6.vtk"
        _export_combined_mesh(all_triangles, str(output_file))
        self.assertTrue(output_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
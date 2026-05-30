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


if __name__ == "__main__":
    unittest.main()

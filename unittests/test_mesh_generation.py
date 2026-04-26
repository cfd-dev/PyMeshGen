#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：网格生成功能
整合了test_mesh_generation.py的测试用例
"""

import sys
from pathlib import Path
import unittest
import time
import json
import glob

# 添加项目根目录和子目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加子目录到Python路径
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    subdir_path = project_root / subdir
    if subdir_path.exists():
        sys.path.insert(0, str(subdir_path))

from PyMeshGen import PyMeshGen, PyMeshGen_mixed
from fileIO.vtk_io import parse_vtk_msh
from data_structure.parameters import Parameters
from optimize.mesh_quality import quadrilateral_quality2


class TestMeshGeneration(unittest.TestCase):
    """网格生成功能测试类"""

    @classmethod
    def setUpClass(cls):
        # 测试用文件路径
        cls.test_dir = Path(__file__).parent / "test_files" / "2d_cases"

        # 创建临时输出目录
        cls.output_dir = (
            Path(__file__).parent / "test_files" / "2d_cases" / "test_outputs"
        )
        cls.output_dir.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """所有测试执行完成后清理临时文件"""
        temp_dir = Path(__file__).parent
        for temp_file in temp_dir.glob("temp_*.json"):
            try:
                temp_file.unlink()
            except Exception:
                pass  # 忽略删除失败的情况

    def _fix_config_paths(self, config_path):
        """修复配置文件中的路径为绝对路径"""
        import os

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 修复input_file和output_file路径
        if 'input_file' in config:
            input_file = config['input_file']
            if input_file.startswith('./'):
                # 移除 ./ 并使用正确的相对路径
                input_file = input_file[2:]
                # 构建正确的绝对路径
                input_file = Path(__file__).parent / "test_files" / "2d_cases" / Path(input_file).name
                config['input_file'] = str(input_file.resolve())

        if 'output_file' in config:
            output_file = config['output_file']
            if output_file.startswith('./'):
                output_file = output_file[2:]
                output_file = Path(__file__).parent / "test_files" / "2d_cases" / "test_outputs" / Path(output_file).name
                config['output_file'] = str(output_file.resolve())

        # 测试时统一关闭可视化，避免渲染导致卡顿
        config['viz_enabled'] = False

        # 保存修复后的配置到临时文件
        temp_config_path = Path(__file__).parent / f"temp_{config_path.stem}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        return temp_config_path

    def _fix_project_case_config(self, case_name):
        """修复项目config目录下算例配置路径并重定向输出"""
        config_path = project_root / "config" / f"{case_name}.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        input_file = Path(config.get('input_file', ''))
        if not input_file.is_absolute():
            config['input_file'] = str((project_root / input_file).resolve())

        output_file = self.output_dir / f"test_{case_name}_output.vtk"
        config['output_file'] = str(output_file.resolve())
        config['viz_enabled'] = False

        temp_config_path = Path(__file__).parent / f"temp_{case_name}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        return temp_config_path, output_file

    @staticmethod
    def _override_case_config(temp_config_path, debug_level=None, wall_multi_direction=None, triangle_to_quad_method=None, output_file=None):
        """覆盖临时算例配置中的部分字段"""
        with open(temp_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if debug_level is not None:
            config['debug_level'] = debug_level

        if wall_multi_direction is not None:
            for part in config.get("parts", []):
                if part.get("PRISM_SWITCH") == "wall":
                    part["multi_direction"] = wall_multi_direction

        if triangle_to_quad_method is not None:
            config['triangle_to_quad_method'] = triangle_to_quad_method
        
        if output_file is not None:
            # ensure output_file is a string (Path objects are not JSON serializable)
            config['output_file'] = str(output_file)

        # Convert any Path objects inside parts to strings to avoid JSON serialization errors
        for part in config.get('parts', []):
            for k, v in list(part.items()):
                if isinstance(v, (str, int, float, bool)):
                    continue
                # stringify Path-like objects
                try:
                    from pathlib import Path
                    if isinstance(v, Path):
                        part[k] = str(v)
                except Exception:
                    # fallback: convert any non-serializable to string
                    if not isinstance(v, (list, dict)):
                        part[k] = str(v)

        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    @staticmethod
    def _count_cell_types(grid):
        tri = sum(1 for cell in grid.cells if len(cell) == 3)
        quad = sum(1 for cell in grid.cells if len(cell) == 4)
        other = grid.num_cells - tri - quad
        return tri, quad, other

    def test_rae2822_generation(self):
        """测试Rae2822网格生成"""
        case_file = self._fix_config_paths(self.test_dir / "rae2822.json")
        output_file = self.output_dir / "test_rae2822_output.vtk"

        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 5908, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 5872, delta=20)
        self.assertLess(cost, 65)

    def test_naca0012_generation(self):
        """测试naca0012网格生成"""
        case_file = self._fix_config_paths(self.test_dir / "naca0012.json")
        output_file = self.output_dir / "test_naca0012_output.vtk"

        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 2961, delta=10)
        self.assertAlmostEqual(grid.num_nodes, 2922, delta=10)
        self.assertLess(cost, 40)

    def test_30p30n_generation(self):
        """测试30p30n网格生成"""
        case_file = self._fix_config_paths(self.test_dir / "30p30n.json")
        output_file = self.output_dir / "test_30p30n_output.vtk"

        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 12499, delta=50)
        self.assertAlmostEqual(grid.num_nodes, 12378, delta=50)
        self.assertLess(cost, 80)

    def test_anw_generation(self):
        """测试anw网格生成"""
        case_file = self._fix_config_paths(self.test_dir / "anw.json")
        output_file = self.output_dir / "test_anw_output.vtk"

        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 2921, delta=10)
        self.assertAlmostEqual(grid.num_nodes, 2901, delta=10)
        self.assertLess(cost, 35)

    def test_convex_match_generation(self):
        """测试anw网格生成"""
        # 模拟参数配置
        case_file = self.test_dir / "convex_match120.json"
        output_file = self.output_dir / "test_convex120.vtk"

        # 执行主函数
        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        # 验证单元数、节点数
        grid = parse_vtk_msh(output_file)

        self.assertAlmostEqual(grid.num_cells, 356, delta=10)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 363, delta=10)  # 预期节点数
        # 耗时比较
        self.assertLess(cost, 4)  # 预期耗时
        
    def test_concave_match_generation(self):
        """测试anw网格生成"""
        # 模拟参数配置
        case_file = self.test_dir / "concav_match120.json"
        output_file = self.output_dir / "test_concav120.vtk"

        # 执行主函数
        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        # 验证单元数、节点数
        grid = parse_vtk_msh(output_file)

        self.assertAlmostEqual(grid.num_cells, 398, delta=10)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 400, delta=10)  # 预期节点数
        # 耗时比较
        self.assertLess(cost, 4)  # 预期耗时

    def test_30p30n_4wall_generation(self):
        """测试30p30n四壁面网格生成"""
        case_file = self._fix_config_paths(self.test_dir / "30p30n_4wall.json")
        output_file = self.output_dir / "test_30p30n_4walls.vtk"

        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 10754, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 10683, delta=10)
        self.assertLess(cost, 90)

    def test_naca0012_multi_generation(self):
        """测试naca0012_multi网格生成"""
        case_file, output_file = self._fix_project_case_config("naca0012_multi")

        try:
            start = time.time()
            PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
            cost = time.time() - start
        finally:
            case_file.unlink(missing_ok=True)

        grid = parse_vtk_msh(output_file)
        tri, quad, other = self._count_cell_types(grid)

        self.assertAlmostEqual(grid.num_cells, 2931, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 2891, delta=20)
        self.assertAlmostEqual(tri, 1720, delta=20)
        self.assertAlmostEqual(quad, 1211, delta=20)
        self.assertEqual(other, 0)
        self.assertLess(cost, 40)

    def test_rae2822_multi_generation(self):
        """测试rae2822_multi网格生成"""
        case_file, output_file = self._fix_project_case_config("rae2822_multi")

        try:
            start = time.time()
            PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
            cost = time.time() - start
        finally:
            case_file.unlink(missing_ok=True)

        grid = parse_vtk_msh(output_file)
        tri, quad, other = self._count_cell_types(grid)

        self.assertAlmostEqual(grid.num_cells, 5895, delta=30)
        self.assertAlmostEqual(grid.num_nodes, 5855, delta=30)
        self.assertAlmostEqual(tri, 1760, delta=30)
        self.assertAlmostEqual(quad, 4135, delta=30)
        self.assertEqual(other, 0)
        self.assertLess(cost, 80)

    def test_quad_quad_multi_generation(self):
        """测试quad_quad_multi网格生成"""
        case_file, output_file = self._fix_project_case_config("quad_quad_multi")

        try:
            start = time.time()
            PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
            cost = time.time() - start
        finally:
            case_file.unlink(missing_ok=True)

        grid = parse_vtk_msh(output_file)
        tri, quad, other = self._count_cell_types(grid)

        self.assertAlmostEqual(grid.num_cells, 172, delta=5)
        self.assertAlmostEqual(grid.num_nodes, 172, delta=5)
        self.assertAlmostEqual(tri, 76, delta=5)
        self.assertAlmostEqual(quad, 96, delta=5)
        self.assertEqual(other, 0)
        self.assertLess(cost, 10)

    def test_anw_multi_generation(self):
        """测试anw_multi网格生成"""
        case_file, output_file = self._fix_project_case_config("anw_multi")

        try:
            start = time.time()
            PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
            cost = time.time() - start
        finally:
            case_file.unlink(missing_ok=True)

        grid = parse_vtk_msh(output_file)
        tri, quad, other = self._count_cell_types(grid)

        self.assertAlmostEqual(grid.num_cells, 2988, delta=30)
        self.assertAlmostEqual(grid.num_nodes, 2966, delta=30)
        self.assertAlmostEqual(tri, 1108, delta=30)
        self.assertAlmostEqual(quad, 1880, delta=30)
        self.assertEqual(other, 0)
        self.assertLess(cost, 60)

    def test_30p30n_multi_generation(self):
        """测试30p30n_multi网格生成"""
        case_file, output_file = self._fix_project_case_config("30p30n_multi")
        self._override_case_config(case_file, debug_level=0)

        try:
            start = time.time()
            PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
            cost = time.time() - start
        finally:
            case_file.unlink(missing_ok=True)

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 12584, delta=30)
        self.assertAlmostEqual(grid.num_nodes, 12461, delta=30)
        self.assertLess(cost, 80)

    def test_30p30n_4wall_multi_generation(self):
        """测试30p30n_4wall_multi网格生成"""
        case_file, output_file = self._fix_project_case_config("30p30n_4wall_multi")
        self._override_case_config(
            case_file,
            debug_level=0,
            wall_multi_direction=True,
        )

        try:
            start = time.time()
            PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
            cost = time.time() - start
        finally:
            case_file.unlink(missing_ok=True)

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 10856, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 10784, delta=20)
        self.assertLess(cost, 80)

    def test_30p30n_mixed_generation(self):
        """测试30p30n_mixed网格生成"""
        case_file, output_file = self._fix_project_case_config("30p30n_mixed")
        output_file = self.output_dir / "test_30p30n_mixed_greedy_merge.vtk"
        self._override_case_config(
            case_file,
            debug_level=0,
            output_file=output_file,
            triangle_to_quad_method="greedy_merge",
        )

        try:
            start = time.time()
            PyMeshGen_mixed(Parameters("FROM_CASE_JSON", case_file))
            cost = time.time() - start
        finally:
            case_file.unlink(missing_ok=True)

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 4410, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 4214, delta=20)
        self.assertLess(cost, 120)

    def test_30p30n_mixed_generation_qmorh(self):
        """测试30p30n_mixed网格生成, q-morph"""
        case_file, output_file = self._fix_project_case_config("30p30n_mixed")
        output_file = self.output_dir / "test_30p30n_mixed_q_morph.vtk"
        self._override_case_config(
            case_file,
            debug_level=0,
            output_file=output_file,
            triangle_to_quad_method="q_morph",
        )

        try:
            start = time.time()
            PyMeshGen_mixed(Parameters("FROM_CASE_JSON", case_file))
            cost = time.time() - start
        finally:
            case_file.unlink(missing_ok=True)

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 4270, delta=120)
        self.assertAlmostEqual(grid.num_nodes, 3315, delta=120)
        self.assertLess(cost, 140)

    def test_anw_mixed_generation(self):
        """测试anw_mixed网格生成"""
        case_file = self._fix_config_paths(self.test_dir / "anw_mixed.json")
        output_file = self.output_dir / "test_anw_mixed_greedy_merge.vtk"
        self._override_case_config(
            case_file,
            debug_level=0,
            output_file=output_file,
            triangle_to_quad_method="greedy_merge",
        )
        start = time.time()
        PyMeshGen_mixed(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        zero_quality_quads = 0
        for cell in grid.cells:
            if len(cell) != 4:
                continue
            points = [grid.node_coords[node_id] for node_id in cell]
            if quadrilateral_quality2(*points) <= 1e-9:
                zero_quality_quads += 1

        self.assertAlmostEqual(grid.num_cells, 1134, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 1099, delta=20)
        self.assertEqual(zero_quality_quads, 0)
        self.assertLess(cost, 40)

    def test_anw_mixed_generation_qmorph(self):
        """测试anw_mixed网格生成"""
        case_file = self._fix_config_paths(self.test_dir / "anw_mixed.json")
        output_file = self.output_dir / "test_anw_mixed.vtk"

        self._override_case_config(
            case_file,
            debug_level=0,
            triangle_to_quad_method="q_morph",
        )

        start = time.time()
        PyMeshGen_mixed(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        zero_quality_quads = 0
        for cell in grid.cells:
            if len(cell) != 4:
                continue
            points = [grid.node_coords[node_id] for node_id in cell]
            if quadrilateral_quality2(*points) <= 1e-9:
                zero_quality_quads += 1

        self.assertAlmostEqual(grid.num_cells, 1116, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 872, delta=20)
        self.assertEqual(zero_quality_quads, 0)
        self.assertLess(cost, 40)

if __name__ == "__main__":
    unittest.main()

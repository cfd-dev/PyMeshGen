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

# 添加项目根目录和子目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加子目录到Python路径
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    subdir_path = project_root / subdir
    if subdir_path.exists():
        sys.path.insert(0, str(subdir_path))

from PyMeshGen import PyMeshGen
from PyMeshGen_mixed import PyMeshGen_mixed
from fileIO.vtk_io import parse_vtk_msh
from data_structure.parameters import Parameters


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
    def _override_case_config(temp_config_path, debug_level=None, wall_multi_direction=None):
        """覆盖临时算例配置中的部分字段"""
        with open(temp_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if debug_level is not None:
            config['debug_level'] = debug_level

        if wall_multi_direction is not None:
            for part in config.get("parts", []):
                if part.get("PRISM_SWITCH") == "wall":
                    part["multi_direction"] = wall_multi_direction

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
        self.assertAlmostEqual(grid.num_cells, 5925, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 5144, delta=20)
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
        self.assertAlmostEqual(grid.num_cells, 2981, delta=10)
        self.assertAlmostEqual(grid.num_nodes, 2216, delta=10)
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
        self.assertAlmostEqual(grid.num_cells, 12406, delta=50)
        self.assertAlmostEqual(grid.num_nodes, 11174, delta=50)
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
        self.assertAlmostEqual(grid.num_cells, 2947, delta=10)
        self.assertAlmostEqual(grid.num_nodes, 2463, delta=10)
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
        self.assertAlmostEqual(grid.num_nodes, 281, delta=10)  # 预期节点数
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

        self.assertAlmostEqual(grid.num_cells, 379, delta=10)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 304, delta=10)  # 预期节点数
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
        self.assertAlmostEqual(grid.num_cells, 10726, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 8932, delta=10)
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

        self.assertAlmostEqual(grid.num_cells, 2923, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 2220, delta=20)
        self.assertAlmostEqual(tri, 1546, delta=20)
        self.assertAlmostEqual(quad, 1377, delta=20)
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

        self.assertAlmostEqual(grid.num_cells, 5929, delta=30)
        self.assertAlmostEqual(grid.num_nodes, 5185, delta=30)
        self.assertAlmostEqual(tri, 1696, delta=30)
        self.assertAlmostEqual(quad, 4233, delta=30)
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
        self.assertAlmostEqual(grid.num_nodes, 164, delta=5)
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

        self.assertAlmostEqual(grid.num_cells, 2993, delta=30)
        self.assertAlmostEqual(grid.num_nodes, 2517, delta=30)
        self.assertAlmostEqual(tri, 1052, delta=30)
        self.assertAlmostEqual(quad, 1941, delta=30)
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
        self.assertAlmostEqual(grid.num_cells, 12440, delta=30)
        self.assertAlmostEqual(grid.num_nodes, 11230, delta=30)
        self.assertLess(cost, 80)

    def test_30p30n_4wall_multi_generation(self):
        """测试30p30n_4wall_multi网格生成"""
        case_file, output_file = self._fix_project_case_config("30p30n_4wall_multi")
        self._override_case_config(
            case_file,
            debug_level=0,
            wall_multi_direction=False,
        )

        try:
            start = time.time()
            PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
            cost = time.time() - start
        finally:
            case_file.unlink(missing_ok=True)

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 10759, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 9006, delta=20)
        self.assertLess(cost, 80)

    def test_30p30n_mixed_generation(self):
        """测试30p30n_mixed网格生成"""
        #  先跳过该测试
        self.skipTest("跳过test_30p30n_mixed_generation!")

        case_file = self._fix_config_paths(self.test_dir / "30p30n_mixed.json")
        output_file = self.output_dir / "test-30p30n-mixed.vtk"

        start = time.time()
        PyMeshGen_mixed(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 4189, delta=30)
        self.assertAlmostEqual(grid.num_nodes, 4035, delta=30)
        self.assertLess(cost, 140)

    def test_anw_mixed_generation(self):
        """测试anw_mixed网格生成"""
        self.skipTest("跳过test_anw_mixed_generation!")
        case_file = self._fix_config_paths(self.test_dir / "anw_mixed.json")
        output_file = self.output_dir / "test_anw_mixed.vtk"

        start = time.time()
        PyMeshGen_mixed(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 1279, delta=10)
        self.assertAlmostEqual(grid.num_nodes, 1085, delta=10)
        self.assertLess(cost, 14)


if __name__ == "__main__":
    unittest.main()

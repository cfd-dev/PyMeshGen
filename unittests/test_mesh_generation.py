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
        import json
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

    def test_rae2822_generation(self):
        """测试Rae2822网格生成"""
        case_file = self._fix_config_paths(self.test_dir / "rae2822.json")
        output_file = self.output_dir / "test_rae2822_output.vtk"

        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 5942, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 5170, delta=20)
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
        self.assertAlmostEqual(grid.num_cells, 2930, delta=10)
        self.assertAlmostEqual(grid.num_nodes, 2200, delta=10)
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
        self.assertAlmostEqual(grid.num_cells, 2926, delta=10)
        self.assertAlmostEqual(grid.num_nodes, 2454, delta=10)
        self.assertLess(cost, 35)

    def test_convex_match_generation(self):
        """测试convex匹配边界网格生成"""
        self.skipTest("Skipping test_convex_match_generation due to known issue with matching boundaries NodeElement type checking")

    def test_concave_match_generation(self):
        """测试concave匹配边界网格生成"""
        self.skipTest("Skipping test_concave_match_generation due to known issue with matching boundaries NodeElement type checking")

    def test_30p30n_4wall_generation(self):
        """测试30p30n四壁面网格生成"""
        case_file = self._fix_config_paths(self.test_dir / "30p30n_4wall.json")
        output_file = self.output_dir / "test_30p30n_4walls.vtk"

        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 10758, delta=20)
        self.assertAlmostEqual(grid.num_nodes, 8992, delta=10)
        self.assertLess(cost, 90)

    def test_30p30n_mixed_generation(self):
        """测试30p30n_mixed网格生成"""
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

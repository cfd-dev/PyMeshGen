import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
sys.path.append(str(Path(__file__).parent / "data_structure"))
import unittest
from unittest.mock import patch
from pathlib import Path
import PyMeshGen
from read_vtk import parse_vtk_msh
from parameters import Parameters


class TestMeshGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 测试用文件路径
        cls.test_dir = Path(__file__).parent / "test_files" / "2d_cases"

        # 创建临时输出目录
        cls.output_dir = (
            Path(__file__).parent / "test_files" / "2d_cases" / "test_outputs"
        )
        cls.output_dir.mkdir(exist_ok=True)

    def test_rae2822_generation(self):
        """测试Rae2822网格生成"""
        with patch("PyMeshGen.Parameters") as mock_params:
            # 模拟参数配置
            case_file = self.test_dir / "rae2822.json"
            output_file = self.output_dir / "test_rae2822_output.vtk"

            # 执行主函数
            PyMeshGen.PyMeshGen(Parameters("FROM_CASE_JSON", case_file))

            # 验证单元数、节点数
            grid = parse_vtk_msh(output_file)
            self.assertEqual(grid.num_cells, 5901)  # 预期单元数
            self.assertEqual(grid.num_nodes, 5149)  # 预期节点数

    def test_naca0012_generation(self):
        """测试naca0012网格生成"""
        with patch("PyMeshGen.Parameters") as mock_params:
            # 模拟参数配置
            case_file = self.test_dir / "naca0012.json"
            output_file = self.output_dir / "test_naca0012_output.vtk"

            # 执行主函数
            PyMeshGen.PyMeshGen(Parameters("FROM_CASE_JSON", case_file))

            # 验证单元数、节点数
            grid = parse_vtk_msh(output_file)
            self.assertEqual(grid.num_cells, 2911)  # 预期单元数
            self.assertEqual(grid.num_nodes, 2200)  # 预期节点数

    def test_30p30n_generation(self):
        """测试30p30n网格生成"""
        with patch("PyMeshGen.Parameters") as mock_params:
            # 模拟参数配置
            case_file = self.test_dir / "30p30n.json"
            output_file = self.output_dir / "test_30p30n_output.vtk"

            # 执行主函数
            PyMeshGen.PyMeshGen(Parameters("FROM_CASE_JSON", case_file))

            # 验证单元数、节点数
            grid = parse_vtk_msh(output_file)
            self.assertEqual(grid.num_cells, 12441)  # 预期单元数
            self.assertEqual(grid.num_nodes, 11187)  # 预期节点数

    # def test_anw_generation(self):
    #     """测试anw网格生成"""
    #     with patch("PyMeshGen.Parameters") as mock_params:
    #         # 模拟参数配置
    #         case_file = self.test_dir / "anw.json"
    #         output_file = self.output_dir / "test_anw_output.vtk"

    #         # 执行主函数
    #         PyMeshGen.PyMeshGen(Parameters("FROM_CASE_JSON", case_file))

    #         # 验证单元数、节点数
    #         grid = parse_vtk_msh(output_file)
    #         self.assertEqual(grid.num_cells, 12441)  # 预期单元数
    #         self.assertEqual(grid.num_nodes, 11187)  # 预期节点数


if __name__ == "__main__":
    unittest.main()

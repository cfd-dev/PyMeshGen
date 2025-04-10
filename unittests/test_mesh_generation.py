import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
sys.path.append(str(Path(__file__).parent / "data_structure"))
import unittest
import time
from PyMeshGen import PyMeshGen
from PyMeshGen_mixed import PyMeshGen_mixed
from vtk_io import parse_vtk_msh
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
        # 模拟参数配置
        case_file = self.test_dir / "rae2822.json"
        output_file = self.output_dir / "test_rae2822_output.vtk"

        # 执行主函数
        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        # 验证单元数、节点数
        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 5942, delta=20)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 5170, delta=20)  # 预期节点数
        # 耗时比较
        self.assertLess(cost, 65)  # 预期耗时

    def test_naca0012_generation(self):
        """测试naca0012网格生成"""
        # 参数配置
        case_file = self.test_dir / "naca0012.json"
        output_file = self.output_dir / "test_naca0012_output.vtk"

        # 执行主函数
        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        # 验证单元数、节点数
        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 2930, delta=10)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 2200, delta=10)  # 预期节点数
        # 耗时比较
        self.assertLess(cost, 40)  # 预期耗时

    def test_30p30n_generation(self):
        """测试30p30n网格生成"""
        # 模拟参数配置
        case_file = self.test_dir / "30p30n.json"
        output_file = self.output_dir / "test_30p30n_output.vtk"

        # 执行主函数
        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        # 验证单元数、节点数
        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 12406, delta=50)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 11174, delta=50)  # 预期节点数

        # 耗时比较
        self.assertLess(cost, 80)  # 预期耗时

    def test_anw_generation(self):
        """测试anw网格生成"""
        # 模拟参数配置
        case_file = self.test_dir / "anw.json"
        output_file = self.output_dir / "test_anw_output.vtk"

        # 执行主函数
        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        # 验证单元数、节点数
        grid = parse_vtk_msh(output_file)
        # self.assertEqual(grid.num_cells, 2926)  # 预期单元数
        # self.assertEqual(grid.num_nodes, 2454)  # 预期节点数

        self.assertAlmostEqual(grid.num_cells, 2926, delta=10)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 2454, delta=10)  # 预期节点数
        # 耗时比较
        self.assertLess(cost, 35)  # 预期耗时
        
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

        self.assertAlmostEqual(grid.num_cells, 368, delta=10)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 298, delta=10)  # 预期节点数
        # 耗时比较
        self.assertLess(cost, 4)  # 预期耗时
        
    def test_30p30n_4wall_generation(self):
        """测试anw网格生成"""
        # 模拟参数配置
        case_file = self.test_dir / "30p30n_4wall.json"
        output_file = self.output_dir / "test_30p30n_4walls.vtk"

        # 执行主函数
        start = time.time()
        PyMeshGen(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        # 验证单元数、节点数
        grid = parse_vtk_msh(output_file)

        self.assertAlmostEqual(grid.num_cells, 10758, delta=10)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 8992, delta=10)  # 预期节点数
        # 耗时比较
        self.assertLess(cost, 90)  # 预期耗时


    def test_30p30n_mixed_generation(self):
        """测试30p30n_mixed网格生成"""
        # 参数配置
        case_file = self.test_dir / "30p30n_mixed.json"
        output_file = self.output_dir / "test-30p30n-mixed.vtk"

        # 执行主函数
        start = time.time()
        PyMeshGen_mixed(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        # 验证单元数、节点数
        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 4189, delta=30)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 4035, delta=30)  # 预期节点数
        # 耗时比较
        self.assertLess(cost, 140)  # 预期耗时

    def test_anw_mixed_generation(self):
        """测试anw_mixed网格生成"""
        # 参数配置
        case_file = self.test_dir / "anw_mixed.json"
        output_file = self.output_dir / "test_anw_mixed.vtk"

        # 执行主函数
        start = time.time()
        PyMeshGen_mixed(Parameters("FROM_CASE_JSON", case_file))
        end = time.time()
        cost = end - start

        # 验证单元数、节点数
        grid = parse_vtk_msh(output_file)
        self.assertAlmostEqual(grid.num_cells, 1131, delta=10)  # 预期单元数
        self.assertAlmostEqual(grid.num_nodes, 1088, delta=10)  # 预期节点数
        # 耗时比较
        self.assertLess(cost, 14)  # 预期耗时

if __name__ == "__main__":
    unittest.main()

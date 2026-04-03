"""
网格导出为 PLT 文件 - 单元测试

测试内容：
1. 从简单网格数据导出 PLT
2. 从 PyMeshGen 网格字典导出 PLT
3. PLT 文件格式验证
4. 异常情况处理
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from fileIO.mesh_to_plt import export_mesh_to_plt, export_from_cas, _extract_from_grid

# 测试文件保存目录
TEST_FILES_DIR = Path(__file__).parent / "test_files" / "plt_export"
TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)


class TestMeshToPLT(unittest.TestCase):
    """网格导出为 PLT 文件的测试"""

    def test_export_simple_2d_mesh(self):
        """测试：导出简单 2D 三角形网格"""
        # 准备数据
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ])
        
        simplices = np.array([
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ])
        
        edge_index = np.array([
            [0, 1, 2, 3, 0, 4],
            [1, 2, 3, 0, 4, 4],
        ])
        
        scalars = {
            "P": np.array([1.0, 1.2, 1.1, 0.9, 1.05]),
        }
        
        output_path = TEST_FILES_DIR / "simple_mesh.plt"
        
        # 执行导出
        result_path = export_mesh_to_plt(
            nodes=nodes,
            simplices=simplices,
            edge_index=edge_index,
            scalars=scalars,
            output_path=str(output_path),
            title="Test Mesh",
        )
        
        # 验证
        self.assertTrue(output_path.exists(), "PLT 文件应该被创建")
        self.assertEqual(result_path, str(output_path))
        
        # 检查文件内容
        content = output_path.read_text()
        self.assertIn("TITLE", content)
        self.assertIn("VARIABLES", content)
        self.assertIn("ZONE", content)
        self.assertIn("Nodes    = 5", content)
        self.assertIn("Faces    = 6", content)
        self.assertIn("Elements = 4", content)

    def test_export_with_multiple_scalars(self):
        """测试：导出带多个标量场的网格"""
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        
        simplices = np.array([[0, 1, 2]])
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        
        scalars = {
            "P": np.array([1.0, 1.2, 1.1]),
            "T": np.array([300.0, 310.0, 305.0]),
            "V": np.array([0.5, 0.6, 0.55]),
        }
        
        output_path = TEST_FILES_DIR / "multi_scalar.plt"
        
        result_path = export_mesh_to_plt(
            nodes=nodes,
            simplices=simplices,
            edge_index=edge_index,
            scalars=scalars,
            output_path=str(output_path),
        )
        
        # 验证文件内容
        content = output_path.read_text()
        self.assertIn("X, Y, P, T, V", content)
        
        # 统计行数（应该有 5 个变量的数据）
        lines = content.strip().split('\n')
        data_lines = [l for l in lines if l.strip() and not l.startswith(('TITLE', 'VARIABLES', 'ZONE', 'Nodes', 'Faces', 'Elements', 'Num', 'Total'))]
        # 简单验证：有数据输出
        self.assertTrue(len(data_lines) > 0)

    def test_export_from_grid_dict(self):
        """测试：从 PyMeshGen 网格字典导出"""
        # 构造网格字典
        grid = {
            "nodes": [
                {"coords": (0.0, 0.0, 0.0)},
                {"coords": (1.0, 0.0, 0.0)},
                {"coords": (1.0, 1.0, 0.0)},
                {"coords": (0.0, 1.0, 0.0)},
            ],
            "zones": {
                "internal": {
                    "type": "faces",
                    "bc_type": "internal",
                    "data": [
                        {"nodes": [1, 2, 3]},
                        {"nodes": [1, 3, 4]},
                    ],
                },
                "wall": {
                    "type": "faces",
                    "bc_type": "wall",
                    "data": [
                        {"nodes": [1, 2]},
                        {"nodes": [2, 3]},
                        {"nodes": [3, 4]},
                        {"nodes": [4, 1]},
                    ],
                },
            },
        }
        
        output_path = TEST_FILES_DIR / "grid_dict.plt"
        
        result_path = export_mesh_to_plt(
            grid=grid,
            output_path=str(output_path),
            title="Grid Dict Test",
        )
        
        # 验证
        self.assertTrue(output_path.exists())
        content = output_path.read_text()
        self.assertIn("Nodes    = 4", content)

    def test_export_without_scalars(self):
        """测试：不带标量场的导出"""
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        
        simplices = np.array([[0, 1, 2]])
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        
        output_path = TEST_FILES_DIR / "no_scalar.plt"
        
        result_path = export_mesh_to_plt(
            nodes=nodes,
            simplices=simplices,
            edge_index=edge_index,
            output_path=str(output_path),
        )
        
        self.assertTrue(output_path.exists())
        content = output_path.read_text()
        self.assertIn("VARIABLES = X, Y", content)

    def test_plt_file_format_valid(self):
        """测试：PLT 文件格式正确性"""
        nodes = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ])
        
        simplices = np.array([
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ])
        
        edge_index = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ])
        
        scalars = {"P": np.array([1.0, 1.1, 1.2, 1.3, 1.4])}
        
        output_path = TEST_FILES_DIR / "format_test.plt"
        
        export_mesh_to_plt(
            nodes=nodes,
            simplices=simplices,
            edge_index=edge_index,
            scalars=scalars,
            output_path=str(output_path),
        )
        
        # 验证文件格式
        content = output_path.read_text()
        lines = content.strip().split('\n')
        
        # 检查文件头
        self.assertTrue(lines[0].startswith('TITLE'))
        self.assertTrue(lines[1].startswith('VARIABLES'))
        self.assertTrue(lines[2] == 'ZONE')
        
        # 检查区域定义
        zone_lines = [l for l in lines[3:] if '=' in l and not l.startswith('TITLE')]
        zone_dict = {}
        for line in zone_lines[:6]:  # 前 6 行区域定义
            if '=' in line:
                key, value = line.split('=', 1)
                zone_dict[key.strip()] = value.strip()
        
        self.assertEqual(zone_dict.get('Nodes'), '5')
        self.assertIn('Faces', zone_dict)
        self.assertIn('Elements', zone_dict)

    def test_invalid_input_raises_error(self):
        """测试：无效输入应抛出异常"""
        # 缺少必要参数
        with self.assertRaises(ValueError):
            export_mesh_to_plt(output_path=str(TEST_FILES_DIR / "fail.plt"))
        
        # 节点为空
        with self.assertRaises(ValueError):
            export_mesh_to_plt(
                nodes=np.array([]),
                output_path=str(TEST_FILES_DIR / "fail.plt"),
            )

    def test_3d_mesh_export(self):
        """测试：3D 网格导出"""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ])
        
        simplices = np.array([
            [0, 1, 2, 4],
            [0, 2, 3, 4],
        ])
        
        # 3D 四面体边
        edge_index = np.array([
            [0, 1, 2, 0, 1, 3],
            [1, 2, 0, 4, 4, 4],
        ])
        
        scalars = {"P": np.array([1.0, 1.1, 1.2, 1.3, 1.4])}
        
        output_path = TEST_FILES_DIR / "3d_mesh.plt"
        
        result_path = export_mesh_to_plt(
            nodes=nodes,
            simplices=simplices,
            edge_index=edge_index,
            scalars=scalars,
            output_path=str(output_path),
            title="3D Test",
        )
        
        self.assertTrue(output_path.exists())
        content = output_path.read_text()
        self.assertIn("X, Y, Z, P", content)


class TestExtractFromGrid(unittest.TestCase):
    """测试 _extract_from_grid 函数"""

    def test_extract_simple_grid(self):
        """测试：提取简单网格数据"""
        grid = {
            "nodes": [
                {"coords": (0.0, 0.0)},
                {"coords": (1.0, 0.0)},
                {"coords": (1.0, 1.0)},
            ],
            "zones": {
                "zone1": {
                    "type": "faces",
                    "bc_type": "internal",
                    "data": [
                        {"nodes": [1, 2, 3]},
                    ],
                },
            },
        }
        
        nodes, faces, simplices, edge_index = _extract_from_grid(grid)
        
        self.assertEqual(nodes.shape, (3, 2))
        self.assertEqual(len(faces), 1)
        self.assertEqual(simplices.shape, (1, 3))

    def test_extract_mixed_faces(self):
        """测试：提取混合类型面（线段和多边形）"""
        grid = {
            "nodes": [
                {"coords": (0.0, 0.0)},
                {"coords": (1.0, 0.0)},
                {"coords": (1.0, 1.0)},
                {"coords": (0.0, 1.0)},
            ],
            "zones": {
                "wall": {
                    "type": "faces",
                    "bc_type": "wall",
                    "data": [
                        {"nodes": [1, 2]},  # 线段
                        {"nodes": [2, 3]},  # 线段
                    ],
                },
                "internal": {
                    "type": "faces",
                    "bc_type": "internal",
                    "data": [
                        {"nodes": [1, 2, 3, 4]},  # 四边形
                    ],
                },
            },
        }
        
        nodes, faces, simplices, edge_index = _extract_from_grid(grid)
        
        # 应该有 2 条边（wall 面）
        self.assertEqual(edge_index.shape[1], 2)
        # 四边形应该被三角剖分为 2 个三角形
        self.assertEqual(simplices.shape[0], 2)


if __name__ == "__main__":
    unittest.main()

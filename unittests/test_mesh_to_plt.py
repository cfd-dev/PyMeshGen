"""
网格导出为 PLT 文件 - 单元测试

测试覆盖：
1. 基本导出功能
   - 简单 2D 三角形网格
   - 不带标量场的导出
   - 带多个标量场的导出
2. 网格格式
   - 从 PyMeshGen 网格字典导出
   - 3D 四面体网格导出
3. 文件格式验证
   - PLT 文件头格式
   - 变量名格式
4. 异常处理
   - 无效输入参数
5. 真实文件测试
   - 从 Fluent .cas 文件导出 (2D)
   - 从 Fluent .cas 文件导出 (3D)

测试工具: unittest
测试文件输出目录: unittests/test_files/test_outputs/
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from fileIO.tecplot_io import export_mesh_to_plt, export_from_cas, _extract_from_grid

# 测试文件保存目录
TEST_FILES_DIR = Path(__file__).parent / "test_files" / "test_outputs"
TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 测试类 1: 基本导出功能
# ============================================================

class TestMeshToPLT(unittest.TestCase):
    """网格导出为 PLT 文件的测试"""

    def test_export_simple_2d_mesh(self):
        """测试：导出简单 2D 三角形网格

        网格结构:
           3 ───── 2
           │╲   ╱│
           │  4  │
           │╱   ╲│
           0 ───── 1

        4 个三角形单元，8 条边，5 个节点
        """
        nodes = np.array([
            [0.0, 0.0],   # 节点 0
            [1.0, 0.0],   # 节点 1
            [1.0, 1.0],   # 节点 2
            [0.0, 1.0],   # 节点 3
            [0.5, 0.5],   # 节点 4 (中心点)
        ])

        simplices = np.array([
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ])

        edge_index = np.array([
            [0, 1, 2, 3, 0, 1, 2, 3],  # 边起点
            [1, 2, 3, 0, 4, 4, 4, 4],  # 边终点
        ])

        scalars = {"P": np.array([1.0, 1.2, 1.1, 0.9, 1.05])}
        output_path = TEST_FILES_DIR / "simple_mesh.plt"

        result_path = export_mesh_to_plt(
            nodes=nodes,
            simplices=simplices,
            edge_index=edge_index,
            scalars=scalars,
            output_path=str(output_path),
            title="Test Mesh",
        )

        self.assertTrue(output_path.exists(), "PLT 文件应该被创建")
        self.assertEqual(result_path, str(output_path))

        content = output_path.read_text()
        self.assertIn("TITLE", content)
        self.assertIn("VARIABLES", content)
        self.assertIn("ZONE", content)
        self.assertIn("Nodes    = 5", content)
        self.assertIn("Faces    = 8", content)
        self.assertIn("Elements = 4", content)
        self.assertIn("TotalNumFaceNodes", content)

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
        self.assertIn('"X", "Y"', content)

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

        content = output_path.read_text()
        self.assertIn('"X", "Y", "P", "T", "V"', content)

        lines = content.strip().split('\n')
        data_lines = [l for l in lines if l.strip() and not l.startswith(
            ('TITLE', 'VARIABLES', 'ZONE', 'Nodes', 'Faces', 'Elements', 'Num', 'Total')
        )]
        self.assertTrue(len(data_lines) > 0)

    def test_3d_mesh_export(self):
        """测试：3D 网格导出（含边界区域）

        网格结构：两个四面体组成的金字塔
        - 节点：底面正方形 4 点 + 顶点 1 点
        - 单元 (Cells)：2 个四面体（导出时被三角剖分为 4 个三角形）
        - 边界 (Zones)：6 个三角形面
        """
        grid = {
            "nodes": [
                {"coords": (0.0, 0.0, 0.0)},
                {"coords": (1.0, 0.0, 0.0)},
                {"coords": (1.0, 1.0, 0.0)},
                {"coords": (0.0, 1.0, 0.0)},
                {"coords": (0.5, 0.5, 1.0)},
            ],
            "cells": [
                {"nodes": [1, 2, 3, 5]},
                {"nodes": [1, 3, 4, 5]},
            ],
            "zones": {
                "wall": {
                    "type": "faces",
                    "bc_type": "wall",
                    "data": [
                        {"nodes": [1, 2, 5]},
                        {"nodes": [2, 3, 5]},
                        {"nodes": [3, 4, 5]},
                        {"nodes": [4, 1, 5]},
                        {"nodes": [1, 2, 3]},
                        {"nodes": [1, 3, 4]},
                    ],
                },
            },
        }

        scalars = {"P": np.array([1.0, 1.1, 1.2, 1.3, 1.4])}
        output_path = TEST_FILES_DIR / "3d_mesh.plt"

        result_path = export_mesh_to_plt(
            grid=grid,
            scalars=scalars,
            output_path=str(output_path),
            title="3D Test",
        )

        self.assertTrue(output_path.exists())
        content = output_path.read_text()

        self.assertIn('"X", "Y", "Z", "P"', content)
        self.assertIn('ZONE T= "wall"', content)
        self.assertIn("ZoneType = FEPolyhedron", content)
        self.assertIn("Elements = 4", content)


# ============================================================
# 测试类 2: 文件格式验证
# ============================================================

class TestPLTFileFormat(unittest.TestCase):
    """PLT 文件格式验证测试"""

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
            [0, 1, 2, 3, 0, 1, 2, 3],
            [1, 2, 3, 0, 4, 4, 4, 4],
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

        content = output_path.read_text()
        lines = content.strip().split('\n')

        self.assertTrue(lines[0].startswith('TITLE'))
        self.assertTrue(lines[1].startswith('VARIABLES'))
        self.assertTrue(lines[2] == 'ZONE')

        zone_lines = [l for l in lines[3:] if '=' in l and not l.startswith('TITLE')]
        zone_dict = {}
        for line in zone_lines[:6]:
            if '=' in line:
                key, value = line.split('=', 1)
                zone_dict[key.strip()] = value.strip()

        self.assertEqual(zone_dict.get('Nodes'), '5')
        self.assertIn('Faces', zone_dict)
        self.assertIn('Elements', zone_dict)


# ============================================================
# 测试类 3: 异常处理
# ============================================================

class TestErrorHandling(unittest.TestCase):
    """异常处理测试"""

    def test_invalid_input_raises_error(self):
        """测试：无效输入应抛出异常"""
        with self.assertRaises(ValueError):
            export_mesh_to_plt(output_path=str(TEST_FILES_DIR / "fail.plt"))

        with self.assertRaises(ValueError):
            export_mesh_to_plt(
                nodes=np.array([]),
                output_path=str(TEST_FILES_DIR / "fail.plt"),
            )


# ============================================================
# 测试类 4: 网格数据提取
# ============================================================

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
                    "data": [{"nodes": [1, 2, 3]}],
                },
            },
        }

        nodes, faces, simplices, edge_index = _extract_from_grid(grid)

        self.assertEqual(nodes.shape, (3, 2))
        self.assertEqual(len(faces), 1)
        self.assertEqual(simplices.shape[0], 1)

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
                        {"nodes": [1, 2]},
                        {"nodes": [2, 3]},
                    ],
                },
                "internal": {
                    "type": "faces",
                    "bc_type": "internal",
                    "data": [{"nodes": [1, 2, 3, 4]}],
                },
            },
        }

        nodes, faces, simplices, edge_index = _extract_from_grid(grid)

        self.assertEqual(edge_index.shape[1], 5)
        self.assertEqual(simplices.shape[0], 2)

    def test_export_from_grid_dict(self):
        """测试：从 PyMeshGen 网格字典导出

        网格结构（2D 四边形，分为 2 个三角形）:
           3 ───── 2
           │╲   │
           │  X  │  <- 对角线 1-3
           │╱   ╲│
           0 ───── 1

        internal: 2 个三角形单元 [1,2,3] 和 [1,3,4]
        wall: 4 条边界边
        """
        grid = {
            "nodes": [
                {"coords": (0.0, 0.0)},
                {"coords": (1.0, 0.0)},
                {"coords": (1.0, 1.0)},
                {"coords": (0.0, 1.0)},
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

        self.assertTrue(output_path.exists())
        content = output_path.read_text()

        self.assertIn("Nodes    = 4", content)
        self.assertIn("Elements = 2", content)
        self.assertIn("ZoneType = FEPolygon", content)
        self.assertIn('"X", "Y"', content)
        self.assertNotIn('"Z"', content)


# ============================================================
# 测试类 5: 真实文件测试
# ============================================================

class TestRealFileExport(unittest.TestCase):
    """真实文件导出测试"""

    def test_export_from_cas_file(self):
        """测试：从真实 2D .cas 文件导出为 PLT"""
        cas_file = TEST_FILES_DIR.parent / "naca0012-tri-coarse.cas"

        if not cas_file.exists():
            alt_paths = [
                root_dir / "config" / "input" / "naca0012-tri-coarse.cas",
                root_dir / "neural" / "GNN_ALM" / "sample_grids" / "training" / "naca4digits" / "naca0012-tri-coarse.cas",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    cas_file = alt_path
                    break
            else:
                self.skipTest("找不到测试文件: naca0012-tri-coarse.cas")

        output_path = TEST_FILES_DIR / "naca0012_tri_coarse.plt"

        result_path = export_from_cas(
            cas_file=str(cas_file),
            output_path=str(output_path),
        )

        self.assertTrue(output_path.exists(), "PLT 文件应该被创建")
        self.assertEqual(result_path, str(output_path))

        content = output_path.read_text()
        self.assertIn("TITLE", content)
        self.assertIn("VARIABLES", content)
        self.assertIn("ZONE", content)
        self.assertIn("ZoneType = FEPolygon", content)
        self.assertIn("TotalNumFaceNodes", content)
        self.assertIn("Nodes", content)
        self.assertIn('"X", "Y"', content)

        print(f"\n[OK] 成功从 .cas 文件导出 PLT: {output_path}")
        print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")

    def test_export_semisphere_3d_cas(self):
        """测试：从 3D 半球混合网格 .cas 文件导出为 PLT"""
        cas_file = root_dir / "examples" / "semisphere" / "semisphere-hybrid.cas"

        if not cas_file.exists():
            self.skipTest("找不到测试文件: semisphere-hybrid.cas")

        output_path = TEST_FILES_DIR / "semisphere_hybrid.plt"

        result_path = export_from_cas(
            cas_file=str(cas_file),
            output_path=str(output_path),
        )

        self.assertTrue(output_path.exists(), "PLT 文件应该被创建")
        self.assertEqual(result_path, str(output_path))

        content = output_path.read_text()
        self.assertIn("TITLE", content)
        self.assertIn("VARIABLES", content)
        self.assertIn("ZONE", content)
        self.assertIn("ZoneType = FEPolyhedron", content)
        self.assertIn('"X", "Y", "Z"', content)
        self.assertIn("TotalNumFaceNodes", content)
        self.assertIn("Nodes", content)

        file_size_kb = output_path.stat().st_size / 1024
        self.assertGreater(file_size_kb, 100, "PLT 文件大小应该大于 100 KB")

        print(f"\n[OK] 成功从 3D 半球混合网格导出 PLT: {output_path}")
        print(f"   文件大小: {file_size_kb:.1f} KB")


# ============================================================
# 测试类 6: GUI 导出流程测试
# ============================================================

class TestGUIExportWorkflow(unittest.TestCase):
    """GUI 导出流程测试（模拟 GUI 的完整导出流程）"""

    def _simulate_gui_export(self, cas_file, output_plt):
        """
        模拟 GUI 的导出流程：
        1. 解析 CAS 文件
        2. 重建网格
        3. 创建 Unstructured_Grid 对象
        4. 导出为 PLT
        """
        from fileIO.read_cas import parse_fluent_msh, reconstruct_mesh_from_cas
        from data_structure.unstructured_grid import Unstructured_Grid

        # 步骤 1：解析 CAS 文件
        raw_cas_data = parse_fluent_msh(cas_file)

        # 步骤 2：重建网格
        unstr_grid = reconstruct_mesh_from_cas(raw_cas_data)

        # 步骤 3：创建 Unstructured_Grid 对象
        mesh_data = Unstructured_Grid.from_cells(
            node_coords=[],
            cells=[],
            boundary_nodes_idx=[],
            grid_dimension=2,
        )
        mesh_data.file_path = cas_file
        mesh_data.mesh_type = 'cas'

        # 复制节点坐标
        if hasattr(unstr_grid, 'node_coords'):
            mesh_data.node_coords = [list(coord) for coord in unstr_grid.node_coords]
        elif hasattr(unstr_grid, 'nodes'):
            mesh_data.node_coords = [list(node.coords) for node in unstr_grid.nodes]

        # 复制单元
        if hasattr(unstr_grid, 'cell_container'):
            cells = []
            for cell in unstr_grid.cell_container:
                if cell is not None and hasattr(cell, 'node_ids'):
                    cells.append(cell.node_ids)
            mesh_data.set_cells(cells)

        # 复制维度信息
        if hasattr(unstr_grid, 'dimension'):
            mesh_data.dimension = int(unstr_grid.dimension)

        # 复制边界信息
        if hasattr(unstr_grid, 'boundary_info'):
            mesh_data.boundary_info = unstr_grid.boundary_info

        mesh_data.update_counts()

        # 步骤 4：导出为 PLT（这就是 GUI 中实际调用的方法）
        title = Path(cas_file).stem
        mesh_data.export_to_plt(
            output_path=output_plt,
            title=title
        )

        return mesh_data

    def test_gui_export_2d_mesh(self):
        """测试：GUI 流程导出 2D 网格

        验证：
        - 从 CAS 文件解析到 Unstructured_Grid 创建
        - 维度判断正确（2D 网格使用 FEPolygon）
        - VARIABLES 只有 X, Y（没有 Z）
        - 内部单元正确输出
        """
        cas_file = TEST_FILES_DIR.parent / "naca0012-tri-coarse.cas"

        if not cas_file.exists():
            alt_paths = [
                root_dir / "config" / "input" / "naca0012-tri-coarse.cas",
                root_dir / "neural" / "GNN_ALM" / "sample_grids" / "training" / "naca4digits" / "naca0012-tri-coarse.cas",
                root_dir / "examples" / "2d_airfoils" / "airplane-bomb-2d.cas",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    cas_file = alt_path
                    break
            else:
                self.skipTest("找不到测试文件: naca0012-tri-coarse.cas 或 airplane-bomb-2d.cas")

        output_path = TEST_FILES_DIR / "gui_export_2d.plt"

        # 模拟 GUI 导出
        mesh_data = self._simulate_gui_export(cas_file, str(output_path))

        # 验证文件存在
        self.assertTrue(output_path.exists(), "PLT 文件应该被创建")

        # 验证文件内容
        content = output_path.read_text(encoding='utf-8')

        # 验证 2D 特征
        self.assertIn('"X", "Y"', content, "2D 网格应该只有 X, Y 变量")
        self.assertNotIn('"Z"', content, "2D 网格不应该有 Z 变量")
        self.assertIn("ZoneType = FEPolygon", content, "2D 网格应该使用 FEPolygon")

        # 验证维度属性
        self.assertEqual(mesh_data.dimension, 2, "网格维度应该是 2D")

        # 验证节点数
        self.assertIn("Nodes", content)

        print(f"\n[OK] GUI 流程成功导出 2D 网格: {output_path}")
        print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
        print(f"   网格维度: {mesh_data.dimension}D")
        print(f"   节点数: {len(mesh_data.node_coords)}")

    def test_gui_export_3d_mesh(self):
        """测试：GUI 流程导出 3D 混合网格

        验证：
        - 从 CAS 文件解析到 Unstructured_Grid 创建
        - 维度判断正确（3D 网格使用 FEPolyhedron）
        - VARIABLES 包含 X, Y, Z
        - 边界区域正确输出（FEQUADRILATERAL 格式）
        - 内部单元正确输出
        """
        cas_file = root_dir / "examples" / "semisphere" / "semisphere-hybrid.cas"

        if not cas_file.exists():
            self.skipTest("找不到测试文件: semisphere-hybrid.cas")

        output_path = TEST_FILES_DIR / "gui_export_3d.plt"

        # 模拟 GUI 导出
        mesh_data = self._simulate_gui_export(cas_file, str(output_path))

        # 验证文件存在
        self.assertTrue(output_path.exists(), "PLT 文件应该被创建")

        # 验证文件内容
        content = output_path.read_text(encoding='utf-8')

        # 验证 3D 特征
        self.assertIn('"X", "Y", "Z"', content, "3D 网格应该包含 X, Y, Z 变量")
        self.assertIn("ZoneType = FEPolyhedron", content, "3D 主区域应该使用 FEPolyhedron")

        # 验证边界区域使用 FEQUADRILATERAL
        self.assertIn("ZoneType = FEQUADRILATERAL", content, "3D 边界区域应该使用 FEQUADRILATERAL")

        # 验证维度属性
        self.assertEqual(mesh_data.dimension, 3, "网格维度应该是 3D")

        # 验证有边界区域
        self.assertTrue(hasattr(mesh_data, 'boundary_info'), "3D 网格应该有边界信息")
        self.assertGreater(len(mesh_data.boundary_info), 0, "3D 网格应该有边界区域")

        # 验证文件大小
        file_size_kb = output_path.stat().st_size / 1024
        self.assertGreater(file_size_kb, 100, "PLT 文件大小应该大于 100 KB")

        print(f"\n[OK] GUI 流程成功导出 3D 网格: {output_path}")
        print(f"   文件大小: {file_size_kb:.1f} KB")
        print(f"   网格维度: {mesh_data.dimension}D")
        print(f"   节点数: {len(mesh_data.node_coords)}")
        print(f"   边界区域数: {len(mesh_data.boundary_info)}")

    def _simulate_cgns_export(self, cgns_file, output_plt):
        """
        模拟从 CGNS 文件导出为 PLT 的流程：
        1. 使用 UniversalCGNSReader 解析 CGNS 文件
        2. 创建 Unstructured_Grid 对象
        3. 导出为 PLT
        """
        import numpy as np
        from fileIO.universal_cgns_reader import UniversalCGNSReader
        from data_structure.unstructured_grid import Unstructured_Grid

        # 步骤 1：解析 CGNS 文件
        reader = UniversalCGNSReader(str(cgns_file))
        success = reader.read()
        self.assertTrue(success, f"CGNS 文件读取失败: {cgns_file.name}")

        # 步骤 2：创建 Unstructured_Grid 对象
        mesh_data = Unstructured_Grid.from_cells(
            node_coords=[],
            cells=[],
            boundary_nodes_idx=[],
            grid_dimension=3,  # CGNS 通常为 3D
        )
        mesh_data.file_path = str(cgns_file)
        mesh_data.mesh_type = 'cgns'

        # 复制节点坐标
        if reader.points is not None:
            mesh_data.node_coords = [list(coord) for coord in reader.points]

        # 复制单元（从 CGNS cells 转换）
        # CGNS cells 是 section 列表，每个 section 包含一种类型的多个单元
        # 体积单元（pyramid, tetra, wedge, hexa）放入主区域
        # 面单元（triangle, quad）放入边界区域
        volume_cells = []
        boundary_faces = {}  # {section_index: {'nodes': [...], 'bc_type': ...}}

        volume_types = {'pyramid', 'tetra', 'wedge', 'hexa', 'pyra'}
        boundary_types = {'triangle', 'quad', 'bar'}

        if reader.cells:
            for section_idx, section in enumerate(reader.cells):
                cell_type = section.get('type', '').lower()
                data = section.get('data')

                if data is None:
                    continue

                # 将 numpy 数组转换为列表
                if isinstance(data, np.ndarray):
                    cell_list = data.tolist()
                else:
                    cell_list = list(data) if data else []

                if not cell_list:
                    continue

                # CGNS 使用 1-based 索引，需要转换为 0-based
                cells_0based = []
                for cell_nodes in cell_list:
                    cells_0based.append([int(n) - 1 for n in cell_nodes])

                if cell_type in volume_types:
                    # 体积单元：添加到主区域
                    volume_cells.extend(cells_0based)
                elif cell_type in boundary_types:
                    # 面单元：添加到边界区域
                    boundary_faces[str(section_idx)] = {
                        'nodes': cells_0based,
                        'bc_type': 'boundary',
                    }

        # 设置体积单元
        if volume_cells:
            mesh_data.set_cells(volume_cells)

        # 设置边界信息
        if boundary_faces:
            mesh_data.boundary_info = {}
            for zone_name, zone_data in boundary_faces.items():
                # 将面节点列表转换为 faces 格式
                faces = [{'nodes': [n + 1 for n in cell]} for cell in zone_data['nodes']]
                mesh_data.boundary_info[zone_name] = {
                    'part_name': zone_name,
                    'bc_type': zone_data['bc_type'],
                    'faces': faces,
                }

        # 复制维度信息
        mesh_data.dimension = 3

        mesh_data.update_counts()

        # 步骤 3：导出为 PLT
        title = Path(cgns_file).stem
        mesh_data.export_to_plt(
            output_path=output_plt,
            title=title
        )

        return mesh_data

    def test_gui_export_cgns_mesh(self):
        """测试：GUI 流程导出 CGNS 网格

        验证：
        - 从 CGNS 文件解析到 Unstructured_Grid 创建
        - 维度判断正确（3D 网格使用 FEPolyhedron）
        - VARIABLES 包含 X, Y, Z
        - 内部单元正确输出
        """
        cgns_file = root_dir / "examples" / "chn-t1" / "grid_chnt-1_coarse.cgns"

        if not cgns_file.exists():
            self.skipTest("找不到测试文件: grid_chnt-1_coarse.cgns")

        output_path = TEST_FILES_DIR / "gui_export_cgns.plt"

        # 模拟 GUI 导出
        mesh_data = self._simulate_cgns_export(cgns_file, str(output_path))

        # 验证文件存在
        self.assertTrue(output_path.exists(), "PLT 文件应该被创建")

        # 验证文件内容
        content = output_path.read_text(encoding='utf-8')

        # 验证 3D 特征
        self.assertIn('"X", "Y", "Z"', content, "CGNS 网格应该包含 X, Y, Z 变量")
        self.assertIn("ZoneType = FEPolyhedron", content, "CGNS 3D 主区域应该使用 FEPolyhedron")

        # 验证维度属性
        self.assertEqual(mesh_data.dimension, 3, "网格维度应该是 3D")

        # 验证节点数
        self.assertGreater(len(mesh_data.node_coords), 0, "节点数应该大于 0")

        # 验证文件大小
        file_size_kb = output_path.stat().st_size / 1024
        self.assertGreater(file_size_kb, 10, "PLT 文件大小应该大于 10 KB")

        print(f"\n[OK] GUI 流程成功导出 CGNS 网格: {output_path}")
        print(f"   文件大小: {file_size_kb:.1f} KB")
        print(f"   网格维度: {mesh_data.dimension}D")
        print(f"   节点数: {len(mesh_data.node_coords)}")
        if hasattr(mesh_data, 'boundary_info') and mesh_data.boundary_info:
            print(f"   边界区域数: {len(mesh_data.boundary_info)}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    unittest.main()

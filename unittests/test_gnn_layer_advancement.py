#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：GNN 层推进网格生成

从测试集随机读取 .cas 文件，使用 GNN 预测方向并通过 PyMeshGen 生成网格
"""

import sys
from pathlib import Path
import unittest
import time
import random
import json

# 添加项目根目录和子目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyMeshGen import PyMeshGen
from data_structure.parameters import Parameters
from fileIO.vtk_io import parse_vtk_msh
from fileIO.read_cas import parse_fluent_msh


class TestGNNLayerAdvancement(unittest.TestCase):
    """GNN 层推进网格生成测试类"""

    @classmethod
    def setUpClass(cls):
        # 数据目录列表
        cls.data_dirs = [
            project_root / "neural" / "GNN_ALM" / "sample_grids" / "naca4digits_series" / "data",
            project_root / "neural" / "GNN_ALM" / "sample_grids" / "uiuc_airfoil_database" / "data",
            project_root / "neural" / "GNN_ALM" / "sample_grids" / "convex" / "data",
        ]
        
        # 创建临时输出目录
        cls.output_dir = Path(__file__).parent / "test_files" / "gnn_test_outputs"
        cls.output_dir.mkdir(exist_ok=True)

    def _get_random_cas_file(self):
        """从数据目录中随机选择一个 .cas 文件"""
        # 固定使用 goe513_N75.cas 用于调试下表面边界层问题
        fixed_cas = project_root / "neural" / "GNN_ALM" / "sample_grids" / "uiuc_airfoil_database" / "data" / "goe513_N75.cas"
        if fixed_cas.exists():
            print(f"\n[调试] 使用固定 CAS 文件: {fixed_cas.name}")
            return fixed_cas

        # 回退到随机选择
        all_cas_files = []
        for data_dir in self.data_dirs:
            if data_dir.exists():
                cas_files = list(data_dir.glob("*.cas"))
                all_cas_files.extend(cas_files)

        if not all_cas_files:
            self.skipTest("未找到任何 .cas 文件，请确保数据目录存在")

        return random.choice(all_cas_files)

    def _get_boundary_part_names(self, cas_file):
        """从 CAS 文件中读取实际的边界 part 名称"""
        result = parse_fluent_msh(str(cas_file))
        zones = result.get("zones", {})
        wall_parts = []
        farfield_parts = []
        for k, v in zones.items():
            if v.get("type") != "faces":
                continue
            bc_type = v.get("bc_type", "")
            part_name = v.get("part_name", "")
            if bc_type == "wall":
                wall_parts.append(part_name)
            elif bc_type in ("pressure-far-field", "farfield", "outlet"):
                farfield_parts.append(part_name)
        return wall_parts, farfield_parts

    def _create_temp_config(self, cas_file, output_file):
        """创建临时配置文件，使用 CAS 文件中的实际 part 名称"""
        wall_parts, farfield_parts = self._get_boundary_part_names(cas_file)

        parts_config = []
        # 添加远场边界
        for part_name in farfield_parts:
            parts_config.append({
                "part_name": part_name,
                "max_size": 2.0,
                "PRISM_SWITCH": "off",
                "first_height": 0.1,
                "max_layers": 5,
                "full_layers": 5,
                "multi_direction": False
            })
        # 添加壁面边界
        for part_name in wall_parts:
            parts_config.append({
                "part_name": part_name,
                "max_size": 2.0,
                "PRISM_SWITCH": "wall",
                "first_height": 0.001,
                "max_layers": 30,
                "full_layers": 5,
                "multi_direction": False
            })

        config = {
            "debug_level": 3,
            "input_file": str(cas_file.resolve()),
            "output_file": str(output_file.resolve()),
            "sizing_decay": 1.45,
            "viz_enabled": False,
            "parts": parts_config
        }
        
        temp_config_path = self.output_dir / f"temp_gnn_test_{cas_file.stem}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        return temp_config_path

    def test_gnn_layer_advancement_random_sample(self):
        """测试 GNN 层推进：从数据目录随机加载 .cas 文件"""
        # 1. 随机选择 .cas 文件
        cas_file = self._get_random_cas_file()
        print(f"\n随机选择的文件: {cas_file.name}")
        print(f"文件路径: {cas_file}")
        
        # 2. 创建临时配置文件
        output_file = self.output_dir / f"gnn_test_{cas_file.stem}.vtk"
        temp_config = self._create_temp_config(cas_file, output_file)
        
        try:
            # 3. 调用 PyMeshGen 生成网格（会自动使用 GNN 方向预测）
            start_time = time.time()
            PyMeshGen(Parameters("FROM_CASE_JSON", temp_config))
            cost = time.time() - start_time
            
            print(f"\n网格生成完成，耗时: {cost:.2f}s")
            
            # 4. 验证输出文件
            self.assertTrue(output_file.exists(), f"输出文件未生成: {output_file}")
            
            # 5. 读取并验证网格
            grid = parse_vtk_msh(output_file)
            print(f"生成的单元数: {grid.num_cells}")
            print(f"生成的节点数: {grid.num_nodes}")
            
            self.assertGreater(grid.num_cells, 0, "未生成任何单元")
            self.assertGreater(grid.num_nodes, 0, "未生成任何节点")
            
            # 6. 统计单元类型
            tri = sum(1 for cell in grid.cells if len(cell) == 3)
            quad = sum(1 for cell in grid.cells if len(cell) == 4)
            print(f"三角形单元: {tri}")
            print(f"四边形单元: {quad}")
            
            print(f"\n测试通过！")
            
        finally:
            # 清理临时配置文件
            if temp_config.exists():
                temp_config.unlink()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 merge_triangles_to_quads 工具函数的单元测试
"""

import sys
import os
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fileIO.read_cas import parse_cas_to_unstr_grid
from utils.mesh_utils import merge_triangles_to_quads
from utils.message import info


class TestMergeTrianglesToQuads(unittest.TestCase):
    """测试三角形合并为四边形的功能"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，设置测试数据"""
        cls.test_files = [
            "examples/2d_simple/quad-tri.cas",
            "examples/2d_simple/quad-hybrid.cas"
        ]
        
        # 检查测试文件是否存在
        cls.available_files = []
        for file_path in cls.test_files:
            if os.path.exists(file_path):
                cls.available_files.append(file_path)
        
        if not cls.available_files:
            raise unittest.SkipTest("没有找到可用的测试网格文件")
    
    def count_cell_types(self, unstr_grid):
        """统计网格中的单元类型数量"""
        num_quads = sum(1 for cell in unstr_grid.cell_container if len(cell.node_ids) == 4)
        num_tris = sum(1 for cell in unstr_grid.cell_container if len(cell.node_ids) == 3)
        return num_quads, num_tris
    
    def test_merge_creates_quads(self):
        """测试合并能够创建四边形单元"""
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 2. 统计初始单元类型
                initial_quads, initial_tris = self.count_cell_types(unstr_grid)
                
                # 3. 调用工具函数进行合并
                merged_grid = merge_triangles_to_quads(unstr_grid)
                
                # 4. 统计合并后的单元类型
                merged_quads, merged_tris = self.count_cell_types(merged_grid)
                
                # 5. 验证四边形数量增加
                self.assertGreaterEqual(merged_quads, initial_quads,
                                    f"四边形数量应该增加或保持不变: {input_file}")
    
    def test_merge_reduces_tris(self):
        """测试合并减少三角形单元数量"""
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 2. 统计初始单元类型
                initial_quads, initial_tris = self.count_cell_types(unstr_grid)
                
                # 3. 调用工具函数进行合并
                merged_grid = merge_triangles_to_quads(unstr_grid)
                
                # 4. 统计合并后的单元类型
                merged_quads, merged_tris = self.count_cell_types(merged_grid)
                
                # 5. 验证三角形数量减少
                self.assertLessEqual(merged_tris, initial_tris,
                                  f"三角形数量应该减少或保持不变: {input_file}")
                
                # 6. 验证单元总数减少（每两个三角形合并为一个四边形）
                total_cells_initial = initial_quads + initial_tris
                total_cells_merged = merged_quads + merged_tris
                self.assertLessEqual(total_cells_merged, total_cells_initial,
                                  f"单元总数应该减少: {input_file}")
    
    def test_merge_preserves_original_grid(self):
        """测试合并不修改原始网格"""
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 2. 统计初始单元类型
                initial_quads, initial_tris = self.count_cell_types(unstr_grid)
                
                # 3. 调用工具函数进行合并
                merged_grid = merge_triangles_to_quads(unstr_grid)
                
                # 4. 再次统计原始网格的单元类型
                final_quads, final_tris = self.count_cell_types(unstr_grid)
                
                # 5. 验证原始网格未被修改
                self.assertEqual(final_quads, initial_quads,
                              f"原始网格的四边形数量应该保持不变: {input_file}")
                self.assertEqual(final_tris, initial_tris,
                              f"原始网格的三角形数量应该保持不变: {input_file}")
    
    def test_merge_preserves_mesh_structure(self):
        """测试合并保持网格结构不变"""
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 2. 记录原始网格结构
                original_num_nodes = unstr_grid.num_nodes
                original_boundary_nodes = set(unstr_grid.boundary_nodes_list)
                
                # 3. 调用工具函数进行合并
                merged_grid = merge_triangles_to_quads(unstr_grid)
                
                # 4. 验证节点数量不变
                self.assertEqual(merged_grid.num_nodes, original_num_nodes,
                              f"节点数量应该保持不变: {input_file}")
                
                # 5. 验证边界节点集合不变
                merged_boundary_nodes = set(merged_grid.boundary_nodes_list)
                self.assertEqual(merged_boundary_nodes, original_boundary_nodes,
                              f"边界节点集合应该保持不变: {input_file}")
    
    def test_merge_creates_convex_quads(self):
        """测试合并创建的四边形是凸多边形"""
        from utils.geom_toolkit import is_convex
        
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 2. 调用工具函数进行合并
                merged_grid = merge_triangles_to_quads(unstr_grid)
                
                # 3. 检查所有四边形是否为凸多边形
                for cell in merged_grid.cell_container:
                    if len(cell.node_ids) == 4:
                        quad_nodes = cell.node_ids
                        coords = [merged_grid.node_coords[i] for i in quad_nodes]
                        
                        # 检查凸性（传递节点索引和坐标列表）
                        is_conv = is_convex(0, 1, 2, 3, coords)
                        self.assertTrue(is_conv, 
                                     f"四边形 {quad_nodes} 应该是凸多边形: {input_file}")
    
    def test_merge_improves_quality(self):
        """测试合并后的四边形质量合理"""
        from optimize.mesh_quality import quadrilateral_quality2, triangle_shape_quality
        
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 2. 统计初始三角形质量
                tri_qualities = []
                for cell in unstr_grid.cell_container:
                    if len(cell.node_ids) == 3:
                        coords = [unstr_grid.node_coords[i] for i in cell.node_ids]
                        try:
                            q = triangle_shape_quality(coords[0], coords[1], coords[2])
                            tri_qualities.append(q)
                        except:
                            tri_qualities.append(0.0)
                
                avg_tri_quality = sum(tri_qualities) / len(tri_qualities) if tri_qualities else 0.0
                
                # 3. 调用工具函数进行合并
                merged_grid = merge_triangles_to_quads(unstr_grid)
                
                # 4. 统计合并后四边形质量
                quad_qualities = []
                for cell in merged_grid.cell_container:
                    if len(cell.node_ids) == 4:
                        coords = [merged_grid.node_coords[i] for i in cell.node_ids]
                        try:
                            q = quadrilateral_quality2(coords[0], coords[1], coords[2], coords[3])
                            quad_qualities.append(q)
                        except:
                            quad_qualities.append(0.0)
                
                avg_quad_quality = sum(quad_qualities) / len(quad_qualities) if quad_qualities else 0.0
                
                # 5. 验证四边形平均质量合理（不低于三角形平均质量的50%）
                self.assertGreaterEqual(avg_quad_quality, avg_tri_quality * 0.5,
                                     f"四边形平均质量应该不低于三角形平均质量的50%: {input_file}")
                
                # 6. 验证四边形平均质量大于0
                self.assertGreater(avg_quad_quality, 0.0,
                                  f"四边形平均质量应该大于0: {input_file}")


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()

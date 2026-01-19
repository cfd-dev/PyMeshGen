#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合网格优化功能的单元测试
测试 hybrid_smooth 函数对混合网格的优化效果
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fileIO.read_cas import parse_cas_to_unstr_grid
from optimize import hybrid_smooth, node_perturbation
from optimize.mesh_quality import triangle_shape_quality, quadrilateral_quality2, quadrilateral_skewness, quadrilateral_shape_quality
from utils.geom_toolkit import is_convex, is_valid_quadrilateral
from utils.message import info, warning, error
from utils.timer import TimeSpan


class TestHybridSmoothOptimization(unittest.TestCase):
    """测试混合网格优化功能"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，设置测试数据"""
        cls.test_files = [
            "examples/2d_simple/cylinder-hybrid.cas",
            "examples/2d_simple/quad-hybrid.cas"
        ]
        
        # 检查测试文件是否存在
        cls.available_files = []
        for file_path in cls.test_files:
            if os.path.exists(file_path):
                cls.available_files.append(file_path)
        
        if not cls.available_files:
            raise unittest.SkipTest("没有找到可用的测试网格文件")
    
    def calculate_mesh_quality(self, unstr_grid):
        """计算网格的整体质量指标"""
        quad_qualities = []
        tri_qualities = []
        
        for cell in unstr_grid.cell_container:
            if hasattr(cell, 'node_ids'):
                coords = [unstr_grid.node_coords[i] for i in cell.node_ids]
                if len(cell.node_ids) == 4:
                    try:
                        q = quadrilateral_quality2(coords[0], coords[1], coords[2], coords[3])
                        quad_qualities.append(q)
                    except Exception:
                        quad_qualities.append(0.0)
                elif len(cell.node_ids) == 3:
                    try:
                        q = triangle_shape_quality(coords[0], coords[1], coords[2])
                        tri_qualities.append(q)
                    except:
                        tri_qualities.append(0.0)
        
        avg_quad_quality = sum(quad_qualities) / len(quad_qualities) if quad_qualities else 0.0
        min_quad_quality = min(quad_qualities) if quad_qualities else 0.0
        
        avg_tri_quality = sum(tri_qualities) / len(tri_qualities) if tri_qualities else 0.0
        min_tri_quality = min(tri_qualities) if tri_qualities else 0.0
        
        return {
            'num_quads': len(quad_qualities),
            'num_tris': len(tri_qualities),
            'avg_quad_quality': avg_quad_quality,
            'min_quad_quality': min_quad_quality,
            'avg_tri_quality': avg_tri_quality,
            'min_tri_quality': min_tri_quality
        }
    
    def test_hybrid_smooth_improves_quality(self):
        """测试 hybrid_smooth 能够改善网格质量"""
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取混合网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 统计单元类型
                num_quads = sum(1 for cell in unstr_grid.cell_container if len(cell.node_ids) == 4)
                num_tris = sum(1 for cell in unstr_grid.cell_container if len(cell.node_ids) == 3)
                
                # 确保是混合网格
                self.assertGreater(num_quads, 0, "网格应该包含四边形")
                self.assertGreater(num_tris, 0, "网格应该包含三角形")
                
                # 2. 计算初始网格质量
                initial_quality = self.calculate_mesh_quality(unstr_grid)
                
                # 3. 对节点进行扰动
                perturbed_grid = node_perturbation(unstr_grid, ratio=0.5)
                
                # 计算扰动后的网格质量
                perturbed_quality = self.calculate_mesh_quality(perturbed_grid)
                
                # 4. 调用 hybrid_smooth 进行优化
                optimized_grid = hybrid_smooth(perturbed_grid, max_iter=10)
                
                # 计算优化后的网格质量
                optimized_quality = self.calculate_mesh_quality(optimized_grid)
                
                # 5. 验证优化效果
                # 四边形平均质量应该改善
                if perturbed_quality['avg_quad_quality'] > 0:
                    quad_improvement = (optimized_quality['avg_quad_quality'] - perturbed_quality['avg_quad_quality']) / perturbed_quality['avg_quad_quality'] * 100
                    self.assertGreater(quad_improvement, 0, 
                                      f"四边形平均质量应该改善: {input_file}")
                
                # 三角形平均质量应该改善或保持不变
                if perturbed_quality['avg_tri_quality'] > 0:
                    tri_improvement = (optimized_quality['avg_tri_quality'] - perturbed_quality['avg_tri_quality']) / perturbed_quality['avg_tri_quality'] * 100
                    self.assertGreaterEqual(tri_improvement, -1, 
                                           f"三角形平均质量应该改善或保持不变: {input_file}")
    
    def test_hybrid_smooth_preserves_mesh_structure(self):
        """测试 hybrid_smooth 保持网格结构不变"""
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取混合网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 2. 对节点进行扰动
                perturbed_grid = node_perturbation(unstr_grid, ratio=0.5)
                
                # 记录原始网格结构
                original_num_nodes = perturbed_grid.num_nodes
                original_num_cells = perturbed_grid.num_cells
                original_boundary_nodes = set(perturbed_grid.boundary_nodes_list)
                
                # 3. 调用 hybrid_smooth 进行优化
                optimized_grid = hybrid_smooth(perturbed_grid, max_iter=10)
                
                # 4. 验证网格结构
                self.assertEqual(optimized_grid.num_nodes, original_num_nodes,
                               "节点数量应该保持不变")
                self.assertEqual(optimized_grid.num_cells, original_num_cells,
                               "单元数量应该保持不变")
                
                # 验证边界节点位置不变
                optimized_boundary_nodes = set(optimized_grid.boundary_nodes_list)
                self.assertEqual(optimized_boundary_nodes, original_boundary_nodes,
                               "边界节点集合应该保持不变")
                
                for node_idx in original_boundary_nodes:
                    original_coord = np.array(perturbed_grid.node_coords[node_idx])
                    optimized_coord = np.array(optimized_grid.node_coords[node_idx])
                    np.testing.assert_array_almost_equal(original_coord, optimized_coord, decimal=10,
                                                         err_msg=f"边界节点 {node_idx} 位置应该保持不变")
    
    def test_node_perturbation_respects_boundary(self):
        """测试节点扰动保持边界节点不变"""
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取混合网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 记录边界节点坐标
                boundary_nodes = set(unstr_grid.boundary_nodes_list)
                original_boundary_coords = {node_idx: np.array(unstr_grid.node_coords[node_idx]) 
                                          for node_idx in boundary_nodes}
                
                # 2. 对节点进行扰动
                perturbed_grid = node_perturbation(unstr_grid, ratio=0.5)
                
                # 3. 验证边界节点位置不变
                for node_idx in boundary_nodes:
                    original_coord = original_boundary_coords[node_idx]
                    perturbed_coord = np.array(perturbed_grid.node_coords[node_idx])
                    np.testing.assert_array_almost_equal(original_coord, perturbed_coord, decimal=10,
                                                         err_msg=f"边界节点 {node_idx} 位置应该保持不变")
    
    def test_hybrid_smooth_handles_zero_quality_cells(self):
        """测试 hybrid_smooth 能够处理质量为0的单元"""
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取混合网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 2. 对节点进行较大扰动以产生质量为0的单元
                perturbed_grid = node_perturbation(unstr_grid, ratio=0.8)
                
                # 计算扰动后的网格质量
                perturbed_quality = self.calculate_mesh_quality(perturbed_grid)
                
                # 3. 调用 hybrid_smooth 进行优化（应该不会抛出异常）
                try:
                    optimized_grid = hybrid_smooth(perturbed_grid, max_iter=10)
                    
                    # 计算优化后的网格质量
                    optimized_quality = self.calculate_mesh_quality(optimized_grid)
                    
                    # 验证优化过程成功完成
                    self.assertIsNotNone(optimized_grid, "优化应该成功完成")
                    
                except Exception as e:
                    self.fail(f"hybrid_smooth 应该能够处理质量为0的单元，但抛出了异常: {e}")
    
    def test_hybrid_smooth_multiple_iterations(self):
        """测试 hybrid_smooth 在多次迭代中的稳定性"""
        for input_file in self.available_files:
            with self.subTest(file=input_file):
                # 1. 读取混合网格
                unstr_grid = parse_cas_to_unstr_grid(input_file)
                
                # 2. 对节点进行扰动
                perturbed_grid = node_perturbation(unstr_grid, ratio=0.5)
                
                # 3. 多次调用 hybrid_smooth
                grid = perturbed_grid
                qualities = []
                
                for iter_idx in range(3):
                    grid = hybrid_smooth(grid, max_iter=5)
                    quality = self.calculate_mesh_quality(grid)
                    qualities.append(quality)
                
                # 验证质量单调递增或保持稳定
                for i in range(1, len(qualities)):
                    if qualities[i]['avg_quad_quality'] > 0:
                        self.assertGreaterEqual(qualities[i]['avg_quad_quality'],
                                               qualities[i-1]['avg_quad_quality'] * 0.99,
                                               f"四边形平均质量应该单调递增或保持稳定: 迭代 {i}")
                    
                    if qualities[i]['avg_tri_quality'] > 0:
                        self.assertGreaterEqual(qualities[i]['avg_tri_quality'],
                                               qualities[i-1]['avg_tri_quality'] * 0.99,
                                               f"三角形平均质量应该单调递增或保持稳定: 迭代 {i}")


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()

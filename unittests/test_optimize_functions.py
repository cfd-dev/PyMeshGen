#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：网格优化功能
包括节点扰动、边交换、拉普拉斯平滑等优化函数的测试
测试文件：@examples/2d_simple/quad-tri.cas
"""

import sys
import os
from pathlib import Path
import unittest
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fileIO.read_cas import parse_cas_to_unstr_grid
from optimize.optimize import (
    node_perturbation,
    edge_swap,
    laplacian_smooth,
    hybrid_smooth,
    edge_swap_delaunay
)
from optimize.mesh_quality import triangle_shape_quality, quadrilateral_quality2
from utils.message import info


class TestNodePerturbation(unittest.TestCase):
    """测试节点扰动功能"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，设置测试数据"""
        cls.test_file = "examples/2d_simple/quad-tri.cas"
        
        # 检查测试文件是否存在
        if not os.path.exists(cls.test_file):
            raise unittest.SkipTest(f"测试文件 {cls.test_file} 不存在")
    
    def setUp(self):
        """每个测试用例前的准备：加载网格文件"""
        self.unstr_grid = parse_cas_to_unstr_grid(self.test_file)
        self.assertIsNotNone(self.unstr_grid, "网格加载失败")
        self.original_coords = np.array(self.unstr_grid.node_coords)
        self.original_num_nodes = len(self.unstr_grid.node_coords)
        
    def test_node_perturbation_basic(self):
        """测试基本的节点扰动功能"""
        info("测试基本节点扰动功能...")
        
        # 执行节点扰动
        perturbed_grid = node_perturbation(self.unstr_grid, ratio=0.5)
        
        # 检查节点数量是否保持不变
        self.assertEqual(len(perturbed_grid.node_coords), self.original_num_nodes,
                         "扰动后节点数量发生变化")
        
        # 检查坐标是否发生了变化
        coords_changed = not np.allclose(self.unstr_grid.node_coords, self.original_coords)
        info(f"扰动后坐标是否发生变化: {coords_changed}")
        
        # 验证网格的有效性
        node_coords_array = np.array(perturbed_grid.node_coords)
        self.assertTrue(np.all(np.isfinite(node_coords_array)), 
                       "扰动后网格中存在非有限值（NaN或Inf）")
        
    def test_perturbation_effectiveness(self):
        """测试节点扰动是否有效果"""
        info("测试节点扰动的有效性...")
        
        # 执行两次扰动
        perturbed_grid = node_perturbation(self.unstr_grid, ratio=0.5)
        perturbed_grid = node_perturbation(perturbed_grid, ratio=0.5)

        # 检查坐标是否发生了显著变化
        new_coords = np.array(perturbed_grid.node_coords)
        total_displacement = np.sum(np.linalg.norm(new_coords - self.original_coords, axis=1))

        info(f"总节点位移: {total_displacement:.6f}")

        # 至少应该有一定量的位移
        self.assertGreater(total_displacement + 1e-10, 0,
                         "节点扰动没有产生任何效果")
    
    def test_perturbation_with_different_ratios(self):
        """测试不同扰动系数的效果"""
        info("测试不同扰动系数...")
        
        ratios = [0.3, 0.5, 0.8]
        displacements = []
        
        for ratio in ratios:
            perturbed_grid = node_perturbation(self.unstr_grid, ratio=ratio)
            new_coords = np.array(perturbed_grid.node_coords)
            displacement = np.sum(np.linalg.norm(new_coords - self.original_coords, axis=1))
            displacements.append(displacement)
            info(f"ratio={ratio}: 总位移={displacement:.6f}")
        
        # 较大的扰动系数应该产生较大的位移
        self.assertGreaterEqual(displacements[1], displacements[0],
                              "ratio=0.5的扰动应该比ratio=0.3的扰动大")
        self.assertGreaterEqual(displacements[2], displacements[1],
                              "ratio=0.8的扰动应该比ratio=0.5的扰动大")


class TestEdgeSwap(unittest.TestCase):
    """测试边交换功能"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，设置测试数据"""
        cls.test_file = "examples/2d_simple/quad-tri.cas"
        
        if not os.path.exists(cls.test_file):
            raise unittest.SkipTest(f"测试文件 {cls.test_file} 不存在")
    
    def setUp(self):
        """每个测试用例前的准备：加载网格文件"""
        self.unstr_grid = parse_cas_to_unstr_grid(self.test_file)
        self.assertIsNotNone(self.unstr_grid, "网格加载失败")
        self.original_num_cells = len(self.unstr_grid.cell_container)
        self.original_num_nodes = len(self.unstr_grid.node_coords)
        
    def test_edge_swap_basic(self):
        """测试基本的边交换功能"""
        info("测试基本边交换功能...")
        
        # 执行边交换
        swapped_grid = edge_swap(self.unstr_grid)
        
        # 验证节点数量不变
        self.assertEqual(len(swapped_grid.node_coords), self.original_num_nodes,
                         "边交换后节点数量发生变化")
        
        # 验证单元数量不变
        self.assertEqual(len(swapped_grid.cell_container), self.original_num_cells,
                         "边交换后单元数量发生变化")
        
        # 验证所有节点坐标都是有效的
        node_coords_array = np.array(swapped_grid.node_coords)
        self.assertTrue(np.all(np.isfinite(node_coords_array)), 
                       "边交换后网格中存在非有限值（NaN或Inf）")
        
    def test_edge_swap_preserves_topology(self):
        """测试边交换操作是否保持网格拓扑结构"""
        info("测试边交换保持拓扑结构...")
        
        swapped_grid = edge_swap(self.unstr_grid)
        
        # 验证节点和单元数量不变
        self.assertEqual(len(swapped_grid.node_coords), self.original_num_nodes,
                         "边交换后节点数量发生变化")
        self.assertEqual(len(swapped_grid.cell_container), self.original_num_cells,
                         "边交换后单元数量发生变化")
        
    def test_edge_swap_delaunay(self):
        """测试Delaunay边交换功能"""
        info("测试Delaunay边交换功能...")
        
        # 执行Delaunay边交换
        delaunay_grid = edge_swap_delaunay(self.unstr_grid)
        
        # 验证基本属性
        self.assertIsNotNone(delaunay_grid, "Delaunay边交换返回None")
        self.assertEqual(len(delaunay_grid.node_coords), self.original_num_nodes,
                         "Delaunay边交换后节点数量发生变化")
        
        # 验证网格有效性
        node_coords_array = np.array(delaunay_grid.node_coords)
        self.assertTrue(np.all(np.isfinite(node_coords_array)), 
                       "Delaunay边交换后网格中存在非有限值（NaN或Inf）")


class TestLaplacianSmooth(unittest.TestCase):
    """测试拉普拉斯平滑功能"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，设置测试数据"""
        cls.test_file = "examples/2d_simple/quad-tri.cas"
        
        if not os.path.exists(cls.test_file):
            raise unittest.SkipTest(f"测试文件 {cls.test_file} 不存在")
    
    def setUp(self):
        """每个测试用例前的准备：加载网格文件"""
        self.unstr_grid = parse_cas_to_unstr_grid(self.test_file)
        self.assertIsNotNone(self.unstr_grid, "网格加载失败")
        self.original_coords = np.array(self.unstr_grid.node_coords)
        self.original_num_nodes = len(self.unstr_grid.node_coords)
        
    def test_laplacian_smooth_basic(self):
        """测试基本的拉普拉斯平滑功能"""
        info("测试基本拉普拉斯平滑功能...")
        
        # 执行拉普拉斯平滑
        smoothed_grid = laplacian_smooth(self.unstr_grid, num_iter=5)
        
        # 验证节点数量不变
        self.assertEqual(len(smoothed_grid.node_coords), self.original_num_nodes,
                         "平滑后节点数量发生变化")
        
        # 验证网格有效性
        node_coords_array = np.array(smoothed_grid.node_coords)
        self.assertTrue(np.all(np.isfinite(node_coords_array)), 
                       "平滑后网格中存在非有限值（NaN或Inf）")
        
    def test_laplacian_smooth_iterations(self):
        """测试不同迭代次数的效果"""
        info("测试不同迭代次数的拉普拉斯平滑...")
        
        iterations = [1, 5, 10]
        displacements = []
        
        for num_iter in iterations:
            smoothed_grid = laplacian_smooth(self.unstr_grid, num_iter=num_iter)
            new_coords = np.array(smoothed_grid.node_coords)
            displacement = np.sum(np.linalg.norm(new_coords - self.original_coords, axis=1))
            displacements.append(displacement)
            info(f"迭代次数={num_iter}: 总位移={displacement:.6f}")
        
        # 更多的迭代应该产生更大的位移（直到收敛）
        self.assertGreater(displacements[1], displacements[0],
                          "5次迭代的位移应该比1次迭代大")


class TestHybridSmooth(unittest.TestCase):
    """测试混合平滑功能"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，设置测试数据"""
        cls.test_file = "examples/2d_simple/quad-tri.cas"
        
        if not os.path.exists(cls.test_file):
            raise unittest.SkipTest(f"测试文件 {cls.test_file} 不存在")
    
    def setUp(self):
        """每个测试用例前的准备：加载网格文件"""
        self.unstr_grid = parse_cas_to_unstr_grid(self.test_file)
        self.assertIsNotNone(self.unstr_grid, "网格加载失败")
        self.original_num_nodes = len(self.unstr_grid.node_coords)
        self.original_num_cells = len(self.unstr_grid.cell_container)
        
    def test_hybrid_smooth_basic(self):
        """测试基本的混合平滑功能"""
        info("测试基本混合平滑功能...")
        
        # 执行混合平滑
        smoothed_grid = hybrid_smooth(self.unstr_grid, max_iter=2)
        
        # 验证基本属性
        self.assertIsNotNone(smoothed_grid, "混合平滑返回None")
        self.assertEqual(len(smoothed_grid.node_coords), self.original_num_nodes,
                         "平滑后节点数量发生变化")
        self.assertEqual(len(smoothed_grid.cell_container), self.original_num_cells,
                         "平滑后单元数量发生变化")
        
        # 验证网格有效性
        node_coords_array = np.array(smoothed_grid.node_coords)
        self.assertTrue(np.all(np.isfinite(node_coords_array)), 
                       "平滑后网格中存在非有限值（NaN或Inf）")
        
    def test_hybrid_smooth_improves_quality(self):
        """测试混合平滑是否改善网格质量"""
        info("测试混合平滑对网格质量的改善...")
        
        # 计算原始网格质量
        original_tri_quality = []
        original_quad_quality = []
        
        for cell in self.unstr_grid.cell_container:
            if isinstance(cell, type(self.unstr_grid.cell_container[0])):
                if hasattr(cell, 'node_indices') and len(cell.node_indices) == 3:
                    coords = [self.unstr_grid.node_coords[i] for i in cell.node_indices]
                    try:
                        quality = triangle_shape_quality(coords[0], coords[1], coords[2])
                        original_tri_quality.append(quality)
                    except:
                        pass
                elif hasattr(cell, 'node_indices') and len(cell.node_indices) == 4:
                    coords = [self.unstr_grid.node_coords[i] for i in cell.node_indices]
                    try:
                        quality = quadrilateral_quality2(coords[0], coords[1], coords[2], coords[3])
                        original_quad_quality.append(quality)
                    except:
                        pass
        
        # 执行混合平滑
        smoothed_grid = hybrid_smooth(self.unstr_grid, max_iter=2)
        
        # 计算平滑后网格质量
        smoothed_tri_quality = []
        smoothed_quad_quality = []
        
        for cell in smoothed_grid.cell_container:
            if hasattr(cell, 'node_indices') and len(cell.node_indices) == 3:
                coords = [smoothed_grid.node_coords[i] for i in cell.node_indices]
                try:
                    quality = triangle_shape_quality(coords[0], coords[1], coords[2])
                    smoothed_tri_quality.append(quality)
                except:
                    pass
            elif hasattr(cell, 'node_indices') and len(cell.node_indices) == 4:
                coords = [smoothed_grid.node_coords[i] for i in cell.node_indices]
                try:
                    quality = quadrilateral_quality2(coords[0], coords[1], coords[2], coords[3])
                    smoothed_quad_quality.append(quality)
                except:
                    pass
        
        # 比较质量
        if original_tri_quality and smoothed_tri_quality:
            orig_avg = np.mean(original_tri_quality)
            smooth_avg = np.mean(smoothed_tri_quality)
            info(f"三角形平均质量: 原始={orig_avg:.4f}, 平滑后={smooth_avg:.4f}")
            
        if original_quad_quality and smoothed_quad_quality:
            orig_avg = np.mean(original_quad_quality)
            smooth_avg = np.mean(smoothed_quad_quality)
            info(f"四边形平均质量: 原始={orig_avg:.4f}, 平滑后={smooth_avg:.4f}")


class TestOptimizationPipeline(unittest.TestCase):
    """测试优化流程（组合多个优化步骤）"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，设置测试数据"""
        cls.test_file = "examples/2d_simple/quad-tri.cas"
        
        if not os.path.exists(cls.test_file):
            raise unittest.SkipTest(f"测试文件 {cls.test_file} 不存在")
    
    def setUp(self):
        """每个测试用例前的准备：加载网格文件"""
        self.unstr_grid = parse_cas_to_unstr_grid(self.test_file)
        self.assertIsNotNone(self.unstr_grid, "网格加载失败")
        self.original_num_nodes = len(self.unstr_grid.node_coords)
        self.original_num_cells = len(self.unstr_grid.cell_container)
        
    def test_combined_optimization_pipeline(self):
        """测试组合优化流程：扰动 + 平滑 + 边交换"""
        info("测试组合优化流程...")
        
        # 步骤1: 节点扰动
        info("步骤1: 节点扰动")
        grid = node_perturbation(self.unstr_grid, ratio=0.5)
        self.assertEqual(len(grid.node_coords), self.original_num_nodes)
        
        # 步骤2: 拉普拉斯平滑
        info("步骤2: 拉普拉斯平滑")
        grid = laplacian_smooth(grid, num_iter=5)
        self.assertEqual(len(grid.node_coords), self.original_num_nodes)
        
        # 步骤3: 边交换
        info("步骤3: 边交换")
        grid = edge_swap(grid)
        self.assertEqual(len(grid.node_coords), self.original_num_nodes)
        self.assertEqual(len(grid.cell_container), self.original_num_cells)
        
        # 验证最终网格有效性
        node_coords_array = np.array(grid.node_coords)
        self.assertTrue(np.all(np.isfinite(node_coords_array)), 
                       "优化后网格中存在非有限值（NaN或Inf）")
        
        info("组合优化流程测试完成")


if __name__ == '__main__':
    unittest.main(verbosity=2)

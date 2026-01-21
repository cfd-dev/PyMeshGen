#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的网格质量计算函数
验证三维单元标准化因子是否正确
"""

import sys
import os
import unittest
import numpy as np

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from optimize.mesh_quality import (
    tetrahedron_shape_quality,
    prism_shape_quality,
    hexahedron_shape_quality,
    pyramid_shape_quality
)


class TestMeshQuality(unittest.TestCase):
    """测试网格质量计算函数"""
    
    def test_ideal_tetrahedron(self):
        """测试理想四面体的质量值应接近1.0"""
        a = 1.0
        p1 = [0, 0, 0]
        p2 = [a, 0, 0]
        p3 = [a/2, a*np.sqrt(3)/2, 0]
        p4 = [a/2, a*np.sqrt(3)/6, a*np.sqrt(2/3)]
        
        tet_quality = tetrahedron_shape_quality(p1, p2, p3, p4)
        self.assertAlmostEqual(tet_quality, 1.0, places=5,
                              msg="理想四面体质量应接近1.0")
    
    def test_ideal_prism(self):
        """测试理想三棱柱的质量值应接近1.0"""
        p1 = [0, 0, 0]
        p2 = [1, 0, 0]
        p3 = [0.5, np.sqrt(3)/2, 0]
        p4 = [0, 0, 1]
        p5 = [1, 0, 1]
        p6 = [0.5, np.sqrt(3)/2, 1]
        
        prism_quality = prism_shape_quality(p1, p2, p3, p4, p5, p6)
        self.assertAlmostEqual(prism_quality, 1.0, places=5,
                              msg="理想三棱柱质量应接近1.0")
    
    def test_ideal_hexahedron(self):
        """测试理想六面体的质量值应接近1.0"""
        p1 = [0, 0, 0]
        p2 = [1, 0, 0]
        p3 = [1, 1, 0]
        p4 = [0, 1, 0]
        p5 = [0, 0, 1]
        p6 = [1, 0, 1]
        p7 = [1, 1, 1]
        p8 = [0, 1, 1]
        
        hex_quality = hexahedron_shape_quality(p1, p2, p3, p4, p5, p6, p7, p8)
        self.assertAlmostEqual(hex_quality, 1.0, places=5,
                              msg="理想六面体质量应接近1.0")
    
    def test_ideal_pyramid(self):
        """测试理想金字塔的质量值应接近1.0"""
        a = 1.0
        h = a / np.sqrt(2)
        p1 = [0, 0, 0]
        p2 = [a, 0, 0]
        p3 = [a, a, 0]
        p4 = [0, a, 0]
        p5 = [a/2, a/2, h]
        
        pyramid_quality = pyramid_shape_quality(p1, p2, p3, p4, p5)
        self.assertAlmostEqual(pyramid_quality, 1.0, places=5,
                              msg="理想金字塔质量应接近1.0")
    
    def test_degenerate_tetrahedron(self):
        """测试退化四面体的质量值应为0或很小"""
        p1 = [0, 0, 0]
        p2 = [1, 0, 0]
        p3 = [0, 1, 0]
        p4 = [1, 1, 0]
        
        tet_quality = tetrahedron_shape_quality(p1, p2, p3, p4)
        self.assertLess(tet_quality, 0.01,
                       msg="退化四面体质量应接近0")
    
    def test_degenerate_prism(self):
        """测试退化三棱柱的质量值应为0或很小"""
        p1 = [0, 0, 0]
        p2 = [1, 0, 0]
        p3 = [0.5, np.sqrt(3)/2, 0]
        p4 = [0, 0, 0]
        p5 = [1, 0, 0]
        p6 = [0.5, np.sqrt(3)/2, 0]
        
        prism_quality = prism_shape_quality(p1, p2, p3, p4, p5, p6)
        self.assertLess(prism_quality, 0.01,
                       msg="退化三棱柱质量应接近0")


if __name__ == "__main__":
    unittest.main()
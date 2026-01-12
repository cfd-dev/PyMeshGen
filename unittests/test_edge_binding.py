#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：OCC Edge与Connector绑定和离散化功能
测试边信息提取、离散化、Connector绑定等功能
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

try:
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeFace
    )
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.GC import GC_MakeSegment
    from OCC.Core.gp import gp_Pnt, gp_Vec
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False

try:
    from fileIO.geometry_io import (
        extract_edges_with_info,
        discretize_edge_by_size,
        discretize_edge_by_count,
        bind_edges_to_connectors,
        bind_edges_by_curve_name,
        create_test_square_shape
    )
    from data_structure.basic_elements import Connector, Part
    from data_structure.parameters import MeshParameters
    from data_structure.front2d import Front
    GEOMETRY_IO_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入模块: {e}")
    GEOMETRY_IO_AVAILABLE = False


class TestEdgeExtraction(unittest.TestCase):
    """边信息提取测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类的准备工作"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            raise unittest.SkipTest("OCC或几何导入模块不可用")

    def setUp(self):
        """每个测试用例前的准备工作"""
        self.shape = None
        if OCC_AVAILABLE and GEOMETRY_IO_AVAILABLE:
            self.shape = create_test_square_shape()

    def test_create_test_shape(self):
        """测试创建测试形状"""
        if not OCC_AVAILABLE:
            self.skipTest("OCC不可用")

        self.assertIsNotNone(self.shape, "创建测试形状失败")

    def test_extract_edges_with_info(self):
        """测试提取边信息"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        edges_info = extract_edges_with_info(self.shape)

        self.assertIsInstance(edges_info, list, "边信息应该是列表")
        self.assertEqual(len(edges_info), 4, "正方形应该有4条边")

        for i, edge_info in enumerate(edges_info):
            self.assertIn('curve', edge_info, f"边{i}缺少curve字段")
            self.assertIn('first', edge_info, f"边{i}缺少first字段")
            self.assertIn('last', edge_info, f"边{i}缺少last字段")
            self.assertIn('length', edge_info, f"边{i}缺少length字段")
            self.assertIn('start_point', edge_info, f"边{i}缺少start_point字段")
            self.assertIn('end_point', edge_info, f"边{i}缺少end_point字段")

            self.assertGreater(edge_info['length'], 0, f"边{i}长度应大于0")
            self.assertEqual(len(edge_info['start_point']), 3, f"边{i}起点应有3个坐标")
            self.assertEqual(len(edge_info['end_point']), 3, f"边{i}终点应有3个坐标")

    def test_edge_length_calculation(self):
        """测试边长度计算"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        edges_info = extract_edges_with_info(self.shape)

        for edge_info in edges_info:
            length = edge_info['length']
            start = np.array(edge_info['start_point'])
            end = np.array(edge_info['end_point'])
            calculated_length = np.linalg.norm(end - start)

            self.assertAlmostEqual(length, calculated_length, places=5,
                                   msg="边长度计算不准确")


class TestEdgeDiscretization(unittest.TestCase):
    """边离散化测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类的准备工作"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            raise unittest.SkipTest("OCC或几何导入模块不可用")

    def setUp(self):
        """每个测试用例前的准备工作"""
        self.shape = None
        self.edges_info = None
        if OCC_AVAILABLE and GEOMETRY_IO_AVAILABLE:
            self.shape = create_test_square_shape()
            self.edges_info = extract_edges_with_info(self.shape)

    def test_discretize_edge_by_size(self):
        """测试按尺寸离散化边"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        max_size = 0.3
        for i, edge_info in enumerate(self.edges_info):
            points = discretize_edge_by_size(edge_info, max_size)

            self.assertIsInstance(points, list, f"边{i}离散化结果应该是列表")
            self.assertGreater(len(points), 0, f"边{i}离散化应至少有1个点")
            self.assertLessEqual(len(points), 5, f"边{i}离散化点数不应超过5")

            for point in points:
                self.assertEqual(len(point), 3, f"边{i}离散点应有3个坐标")

    def test_discretize_edge_by_count(self):
        """测试按点数离散化边"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        num_points = 5
        for i, edge_info in enumerate(self.edges_info):
            points = discretize_edge_by_count(edge_info, num_points)

            self.assertIsInstance(points, list, f"边{i}离散化结果应该是列表")
            self.assertEqual(len(points), num_points,
                             f"边{i}离散化应有{num_points}个点")

            for point in points:
                self.assertEqual(len(point), 3, f"边{i}离散点应有3个坐标")

    def test_discretization_point_distribution(self):
        """测试离散化点分布"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        num_points = 5
        edge_info = self.edges_info[0]
        points = discretize_edge_by_count(edge_info, num_points)

        start = np.array(edge_info['start_point'])
        end = np.array(edge_info['end_point'])

        for i, point in enumerate(points):
            expected_t = i / (num_points - 1)
            expected_point = start + (end - start) * expected_t
            actual_point = np.array(point)

            np.testing.assert_array_almost_equal(
                actual_point, expected_point, decimal=5,
                err_msg=f"第{i}个点位置不正确"
            )

    def test_discretization_edge_case(self):
        """测试离散化边界情况"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        edge_info = self.edges_info[0]

        points_1 = discretize_edge_by_count(edge_info, 2)
        self.assertEqual(len(points_1), 2, "2个点离散化应返回2个点")

        points_2 = discretize_edge_by_count(edge_info, 100)
        self.assertEqual(len(points_2), 100, "100个点离散化应返回100个点")


class TestConnectorBinding(unittest.TestCase):
    """Connector绑定测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类的准备工作"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            raise unittest.SkipTest("OCC或几何导入模块不可用")

    def setUp(self):
        """每个测试用例前的准备工作"""
        self.shape = None
        self.parts = None
        if OCC_AVAILABLE and GEOMETRY_IO_AVAILABLE:
            self.shape = create_test_square_shape()

            param = MeshParameters(
                part_name="test_part",
                max_size=0.3,
                first_height=0.01,
                growth_rate=1.2
            )
            connector = Connector("test_part", "default", param)
            part = Part("test_part", param, [connector])
            self.parts = [part]

    def test_bind_edges_to_connectors(self):
        """测试绑定边到Connector"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        bind_edges_to_connectors(self.shape, self.parts)

        connector = self.parts[0].connectors[0]

        self.assertIsNotNone(connector.cad_obj, "Connector的cad_obj不应为None")
        self.assertIsInstance(connector.cad_obj, dict, "cad_obj应该是字典")
        self.assertIn('edge', connector.cad_obj, "cad_obj应包含edge字段")
        self.assertIn('curve', connector.cad_obj, "cad_obj应包含curve字段")

        self.assertIsInstance(connector.front_list, list, "front_list应该是列表")
        self.assertGreater(len(connector.front_list), 0, "front_list应至少包含1个Front")

    def test_front_list_content(self):
        """测试front_list内容"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        bind_edges_to_connectors(self.shape, self.parts)
        connector = self.parts[0].connectors[0]

        for i, front in enumerate(connector.front_list):
            self.assertIsInstance(front, Front, f"front_list[{i}]应该是Front对象")
            self.assertIsNotNone(front.node_elems, f"front_list[{i}]的node_elems不应为None")
            self.assertEqual(len(front.node_elems), 2, f"front_list[{i}]应有2个node_elems")

            self.assertGreater(front.length, 0, f"front_list[{i}]长度应大于0")
            self.assertIsNotNone(front.direction, f"front_list[{i}]方向向量不应为None")
            self.assertIsNotNone(front.normal, f"front_list[{i}]法向量不应为None")

    def test_front_geometry(self):
        """测试Front几何属性"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        bind_edges_to_connectors(self.shape, self.parts)
        connector = self.parts[0].connectors[0]

        if len(connector.front_list) > 0:
            front = connector.front_list[0]

            self.assertEqual(len(front.center), 3, "Front中心应有3个坐标")
            self.assertEqual(len(front.direction), 3, "Front方向应有3个坐标")
            self.assertEqual(len(front.normal), 2, "Front法向应有2个坐标（2D）")

            direction_norm = np.linalg.norm(front.direction)
            normal_norm = np.linalg.norm(front.normal)

            self.assertAlmostEqual(direction_norm, 1.0, places=5,
                                   msg="方向向量应该是单位向量")
            self.assertAlmostEqual(normal_norm, 1.0, places=5,
                                   msg="法向量应该是单位向量")

    def test_bind_edges_by_curve_name(self):
        """测试按曲线名称绑定边"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        param1 = MeshParameters(
            part_name="test_part",
            max_size=0.3,
            first_height=0.01,
            growth_rate=1.2
        )
        param2 = MeshParameters(
            part_name="test_part",
            max_size=0.25,
            first_height=0.01,
            growth_rate=1.2
        )

        connector1 = Connector("test_part", "curve_1", param1)
        connector2 = Connector("test_part", "curve_2", param2)
        part = Part("test_part", param1, [connector1, connector2])

        edge_curve_mapping = {
            0: "curve_1",
            1: "curve_2"
        }

        bind_edges_by_curve_name(self.shape, [part], edge_curve_mapping)

        self.assertGreater(len(connector1.front_list), 0,
                          "connector1的front_list应包含Front")
        self.assertGreater(len(connector2.front_list), 0,
                          "connector2的front_list应包含Front")


class TestIntegration(unittest.TestCase):
    """集成测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类的准备工作"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            raise unittest.SkipTest("OCC或几何导入模块不可用")

    def test_full_workflow(self):
        """测试完整工作流程"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        shape = create_test_square_shape()

        edges_info = extract_edges_with_info(shape)
        self.assertEqual(len(edges_info), 4, "应提取到4条边")

        param = MeshParameters(
            part_name="wall",
            max_size=0.2,
            first_height=0.01,
            growth_rate=1.2
        )
        connector = Connector("wall", "default", param)
        part = Part("wall", param, [connector])

        bind_edges_to_connectors(shape, [part])

        self.assertGreater(len(connector.front_list), 0,
                          "front_list应包含Front")
        self.assertIsNotNone(connector.cad_obj,
                            "cad_obj不应为None")

        total_length = sum(front.length for front in connector.front_list)
        self.assertGreater(total_length, 0, "总长度应大于0")

    def test_multiple_connectors(self):
        """测试多个Connector"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        shape = create_test_square_shape()

        param1 = MeshParameters(
            part_name="inlet",
            max_size=0.15,
            first_height=0.005,
            growth_rate=1.1
        )
        param2 = MeshParameters(
            part_name="outlet",
            max_size=0.2,
            first_height=0.01,
            growth_rate=1.2
        )

        connector1 = Connector("inlet", "inlet_curve", param1)
        connector2 = Connector("outlet", "outlet_curve", param2)

        part1 = Part("inlet", param1, [connector1])
        part2 = Part("outlet", param2, [connector2])

        edge_curve_mapping = {
            0: "inlet_curve",
            1: "outlet_curve"
        }

        bind_edges_by_curve_name(shape, [part1, part2], edge_curve_mapping)

        self.assertGreater(len(connector1.front_list), 0,
                          "connector1应有Front")
        self.assertGreater(len(connector2.front_list), 0,
                          "connector2应有Front")


class TestEdgeCases(unittest.TestCase):
    """边界情况测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类的准备工作"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            raise unittest.SkipTest("OCC或几何导入模块不可用")

    def test_empty_shape(self):
        """测试空形状"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire

        wire_maker = BRepBuilderAPI_MakeWire()
        if wire_maker.IsDone():
            wire = wire_maker.Wire()
        else:
            wire = None

        if wire is not None:
            edges_info = extract_edges_with_info(wire)
        else:
            edges_info = []

        self.assertEqual(len(edges_info), 0, "空形状应没有边")

    def test_single_point_discretization(self):
        """测试单点离散化"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        shape = create_test_square_shape()
        edges_info = extract_edges_with_info(shape)

        points = discretize_edge_by_count(edges_info[0], 1)
        self.assertEqual(len(points), 2, "单点离散化应返回2个点（起点和终点）")

    def test_large_max_size(self):
        """测试大尺寸离散化"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        shape = create_test_square_shape()
        edges_info = extract_edges_with_info(shape)

        points = discretize_edge_by_size(edges_info[0], max_size=100.0)
        self.assertEqual(len(points), 2, "大尺寸应只返回起点和终点")

    def test_small_max_size(self):
        """测试小尺寸离散化"""
        if not OCC_AVAILABLE or not GEOMETRY_IO_AVAILABLE:
            self.skipTest("OCC或几何导入模块不可用")

        shape = create_test_square_shape()
        edges_info = extract_edges_with_info(shape)

        points = discretize_edge_by_size(edges_info[0], max_size=0.001)
        self.assertGreater(len(points), 100, "小尺寸应返回很多点")


if __name__ == '__main__':
    unittest.main(verbosity=2)

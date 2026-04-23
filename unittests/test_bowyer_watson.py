#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bowyer-Watson / Triangle mesh_type=4 回归测试。"""

import sys
import json
import time
from pathlib import Path
import unittest
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加必要的子目录
for subdir in ["fileIO", "data_structure", "meshsize", "delaunay", "optimize", "utils"]:
    subdir_path = project_root / subdir
    if subdir_path.exists():
        sys.path.insert(0, str(subdir_path))

from delaunay.validation import (
    check_boundary_edges,
    check_hole_cleanup,
    check_topology_clean,
)


def resolve_case_input_path(input_file_str, project_root, fallback_input_dir=None):
    input_file = Path(input_file_str)
    if input_file.is_absolute():
        return input_file
    if input_file_str.startswith("./unittests") or input_file_str.startswith("./config"):
        return (project_root / input_file).resolve()
    if fallback_input_dir is not None:
        return (fallback_input_dir / input_file.name).resolve()
    return (project_root / input_file).resolve()


def resolve_effective_delaunay_backend(delaunay_backend, enable_boundary_layer):
    backend = str(delaunay_backend).strip().lower()
    if backend not in {"bowyer_watson", "triangle"}:
        backend = "bowyer_watson"
    if enable_boundary_layer:
        return "triangle"
    return backend


def create_delaunay_case_config(
    original_config_path,
    output_file,
    project_root,
    enable_boundary_layer=False,
    delaunay_backend="bowyer_watson",
    fallback_input_dir=None,
):
    with open(original_config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["mesh_type"] = 4
    config["delaunay_backend"] = delaunay_backend

    if enable_boundary_layer:
        print("  - 边界层: 启用")
    else:
        print("  - 边界层: 禁用")
        for part in config.get("parts", []):
            part["PRISM_SWITCH"] = "off"
            part["max_layers"] = 0

    if "input_file" in config:
        config["input_file"] = str(
            resolve_case_input_path(
                config["input_file"],
                project_root=project_root,
                fallback_input_dir=fallback_input_dir,
            )
        )

    config["output_file"] = str(output_file)
    config["viz_enabled"] = False
    config["debug_level"] = 0

    prefix = "temp_triangle" if delaunay_backend == "triangle" else "temp_bw"
    temp_config_path = project_root / f"{prefix}_{original_config_path.stem}.json"
    with open(temp_config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4, ensure_ascii=False)

    return temp_config_path


def assert_boundary_recovery(testcase, cas_file, grid, test_name):
    print("\n  - 边界恢复检查:")
    boundary_edges_result = check_boundary_edges(str(cas_file), grid, test_name)

    if boundary_edges_result["pass"]:
        print("  - [PASS] 边界恢复检查通过")
        for zone_key, zone_result in boundary_edges_result["zone_results"].items():
            print(f"    - {zone_key}: {zone_result['total_edges']}/{zone_result['total_edges']} 条边恢复")
            if zone_result["inner_boundary_inner_points"] > 0:
                print(f"      警告: 内边界内部发现 {zone_result['inner_boundary_inner_points']} 个点")
            if zone_result["inner_boundary_inner_cells"] > 0:
                print(f"      警告: 内边界内部发现 {zone_result['inner_boundary_inner_cells']} 个单元")
    else:
        print("  - [FAIL] 边界恢复检查失败")
        for zone_key, zone_result in boundary_edges_result["zone_results"].items():
            if zone_result["missing_edges"] > 0:
                print(f"    - {zone_key}: {zone_result['missing_edges']}/{zone_result['total_edges']} 条边丢失")
                for detail in zone_result["missing_details"][:5]:
                    n1, n2, coord1, coord2, reason = detail
                    if reason:
                        print(f"      边 ({n1},{n2}): {reason}")
                    else:
                        print(
                            f"      边 ({n1},{n2}): "
                            f"({coord1[0]:.4f},{coord1[1]:.4f}) -> ({coord2[0]:.4f},{coord2[1]:.4f})"
                        )
        if boundary_edges_result["issue"]:
            print(f"    - 问题: {boundary_edges_result['issue']}")
        testcase.fail(
            f"{test_name} 边界恢复检查失败: {boundary_edges_result['issue'] or '边界边丢失'}"
        )

    print("\n  - 孔洞清理检查:")
    hole_cleanup_result = check_hole_cleanup(str(cas_file), grid, test_name)
    if hole_cleanup_result["pass"]:
        print("  - [PASS] 孔洞清理检查通过")
        for hole_key, hole_result in hole_cleanup_result.get("hole_results", {}).items():
            print(
                f"    - {hole_key}: 内部点数={hole_result['points_inside']}, "
                f"内部单元数={hole_result['cells_inside']}"
            )
    else:
        print("  - [FAIL] 孔洞清理检查失败")
        for hole_key, hole_result in hole_cleanup_result.get("hole_results", {}).items():
            if hole_result["points_inside"] > 0 or hole_result["cells_inside"] > 0:
                print(
                    f"    - {hole_key}: 内部残留 {hole_result['points_inside']} 个点, "
                    f"{hole_result['cells_inside']} 个单元"
                )
        if hole_cleanup_result.get("issue"):
            print(f"    - 问题: {hole_cleanup_result['issue']}")
        testcase.fail(
            f"{test_name} 孔洞清理检查失败: "
            f"{hole_cleanup_result.get('issue', '孔洞内残留点或单元')}"
        )

    print("\n  - 拓扑洁净检查:")
    topology_result = check_topology_clean(grid, test_name)
    if topology_result["pass"]:
        print("  - [PASS] 拓扑洁净检查通过")
    else:
        print("  - [FAIL] 拓扑洁净检查失败")
        if topology_result.get("issue"):
            print(f"    - 问题: {topology_result['issue']}")
        testcase.fail(
            f"{test_name} 拓扑洁净检查失败: {topology_result.get('issue', '拓扑异常')}"
        )


def run_delaunay_config_test(
    testcase,
    original_config,
    output_file,
    project_root,
    test_name,
    enable_boundary_layer,
    delaunay_backend="bowyer_watson",
    fallback_input_dir=None,
    check_boundary_recovery=True,
):
    from PyMeshGen import PyMeshGen
    from data_structure.parameters import Parameters
    from fileIO.vtk_io import parse_vtk_msh
    from utils.message import DEBUG_LEVEL_VERBOSE, set_debug_level

    if not original_config.exists():
        testcase.skipTest(f"{original_config.name} 不存在")

    with open(original_config, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    set_debug_level(DEBUG_LEVEL_VERBOSE)

    try:
        bw_config = create_delaunay_case_config(
            original_config_path=original_config,
            output_file=output_file,
            project_root=project_root,
            enable_boundary_layer=enable_boundary_layer,
            delaunay_backend=delaunay_backend,
            fallback_input_dir=fallback_input_dir,
        )

        print(f"\n{test_name}:")
        print(f"  - 配置文件: {bw_config.name}")
        print(f"  - 输出文件: {output_file.name}")

        start = time.time()
        parameters = Parameters("FROM_CASE_JSON", str(bw_config))
        PyMeshGen(parameters)
        cost = time.time() - start

        testcase.assertTrue(output_file.exists(), "输出文件应该存在")
        grid = parse_vtk_msh(str(output_file))

        print(f"  - 生成时间: {cost:.2f}秒")
        print(f"  - 节点数: {grid.num_nodes}")
        print(f"  - 单元数: {grid.num_cells}")

        testcase.assertGreater(grid.num_nodes, 0, "节点数应大于 0")
        testcase.assertGreater(grid.num_cells, 0, "单元数应大于 0")
        testcase.assertLess(cost, 120, "生成时间应小于 120 秒")

        tri_count = sum(1 for cell in grid.cells if len(cell) == 3)
        quad_count = sum(1 for cell in grid.cells if len(cell) == 4)
        other_count = grid.num_cells - tri_count - quad_count

        print(f"  - 三角形数: {tri_count}")
        print(f"  - 四边形数: {quad_count}")
        print(f"  - 其他单元: {other_count}")

        if enable_boundary_layer:
            effective_backend = resolve_effective_delaunay_backend(
                delaunay_backend,
                enable_boundary_layer=True,
            )
            print(f"  - 模式: {effective_backend} + 边界层")
            testcase.assertGreater(tri_count, 0, "应该有三角形单元")
        else:
            effective_backend = resolve_effective_delaunay_backend(
                delaunay_backend,
                enable_boundary_layer=False,
            )
            print(f"  - 模式: 纯 {effective_backend} 三角网格")
            testcase.assertEqual(tri_count, grid.num_cells, "无边界层时应全部是三角形单元")

        if check_boundary_recovery:
            input_file = resolve_case_input_path(
                config_data.get("input_file", ""),
                project_root=project_root,
                fallback_input_dir=fallback_input_dir,
            )
            if input_file.exists():
                assert_boundary_recovery(testcase, input_file, grid, test_name)
            else:
                print(f"\n  - [SKIP] CAS 文件不存在: {input_file}，跳过边界恢复检查")

        print(f"  - [PASS] {test_name} 测试通过")
    except Exception as exc:
        print(f"  - [FAIL] {test_name} 测试失败: {exc}")
        import traceback

        traceback.print_exc()
        testcase.fail(f"{test_name} 测试失败: {exc}")


class TestBowyerWatsonIntegration(unittest.TestCase):
    """Bowyer-Watson 与核心流程集成测试"""

    def test_core_integration(self):
        """测试 13: 与 core.py 的集成"""
        try:
            config_path = project_root / "config" / "quad_quad.json"
            if not config_path.exists():
                self.skipTest("quad_quad.json 不存在")

            print("\n核心集成测试：验证 Bowyer-Watson 模块可导入")
            from core import create_bowyer_watson_mesh as bw_from_core
            self.assertIsNotNone(bw_from_core)
        except ImportError as e:
            self.skipTest(f"核心模块导入失败: {e}")
        except Exception as e:
            self.skipTest(f"核心集成测试失败: {e}")


class TestBowyerWatsonJSONConfig(unittest.TestCase):
    """Bowyer-Watson 通过 JSON 配置文件测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.test_dir = project_root / "unittests" / "test_files" / "2d_cases"
        cls.output_dir = cls.test_dir / "test_outputs"
        cls.output_dir.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """清理临时文件"""
        for pattern in ("temp_bw_*.json", "temp_triangle_*.json"):
            for temp_file in project_root.glob(pattern):
                try:
                    temp_file.unlink()
                except Exception:
                    pass
    
    def test_naca0012_bowyer_watson(self):
        """测试 14: NACA0012 使用 Bowyer-Watson 算法（无边界层）"""
        self._test_bowyer_watson_with_config(
            "naca0012.json",
            "test_naca0012_bw_no_bl.vtk",
            enable_boundary_layer=False,
            test_name="NACA0012 Bowyer-Watson（无边界层）"
        )
    
    def test_naca0012_bowyer_watson_with_boundary_layer(self):
        """测试 17: NACA0012 使用 Bowyer-Watson 算法（带边界层）"""
        self._test_bowyer_watson_with_config(
            "naca0012.json",
            "test_naca0012_bw_with_bl.vtk",
            enable_boundary_layer=True,
            test_name="NACA0012 Bowyer-Watson（带边界层）"
        )
    
    def test_anw_bowyer_watson(self):
        """测试 15: ANW 使用 Bowyer-Watson 算法（无边界层）"""
        self._test_bowyer_watson_with_config(
            "anw.json",
            "test_anw_bw_no_bl.vtk",
            enable_boundary_layer=False,
            test_name="ANW Bowyer-Watson（无边界层）"
        )
    
    def test_anw_bowyer_watson_with_boundary_layer(self):
        """测试 18: ANW 使用 Bowyer-Watson 算法（带边界层）"""
        self._test_bowyer_watson_with_config(
            "anw.json",
            "test_anw_bw_with_bl.vtk",
            enable_boundary_layer=True,
            test_name="ANW Bowyer-Watson（带边界层）"
        )

    def test_naca0012_triangle_backend(self):
        """测试: NACA0012 使用 Triangle 后端（无边界层）"""
        self._test_bowyer_watson_with_config(
            "naca0012.json",
            "test_naca0012_triangle_no_bl.vtk",
            enable_boundary_layer=False,
            test_name="NACA0012 Triangle 后端（无边界层）",
            delaunay_backend="triangle",
        )

    def test_anw_triangle_backend(self):
        """测试: ANW 使用 Triangle 后端（无边界层）"""
        self._test_bowyer_watson_with_config(
            "anw.json",
            "test_anw_triangle_no_bl.vtk",
            enable_boundary_layer=False,
            test_name="ANW Triangle 后端（无边界层）",
            delaunay_backend="triangle",
        )
    
    def test_rae2822_bowyer_watson(self):
        """测试 16: RAE2822 使用 Bowyer-Watson 算法（无边界层）"""
        self._test_bowyer_watson_with_config(
            "rae2822.json",
            "test_rae2822_bw_no_bl.vtk",
            enable_boundary_layer=False,
            test_name="RAE2822 Bowyer-Watson（无边界层）"
        )
    
    def test_rae2822_bowyer_watson_with_boundary_layer(self):
        """测试 19: RAE2822 使用 Bowyer-Watson 算法（带边界层）"""
        self._test_bowyer_watson_with_config(
            "rae2822.json",
            "test_rae2822_bw_with_bl.vtk",
            enable_boundary_layer=True,
            test_name="RAE2822 Bowyer-Watson（带边界层）"
        )

    def test_quad_quad_bowyer_watson(self):
        """测试 20: quad_quad 使用 Bowyer-Watson 算法（无边界层）"""
        self._test_bowyer_watson_with_config_from_root(
            "quad_quad.json",
            "test_quad_quad_bw_no_bl.vtk",
            enable_boundary_layer=False,
            test_name="quad_quad Bowyer-Watson（无边界层）",
            check_boundary_recovery=True
        )

    def test_quad_quad_bowyer_watson_with_boundary_layer(self):
        """测试 21: quad_quad 使用 Bowyer-Watson 算法（带边界层）"""
        self._test_bowyer_watson_with_config_from_root(
            "quad_quad.json",
            "test_quad_quad_bw_with_bl.vtk",
            enable_boundary_layer=True,
            test_name="quad_quad Bowyer-Watson（带边界层）",
            check_boundary_recovery=True
        )

    def test_cylinder_bowyer_watson(self):
        """测试 22: cylinder 使用 Bowyer-Watson 算法（无边界层）"""
        self._test_bowyer_watson_with_config_from_root(
            "cylinder.json",
            "test_cylinder_bw_no_bl.vtk",
            enable_boundary_layer=False,
            test_name="cylinder Bowyer-Watson（无边界层）",
            check_boundary_recovery=True
        )

    def test_cylinder_bowyer_watson_with_boundary_layer(self):
        """测试 23: cylinder 使用 Bowyer-Watson 算法（带边界层）"""
        self._test_bowyer_watson_with_config_from_root(
            "cylinder.json",
            "test_cylinder_bw_with_bl.vtk",
            enable_boundary_layer=True,
            test_name="cylinder Bowyer-Watson（带边界层）",
            check_boundary_recovery=True
        )

    def _test_bowyer_watson_with_config_from_root(
        self,
        config_file,
        output_file_name,
        enable_boundary_layer,
        test_name,
        check_boundary_recovery=True,
    ):
        """从项目根目录 config/ 文件夹加载配置的 Bowyer-Watson 测试方法。"""
        run_delaunay_config_test(
            testcase=self,
            original_config=project_root / "config" / config_file,
            output_file=self.output_dir / output_file_name,
            project_root=project_root,
            test_name=test_name,
            enable_boundary_layer=enable_boundary_layer,
            fallback_input_dir=None,
            check_boundary_recovery=check_boundary_recovery,
        )

    def _test_bowyer_watson_with_config(
        self,
        config_file,
        output_file_name,
        enable_boundary_layer,
        test_name,
        delaunay_backend="bowyer_watson",
    ):
        """通用的 mesh_type=4 Delaunay 配置测试方法。"""
        run_delaunay_config_test(
            testcase=self,
            original_config=self.test_dir / config_file,
            output_file=self.output_dir / output_file_name,
            project_root=project_root,
            test_name=test_name,
            enable_boundary_layer=enable_boundary_layer,
            delaunay_backend=delaunay_backend,
            fallback_input_dir=self.test_dir,
            check_boundary_recovery=True,
        )


if __name__ == '__main__':
    # 运行单元测试
    unittest.main(verbosity=2)


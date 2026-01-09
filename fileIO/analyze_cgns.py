#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 CGNS 文件结构
"""

import sys
from pathlib import Path
import h5py

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加 meshio 到 Python 路径
meshio_path = project_root / "3rd_party" / "meshio" / "src"
if meshio_path.exists():
    sys.path.insert(0, str(meshio_path))


def analyze_cgns_structure(file_path: Path):
    """
    分析 CGNS 文件结构

    Args:
        file_path: CGNS 文件路径
    """
    print(f"\n{'='*70}")
    print(f"分析文件: {file_path.name}")
    print(f"{'='*70}")

    try:
        with h5py.File(file_path, 'r') as f:
            # 查找 Base
            base_name = None
            for key in f.keys():
                if "Base" in key or "base" in key or "BASE" in key:
                    base_name = key
                    break

            if not base_name:
                print("❌ 未找到 Base 节点")
                return

            print(f"Base: {base_name}")
            base = f[base_name]

            # 查找 Zone
            zone_name = None
            for key in base.keys():
                obj = base[key]
                if isinstance(obj, h5py.Group):
                    if "GridCoordinates" in obj.keys():
                        zone_name = key
                        break

            if not zone_name:
                print("❌ 未找到 Zone 节点")
                return

            print(f"Zone: {zone_name}")
            zone = base[zone_name]

            # 读取节点坐标
            grid_coords = zone["GridCoordinates"]
            x = None
            y = None
            z = None

            for key in grid_coords.keys():
                if "CoordinateX" in key or "coordinateX" in key or "COORDINATEX" in key:
                    x = grid_coords[key][" data"]
                elif "CoordinateY" in key or "coordinateY" in key or "COORDINATEY" in key:
                    y = grid_coords[key][" data"]
                elif "CoordinateZ" in key or "coordinateZ" in key or "COORDINATEZ" in key:
                    z = grid_coords[key][" data"]

            if x is not None:
                print(f"节点数: {len(x)}")

            # 查找所有单元组
            print(f"\n单元组:")
            for key in zone.keys():
                obj = zone[key]
                if isinstance(obj, h5py.Group):
                    if "ElementConnectivity" in obj.keys():
                        elem_conn = obj["ElementConnectivity"][" data"]
                        elem_range = obj["ElementRange"][" data"]

                        print(f"\n  {key}:")
                        print(f"    ElementRange: {elem_range}")
                        print(f"    ElementConnectivity shape: {elem_conn.shape}")
                        print(f"    ElementConnectivity size: {elem_conn.shape[0]}")

                        # 尝试推断节点数
                        idx_min = elem_range[0]
                        idx_max_or_count = elem_range[1]

                        # 尝试不同的节点数
                        possible_nodes_per_cell = [2, 3, 4, 5, 6, 8, 9, 10, 20, 27]
                        for nodes_per_cell in possible_nodes_per_cell:
                            expected_cells_from_data = elem_conn.shape[0] // nodes_per_cell
                            expected_cells_from_range = idx_max_or_count - idx_min + 1

                            if expected_cells_from_data == expected_cells_from_range:
                                print(f"    ✓ 推断: {nodes_per_cell} 节点/单元, {expected_cells_from_data} 个单元 (格式1)")
                                break
                            elif expected_cells_from_data == idx_max_or_count:
                                print(f"    ✓ 推断: {nodes_per_cell} 节点/单元, {expected_cells_from_data} 个单元 (格式2)")
                                break
                        else:
                            print(f"    ✗ 无法推断节点数")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 分析失败的文件
    failed_file = project_root / "examples" / "2d_airfoils" / "sd7003_vis_p2.cgns"

    if not failed_file.exists():
        print(f"❌ 文件不存在: {failed_file}")
        sys.exit(1)

    analyze_cgns_structure(failed_file)

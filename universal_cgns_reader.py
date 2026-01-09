#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用h5py实现的通用CGNS文件读取器
支持所有类型的CGNS文件，包括高阶单元
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("❌ h5py 未安装，无法读取 CGNS 文件")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 添加 meshio 到 Python 路径
meshio_path = project_root / "3rd_party" / "meshio" / "src"
if meshio_path.exists():
    sys.path.insert(0, str(meshio_path))


class UniversalCGNSReader:
    """通用CGNS文件读取器"""

    def __init__(self, file_path: str):
        """
        初始化CGNS读取器

        Args:
            file_path: CGNS文件路径
        """
        self.file_path = file_path
        self.points = None
        self.cells = []
        self.cell_info = []
        self.metadata = {}

    def read(self) -> bool:
        """
        读取CGNS文件

        Returns:
            bool: 是否成功读取
        """
        try:
            with h5py.File(self.file_path, 'r') as f:
                # 查找Base
                base_name = self._find_base(f)
                if not base_name:
                    print(f"❌ 未找到Base节点")
                    return False

                base = f[base_name]

                # 查找Zone
                zone_name = self._find_zone(base)
                if not zone_name:
                    print(f"❌ 未找到Zone节点")
                    return False

                zone = base[zone_name]

                # 读取节点坐标
                self.points = self._read_points(zone)
                if self.points is None:
                    print(f"❌ 读取节点坐标失败")
                    return False

                # 读取单元数据
                self.cells = self._read_cells(zone)
                if not self.cells:
                    print(f"❌ 读取单元数据失败")
                    return False

                # 读取元数据
                self.metadata = self._read_metadata(f, base, zone)

                return True

        except Exception as e:
            print(f"❌ 读取CGNS文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _find_base(self, f: h5py.File) -> Optional[str]:
        """查找Base节点"""
        for key in f.keys():
            if "Base" in key or "base" in key or "BASE" in key:
                return key
        return None

    def _find_zone(self, base: h5py.Group) -> Optional[str]:
        """查找Zone节点"""
        for key in base.keys():
            obj = base[key]
            if isinstance(obj, h5py.Group):
                # 检查是否包含GridCoordinates（Zone的特征）
                if "GridCoordinates" in obj.keys():
                    return key
        return None

    def _read_points(self, zone: h5py.Group) -> Optional[np.ndarray]:
        """读取节点坐标"""
        try:
            grid_coords = zone["GridCoordinates"]

            # 查找X坐标
            x = None
            for key in grid_coords.keys():
                if "CoordinateX" in key or "coordinateX" in key or "COORDINATEX" in key:
                    x = grid_coords[key][" data"]
                    break

            # 查找Y坐标
            y = None
            for key in grid_coords.keys():
                if "CoordinateY" in key or "coordinateY" in key or "COORDINATEY" in key:
                    y = grid_coords[key][" data"]
                    break

            # 查找Z坐标
            z = None
            for key in grid_coords.keys():
                if "CoordinateZ" in key or "coordinateZ" in key or "COORDINATEZ" in key:
                    z = grid_coords[key][" data"]
                    break

            if x is None or y is None:
                return None

            # 如果没有Z坐标，则创建全零的Z坐标
            if z is None:
                z = np.zeros_like(x)

            return np.column_stack([x, y, z])

        except Exception as e:
            print(f"❌ 读取节点坐标失败: {e}")
            return None

    def _read_cells(self, zone: h5py.Group) -> List[Dict[str, Any]]:
        """读取单元数据"""
        cells = []
        self.cell_info = []

        for key in zone.keys():
            obj = zone[key]
            if isinstance(obj, h5py.Group):
                # 检查是否包含ElementConnectivity（单元组的特征）
                if "ElementConnectivity" in obj.keys():
                    try:
                        # 读取ElementConnectivity数据
                        elem_conn = obj["ElementConnectivity"][" data"]
                        
                        # 读取ElementRange数据
                        elem_range = obj["ElementRange"][" data"]
                        idx_min = elem_range[0]
                        idx_max_or_count = elem_range[1]

                        # 根据单元组名称推断单元类型
                        cell_type = self._infer_cell_type(key)

                        # 根据ElementConnectivity数据推断节点数
                        # 需要确定正确的单元数
                        # ElementRange有两种格式：
                        # 1. [起始索引, 结束索引] - 单元数 = 结束索引 - 起始索引 + 1
                        # 2. [起始索引, 单元数] - 单元数 = 第二个值
                        
                        # 先尝试几种常见的节点数
                        possible_nodes_per_cell = [2, 3, 4, 5, 6, 8, 9, 10, 20, 27]
                        
                        for nodes_per_cell in possible_nodes_per_cell:
                            expected_cells_from_data = elem_conn.shape[0] // nodes_per_cell
                            expected_cells_from_range = idx_max_or_count - idx_min + 1
                            
                            if expected_cells_from_data == expected_cells_from_range:
                                # 格式1: [起始索引, 结束索引]
                                num_cells = expected_cells_from_data
                                break
                            elif expected_cells_from_data == idx_max_or_count:
                                # 格式2: [起始索引, 单元数]
                                num_cells = expected_cells_from_data
                                break
                        else:
                            # 如果都不匹配，使用格式2
                            num_cells = idx_max_or_count
                            # 根据数据大小推断节点数
                            nodes_per_cell = elem_conn.shape[0] // num_cells

                        # 重塑数据
                        cell_data = np.array(elem_conn).reshape(num_cells, nodes_per_cell) - 1

                        # 保存单元信息
                        cell_info = {
                            'name': key,
                            'type': cell_type,
                            'count': num_cells,
                            'nodes_per_cell': nodes_per_cell
                        }
                        self.cell_info.append(cell_info)

                        # 保存单元数据
                        cells.append({
                            'type': cell_type,
                            'data': cell_data,
                            'num_nodes': nodes_per_cell,
                            'num_cells': num_cells
                        })

                    except Exception as e:
                        print(f"❌ 读取单元组 {key} 失败: {e}")
                        continue

        return cells

    def _infer_cell_type(self, name: str) -> str:
        """根据单元组名称推断单元类型"""
        name_lower = name.lower()
        
        if "tet" in name_lower:
            return "tetra"
        elif "tri" in name_lower:
            return "triangle"
        elif "hex" in name_lower:
            return "hexahedron"
        elif "wedge" in name_lower or "prism" in name_lower:
            return "wedge"
        elif "pyramid" in name_lower:
            return "pyramid"
        elif "quad" in name_lower:
            return "quad"
        elif "bar" in name_lower or "line" in name_lower:
            return "line"
        else:
            # 默认为四面体
            return "tetra"

    def _read_metadata(self, f: h5py.File, base: h5py.Group, zone: h5py.Group) -> Dict[str, Any]:
        """读取元数据"""
        metadata = {}

        # 读取文件信息
        if "CGNSLibraryVersion" in f.keys():
            try:
                version_obj = f["CGNSLibraryVersion"]
                if isinstance(version_obj, h5py.Dataset):
                    metadata['CGNSLibraryVersion'] = version_obj[()]
                else:
                    metadata['CGNSLibraryVersion'] = str(version_obj)
            except Exception:
                pass

        # 读取Base名称
        metadata['BaseName'] = base.name

        # 读取Zone名称
        metadata['ZoneName'] = zone.name

        # 读取GridCoordinates
        if "GridCoordinates" in zone.keys():
            metadata['GridCoordinates'] = True

        # 读取Base信息
        if hasattr(base, 'attrs'):
            for key, value in base.attrs.items():
                metadata[f'base_{key}'] = value

        # 读取Zone信息
        if hasattr(zone, 'attrs'):
            for key, value in zone.attrs.items():
                metadata[f'zone_{key}'] = value

        return metadata

    def to_meshio_format(self):
        """
        转换为meshio格式

        Returns:
            meshio.Mesh: meshio网格对象，如果失败则返回None
        """
        try:
            import meshio

            # 转换单元格式
            cells = []
            for cell in self.cells:
                cells.append((cell['type'], cell['data']))

            mesh = meshio.Mesh(
                points=self.points,
                cells=cells
            )

            return mesh

        except ImportError:
            print(f"❌ meshio未安装")
            return None

        except Exception as e:
            print(f"❌ 转换失败: {e}")
            return None

    def print_summary(self):
        """打印读取结果摘要"""
        print(f"\n{'='*70}")
        print(f"CGNS文件读取摘要: {Path(self.file_path).name}")
        print(f"{'='*70}")
        print(f"节点数: {len(self.points)}")
        print(f"单元组数: {len(self.cells)}")
        print(f"单元总数: {sum(cell['data'].shape[0] for cell in self.cells)}")
        
        print(f"\n单元类型详情:")
        for cell_info in self.cell_info:
            print(f"  {cell_info['type']} ({cell_info['name']}): "
                  f"{cell_info['count']} 个单元, {cell_info['nodes_per_cell']} 节点/单元")
        
        # 计算坐标范围
        x_min, x_max = self.points[:, 0].min(), self.points[:, 0].max()
        y_min, y_max = self.points[:, 1].min(), self.points[:, 1].max()
        z_min, z_max = self.points[:, 2].min(), self.points[:, 2].max()
        
        print(f"\n坐标范围:")
        print(f"  X: [{x_min:.6f}, {x_max:.6f}]")
        print(f"  Y: [{y_min:.6f}, {y_max:.6f}]")
        print(f"  Z: [{z_min:.6f}, {z_max:.6f}]")


def test_all_cgns_files(meshes_dir: Path):
    """
    测试meshes目录下所有CGNS文件

    Args:
        meshes_dir: meshes目录路径
    """
    print(f"\n{'='*70}")
    print(f"使用通用CGNS读取器测试所有CGNS文件")
    print(f"{'='*70}")

    # 查找所有CGNS文件
    cgns_files = sorted(meshes_dir.glob("*.cgns"))

    if not cgns_files:
        print(f"\n❌ 未找到任何CGNS文件")
        return

    print(f"\n找到 {len(cgns_files)} 个CGNS文件")

    # 测试每个文件
    results = []
    success_count = 0
    fail_count = 0

    for cgns_file in cgns_files:
        print(f"\n{'='*70}")
        print(f"测试文件: {cgns_file.name}")
        print(f"{'='*70}")

        reader = UniversalCGNSReader(str(cgns_file))
        success = reader.read()

        if success:
            reader.print_summary()
            success_count += 1
            
            # 尝试转换为meshio格式
            mesh = reader.to_meshio_format()
            if mesh:
                print(f"\n✅ 成功转换为meshio格式")
            else:
                print(f"\n⚠️ 转换为meshio格式失败")
        else:
            fail_count += 1

        results.append({
            'file': cgns_file.name,
            'success': success,
            'reader': reader
        })

    # 打印汇总结果
    print(f"\n{'='*70}")
    print(f"测试结果汇总")
    print(f"{'='*70}")
    print(f"总文件数: {len(cgns_files)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")

    if fail_count > 0:
        print(f"\n失败的文件:")
        for result in results:
            if not result['success']:
                print(f"  - {result['file']}")

    print(f"\n{'='*70}")

    # 返回所有结果
    return results


if __name__ == "__main__":
    # 设置meshes目录路径
    meshes_dir = project_root / "meshes"

    if not meshes_dir.exists():
        print(f"❌ meshes目录不存在: {meshes_dir}")
        sys.exit(1)

    # 测试所有CGNS文件
    results = test_all_cgns_files(meshes_dir)

    # 检查是否所有文件都成功读取
    all_success = all(result['success'] for result in results)

    if all_success:
        print(f"\n✅ 所有CGNS文件都成功读取和解析！")
        sys.exit(0)
    else:
        print(f"\n❌ 部分CGNS文件读取失败")
        sys.exit(1)
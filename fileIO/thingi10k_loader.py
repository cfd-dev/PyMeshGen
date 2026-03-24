#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thingi10K数据集加载器
用于从Thingi10K数据集中导入指定ID的网格模型
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


def init_thingi10k(cache_dir: Optional[str] = None) -> bool:
    """
    初始化Thingi10K数据集
    
    Args:
        cache_dir: 缓存目录路径，如果为None则使用默认位置
        
    Returns:
        bool: 初始化是否成功
    """
    try:
        import thingi10k
        
        if cache_dir:
            thingi10k.init(cache_dir=cache_dir)
        else:
            thingi10k.init()
        
        return True
    except ImportError:
        return False
    except Exception:
        return False


def get_thingi10k_dataset_info(cache_dir: Optional[str] = None, variant: str = None) -> List[Dict[str, Any]]:
    """
    获取Thingi10K数据集的元信息列表

    Args:
        cache_dir: 缓存目录路径
        variant: 数据集变体类型，'raw' 或 'tetwild'，如果为None则使用默认

    Returns:
        数据集条目列表，每个条目包含file_id, author, license等信息
    """
    try:
        import thingi10k

        if cache_dir:
            if variant:
                thingi10k.init(cache_dir=cache_dir, variant=variant)
            else:
                thingi10k.init(cache_dir=cache_dir)
        else:
            if variant:
                thingi10k.init(variant=variant)
            else:
                thingi10k.init()

        dataset = []
        for entry in thingi10k.dataset():
            dataset.append({
                'file_id': entry['file_id'],
                'author': entry.get('author', 'Unknown'),
                'license': entry.get('license', 'Unknown'),
                'name': entry.get('name', 'Unnamed'),
                'file_path': entry.get('file_path', ''),
            })

        return dataset
    except Exception:
        return []


def search_thingi10k_by_id(
    file_id: str,
    cache_dir: Optional[str] = None,
    variant: str = None
) -> Optional[Dict[str, Any]]:
    """
    根据ID搜索Thingi10K数据集条目

    Args:
        file_id: 文件ID
        cache_dir: 缓存目录路径
        variant: 数据集变体类型，'raw' 或 'tetwild'

    Returns:
        数据集条目信息，如果未找到返回None
    """
    dataset = get_thingi10k_dataset_info(cache_dir, variant)

    try:
        file_id_int = int(file_id)
    except ValueError:
        file_id_int = None

    for entry in dataset:
        entry_file_id = entry.get('file_id')
        if entry_file_id == file_id or (file_id_int is not None and entry_file_id == file_id_int):
            return entry

    return None


def load_thingi10k_mesh(
    file_id: str,
    cache_dir: Optional[str] = None,
    variant: str = None
) -> Optional[Tuple[List[List[float]], List[List[int]]]]:
    """
    根据ID加载Thingi10K网格模型

    Args:
        file_id: 文件ID
        cache_dir: 缓存目录路径
        variant: 数据集变体类型，'raw' 或 'tetwild'

    Returns:
        (vertices, facets) 元组，如果加载失败返回None
        vertices: 节点坐标列表 [[x, y, z], ...]
        facets: 面单元列表 [[node1, node2, node3], ...]
    """
    try:
        import thingi10k

        if cache_dir:
            if variant:
                thingi10k.init(cache_dir=cache_dir, variant=variant)
            else:
                thingi10k.init(cache_dir=cache_dir)
        else:
            if variant:
                thingi10k.init(variant=variant)
            else:
                thingi10k.init()

        entry = search_thingi10k_by_id(file_id, cache_dir, variant)

        if entry is None:
            return None

        vertices, facets = thingi10k.load_file(entry['file_path'])

        return (vertices.tolist(), facets.tolist())
    except Exception:
        return None


def load_thingi10k_mesh_as_stl(
    file_id: str,
    output_path: str,
    cache_dir: Optional[str] = None,
    variant: str = None
) -> bool:
    """
    根据ID加载Thingi10K网格并保存为STL文件

    Args:
        file_id: 文件ID
        output_path: 输出STL文件路径
        cache_dir: 缓存目录路径
        variant: 数据集变体类型，'raw' 或 'tetwild'

    Returns:
        bool: 保存是否成功
    """
    try:
        mesh_data = load_thingi10k_mesh(file_id, cache_dir, variant)

        if mesh_data is None:
            return False

        vertices, facets = mesh_data

        with open(output_path, 'w') as f:
            f.write("solid thingi10k_model\n")

            for facet in facets:
                if len(facet) >= 3:
                    p1 = vertices[facet[0]]
                    p2 = vertices[facet[1]]
                    p3 = vertices[facet[2]]

                    f.write("facet normal 0.0 0.0 0.0\n")
                    f.write("    outer loop\n")
                    f.write(f"        vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
                    f.write(f"        vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n")
                    f.write(f"        vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n")
                    f.write("    endloop\n")
                    f.write("endfacet\n")

            f.write("endsolid thingi10k_model\n")

        return True
    except Exception:
        return False


def get_thingi10k_mesh_statistics(
    file_id: str,
    cache_dir: Optional[str] = None,
    variant: str = None
) -> Optional[Dict[str, int]]:
    """
    获取Thingi10K网格的统计信息

    Args:
        file_id: 文件ID
        cache_dir: 缓存目录路径
        variant: 数据集变体类型，'raw' 或 'tetwild'

    Returns:
        统计信息字典，如果加载失败返回None
    """
    mesh_data = load_thingi10k_mesh(file_id, cache_dir, variant)

    if mesh_data is None:
        return None

    vertices, facets = mesh_data

    num_vertices = len(vertices)
    num_triangles = sum(1 for f in facets if len(f) == 3)
    num_quads = sum(1 for f in facets if len(f) == 4)

    return {
        'num_vertices': num_vertices,
        'num_triangles': num_triangles,
        'num_quads': num_quads,
        'num_total_faces': len(facets),
    }

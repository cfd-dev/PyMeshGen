"""曲面网格生成模块

将 sfmesh 3D 曲面网格生成接入 GUI 流程：
- 从 Connector + OCC Face 生成曲面三角形网格
- 将 SurfaceTriangle 列表转换为 Unstructured_Grid
"""

from typing import List, Optional, Tuple
from data_structure.unstructured_grid import Unstructured_Grid
from utils.message import info


def generate_surface_mesh_from_domain(
    connectors: list,
    occ_face,
    spacing: float = 1.0,
) -> 'Unstructured_Grid':
    """
    从 Connector 列表和 OCC Face 生成曲面网格

    Args:
        connectors: Connector 对象列表（包含预离散的边界阵面）
        occ_face: OCC TopoDS_Face 曲面对象
        spacing: 全局网格尺寸

    Returns:
        生成的 Unstructured_Grid 对象
    """
    from sfmesh.mesh_3d_afm import SurfaceMeshGenerator

    generator = SurfaceMeshGenerator(
        surface=occ_face,
        global_spacing=spacing,
        connectors=connectors,
    )
    triangles = generator.generate()
    info(f"曲面网格生成完成: {len(triangles)} 三角形")

    return convert_surface_triangles_to_unstructured_grid(triangles)


def convert_surface_triangles_to_unstructured_grid(
    triangles: list,
    part_name: str = "surface_mesh",
) -> Unstructured_Grid:
    """
    将 SurfaceTriangle 列表转换为 Unstructured_Grid

    通过 NodeElement3D.hash 去重节点，构建三角形单元。

    Args:
        triangles: SurfaceTriangle 列表
        part_name: 部件名称

    Returns:
        Unstructured_Grid 对象
    """
    node_hash_to_idx = {}
    node_coords = []
    cells = []

    for tri in triangles:
        tri_node_ids = []
        for node in tri.nodes:
            h = node.hash
            if h not in node_hash_to_idx:
                node_hash_to_idx[h] = len(node_coords)
                node_coords.append(list(node.coords))
            tri_node_ids.append(node_hash_to_idx[h])
        cells.append(tri_node_ids)

    grid = Unstructured_Grid.from_cells(
        node_coords=node_coords,
        cells=cells,
        grid_dimension=3,
        cell_dimension=2,
    )

    info(f"曲面网格转换完成: {len(node_coords)} 节点, {len(cells)} 三角形")
    return grid

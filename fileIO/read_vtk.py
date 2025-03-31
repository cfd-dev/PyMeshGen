import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "utils"))
from geometry_info import Unstructured_Grid, NodeElement, Triangle, Quadrilateral

VTK_ELEMENT_TYPE = {
    1: "POINT",
    3: "LINE",
    5: "TRI",
    9: "QUAD",
    10: "TETRA",
    12: "HEX",
    13: "PRISM",
    14: "PYRAMID",
}


def read_vtk(filename):
    node_coords = []
    cell_idx_container = []
    boundary_nodes_idx = []
    cell_type_container = []

    with open(filename, "r") as f:
        lines = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("POINTS"):
            _, nNodes, precision = line.split()
            nNodes = int(nNodes)
            node_coords = []
            for _ in range(nNodes):
                i += 1
                coords = list(map(float, lines[i].split()))[:2]  # 只取前2个坐标
                node_coords.append(coords)
        elif line.startswith("CELLS"):
            _, nCells, dataLength = line.split()
            dataLength = int(dataLength)
            nCells = int(nCells)
            cell_idx_container = []
            for _ in range(nCells):
                i += 1
                parts = list(map(int, lines[i].split()))
                num_nodes = parts[0]
                nodes = [n for n in parts[1:]]  # 注意为0-based索引
                if len(nodes) != num_nodes:
                    raise ValueError("节点数量与指定的不一致!")
                cell_idx_container.append(nodes)
            if len(cell_idx_container) != nCells:
                raise ValueError("单元数量与指定的不一致!")
            if sum(len(cell) + 1 for cell in cell_idx_container) != dataLength:
                raise ValueError("数据长度与指定的不一致!")
        elif line.startswith("CELL_TYPES"):
            _, nCells = line.split()
            nCells = int(nCells)
            for _ in range(nCells):
                i += 1
                cell_type = int(lines[i])
                # 检查是否为支持的类型
                if cell_type not in VTK_ELEMENT_TYPE:
                    raise ValueError(f"不支持的单元类型: {cell_type}")
                cell_type_container.append(cell_type)
        elif line.startswith("POINT_DATA"):
            _, nNodes = line.split()
            nNodes = int(nNodes)
            i += 1  # 跳过SCALARS node_id int 1行
            if lines[i].startswith("SCALARS node_id"):
                i += 1  # 跳过LOOKUP_TABLE行
                if lines[i].startswith("LOOKUP_TABLE default"):
                    NodeTypes = []
                    for _ in range(nNodes):
                        i += 1
                        val = int(lines[i])
                        NodeTypes.append(val)
                    boundary_nodes_idx = [
                        idx for idx, v in enumerate(NodeTypes) if v == 1
                    ]
        elif line.startswith("CELL_DATA"):
            _, nCells = line.split()
            nCells = int(nCells)
            i += 1  # 跳过SCALARS cell_id int 1行
            if lines[i].startswith("SCALARS cell_id"):
                i += 1  # 跳过LOOKUP_TABLE行
                if lines[i].startswith("LOOKUP_TABLE default"):
                    Cell_ID = []
                    for _ in range(nCells):
                        i += 1
                        val = int(lines[i])
                        Cell_ID.append(val)
        i += 1

    return node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container


def reconstruct_mesh_from_vtk(
    node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container
):

    num_nodes = len(node_coords)
    num_cells = len(cell_idx_container)

    node_container = [NodeElement(node_coords[idx], idx) for idx in range(num_nodes)]
    # 初始化长度为num_cells的列表，每个元素为None
    cell_container = [None] * num_cells
    # 遍历每个单元，根据单元类型创建相应的对象
    for idx in range(num_cells):
        node1 = node_container[cell_idx_container[idx][0]]
        node2 = node_container[cell_idx_container[idx][1]]
        node3 = node_container[cell_idx_container[idx][2]]

        cell_type = VTK_ELEMENT_TYPE.get(cell_type_container[idx], "未知类型")
        if cell_type == "TRI":  # 三角形
            cell = Triangle(node1, node2, node3, idx)
        elif cell_type == "QUAD":  # 四边形
            node4 = node_container[cell_idx_container[idx][3]]
            cell = Quadrilateral(node1, node2, node3, node4, idx)
        else:
            raise ValueError(f"Unsupported cell type: {cell_type_container[idx]}")
        cell_container[idx] = cell

        boundary_nodes = [node_container[idx] for idx in boundary_nodes_idx]
        unstr_grid = Unstructured_Grid(
            cell_container, node_coords, boundary_nodes
        )  # 注意这里的cell_container是修改过的，已经包含了单元对象而不是索引列表
    return unstr_grid


def parse_vtk_msh(filename):
    node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container = read_vtk(
        filename
    )
    unstr_grid = reconstruct_mesh_from_vtk(
        node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container
    )
    return unstr_grid

import numpy as np
from enum import IntEnum

from utils.message import info


class VTK_ELEMENT_TYPE(IntEnum):
    """VTK元素类型枚举"""

    EMPTY_CELL = 0
    VERTEX = 1
    LINE = 3
    TRI = 5  # 三角形单元
    QUAD = 9  # 四边形单元
    TETRA = 10  # 四面体
    HEXA = 12  # 六面体
    PRISM = 13  # 三棱柱
    PYRAMID = 14  # 金字塔


def write_vtk(
    filename, node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container, cell_part_names=None
):
    """
    将网格数据写入VTK文件。暂只支持二维三角形、四边形单元。

    Args:
        filename (str): 输出文件名。
        node_coords (list): 节点坐标列表，每个元素为一个包含x, y坐标的列表。
        cell_idx_container (list): 单元索引列表，每个元素为一个包含节点索引的列表。
        boundary_nodes_idx (list): 边界节点索引列表。
        cell_type_container (list): 单元类型列表，每个元素为一个整数。
        cell_part_names (list, optional): 单元部件名称列表，每个元素为一个字符串。
    """

    num_nodes = len(node_coords)
    num_cells = len(cell_idx_container)

    cell_data = []
    num_tri = 0
    num_quad = 0
    num_tetra = 0
    for i, cell in enumerate(cell_idx_container):
        if cell_type_container[i] == VTK_ELEMENT_TYPE.TRI.value:  # 三角形
            cell_data.append(f"3 {cell[0]} {cell[1]} {cell[2]}")
            num_tri += 1
        elif cell_type_container[i] == VTK_ELEMENT_TYPE.QUAD.value:  # 四边形
            cell_data.append(f"4 {cell[0]} {cell[1]} {cell[2]} {cell[3]}")
            num_quad += 1
        elif cell_type_container[i] == VTK_ELEMENT_TYPE.TETRA.value:  # 四面体
            cell_data.append(f"4 {cell[0]} {cell[1]} {cell[2]} {cell[3]}")
            num_tetra += 1

    with open(filename, "w") as file:
        file.write("# vtk DataFile Version 2.0\n")
        file.write("Unstructured Grid\n")
        file.write("ASCII\n")
        file.write("DATASET UNSTRUCTURED_GRID\n")
        file.write(f"POINTS {num_nodes} float\n")
        for coord in node_coords:
            coord_tmp = coord.copy()
            if len(coord) == 2:
                coord_tmp.append(0.0)  # 添加Z坐标
            file.write(" ".join(map(str, coord_tmp)) + "\n")

        # 写入单元信息
        total_cells = num_tri + num_quad + num_tetra
        if total_cells != num_cells:
            raise ValueError("单元数量与指定的不一致!")
        total_data_size = num_tri * 4 + num_quad * 5 + num_tetra * 5  # 3+1, 4+1 和 4+1
        file.write(f"CELLS {total_cells} {total_data_size}\n")
        file.write("\n".join(cell_data) + "\n")

        file.write(f"CELL_TYPES {total_cells}\n")
        file.write("\n".join(map(str, cell_type_container)) + "\n")

        # 写入边界节点标记
        file.write(f"POINT_DATA {num_nodes}\n")
        file.write("SCALARS fixed int 1\n")
        file.write("LOOKUP_TABLE default\n")
        for i in range(num_nodes):
            file.write(f"{1 if i in boundary_nodes_idx else 0}\n")  # 标记边界节点

        file.write(f"CELL_DATA {total_cells}\n")

        # 写入单元ID
        file.write("SCALARS cell_id int 1\n")
        file.write("LOOKUP_TABLE default\n")
        file.write("\n".join(map(str, range(total_cells))) + "\n")

        # 写入部件信息（如果提供）
        if cell_part_names and len(cell_part_names) == total_cells:
            # 创建部件名称到整数的映射
            unique_part_names = list(set(str(name) for name in cell_part_names))  # Ensure all are strings
            unique_part_names.sort()  # Sort to ensure consistent mapping
            part_name_to_id = {name: idx for idx, name in enumerate(unique_part_names)}

            # 写入部件ID
            file.write("SCALARS part_id int 1\n")
            part_ids = [part_name_to_id[name] for name in cell_part_names]
            file.write("\n".join(map(str, part_ids)) + "\n")

    info(f"网格已保存到 {filename}")


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
                coords = list(map(float, lines[i].split()))
                # 如果只有两个坐标，添加z=0.0
                if len(coords) == 2:
                    coords.append(0.0)
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
                try:
                    VTK_ELEMENT_TYPE(cell_type)  # 尝试转换为枚举实例
                except ValueError:
                    raise ValueError(f"不支持的单元类型: {cell_type}")

                cell_type_container.append(cell_type)
        elif line.startswith("POINT_DATA"):
            _, nNodes = line.split()
            nNodes = int(nNodes)
            i += 1  # 跳过SCALARS fixed int 1行
            if lines[i].startswith("SCALARS fixed int"):
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
            elif lines[i].startswith("SCALARS part_id"):
                i += 1  # 跳过LOOKUP_TABLE行
                if lines[i].startswith("LOOKUP_TABLE"):
                    i += 1  # 跳过LOOKUP_TABLE part_name_lut line
                    part_ids = []
                    for _ in range(nCells):
                        i += 1
                        val = int(lines[i])
                        part_ids.append(val)
        i += 1

    # Return part_ids if available, otherwise return None
    result = node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container
    if 'part_ids' in locals():
        result = result + (part_ids,)
    else:
        result = result + (None,)

    return result


def reconstruct_mesh_from_vtk(
    node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container, cell_part_ids=None
):
    from data_structure.basic_elements import NodeElement, Triangle, Quadrilateral, Tetrahedron
    from data_structure.unstructured_grid import Unstructured_Grid

    num_nodes = len(node_coords)
    num_cells = len(cell_idx_container)

    node_container = [NodeElement(node_coords[idx], idx) for idx in range(num_nodes)]
    # 初始化长度为num_cells的列表，每个元素为None
    cell_container = [None] * num_cells
    # 遍历每个单元，根据单元类型创建相应的对象
    for idx in range(num_cells):
        # 检查单元索引是否有效
        if idx >= len(cell_idx_container):
            raise ValueError(f"单元索引 {idx} 超出范围 {len(cell_idx_container)}")

        # 检查单元是否有足够的节点
        if len(cell_idx_container[idx]) < 3:
            raise ValueError(f"单元 {idx} 节点数量不足: {len(cell_idx_container[idx])}")

        # 检查节点索引是否有效，如果无效则跳过该单元
        valid_nodes = True
        for node_idx in cell_idx_container[idx]:
            if node_idx >= len(node_container):
                valid_nodes = False
                break

        if not valid_nodes:
            # 跳过无效单元，不创建网格对象
            continue

        node1 = node_container[cell_idx_container[idx][0]]
        node2 = node_container[cell_idx_container[idx][1]]
        node3 = node_container[cell_idx_container[idx][2]]

        cell_type = VTK_ELEMENT_TYPE(cell_type_container[idx])
        if cell_type == VTK_ELEMENT_TYPE.TRI:  # 三角形
            cell = Triangle(node1, node2, node3, "interior-triangle", idx)
        elif cell_type == VTK_ELEMENT_TYPE.QUAD:  # 四边形
            if len(cell_idx_container[idx]) < 4:
                raise ValueError(f"四边形单元 {idx} 节点数量不足: {len(cell_idx_container[idx])}")
            node4 = node_container[cell_idx_container[idx][3]]
            cell = Quadrilateral(node1, node2, node3, node4, "interior-quadrilateral", idx)
        elif cell_type == VTK_ELEMENT_TYPE.TETRA:  # 四面体
            if len(cell_idx_container[idx]) < 4:
                raise ValueError(f"四面体单元 {idx} 节点数量不足: {len(cell_idx_container[idx])}")
            node4 = node_container[cell_idx_container[idx][3]]
            cell = Tetrahedron(node1, node2, node3, node4, "interior-tetrahedron", idx)
        else:
            raise ValueError(f"Unsupported cell type: {cell_type_container[idx]}")

        # 如果有部件ID信息，设置部件名称
        if cell_part_ids and idx < len(cell_part_ids):
            # 检查cell_part_ids中的值是否是实际的part名称（字符串）还是ID（数字）
            part_id = cell_part_ids[idx]
            if isinstance(part_id, str) and part_id:  # 如果是字符串，直接使用
                cell.part_name = part_id
            else:  # 如果是数字ID，转换为part名称格式
                cell.part_name = f"part_{part_id}"  # Convert to string format
        # 否则保持单元原有的part_name（如果已存在），避免覆盖有意义的part名称
        # 只有在part_name为None或空的情况下才设置默认值
        elif not hasattr(cell, 'part_name') or cell.part_name is None or cell.part_name == '':
            cell.part_name = "Fluid"  # 默认为Fluid

        cell_container[idx] = cell

    boundary_nodes = [node_container[idx] for idx in boundary_nodes_idx if idx < len(node_container)]
    unstr_grid = Unstructured_Grid(
            cell_container, node_coords, boundary_nodes
    )  # 注意这里的cell_container是修改过的，已经包含了单元对象而不是索引列表
    return unstr_grid


def parse_vtk_msh(filename):
    result = read_vtk(filename)
    if len(result) == 5:
        node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container, cell_part_ids = result
    else:
        node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container, cell_part_ids = result + (None,)

    unstr_grid = reconstruct_mesh_from_vtk(
        node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container, cell_part_ids
    )
    return unstr_grid

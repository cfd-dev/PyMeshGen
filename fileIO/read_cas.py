import re

from utils.timer import TimeSpan

# Fluent网格类型定义
FLUENT_FACE_TYPES = {"MIXED": 0, "LINEAR": 2, "TRI": 3, "QUAD": 4}

# Fluent单元类型定义
FLUENT_CELL_TYPES = {
    "MIXED": 0,
    "TRI": 1,
    "TET": 2,
    "QUAD": 3,
    "HEX": 4,
    "PYRAMID": 5,
    "WEDGE": 6,
}

# Fluent边界条件类型定义
FLUENT_BOUNDARY_TYPES = {
    "INTERIOR": 2,  # 内部面
    "WALL": 3,  # 壁面
    "PRESSURE_INLET": 4,  # 压力入口
    "PRESSURE_OUTLET": 5,  # 压力出口
    "SYMMETRY": 7,  # 对称面
    "PRESSURE_FAR": 9,  # 远场压力
    "VELOCITY_INLET": 10,  # 速度入口
    "PERIODIC": 12,  # 周期性边界
    "MASS_FLOW_INLET": 20,  # 质量流入口
    "INTERFACE": 24,  # 交界面
    "OUTFLOW": 36,  # 出流边界
    "AXIS": 37,  # 轴边界
}

# CELL区域类型
CELL_ZONE_TYPE = {"DEAD": 0, "FLUID": 1}


def parse_fluent_msh(file_path):
    timer = TimeSpan("解析fluent .cas网格...")

    raw_cas_data = {
        "nodes": [],
        "faces": [],
        "cells": [],
        "zones": {},
        "comments": [],
        "output_prompts": [],
        "dimensions": 0,
    }

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    current_section = None
    current_zone = None

    # 正则表达式模式
    hex_pattern = re.compile(r"[0-9a-fA-F]+")
    node_section_pattern = re.compile(r"\(10 \(1")
    face_section_pattern = re.compile(
        r"\(\s*13\s*\(\s*(\d+)\s+([0-9A-Fa-f]+)\s+([0-9A-Fa-f]+)\s+(\d+)\s+(\d+)"
    )
    cell_section_pattern = re.compile(
        r"\(\s*12\s*\(\s*(\d+)\s+([0-9A-Fa-f]+)\s+([0-9A-Fa-f]+)\s+(\d+)\s+(\d+)"
    )
    bc_pattern = re.compile(
        r"^\(\s*45\s+\(\s*(\d+)\s+([\w-]+)\s+([\w-]+)\s*\)\s*\(\s*\)\s*\)$"
    )

    for line in lines:
        # 处理注释和输出提示
        if line.startswith("(0"):
            raw_cas_data["comments"].append(line[2:].strip())
            continue
        elif (
            line.startswith("(1 ")
            and not line.startswith("(10 ")
            and not line.startswith("(12 ")
            and not line.startswith("(13 ")
        ):
            raw_cas_data["output_prompts"].append(line[2:].strip())
            continue

        # 处理维度信息
        if line.startswith("(2 "):
            # 使用正则表达式提取所有数字
            numbers = re.findall(r"\d+", line)
            if len(numbers) >= 2:
                raw_cas_data["dimensions"] = int(numbers[1])
            else:
                raise ValueError(f"Invalid dimension line: {line}")
            continue

        # 处理节点数量
        if line.startswith("(10 (0"):
            raw_cas_data["node_count"] = int(line.split()[3], 16)
            continue

        # 处理面数量
        if line.startswith("(13 (0"):
            raw_cas_data["face_count"] = int(line.split()[3], 16)
            continue

        # 处理单元数量
        if line.startswith("(12 (0"):
            raw_cas_data["cell_count"] = int(line.split()[3], 16)
            continue

        # 处理节点坐标
        if node_section_pattern.match(line):
            current_section = "nodes"
            continue

        # 处理面数据
        face_match = face_section_pattern.match(line)
        if face_match:
            current_section = "faces"
            zone_id = int(face_match.group(1))
            face_start_idx = int(face_match.group(2), 16)
            face_end_idx = int(face_match.group(3), 16)
            face_count_section = face_end_idx - face_start_idx + 1
            bc_type = int(face_match.group(4))
            face_type = int(face_match.group(5))
            current_zone = {
                "type": "faces",
                "zone_id": zone_id,
                "start_idx": face_start_idx,
                "end_idx": face_end_idx,
                "bc_type": bc_type,
                "face_type": face_type,
                "part_name": [],
                "face_count_section": face_count_section,
                "data": [],
            }
            raw_cas_data["zones"][f"zone_{zone_id}"] = current_zone
            continue

        # 处理单元数据
        cell_match = cell_section_pattern.match(line)
        if cell_match:
            current_section = "cells"
            zone_id = int(cell_match.group(1))
            cell_start_idx = int(cell_match.group(2), 16)
            cell_end_idx = int(cell_match.group(3), 16)
            cell_zone_type = int(cell_match.group(4))
            cell_type = int(cell_match.group(5))
            cell_count = cell_end_idx - cell_start_idx + 1
            current_zone = {
                "type": "cells",
                "zone_id": zone_id,
                "start_idx": cell_start_idx,
                "end_idx": cell_end_idx,
                "cell_zone_type": cell_zone_type,
                "cell_type": cell_type,
                "cell_type_array": [],
                "part_name": [],
                "cell_count": cell_count,
            }
            raw_cas_data["zones"][f"zone_{zone_id}"] = current_zone
            continue

        # 处理边界条件
        if line.startswith("(45"):
            match = bc_pattern.match(line)
            if match:
                zone_id = int(match.group(1))
                bc_type = match.group(2)
                part_name = match.group(3).strip() if match.group(3) else None

                # 清理边界名称中的多余字符
                if part_name:
                    part_name = part_name.split(")", 1)[0].strip()

                if f"zone_{zone_id}" in raw_cas_data["zones"]:
                    zone = raw_cas_data["zones"][f"zone_{zone_id}"]
                    zone["bc_type"] = bc_type
                    zone["part_name"] = part_name
                else:
                    print(f"Warning: Zone {zone_id} not found for BC: {line}")
            else:
                print(f"Warning: Unparsed BC line: {line}")
            continue

        # 处理当前section的数据
        if current_section == "nodes":
            if line == "))":
                current_section = None
            else:
                coords = list(map(float, line.split()))
                for i in range(0, len(coords), raw_cas_data["dimensions"]):
                    raw_cas_data["nodes"].append(coords[i : i + raw_cas_data["dimensions"]])
            continue

        if current_section == "faces":
            if line == "))":
                current_section = None
            else:
                # 处理十六进制面数据
                hex_values = hex_pattern.findall(line)
                dec_values = [int(h, 16) for h in hex_values]

                if face_type == FLUENT_FACE_TYPES["MIXED"]:
                    nnodes = dec_values[0]
                    face = {
                        "nnodes": dec_values[0],
                        "nodes": dec_values[1 : 1 + nnodes],
                        "left_cell": dec_values[1 + nnodes],
                        "right_cell": dec_values[2 + nnodes],
                    }
                else:
                    face = {"nnodes": 0, "nodes": [], "left_cell": [], "right_cell": []}
                    if face_type == FLUENT_FACE_TYPES["LINEAR"]:
                        face["nnodes"] = 2
                    elif face_type == FLUENT_FACE_TYPES["TRI"]:
                        face["nnodes"] = 3
                    elif face_type == FLUENT_FACE_TYPES["QUAD"]:
                        face["nnodes"] = 4

                    nnodes = face["nnodes"]
                    face["nodes"] = dec_values[0:nnodes]
                    face["left_cell"] = dec_values[nnodes]
                    face["right_cell"] = dec_values[1 + nnodes]

                current_zone["data"].append(face)
            continue

        if current_section == "cells":
            if line == "))":
                current_section = None
            else:
                dec_values = line.split()
                # 分离单元类型
                for h in dec_values:
                    cell_type = int(h)
                    current_zone["cell_type_array"].append(cell_type)
            continue

    # 收集faces数据到cas_data['faces']
    for zone in raw_cas_data["zones"].values():
        if zone["type"] == "faces":
            bc_type = zone.get("bc_type", "internal")
            part_name = zone.get("part_name", "unspecified")
            for face in zone["data"]:
                face_with_bc = {
                    "nodes": face["nodes"],
                    "left_cell": face["left_cell"],
                    "right_cell": face["right_cell"],
                    "bc_type": bc_type,
                    "part_name": part_name,
                }
                raw_cas_data["faces"].append(face_with_bc)

    timer.show_to_console("解析fluent .cas网格..., Done.")

    return raw_cas_data


def reconstruct_mesh_from_cas(raw_cas_data):
    """
    将cas文件解析后的数据转换为Unstructured_Grid对象

    Args:
        raw_cas_data (dict): parse_fluent_msh函数返回的数据结构

    Returns:
        Unstructured_Grid: 转换后的非结构化网格对象
    """
    from data_structure.basic_elements import (
        Unstructured_Grid,
        NodeElement,
        Triangle,
        Quadrilateral,
    )

    # 提取节点坐标
    node_coords = raw_cas_data["nodes"]
    num_nodes = len(node_coords)

    # 确保所有节点都有3个坐标值(x,y,z)
    # 如果是2D网格，添加z=0.0坐标
    for i in range(num_nodes):
        if len(node_coords[i]) == 2:
            # 2D坐标，添加z=0.0
            node_coords[i] = [node_coords[i][0], node_coords[i][1], 0.0]
        elif len(node_coords[i]) < 2:
            # 异常情况，至少需要x,y坐标
            print(
                f"Warning: Node {i} has only {len(node_coords[i])} coordinates, skipping"
            )
            continue

    # 创建节点对象
    node_container = [NodeElement(node_coords[idx], idx) for idx in range(num_nodes)]

    # 从面数据构建单元
    cell_container = []
    cell_type_container = []

    # 收集所有单元的面连接关系
    cell_faces = {}  # {cell_id: [face_data]}

    for face in raw_cas_data["faces"]:
        left_cell = face["left_cell"]  # 1-based index
        right_cell = face["right_cell"]  # 1-based index

        # 添加到左侧单元的面列表
        if left_cell > 0:  # 1-based index, 0表示没有单元
            if left_cell not in cell_faces:
                cell_faces[left_cell] = []
            cell_faces[left_cell].append(face)

        # 添加到右侧单元的面列表
        if right_cell > 0:  # 1-based index, 0表示没有单元
            if right_cell not in cell_faces:
                cell_faces[right_cell] = []
            cell_faces[right_cell].append(face)

    # 根据面连接关系构建单元
    for cell_id, faces in cell_faces.items():
        # 收集单元的所有节点
        cell_nodes = set()
        for face in faces:
            for node_idx in face["nodes"]:
                cell_nodes.add(node_idx - 1)  # 转换为0基索引

        # 转换为列表
        cell_nodes = list(cell_nodes)

        # 根据节点数量确定单元类型
        if len(cell_nodes) == 3:
            # 三角形单元
            node1 = node_container[cell_nodes[0]]
            node2 = node_container[cell_nodes[1]]
            node3 = node_container[cell_nodes[2]]
            cell = Triangle(
                node1, node2, node3, "interior-triangle", idx=len(cell_container)
            )
            cell_container.append(cell)
            cell_type_container.append(5)  # VTK_TRIANGLE
        elif len(cell_nodes) == 4:
            # 四边形单元
            node1 = node_container[cell_nodes[0]]
            node2 = node_container[cell_nodes[1]]
            node3 = node_container[cell_nodes[2]]
            node4 = node_container[cell_nodes[3]]
            cell = Quadrilateral(
                node1,
                node2,
                node3,
                node4,
                "interior-quadrilateral",
                idx=len(cell_container),
            )
            cell_container.append(cell)
            cell_type_container.append(9)  # VTK_QUAD
        else:
            # 其他类型暂时不支持
            print(f"Warning: Unsupported cell with {len(cell_nodes)} nodes, skipping")
            continue

    # 确定边界节点
    boundary_nodes_idx = set()

    # 遍历所有面，找出边界节点
    for face in raw_cas_data["faces"]:
        # 如果面的左侧或右侧没有单元，则这是边界
        if face["left_cell"] == 0 or face["right_cell"] == 0:
            for node_idx in face["nodes"]:
                boundary_nodes_idx.add(node_idx - 1)  # 转换为0基索引

    # 创建边界节点对象
    boundary_nodes = [node_container[idx] for idx in boundary_nodes_idx]

    # 设置边界节点的边界类型
    for face in raw_cas_data["faces"]:
        if face["left_cell"] == 0 or face["right_cell"] == 0:
            bc_type = face.get("bc_type", "unspecified")
            part_name = face.get("part_name", "unspecified")

            for node_idx in face["nodes"]:
                node_idx_0 = node_idx - 1  # 转换为0基索引
                if node_idx_0 < len(node_container):
                    node_container[node_idx_0].bc_type = bc_type
                    node_container[node_idx_0].part_name = part_name

    # 创建边界信息字典，用于可视化
    boundary_info = {}

    # 按边界类型和区域名称组织边界信息
    for zone_id, zone in raw_cas_data["zones"].items():
        if zone["type"] == "faces":
            bc_type = zone.get("bc_type", "unspecified")
            part_name = zone.get("part_name", f"zone_{zone_id}")

            # 收集该边界的所有面
            boundary_faces = []
            for face in zone["data"]:
                boundary_faces.append(
                    {
                        "nodes": [node - 1 for node in face["nodes"]],  # 转换为0基索引
                        "left_cell": face["left_cell"] - 1,  # 转换为0基索引
                        "right_cell": face["right_cell"] - 1,  # 转换为0基索引
                    }
                )

            # 添加到边界信息，已经转换为0-based索引
            boundary_info[part_name] = {"bc_type": bc_type, "faces": boundary_faces}

    # 创建Unstructured_Grid对象
    unstr_grid = Unstructured_Grid(cell_container, node_coords, boundary_nodes)

    # 设置边界信息
    unstr_grid.boundary_info = boundary_info

    # 保存原始部件信息以便后续使用
    unstr_grid.parts_info = boundary_info

    return unstr_grid


def parse_cas_to_unstr_grid(cas_file_path):
    """
    将cas文件直接转换为Unstructured_Grid对象

    Args:
        cas_file_path (str): cas文件路径

    Returns:
        Unstructured_Grid: 转换后的非结构化网格对象
    """
    # 首先解析cas文件
    raw_cas_data = parse_fluent_msh(cas_file_path)

    # 然后将解析后的数据转换为Unstructured_Grid对象
    unstr_grid = reconstruct_mesh_from_cas(raw_cas_data)

    return unstr_grid

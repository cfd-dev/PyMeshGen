import re
from timer import TimeSpan

# Fluent网格类型定义
FACE_TYPES = {"MIXED": 0, "LINEAR": 2, "TRI": 3, "QUAD": 4}

CELL_TYPES = {
    "MIXED": 0,
    "TRI": 1,
    "TET": 2,
    "QUAD": 3,
    "HEX": 4,
    "PYRAMID": 5,
    "WEDGE": 6,
}

BOUNDARY_TYPES = {
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

    data = {
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
            data["comments"].append(line[2:].strip())
            continue
        elif (
            line.startswith("(1 ")
            and not line.startswith("(10 ")
            and not line.startswith("(12 ")
            and not line.startswith("(13 ")
        ):
            data["output_prompts"].append(line[2:].strip())
            continue

        # 处理维度信息
        if line.startswith("(2 "):
            # 使用正则表达式提取所有数字
            numbers = re.findall(r"\d+", line)
            if len(numbers) >= 2:
                data["dimensions"] = int(numbers[1])
            else:
                raise ValueError(f"Invalid dimension line: {line}")
            continue

        # 处理节点数量
        if line.startswith("(10 (0"):
            data["node_count"] = int(line.split()[3], 16)
            continue

        # 处理面数量
        if line.startswith("(13 (0"):
            data["face_count"] = int(line.split()[3], 16)
            continue

        # 处理单元数量
        if line.startswith("(12 (0"):
            data["cell_count"] = int(line.split()[3], 16)
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
            data["zones"][f"zone_{zone_id}"] = current_zone
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
            data["zones"][f"zone_{zone_id}"] = current_zone
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

                if f"zone_{zone_id}" in data["zones"]:
                    zone = data["zones"][f"zone_{zone_id}"]
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
                for i in range(0, len(coords), data["dimensions"]):
                    data["nodes"].append(coords[i : i + data["dimensions"]])
            continue

        if current_section == "faces":
            if line == "))":
                current_section = None
            else:
                # 处理十六进制面数据
                hex_values = hex_pattern.findall(line)
                dec_values = [int(h, 16) for h in hex_values]

                if face_type == FACE_TYPES["MIXED"]:
                    nnodes = dec_values[0]
                    face = {
                        "nnodes": dec_values[0],
                        "nodes": dec_values[1 : 1 + nnodes],
                        "left_cell": dec_values[1 + nnodes],
                        "right_cell": dec_values[2 + nnodes],
                    }
                else:
                    face = {"nnodes": 0, "nodes": [], "left_cell": [], "right_cell": []}
                    if face_type == FACE_TYPES["LINEAR"]:
                        face["nnodes"] = 2
                    elif face_type == FACE_TYPES["TRI"]:
                        face["nnodes"] = 3
                    elif face_type == FACE_TYPES["QUAD"]:
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

    # 收集faces数据到data['faces']
    for zone in data["zones"].values():
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
                data["faces"].append(face_with_bc)

    timer.show_to_console("解析fluent .cas网格..., Done.")
    return data

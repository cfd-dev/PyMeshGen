"""
Tecplot PLT 文件输入输出模块

功能：
- 从 PyMeshGen 网格数据导出为 Tecplot PLT 格式
- 支持 2D (FEPolygon) 和 3D (FEPolyhedron) 非结构网格
- 支持标量场数据导出
- 3D 网格自动输出边界区域 Zone

Tecplot 格式说明：
- 2D 网格使用 FEPolygon 格式
- 3D 网格使用 FEPolyhedron 格式，边界区域使用 FEQUADRILATERAL 格式
- 数据采用 BLOCK 打包方式
- 索引均为 1-based

用法：
    from fileIO.tecplot_io import export_mesh_to_plt

    # 从网格字典导出
    export_mesh_to_plt(grid, output_path="output.plt")

    # 从节点和面数据直接导出
    export_mesh_to_plt(
        nodes=nodes,
        simplices=simplices,
        edge_index=edge_index,
        scalars={"P": pressure},
        output_path="output.plt"
    )
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def export_mesh_to_plt(
    grid: Optional[Dict] = None,
    nodes: Optional[np.ndarray] = None,
    faces: Optional[List[Dict]] = None,
    simplices: Optional[np.ndarray] = None,
    edge_index: Optional[np.ndarray] = None,
    scalars: Optional[Dict[str, np.ndarray]] = None,
    output_path: str = "output.plt",
    title: str = "Mesh Data",
) -> str:
    """
    将非结构网格导出为 Tecplot PLT 文件

    Args:
        grid: PyMeshGen 网格数据字典
        nodes: 节点坐标数组 [num_nodes, 2] 或 [num_nodes, 3]
        faces: 面列表，每个面包含 "nodes" 字段（1-based 索引）
        simplices: 单元连接数组，形状 [num_cells, 3] 或 [num_cells, 4]
        edge_index: 边索引数组，形状 [2, num_edges]
        scalars: 标量字段字典，如 {"P": pressure}
        output_path: 输出 PLT 文件路径
        title: 文件标题（会自动添加 " exported from PyMeshGen" 后缀）

    Returns:
        输出文件路径

    Raises:
        ValueError: 输入参数无效时抛出
    """
    # 1. 提取网格数据
    if grid is not None:
        nodes, faces, simplices, edge_index = _extract_from_grid(grid)
        if not isinstance(nodes, np.ndarray):
            nodes = np.array(nodes)
    elif nodes is None or (faces is None and edge_index is None):
        raise ValueError("必须提供 grid 参数，或同时提供 nodes 和 faces/edge_index")

    if nodes is None or len(nodes) == 0:
        raise ValueError("节点数据为空")

    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes)

    # 2. 准备标量数据
    if scalars is None:
        scalars = {}

    # 3. 格式化文件标题
    if not title:
        title = "Mesh"
    if not title.endswith(" exported from PyMeshGen"):
        title = f"{title} exported from PyMeshGen"

    # 4. 3D 时初始化文件并写入边界区域
    is_3d = nodes.shape[1] == 3
    if is_3d:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f'TITLE = "{title}"\n')

        # 3D 需要先输出边界区域，再输出主区域
        if grid is not None and grid.get("zones"):
            _write_boundary_zones_first(
                nodes=nodes,
                grid=grid,
                output_path=output_path,
            )
            _append_main_zone(
                nodes=nodes,
                faces=faces,
                simplices=simplices,
                edge_index=edge_index,
                scalars=scalars,
                output_path=output_path,
                title=title,
            )
        else:
            _write_plt_file(
                nodes=nodes,
                faces=faces,
                simplices=simplices,
                edge_index=edge_index,
                scalars=scalars,
                output_path=output_path,
                title=title,
            )
    else:
        # 2D 直接写入主区域
        _write_plt_file(
            nodes=nodes,
            faces=faces,
            simplices=simplices,
            edge_index=edge_index,
            scalars=scalars,
            output_path=output_path,
            title=title,
        )

    print(f"PLT 文件已保存: {output_path}")
    return output_path


def export_from_cas(cas_file: str, output_path: str,
                    scalars: Optional[Dict[str, np.ndarray]] = None) -> str:
    """
    从 Fluent .cas 文件导出为 PLT 文件

    Args:
        cas_file: Fluent .cas 文件路径
        output_path: 输出 PLT 文件路径
        scalars: 标量字段字典（可选）

    Returns:
        输出文件路径
    """
    from fileIO.read_cas import parse_fluent_msh
    from data_structure.mesh_reconstruction import preprocess_grid

    grid = parse_fluent_msh(cas_file)
    preprocess_grid(grid)

    title = Path(cas_file).stem
    return export_mesh_to_plt(
        grid=grid,
        output_path=output_path,
        title=title,
        scalars=scalars,
    )


def _extract_from_grid(grid: Dict):
    """
    从 PyMeshGen 网格字典中提取节点、面、单元信息

    支持三种网格格式：
    - 包含 cells 的网格
    - Fluent 面基网格 (faces 格式)
    - zones 格式的旧版网格

    Returns:
        tuple: (node_array, all_faces, simplices, edge_index)
    """
    # 提取节点坐标
    if grid["nodes"] and isinstance(grid["nodes"][0], dict):
        node_array = np.array([node["coords"] for node in grid["nodes"]])
    else:
        node_array = np.array(grid["nodes"])

    if node_array.ndim == 1:
        dim = grid.get("dimension", 2)
        node_array = node_array.reshape(-1, dim)

    # 收集所有面
    all_faces = []
    for zone in grid.get("zones", {}).values():
        if zone.get("type") == "faces":
            all_faces.extend(zone.get("data", []))

    # 提取单元 (simplices)
    simplices_list = []

    if "cells" in grid and len(grid["cells"]) > 0:
        for cell in grid["cells"]:
            if "nodes" in cell:
                cell_nodes = cell["nodes"]
                cell_nodes_0based = [n - 1 if n > 0 else -n - 1 for n in cell_nodes]
                if len(cell_nodes_0based) == 3:
                    simplices_list.append(cell_nodes_0based)
                elif len(cell_nodes_0based) == 4:
                    simplices_list.append([cell_nodes_0based[0], cell_nodes_0based[1], cell_nodes_0based[2]])
                    simplices_list.append([cell_nodes_0based[0], cell_nodes_0based[2], cell_nodes_0based[3]])

    elif "faces" in grid and len(grid.get("faces", [])) > 0:
        cell_faces_map = {}
        for face in grid["faces"]:
            left_cell = face.get("left_cell", 0)
            right_cell = face.get("right_cell", 0)
            face_nodes = face.get("nodes", [])

            if left_cell > 0:
                cell_faces_map.setdefault(left_cell, []).extend(face_nodes)
            if right_cell > 0:
                cell_faces_map.setdefault(right_cell, []).extend(face_nodes)

        for nodes_list in cell_faces_map.values():
            unique_nodes = sorted(set(nodes_list))
            if len(unique_nodes) == 3:
                simplices_list.append([n - 1 for n in unique_nodes])
            elif len(unique_nodes) == 4:
                simplices_list.append([unique_nodes[0] - 1, unique_nodes[1] - 1, unique_nodes[2] - 1])
                simplices_list.append([unique_nodes[0] - 1, unique_nodes[2] - 1, unique_nodes[3] - 1])
    else:
        for face in all_faces:
            face_nodes = face.get("nodes", [])
            if len(face_nodes) == 3:
                simplices_list.append([n - 1 for n in face_nodes])
            elif len(face_nodes) == 4:
                simplices_list.append([face_nodes[0] - 1, face_nodes[1] - 1, face_nodes[2] - 1])
                simplices_list.append([face_nodes[0] - 1, face_nodes[2] - 1, face_nodes[3] - 1])

    # 构建边索引
    edge_set = set()
    if simplices_list:
        for cell_nodes in simplices_list:
            for i in range(len(cell_nodes)):
                n1, n2 = cell_nodes[i], cell_nodes[(i + 1) % len(cell_nodes)]
                edge_set.add((min(n1, n2), max(n1, n2)))
    else:
        for face in all_faces:
            face_nodes = face.get("nodes", [])
            if len(face_nodes) == 2:
                n1, n2 = face_nodes[0] - 1, face_nodes[1] - 1
                edge_set.add((min(n1, n2), max(n1, n2)))
            elif len(face_nodes) >= 3:
                for i in range(len(face_nodes)):
                    n1, n2 = face_nodes[i] - 1, face_nodes[(i + 1) % len(face_nodes)] - 1
                    edge_set.add((min(n1, n2), max(n1, n2)))

    edge_index = np.array(list(edge_set)).T if edge_set else None
    simplices = np.array(simplices_list) if simplices_list else None

    return node_array, all_faces, simplices, edge_index


def _write_plt_file(
    nodes: np.ndarray,
    faces: Optional[List[Dict]],
    simplices: Optional[np.ndarray],
    edge_index: Optional[np.ndarray],
    scalars: Dict[str, np.ndarray],
    output_path: str,
    title: str,
    append_mode: bool = False,
):
    """
    写入 Tecplot PLT 文件主区域

    Args:
        nodes: 节点坐标数组 [num_nodes, 2] 或 [num_nodes, 3]
        faces: 面列表 (可选)
        simplices: 单元连接数组
        edge_index: 边索引数组 [2, num_edges]
        scalars: 标量字段字典
        output_path: 输出文件路径
        title: 文件标题
        append_mode: 是否追加模式（3D 主区域时使用）
    """
    num_nodes = len(nodes)
    is_3d = nodes.shape[1] == 3

    cell_type = "FEPolyhedron" if is_3d else "FEPolygon"
    num_cells = len(simplices) if simplices is not None else 0
    num_faces = edge_index.shape[1] if edge_index is not None else (len(faces) if faces else 0)

    var_names = ["X", "Y", "Z"] if is_3d else ["X", "Y"]
    var_names.extend(scalars.keys())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    words_per_line = 5

    # 预先构建拓扑关系
    face_to_nodes = []
    left_cell = np.zeros(num_faces, dtype=int)
    right_cell = np.zeros(num_faces, dtype=int)

    if num_faces > 0:
        if edge_index is not None:
            for i in range(edge_index.shape[1]):
                n1, n2 = edge_index[0, i] + 1, edge_index[1, i] + 1
                face_to_nodes.append([n1, n2])
        elif faces is not None:
            for face in faces:
                face_nodes = [n + 1 for n in face.get("nodes", [])]
                face_to_nodes.append(face_nodes)

        total_num_face_nodes = sum(len(fn) for fn in face_to_nodes)

        # 构建 LeftCell/RightCell
        if num_cells > 0 and simplices is not None:
            edge_to_face = {}
            for face_idx in range(num_faces):
                if edge_index is not None:
                    e1, e2 = edge_index[0, face_idx], edge_index[1, face_idx]
                    edge_key = (min(e1, e2), max(e1, e2))
                elif face_idx < len(face_to_nodes) and len(face_to_nodes[face_idx]) == 2:
                    e1, e2 = face_to_nodes[face_idx][0] - 1, face_to_nodes[face_idx][1] - 1
                    edge_key = (min(e1, e2), max(e1, e2))
                else:
                    continue
                edge_to_face[edge_key] = face_idx

            for cell_idx, simplex in enumerate(simplices):
                for i in range(len(simplex)):
                    p1, p2 = simplex[i], simplex[(i + 1) % len(simplex)]
                    edge_key = (min(p1, p2), max(p1, p2))
                    face_idx = edge_to_face.get(edge_key)
                    if face_idx is not None:
                        if left_cell[face_idx] == 0:
                            left_cell[face_idx] = cell_idx + 1
                        else:
                            right_cell[face_idx] = cell_idx + 1
    else:
        total_num_face_nodes = 0

    # 构建单元到面的映射
    cell_to_faces = {}
    if num_faces > 0 and num_cells > 0:
        for face_idx in range(num_faces):
            lc, rc = left_cell[face_idx], right_cell[face_idx]
            if lc > 0:
                cell_to_faces.setdefault(lc, []).append(face_idx + 1)
            if rc > 0:
                cell_to_faces.setdefault(rc, []).append(face_idx + 1)

    # 写入文件
    file_mode = "a" if append_mode else "w"
    with open(output_path, file_mode) as f:
        if not append_mode:
            f.write(f'TITLE = "{title}"\n')

        var_names_quoted = ', '.join(f'"{v}"' for v in var_names)
        f.write(f"VARIABLES = {var_names_quoted}\n")
        f.write("ZONE\n")
        f.write(f"ZoneType = {cell_type}\n")
        f.write(f"Nodes    = {num_nodes}\n")
        f.write(f"Faces    = {num_faces}\n")
        f.write(f"Elements = {num_cells}\n")
        f.write(f"Datapacking = BLOCK\n")

        if num_faces > 0:
            f.write(f"TotalNumFaceNodes = {total_num_face_nodes}\n")
            f.write("NumConnectedBoundaryFaces = 0\n")
            f.write("TotalNumBoundaryConnections = 0\n")

        # 节点坐标数据 (BLOCK 格式)
        for dim in range(nodes.shape[1]):
            for i in range(num_nodes):
                f.write(f"{nodes[i, dim]:.10f} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_nodes % words_per_line != 0:
                f.write("\n")

        # 标量场数据
        for var_data in scalars.values():
            for i in range(num_nodes):
                f.write(f"{var_data[i]:.10f} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_nodes % words_per_line != 0:
                f.write("\n")

        # 拓扑数据
        if num_faces > 0 and num_cells > 0:
            # FaceNodeNumber (3D 专用)
            if is_3d:
                for iFace, fn in enumerate(face_to_nodes):
                    f.write(f"{len(fn)} ")
                    if (iFace + 1) % words_per_line == 0:
                        f.write("\n")
                if len(face_to_nodes) % words_per_line == 0:
                    f.write("\n")

            # FaceNodesLink
            node_count = 0
            for fn in face_to_nodes:
                for nid in fn:
                    f.write(f"{nid} ")
                    node_count += 1
                    if node_count % words_per_line == 0:
                        f.write("\n")
            if node_count % words_per_line != 0:
                f.write("\n")

            # LeftCell
            for i in range(num_faces):
                f.write(f"{left_cell[i]} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_faces % words_per_line != 0:
                f.write("\n")

            # RightCell
            for i in range(num_faces):
                f.write(f"{right_cell[i]} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_faces % words_per_line != 0:
                f.write("\n")

            # FaceElementLink
            for cell_idx in range(1, num_cells + 1):
                faces_in_cell = cell_to_faces.get(cell_idx, [])
                for face_id in faces_in_cell:
                    f.write(f"{face_id} ")
                f.write("\n")


def _write_boundary_zones_first(nodes: np.ndarray, grid: Dict, output_path: str):
    """
    3D 时先写入边界区域作为独立的 PLT Zone

    每个边界类型（wall, inlet, outlet 等）作为独立的 FEQUADRILATERAL Zone。
    三角形面通过重复第 4 个节点转换为四边形。

    Args:
        nodes: 节点坐标数组 [num_nodes, 3]
        grid: PyMeshGen 网格字典，包含 zones 信息
        output_path: 输出文件路径（追加模式）
    """
    words_per_line = 5

    # 收集所有边界区域
    boundary_zones = []
    for zone_name, zone_data in grid.get("zones", {}).items():
        bc_type = zone_data.get("bc_type", "internal")
        if bc_type and bc_type.lower() not in ("internal", "interior"):
            boundary_zones.append((zone_name, zone_data))

    if not boundary_zones:
        return

    with open(output_path, "a") as f:
        for zone_name, zone_data in boundary_zones:
            part_name = zone_data.get("part_name", zone_name)
            zone_title = part_name if part_name else zone_name

            faces_data = zone_data.get("data", [])
            if not faces_data:
                continue

            # 收集边界区域节点
            boundary_node_set = set()
            for face in faces_data:
                for node_idx in face.get("nodes", []):
                    boundary_node_set.add(node_idx - 1)

            boundary_nodes_list = sorted(list(boundary_node_set))
            node_index_map = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(boundary_nodes_list)}

            num_boundary_nodes = len(boundary_nodes_list)
            num_boundary_faces = len(faces_data)

            if num_boundary_nodes == 0 or num_boundary_faces == 0:
                continue

            # 写入边界区域 Zone 头
            f.write('VARIABLES = "X", "Y", "Z"\n')
            f.write(f'ZONE T= "{zone_title}"\n')
            f.write(f"ZoneType = FEQUADRILATERAL\n")
            f.write(f"Nodes    = {num_boundary_nodes}\n")
            f.write(f"Elements = {num_boundary_faces}\n")
            f.write(f"Datapacking = BLOCK\n")

            # 边界节点坐标
            for dim in range(3):
                for node_idx in boundary_nodes_list:
                    f.write(f"{nodes[node_idx, dim]:.10f} ")
                    if (node_index_map[node_idx]) % words_per_line == 0:
                        f.write("\n")
                if num_boundary_nodes % words_per_line != 0:
                    f.write("\n")

            # 单元连接（FEQUADRILATERAL）
            for face in faces_data:
                face_nodes = face.get("nodes", [])
                if len(face_nodes) >= 3:
                    n1 = node_index_map.get(face_nodes[0] - 1, 0)
                    n2 = node_index_map.get(face_nodes[1] - 1, 0)
                    n3 = node_index_map.get(face_nodes[2] - 1, 0)
                    if len(face_nodes) >= 4:
                        n4 = node_index_map.get(face_nodes[3] - 1, 0)
                        f.write(f"{n1} {n2} {n3} {n4}\n")
                    else:
                        f.write(f"{n1} {n2} {n3} {n3}\n")


def _append_main_zone(
    nodes: np.ndarray,
    faces,
    simplices,
    edge_index,
    scalars: Dict[str, np.ndarray],
    output_path: str,
    title: str,
):
    """
    追加主网格区域到 PLT 文件（用于 3D）

    3D 时边界区域先写入，然后调用此函数追加主区域。
    """
    _write_plt_file(
        nodes=nodes,
        faces=faces,
        simplices=simplices,
        edge_index=edge_index,
        scalars=scalars,
        output_path=output_path,
        title=title,
        append_mode=True,
    )

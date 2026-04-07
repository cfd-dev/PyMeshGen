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

    # 4. 判断维度（2D 或 3D）
    # 优先使用 grid 中的 dimension 字段，其次使用节点坐标的维度
    if grid is not None and "dimension" in grid:
        dimension = grid["dimension"]
    else:
        dimension = nodes.shape[1] if nodes.ndim == 2 else 2
    
    is_3d = dimension == 3 or (nodes.ndim == 2 and nodes.shape[1] == 3)
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


def export_unstructured_grid_to_plt(
    unstructured_grid,
    output_path: str,
    title: str = "Mesh Data",
    scalars: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """
    从 Unstructured_Grid 对象导出为 Tecplot PLT 文件

    Args:
        unstructured_grid: Unstructured_Grid 对象
        output_path: 输出 PLT 文件路径
        title: 文件标题
        scalars: 标量字段字典（可选）

    Returns:
        输出文件路径
    """
    # 提取节点坐标
    nodes = np.array(unstructured_grid.node_coords)
    # 判断维度：优先使用 unstructured_grid 的 dimension 属性
    # 因为即使原始网格是2D，节点坐标也可能被存储为3维（第三维为0）
    if hasattr(unstructured_grid, 'dimension'):
        dimension = unstructured_grid.dimension
    else:
        dimension = nodes.shape[1] if nodes.ndim == 2 else 2
    
    is_3d = dimension == 3
    
    # 提取单元连接
    simplices_list = []
    for cell in unstructured_grid.cell_container:
        if cell is None:
            continue
        if hasattr(cell, 'node_ids'):
            node_ids = cell.node_ids
            if node_ids and hasattr(node_ids[0], 'idx'):
                simplices_list.append([node.idx for node in node_ids])
            else:
                simplices_list.append(list(node_ids))
    
    # 按节点数分组处理混合单元
    from collections import defaultdict
    cells_by_nodes = defaultdict(list)
    for cell_nodes in simplices_list:
        cells_by_nodes[len(cell_nodes)].append(cell_nodes)
    
    # 选择主要单元类型
    main_type = max(cells_by_nodes.keys(), key=lambda k: len(cells_by_nodes[k]))
    simplices = np.array(cells_by_nodes[main_type])
    
    # 构建边索引
    edge_set = set()
    node_count = main_type
    for cell_nodes in cells_by_nodes[main_type]:
        if node_count == 3 or node_count == 4:
            for i in range(node_count):
                n1, n2 = cell_nodes[i], cell_nodes[(i + 1) % node_count]
                edge_set.add((min(n1, n2), max(n1, n2)))
        elif node_count >= 4:
            for i in range(min(4, node_count)):
                n1, n2 = cell_nodes[i], cell_nodes[(i + 1) % min(4, node_count)]
                edge_set.add((min(n1, n2), max(n1, n2)))
    
    edge_index = np.array(list(edge_set)).T if edge_set else None
    
    # 提取边界区域信息
    boundary_zones = _extract_boundary_zones_from_grid(unstructured_grid)

    # 3D 网格需要特殊处理边界
    # 2D 网格即使有边界区域，也不需要先写入边界区域（Tecplot 2D 格式不支持独立边界 Zone）
    if is_3d and boundary_zones:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if not title.endswith(" exported from PyMeshGen"):
            title = f"{title} exported from PyMeshGen" if title else "Mesh exported from PyMeshGen"

        # 写入 TITLE
        with open(output_path, "w") as f:
            f.write(f'TITLE = "{title}"\n')

        # 先写入边界区域
        _write_boundary_zones_from_dict(
            nodes=nodes,
            boundary_zones=boundary_zones,
            output_path=output_path,
        )

        # 再追加主区域
        _append_main_zone(
            nodes=nodes,
            faces=None,
            simplices=simplices,
            edge_index=edge_index,
            scalars=scalars or {},
            output_path=output_path,
            title=title,
            dimension=dimension,  # 传递维度参数
        )
    else:
        # 2D 或无边界区域，直接写入
        if scalars is None:
            scalars = {}

        _write_plt_file(
            nodes=nodes,
            faces=None,
            simplices=simplices,
            edge_index=edge_index,
            scalars=scalars,
            output_path=output_path,
            title=title,
            dimension=dimension,  # 传递维度参数
        )
    
    print(f"PLT 文件已保存: {output_path}")
    return output_path


def _extract_boundary_zones_from_grid(unstructured_grid) -> List[Dict]:
    """
    从 Unstructured_Grid 对象中提取边界区域信息

    优先级：boundary_info > parts_info（仅提取边界类型的区域）

    Args:
        unstructured_grid: Unstructured_Grid 对象

    Returns:
        边界区域列表，每个区域包含 part_name, bc_type, faces 信息
    """
    boundary_zones = []
    seen_part_names = set()
    
    # 优先从 boundary_info 中提取（包含正确的 faces 数据）
    if hasattr(unstructured_grid, 'boundary_info') and unstructured_grid.boundary_info:
        for zone_name, zone_data in unstructured_grid.boundary_info.items():
            if 'faces' in zone_data and zone_data.get('faces'):
                part_name = zone_data.get('part_name', zone_name)
                bc_type = zone_data.get('bc_type', 'boundary')
                
                # 跳过内部边界
                if bc_type and bc_type.lower() in ("internal", "interior"):
                    continue
                
                boundary_zones.append({
                    'part_name': part_name,
                    'bc_type': bc_type,
                    'data': zone_data['faces']
                })
                seen_part_names.add(part_name)
    
    # 如果 boundary_info 为空，尝试从 parts_info 中提取
    if not boundary_zones and hasattr(unstructured_grid, 'parts_info') and unstructured_grid.parts_info:
        for part_name, part_data in unstructured_grid.parts_info.items():
            # 跳过已添加的部件
            if part_name in seen_part_names:
                continue
            
            bc_type = part_data.get('bc_type', '')
            # 只提取边界类型的部件
            if bc_type and bc_type.lower() not in ("internal", "interior", ""):
                if 'faces' in part_data and part_data['faces']:
                    boundary_zones.append({
                        'part_name': part_data.get('part_name', part_name),
                        'bc_type': bc_type,
                        'data': part_data['faces']
                    })
                    seen_part_names.add(part_name)
    
    return boundary_zones


def _write_boundary_zones_from_dict(nodes: np.ndarray, boundary_zones: List[Dict], output_path: str):
    """
    从边界区域字典写入 PLT 文件

    Args:
        nodes: 节点坐标数组
        boundary_zones: 边界区域列表
        output_path: 输出文件路径（追加模式）
    """
    words_per_line = 5
    
    if not boundary_zones:
        return
    
    with open(output_path, "a") as f:
        for zone_data in boundary_zones:
            part_name = zone_data.get('part_name', 'Unknown')
            zone_title = part_name
            
            faces_data = zone_data.get('data', [])
            if not faces_data:
                continue
            
            # 收集边界区域节点
            boundary_node_set = set()
            for face in faces_data:
                # face 可能是 dict 格式或 list 格式
                if isinstance(face, dict):
                    node_indices = face.get('nodes', [])
                else:
                    node_indices = face
                
                for node_idx in node_indices:
                    node_idx = int(node_idx)
                    # 注意：boundary_info中的faces数据已经是0-based的（来自_extract_faces_from_element_indices）
                    # 直接使用，不需要转换
                    boundary_node_set.add(node_idx)
            
            boundary_nodes_list = sorted(list(boundary_node_set))
            if len(boundary_nodes_list) == 0:
                continue
            
            node_index_map = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(boundary_nodes_list)}

            num_boundary_nodes = len(boundary_nodes_list)

            # 判断是2D还是3D边界
            is_3d = nodes.ndim == 2 and nodes.shape[1] >= 3
            
            # 第一步：预处理所有面，收集有效的面和确定ZoneType
            face_node_counts = []
            valid_faces = []
            
            for face in faces_data:
                # face 可能是 dict 格式或 list 格式
                if isinstance(face, dict):
                    face_nodes = face.get('nodes', [])
                else:
                    face_nodes = face

                if len(face_nodes) == 0:
                    continue

                # 注意：boundary_info中的faces数据已经是0-based的
                # 直接使用，不需要转换
                face_nodes_0based = [int(n) for n in face_nodes]
                
                # 使用 node_index_map 转换为新的 1-based 索引
                face_nodes_new_1based = []
                valid = True
                for node_idx in face_nodes_0based:
                    mapped_idx = node_index_map.get(node_idx)
                    if mapped_idx is None:
                        # 如果找不到节点，跳过这个面
                        valid = False
                        break
                    face_nodes_new_1based.append(mapped_idx)
                
                if not valid or len(face_nodes_new_1based) < 2:
                    continue
                
                face_node_counts.append(len(face_nodes_new_1based))
                valid_faces.append(face_nodes_new_1based)
            
            # 如果没有有效面，跳过这个zone
            if not valid_faces:
                continue

            num_boundary_faces = len(valid_faces)

            # 对于 3D 边界区域，强制使用 FEQUADRILATERAL 格式
            # 三角形面通过重复第 4 个节点转换为四边形（与 _write_boundary_zones_first 一致）
            zone_type = "FEQUADRILATERAL"

            # 写入边界区域 Zone 头
            f.write('VARIABLES = "X", "Y", "Z"\n')
            f.write(f'ZONE T= "{zone_title}"\n')
            f.write(f"ZoneType = {zone_type}\n")
            f.write(f"Nodes    = {num_boundary_nodes}\n")
            f.write(f"Elements = {num_boundary_faces}\n")
            f.write(f"Datapacking = BLOCK\n")
            
            # 边界节点坐标
            # 根据 is_3d 决定写入的维度数
            num_dims_to_write = 3 if is_3d else 2
            for dim in range(num_dims_to_write):
                for node_idx in boundary_nodes_list:
                    coord = nodes[node_idx, dim] if dim < nodes.shape[1] else 0.0
                    f.write(f"{coord:.10f} ")
                    if (node_index_map[node_idx]) % words_per_line == 0:
                        f.write("\n")
                if num_boundary_nodes % words_per_line != 0:
                    f.write("\n")
            
            # 单元连接（使用预处理的有效面）
            # 由于固定使用 FEQUADRILATERAL，所有面都输出 4 个节点
            for face_nodes_new_1based in valid_faces:
                if len(face_nodes_new_1based) >= 4:
                    # 四边形：直接输出 4 个节点
                    f.write(f"{face_nodes_new_1based[0]} {face_nodes_new_1based[1]} {face_nodes_new_1based[2]} {face_nodes_new_1based[3]}\n")
                elif len(face_nodes_new_1based) == 3:
                    # 三角形：重复第 4 个节点转换为四边形
                    f.write(f"{face_nodes_new_1based[0]} {face_nodes_new_1based[1]} {face_nodes_new_1based[2]} {face_nodes_new_1based[2]}\n")
                elif len(face_nodes_new_1based) == 2:
                    # 线段：重复第 3、4 个节点（理论上不应该出现）
                    f.write(f"{face_nodes_new_1based[0]} {face_nodes_new_1based[1]} {face_nodes_new_1based[1]} {face_nodes_new_1based[1]}\n")


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
    
    # 确保节点坐标维度正确
    if node_array.ndim == 2 and node_array.shape[1] < 2:
        raise ValueError(f"节点坐标维度不足，期望至少 2 维，实际 {node_array.shape}")

    # 收集所有面
    all_faces = []
    for zone in grid.get("zones", {}).values():
        if zone.get("type") == "faces":
            all_faces.extend(zone.get("data", []))

    # 提取单元 (simplices)
    simplices_list = []

    # 优先从 cells 中提取
    if "cells" in grid and len(grid["cells"]) > 0:
        for cell in grid["cells"]:
            if "nodes" in cell:
                cell_nodes = cell["nodes"]
                # 转换为 0-based 索引
                cell_nodes_0based = [n - 1 if n > 0 else -n - 1 for n in cell_nodes]
                if len(cell_nodes_0based) == 3:
                    simplices_list.append(cell_nodes_0based)
                elif len(cell_nodes_0based) == 4:
                    simplices_list.append([cell_nodes_0based[0], cell_nodes_0based[1], cell_nodes_0based[2]])
                    simplices_list.append([cell_nodes_0based[0], cell_nodes_0based[2], cell_nodes_0based[3]])

    # 如果 cells 为空，尝试从 faces 构建单元
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
    
    # 如果仍为空，尝试从 zones 中的内部区域提取
    if not simplices_list:
        for zone_name, zone_data in grid.get("zones", {}).items():
            # 跳过边界区域，只处理内部区域
            bc_type = zone_data.get("bc_type", "")
            if bc_type and bc_type.lower() not in ("internal", "interior"):
                continue
            
            # 从 zone 数据中提取单元
            zone_data_list = zone_data.get("data", [])
            for item in zone_data_list:
                if isinstance(item, dict):
                    if "nodes" in item:
                        face_nodes = item["nodes"]
                        if len(face_nodes) == 3:
                            simplices_list.append([n - 1 for n in face_nodes])
                        elif len(face_nodes) == 4:
                            simplices_list.append([face_nodes[0] - 1, face_nodes[1] - 1, face_nodes[2] - 1])
                            simplices_list.append([face_nodes[0] - 1, face_nodes[2] - 1, face_nodes[3] - 1])
                elif isinstance(item, list):
                    # 直接的节点列表
                    if len(item) == 3:
                        simplices_list.append([n - 1 for n in item])
                    elif len(item) == 4:
                        simplices_list.append([item[0] - 1, item[1] - 1, item[2] - 1])
                        simplices_list.append([item[0] - 1, item[2] - 1, item[3] - 1])

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
    dimension: int = None,  # 新增：允许显式指定维度
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
        dimension: 网格维度（2 或 3），如果不指定则从 nodes.shape 推断
    """
    num_nodes = len(nodes)
    # 判断维度：优先使用传入的 dimension 参数，否则从节点坐标的列数推断
    if dimension is not None:
        is_3d = dimension == 3
    else:
        is_3d = nodes.ndim == 2 and nodes.shape[1] == 3

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
        # 根据 is_3d 决定写入的维度数，而不是 nodes.shape[1]
        # 因为 2D 网格可能存储为 3 维坐标（第三维为 0）
        num_dims_to_write = 3 if is_3d else 2
        for dim in range(num_dims_to_write):
            for i in range(num_nodes):
                coord = nodes[i, dim] if dim < nodes.shape[1] else 0.0
                f.write(f"{coord:.10f} ")
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
    dimension: int = None,  # 新增：允许显式指定维度
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
        dimension=dimension,  # 传递维度参数
    )

"""
非结构网格导出为 Tecplot PLT 文件

功能：
1. 从 PyMeshGen 网格数据导出为 Tecplot PLT 格式
2. 支持 2D/3D 网格 (FEPolygon / FEPolyhedron)
3. 支持标量场数据（如压力、温度等）

Tecplot FEPolygon/FEPolyhedron 格式说明：
参考: https://github.com/su2code/SU2/raw/master/externals/tecio/360_data_format_guide.pdf

文件格式结构：
┌─────────────────────────────────────────────────┐
│ 文件头 (Header)                                 │
│ ├─ TITLE = "..."                                │
│ ├─ VARIABLES = "X", "Y", "P", ...               │
│ └─ ZONE                                         │
│    ├─ ZoneType = FEPolygon / FEPolyhedron       │
│    ├─ Nodes, Faces, Elements                     │
│    ├─ Datapacking = BLOCK                        │
│    ├─ TotalNumFaceNodes                          │
│    └─ NumConnectedBoundaryFaces = 0              │
├─────────────────────────────────────────────────┤
│ 数据区 (Data) - BLOCK 格式                      │
│ ├─ 节点坐标: X[nodes], Y[nodes], Z[nodes](3D)   │
│ └─ 标量场数据: P[nodes], T[nodes], ...          │
├─────────────────────────────────────────────────┤
│ 拓扑区 (Topology)                               │
│ ├─ FaceNodeNumber (3D 专用): 每个面的节点数      │
│ ├─ FaceNodesLink: 面-节点连接索引 (1-based)      │
│ ├─ LeftCell: 每个面的左单元索引 (1-based)        │
│ ├─ RightCell: 每个面的右单元索引 (1-based)       │
│ └─ FaceElementLink: 每个单元包含的面列表         │
└─────────────────────────────────────────────────┘

边界外表面识别：
Tecplot 通过 LeftCell/RightCell 中的 0 值自动识别边界外表面：
- LeftCell[i] == 0 或 RightCell[i] == 0 的面即为边界外表面

用法：
    from fileIO.mesh_to_plt import export_mesh_to_plt

    # 方式 1: 从 PyMeshGen 网格字典导出
    export_mesh_to_plt(grid, output_path="output.plt")

    # 方式 2: 从节点和面数据直接导出
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


# ============================================================
# 公共接口
# ============================================================

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

    支持两种输入方式：
    1. PyMeshGen 网格字典（grid 参数）
    2. 直接的节点/面数据（nodes, simplices, edge_index）

    Args:
        grid: PyMeshGen 网格数据字典，包含 nodes, faces, zones 等字段
        nodes: 节点坐标数组，形状 [num_nodes, 2] 或 [num_nodes, 3]
        faces: 面列表，每个面包含 "nodes" 字段（1-based 索引）
        simplices: 单元连接数组，形状 [num_cells, 3] 三角形 或 [num_cells, 4] 四面体
        edge_index: 边索引数组，形状 [2, num_edges]，2D 时需要
        scalars: 标量字段字典，如 {"P": pressure, "T": temperature}
        output_path: 输出 PLT 文件路径
        title: 文件标题

    Returns:
        输出文件路径

    Raises:
        ValueError: 输入参数无效时抛出
    """
    # ── 步骤 1: 提取网格数据 ──────────────────────────────────
    if grid is not None:
        # 从 PyMeshGen 网格字典提取
        nodes, faces, simplices, edge_index = _extract_from_grid(grid)
        if not isinstance(nodes, np.ndarray):
            nodes = np.array(nodes)
    elif nodes is None or (faces is None and edge_index is None):
        raise ValueError("必须提供 grid 参数，或同时提供 nodes 和 faces/edge_index")

    # 验证节点数据
    if nodes is None or len(nodes) == 0:
        raise ValueError("节点数据为空")

    # 确保 nodes 是 numpy 数组
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes)

    # ── 步骤 2: 准备标量数据 ──────────────────────────────────
    if scalars is None:
        scalars = {}

    # ── 步骤 3: 写入 PLT 文件 ──────────────────────────────────
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

    # 解析网格并预处理
    grid = parse_fluent_msh(cas_file)
    preprocess_grid(grid)

    # 导出为 PLT
    title = Path(cas_file).stem
    return export_mesh_to_plt(
        grid=grid,
        output_path=output_path,
        title=title,
        scalars=scalars,
    )


# ============================================================
# 内部函数：网格数据提取
# ============================================================

def _extract_from_grid(grid: Dict):
    """
    从 PyMeshGen 网格字典中提取节点、面、单元信息

    支持多种网格格式：
    - 包含 cells 的网格
    - Fluent 面基网格 (faces 格式)
    - zones 格式的旧版网格

    Args:
        grid: PyMeshGen 网格字典

    Returns:
        tuple: (node_array, all_faces, simplices, edge_index)
            - node_array: 节点坐标数组 [num_nodes, dim]
            - all_faces: 所有面的列表
            - simplices: 单元连接数组
            - edge_index: 边索引数组 [2, num_edges]
    """
    # ── 提取节点坐标 ──────────────────────────────────────────
    # 支持两种格式：字典列表 {"coords": (...)} 或元组列表
    if grid["nodes"] and isinstance(grid["nodes"][0], dict):
        node_array = np.array([node["coords"] for node in grid["nodes"]])
    else:
        node_array = np.array(grid["nodes"])

    # 确保是 2D 数组 [num_nodes, dim]
    if node_array.ndim == 1:
        dim = grid.get("dimension", 2)
        node_array = node_array.reshape(-1, dim)

    # ── 收集所有面 ────────────────────────────────────────────
    all_faces = []
    for zone in grid.get("zones", {}).values():
        if zone.get("type") == "faces":
            all_faces.extend(zone.get("data", []))

    # ── 提取单元 (simplices) ──────────────────────────────────
    simplices_list = []

    # 情况 1: 网格包含 cells
    if "cells" in grid and len(grid["cells"]) > 0:
        for cell in grid["cells"]:
            if "nodes" in cell:
                cell_nodes = cell["nodes"]
                # 转为 0-based 索引
                cell_nodes_0based = [n - 1 if n > 0 else -n - 1 for n in cell_nodes]
                if len(cell_nodes_0based) == 3:
                    simplices_list.append(cell_nodes_0based)  # 三角形
                elif len(cell_nodes_0based) == 4:
                    # 四边形三角剖分
                    simplices_list.append([cell_nodes_0based[0], cell_nodes_0based[1], cell_nodes_0based[2]])
                    simplices_list.append([cell_nodes_0based[0], cell_nodes_0based[2], cell_nodes_0based[3]])

    # 情况 2: Fluent 面基网格
    elif "faces" in grid and len(grid.get("faces", [])) > 0:
        # 通过面重建单元：收集每个单元包含的面
        cell_faces_map = {}  # cell_id -> list of face node indices
        for face in grid["faces"]:
            left_cell = face.get("left_cell", 0)
            right_cell = face.get("right_cell", 0)
            face_nodes = face.get("nodes", [])

            if left_cell > 0:
                cell_faces_map.setdefault(left_cell, []).extend(face_nodes)
            if right_cell > 0:
                cell_faces_map.setdefault(right_cell, []).extend(face_nodes)

        # 从单元的面中提取唯一节点
        for nodes_list in cell_faces_map.values():
            unique_nodes = sorted(set(nodes_list))
            if len(unique_nodes) == 3:
                simplices_list.append([n - 1 for n in unique_nodes])  # 三角形
            elif len(unique_nodes) == 4:
                # 四边形三角剖分
                simplices_list.append([unique_nodes[0] - 1, unique_nodes[1] - 1, unique_nodes[2] - 1])
                simplices_list.append([unique_nodes[0] - 1, unique_nodes[2] - 1, unique_nodes[3] - 1])

    # 情况 3: 从 zones 中的 face 数据提取 (旧格式)
    else:
        for face in all_faces:
            face_nodes = face.get("nodes", [])
            if len(face_nodes) == 3:
                simplices_list.append([n - 1 for n in face_nodes])
            elif len(face_nodes) == 4:
                # 四边形三角剖分
                simplices_list.append([face_nodes[0] - 1, face_nodes[1] - 1, face_nodes[2] - 1])
                simplices_list.append([face_nodes[0] - 1, face_nodes[2] - 1, face_nodes[3] - 1])

    # ── 构建边索引 ────────────────────────────────────────────
    edge_set = set()

    if simplices_list:
        # 从单元中提取边
        for cell_nodes in simplices_list:
            for i in range(len(cell_nodes)):
                n1, n2 = cell_nodes[i], cell_nodes[(i + 1) % len(cell_nodes)]
                edge_set.add((min(n1, n2), max(n1, n2)))
    else:
        # 从面中提取边
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


# ============================================================
# 内部函数：PLT 文件写入
# ============================================================

def _write_plt_file(
    nodes: np.ndarray,
    faces: Optional[List[Dict]],
    simplices: Optional[np.ndarray],
    edge_index: Optional[np.ndarray],
    scalars: Dict[str, np.ndarray],
    output_path: str,
    title: str,
):
    """
    写入 Tecplot PLT 文件

    PLT 文件格式 (Tecplot 360):
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 文件头: TITLE, VARIABLES, ZONE 定义                  │
    │ 2. 节点坐标数据: X[], Y[], Z[](3D)                      │
    │ 3. 标量场数据: P[], T[], ...(BLOCK 格式)                │
    │ 4. 拓扑数据: FaceNodeNumber(3D), FaceNodesLink,         │
    │             LeftCell, RightCell, FaceElementLink        │
    └─────────────────────────────────────────────────────────┘

    Args:
        nodes: 节点坐标数组 [num_nodes, 2] 或 [num_nodes, 3]
        faces: 面列表 (可选)
        simplices: 单元连接数组
        edge_index: 边索引数组 [2, num_edges]
        scalars: 标量字段字典
        output_path: 输出文件路径
        title: 文件标题
    """
    num_nodes = len(nodes)
    is_3d = nodes.shape[1] == 3

    # ── 确定网格类型 ──────────────────────────────────────────
    cell_type = "FEPolyhedron" if is_3d else "FEPolygon"
    num_cells = len(simplices) if simplices is not None else 0
    num_faces = edge_index.shape[1] if edge_index is not None else (len(faces) if faces else 0)

    # ── 构建变量列表 ──────────────────────────────────────────
    var_names = ["X", "Y", "Z"] if is_3d else ["X", "Y"]
    var_names.extend(scalars.keys())

    # ── 确保输出目录存在 ──────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    words_per_line = 5  # Tecplot 标准：每行最多 5 个数据

    # ============================================================
    # 预先构建拓扑关系 (在写入文件前计算)
    # ============================================================
    face_to_nodes = []
    left_cell = np.zeros(num_faces, dtype=int)
    right_cell = np.zeros(num_faces, dtype=int)

    if num_faces > 0:
        # ── 构建 face_to_nodes (1-based 索引) ─────────────────
        if edge_index is not None:
            for i in range(edge_index.shape[1]):
                n1, n2 = edge_index[0, i] + 1, edge_index[1, i] + 1
                face_to_nodes.append([n1, n2])
        elif faces is not None:
            for face in faces:
                face_nodes = [n + 1 for n in face.get("nodes", [])]
                face_to_nodes.append(face_nodes)

        # ── 计算 TotalNumFaceNodes ────────────────────────────
        total_num_face_nodes = sum(len(fn) for fn in face_to_nodes)

        # ── 构建 LeftCell/RightCell ───────────────────────────
        # 通过边-面映射，确定每个面的左右单元
        if num_cells > 0 and simplices is not None:
            # 构建边到面的映射 (O(faces))
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

            # 遍历单元，查找匹配的边 (O(cells × edges_per_cell))
            for cell_idx, simplex in enumerate(simplices):
                for i in range(len(simplex)):
                    p1, p2 = simplex[i], simplex[(i + 1) % len(simplex)]
                    edge_key = (min(p1, p2), max(p1, p2))
                    face_idx = edge_to_face.get(edge_key)
                    if face_idx is not None:
                        if left_cell[face_idx] == 0:
                            left_cell[face_idx] = cell_idx + 1  # 1-based
                        else:
                            right_cell[face_idx] = cell_idx + 1
    else:
        total_num_face_nodes = 0

    # ── 构建单元到面的映射 (用于 FaceElementLink) ─────────────
    cell_to_faces = {}
    if num_faces > 0 and num_cells > 0:
        for face_idx in range(num_faces):
            lc, rc = left_cell[face_idx], right_cell[face_idx]
            if lc > 0:
                cell_to_faces.setdefault(lc, []).append(face_idx + 1)
            if rc > 0:
                cell_to_faces.setdefault(rc, []).append(face_idx + 1)

    # ============================================================
    # 写入文件
    # ============================================================
    with open(output_path, "w") as f:
        # ── 1. 文件头 ─────────────────────────────────────────
        f.write(f'TITLE = "{title}"\n')
        # 变量名需要用双引号包裹 (Tecplot 标准)
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
            # 注意：NumConnectedBoundaryFaces 和 TotalNumBoundaryConnections
            # 必须同时为 0 或同时非零 (Tecplot 格式要求)
            f.write("NumConnectedBoundaryFaces = 0\n")
            f.write("TotalNumBoundaryConnections = 0\n")

        # ── 2. 节点坐标数据 (BLOCK 格式) ──────────────────────
        for dim in range(nodes.shape[1]):
            for i in range(num_nodes):
                f.write(f"{nodes[i, dim]:.10f} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_nodes % words_per_line != 0:
                f.write("\n")

        # ── 3. 标量场数据 ─────────────────────────────────────
        for var_name, var_data in scalars.items():
            for i in range(num_nodes):
                f.write(f"{var_data[i]:.10f} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_nodes % words_per_line != 0:
                f.write("\n")

        # ── 4. 拓扑数据 ───────────────────────────────────────
        if num_faces > 0 and num_cells > 0:
            # 4a. FaceNodeNumber (3D 专用): 每个面的节点数量
            if is_3d:
                for iFace, fn in enumerate(face_to_nodes):
                    f.write(f"{len(fn)} ")
                    if (iFace + 1) % words_per_line == 0:
                        f.write("\n")
                if len(face_to_nodes) % words_per_line == 0:
                    f.write("\n")

            # 4b. FaceNodesLink: 面-节点连接列表 (1-based 索引)
            node_count = 0
            for fn in face_to_nodes:
                for nid in fn:
                    f.write(f"{nid} ")
                    node_count += 1
                    if node_count % words_per_line == 0:
                        f.write("\n")
            if node_count % words_per_line != 0:
                f.write("\n")

            # 4c. LeftCell: 每个面的左单元索引 (1-based, 0 表示边界)
            for i in range(num_faces):
                f.write(f"{left_cell[i]} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_faces % words_per_line != 0:
                f.write("\n")

            # 4d. RightCell: 每个面的右单元索引 (1-based, 0 表示边界)
            for i in range(num_faces):
                f.write(f"{right_cell[i]} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_faces % words_per_line != 0:
                f.write("\n")

            # 4e. FaceElementLink: 每个单元包含的面列表
            for cell_idx in range(1, num_cells + 1):
                faces_in_cell = cell_to_faces.get(cell_idx, [])
                for face_id in faces_in_cell:
                    f.write(f"{face_id} ")
                f.write("\n")

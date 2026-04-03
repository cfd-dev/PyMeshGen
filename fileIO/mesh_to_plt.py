"""
非结构网格导出为 Tecplot PLT 文件

功能：
1. 从 PyMeshGen 网格数据导出为 Tecplot PLT 格式
2. 支持 2D/3D 网格
3. 支持标量场数据（如压力、温度等）

用法：
    from fileIO.mesh_to_plt import export_mesh_to_plt
    
    # 从网格数据导出
    export_mesh_to_plt(grid, output_path="output.plt")
    
    # 或从节点和面数据导出
    export_mesh_to_plt(
        nodes=nodes,
        faces=faces,
        scalars={"P": pressure},
        output_path="output.plt"
    )
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


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
    2. 直接的节点/面数据（nodes, faces, simplices, edge_index）

    Args:
        grid: PyMeshGen 网格数据字典（包含 nodes, faces, zones）
        nodes: 节点坐标，形状 [num_nodes, 2] 或 [num_nodes, 3]
        faces: 面列表，每个面包含 "nodes" 字段（1-based 索引）
        simplices: 三角形/四面体单元连接，形状 [num_cells, 3] 或 [num_cells, 4]
        edge_index: 边索引，形状 [2, num_edges]（2D 时需要）
        scalars: 标量字段字典，如 {"P": pressure, "T": temperature}
        output_path: 输出文件路径
        title: 文件标题

    Returns:
        输出文件路径
    """
    # 提取网格数据
    if grid is not None:
        nodes, faces, simplices, edge_index = _extract_from_grid(grid)
    elif nodes is None or (faces is None and edge_index is None):
        raise ValueError("必须提供 grid 参数，或同时提供 nodes 和 faces/edge_index")

    if nodes is None or len(nodes) == 0:
        raise ValueError("节点数据为空")

    # 确保 nodes 是 numpy 数组
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes)

    # 准备标量数据
    if scalars is None:
        scalars = {}

    # 写入 PLT 文件
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


def _extract_from_grid(grid: Dict):
    """
    从 PyMeshGen 网格数据中提取节点、面、单元信息

    Args:
        grid: PyMeshGen 网格字典

    Returns:
        nodes, faces, simplices, edge_index
    """
    # 提取节点
    nodes = np.array([node["coords"] for node in grid["nodes"]])

    # 收集所有面
    all_faces = []
    for zone in grid.get("zones", {}).values():
        if zone.get("type") == "faces":
            all_faces.extend(zone.get("data", []))

    # 构建边索引（2D）
    edge_set = set()
    simplices_list = []

    for face in all_faces:
        face_nodes = face.get("nodes", [])
        if len(face_nodes) == 2:
            # 线段面
            n1, n2 = face_nodes[0] - 1, face_nodes[1] - 1  # 转为 0-based
            edge_set.add((min(n1, n2), max(n1, n2)))
        elif len(face_nodes) >= 3:
            # 多边形面，三角剖分
            # 简单扇形三角剖分
            for i in range(1, len(face_nodes) - 1):
                simplices_list.append([
                    face_nodes[0] - 1,
                    face_nodes[i] - 1,
                    face_nodes[i + 1] - 1,
                ])

    edge_index = np.array(list(edge_set)).T if edge_set else None
    simplices = np.array(simplices_list) if simplices_list else None

    return nodes, all_faces, simplices, edge_index


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

    Tecplot FEPolygon 格式（2D）：
    - TITLE: 标题
    - VARIABLES: 变量列表
    - ZONE: 区域定义
    - 节点数据
    - FaceNodesLink: 面-节点连接
    - LeftCell/RightCell: 面左右单元
    - Elements: 单元节点连接
    """
    num_nodes = len(nodes)
    is_3d = nodes.shape[1] == 3

    # 确定拓扑信息
    # ZoneType 只有两种：FEPolygon（2D）和 FEPolyhedron（3D）
    cell_type = "FEPolyhedron" if is_3d else "FEPolygon"

    if simplices is not None:
        num_cells = len(simplices)
    else:
        num_cells = 0

    if edge_index is not None:
        num_faces = edge_index.shape[1]
    elif faces is not None:
        num_faces = len(faces)
    else:
        num_faces = 0

    # 构建变量列表
    var_names = ["X", "Y", "Z"] if is_3d else ["X", "Y"]
    var_names.extend(scalars.keys())

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    words_per_line = 5  # 每行最多输出 5 个数据

    with open(output_path, "w") as f:
        # 文件头
        f.write(f'TITLE = "{title}"\n')
        f.write(f"VARIABLES = {', '.join(var_names)}\n")
        f.write("ZONE\n")
        f.write(f"ZoneType = {cell_type}\n")
        f.write(f"Nodes    = {num_nodes}\n")
        f.write(f"Faces    = {num_faces}\n")
        f.write(f"Elements = {num_cells}\n")

        if num_faces > 0:
            f.write("NumConnectedBoundaryFaces = 0\n")
            f.write("TotalNumBoundaryConnections = 0\n")

        # 输出节点坐标
        for dim in range(nodes.shape[1]):
            for i in range(num_nodes):
                f.write(f"{nodes[i, dim]:.10f} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_nodes % words_per_line != 0:
                f.write("\n")

        # 输出标量字段
        for var_name, var_data in scalars.items():
            for i in range(num_nodes):
                f.write(f"{var_data[i]:.10f} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_nodes % words_per_line != 0:
                f.write("\n")

        # 如果有面数据，输出拓扑信息
        if num_faces > 0 and num_cells > 0:
            # 构建面-节点连接关系
            if edge_index is not None:
                face_to_nodes = []
                for i in range(edge_index.shape[1]):
                    n1, n2 = edge_index[0, i] + 1, edge_index[1, i] + 1
                    face_to_nodes.append([n1, n2])
            elif faces is not None:
                face_to_nodes = []
                for face in faces:
                    face_nodes = [n + 1 for n in face.get("nodes", [])]
                    face_to_nodes.append(face_nodes)
            else:
                face_to_nodes = []

            # 3D 专用：FaceNodeNumber - 每个面的节点数量
            if is_3d:
                for fn in face_to_nodes:
                    f.write(f"{len(fn)} ")
                    if (len(fn) + 1) % words_per_line == 0:
                        f.write("\n")
                if len(face_to_nodes) % words_per_line != 0:
                    f.write("\n")

            # FaceNodesLink: 面-节点连接列表
            node_count = 0
            for fn in face_to_nodes:
                for nid in fn:
                    f.write(f"{nid} ")
                    node_count += 1
                    if node_count % words_per_line == 0:
                        f.write("\n")
            if node_count % words_per_line != 0:
                f.write("\n")

            # LeftCell/RightCell: 面左右单元
            left_cell = np.zeros(num_faces, dtype=int)
            right_cell = np.zeros(num_faces, dtype=int)

            if simplices is not None:
                for cell_idx, simplex in enumerate(simplices):
                    # 遍历单元的边
                    for i in range(len(simplex)):
                        p1, p2 = simplex[i], simplex[(i + 1) % len(simplex)]
                        edge_key = (min(p1, p2), max(p1, p2))

                        # 查找匹配的边
                        for face_idx in range(num_faces):
                            if edge_index is not None:
                                e1, e2 = edge_index[0, face_idx], edge_index[1, face_idx]
                            elif face_idx < len(face_to_nodes) and len(face_to_nodes[face_idx]) == 2:
                                e1, e2 = face_to_nodes[face_idx][0] - 1, face_to_nodes[face_idx][1] - 1
                            else:
                                continue

                            if (min(e1, e2), max(e1, e2)) == edge_key:
                                if left_cell[face_idx] == 0:
                                    left_cell[face_idx] = cell_idx + 1
                                else:
                                    right_cell[face_idx] = cell_idx + 1
                                break

            # 输出左单元
            for i in range(num_faces):
                f.write(f"{left_cell[i]} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_faces % words_per_line != 0:
                f.write("\n")

            # 输出右单元
            for i in range(num_faces):
                f.write(f"{right_cell[i]} ")
                if (i + 1) % words_per_line == 0:
                    f.write("\n")
            if num_faces % words_per_line != 0:
                f.write("\n")

            # Element connections: 单元节点连接
            for simplex in simplices:
                for node_id in simplex:
                    f.write(f"{node_id + 1} ")
                f.write("\n")


def export_from_cas(cas_file: str, output_path: str, scalars: Optional[Dict[str, np.ndarray]] = None) -> str:
    """
    从 Fluent .cas 文件导出为 PLT 文件

    Args:
        cas_file: .cas 文件路径
        output_path: 输出 PLT 文件路径
        scalars: 标量字段字典（可选）

    Returns:
        输出文件路径
    """
    from fileIO.read_cas import parse_fluent_msh
    from data_structure.mesh_reconstruction import preprocess_grid

    # 解析网格
    grid = parse_fluent_msh(cas_file)
    preprocess_grid(grid)

    # 导出
    title = Path(cas_file).stem
    return export_mesh_to_plt(
        grid=grid,
        output_path=output_path,
        title=title,
        scalars=scalars,
    )

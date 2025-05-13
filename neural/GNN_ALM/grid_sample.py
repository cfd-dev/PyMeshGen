import os
import sys
from pathlib import Path
import numpy as np

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from data_structure.mesh_reconstruction import get_adjacent_node
from utils.geom_toolkit import unit_direction_vector
from fileIO.read_cas import parse_fluent_msh
from data_structure.mesh_reconstruction import preprocess_grid
from utils.message import info, warning, error, debug
from visualization.mesh_visualization import visualize_wall_structure_2d


def get_march_vector(grid, node_1based, current_face):
    """
    获取节点的推进向量，即节点到相邻节点的单位方向向量。

    参数:
    grid (dict): 网格数据
    node_1based (int): 节点的1-based索引
    current_face (dict): 当前面的信息

    返回:
    tuple: 推进向量的x, y, z分量
    """

    adjacent_faces = get_adjacent_node(grid, node_1based, current_face)

    if len(adjacent_faces) > 2:
        # 处理翼型尾部的非四边形边界层
        # 计算adjacent_faces的空间长短，取长度最小的面计算march_vector
        min_length = float("inf")
        for adj_face in adjacent_faces:
            face = adj_face["face"]
            nodes = face["nodes"]
            node1_coord = grid["nodes"][nodes[0] - 1]
            node2_coord = grid["nodes"][nodes[1] - 1]
            length = np.linalg.norm(np.array(node1_coord) - np.array(node2_coord))
            if length < min_length:
                min_length = length
                min_face = face
        adjacent_faces = [{"face": min_face}]

    if len(adjacent_faces) == 0:
        raise ValueError("No adjacent faces found")
    if len(adjacent_faces) > 1:
        raise ValueError("More than one adjacent face found")

    for adj_face in adjacent_faces:
        face = adj_face["face"]
        nodes = face["nodes"]

        try:
            idx = nodes.index(node_1based)
            adj_node = nodes[(idx + 1) % len(nodes)]  # 取下一个节点
            break
        except ValueError:
            # 处理节点不在当前面的情况
            adj_node = None

        # if nodes[0] == node_1based:
        #     adj_node = nodes[1]
        # else:
        #     adj_node = nodes[0]

    node_0based = node_1based - 1
    adj_node = adj_node - 1
    node1_coord = grid["nodes"][node_0based]
    node2_coord = grid["nodes"][adj_node]

    return unit_direction_vector(node1_coord, node2_coord)


def process_single_file(file_path, visualize=False):
    """
    处理单个网格文件，提取wall节点和推进向量

    参数:
    file_path (str): 文件路径

    返回:
    dict: 包含wall_faces, wall_nodes, valid_wall_nodes等信息
    """
    info(f"Processing file: {file_path}")

    # 解析网格文件
    try:
        grid = parse_fluent_msh(file_path)
    except Exception as e:
        error(f"Error parsing file {file_path}: {e}")
        return None

    # 预处理网格
    preprocess_grid(grid)

    # === 新增归一化预处理 ===
    # 获取所有节点坐标
    all_nodes = grid["nodes"]
    if not all_nodes:
        warning("No nodes found in grid.")
        return None

    # 以wall_faces的节点坐标范围进行归一化，取出所有wall面的节点
    wall_nodes_1based = []
    for zone in grid["zones"].values():
        if zone["type"] == "faces" and zone.get("bc_type") == "wall":
            for face in zone["data"]:
                wall_nodes_1based.extend(face["nodes"])
    wall_nodes_1based = list(set(wall_nodes_1based))

    # 计算wall_nodes_1based各维度范围
    wall_nodes_coord = [grid["nodes"][n - 1] for n in wall_nodes_1based]
    x_min, x_max = min(n[0] for n in wall_nodes_coord), max(
        n[0] for n in wall_nodes_coord
    )
    y_min, y_max = min(n[1] for n in wall_nodes_coord), max(
        n[1] for n in wall_nodes_coord
    )
    z_min, z_max = (
        (
            min(n[2] for n in wall_nodes_coord),
            max(n[2] for n in wall_nodes_coord),
        )
        if len(wall_nodes_coord[0]) > 2
        else (0, 0)
    )

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    ref_d = max(x_range, y_range, z_range) + 1e-8  # 防止除零

    # 对坐标进行归一化处理
    normalized_nodes = []
    for node in all_nodes:
        norm_x = (node[0] - x_min) / ref_d

        y_range = y_max - y_min
        norm_y = (node[1] - y_min) / ref_d

        if len(node) > 2:
            norm_z = (node[2] - z_min) / ref_d
            normalized_nodes.append((norm_x, norm_y, norm_z))
        else:
            normalized_nodes.append((norm_x, norm_y))

    if visualize:  # 可视化时无需进行归一化，直接使用原始坐标
        normalized_nodes = all_nodes

    # 初始化存储结构
    wall_faces = []
    node_dict = {}  # 使用字典暂存节点信息，避免重复

    # 遍历所有区域
    for zone in grid["zones"].values():
        # 检查是否为面区域且边界类型为wall
        if zone["type"] == "faces" and zone.get("bc_type") == "wall":
            # 遍历该区域的所有面
            for face in zone["data"]:
                wall_faces.append(face)

                # 获取面的节点索引（注意Fluent节点索引从1开始）
                node_indices_0based = [
                    n - 1 for n in face["nodes"]
                ]  # 转换为Python的0基索引

                for node_id_xy in node_indices_0based:
                    # 初始化节点信息
                    if node_id_xy not in node_dict:
                        node_dict[node_id_xy] = {
                            "original_indices": node_id_xy,
                            "coords": normalized_nodes[node_id_xy],
                            "node_wall_faces": [],  # 存储与该节点相关的所有wall面
                            "march_vector": None,
                        }

                    # 添加当前面到节点的faces列表
                    node_dict[node_id_xy]["node_wall_faces"].append(face)

    # 转换为列表并计算推进向量
    wall_nodes = list(node_dict.values())
    for node_info in wall_nodes:
        node_1based = node_info["original_indices"] + 1
        # 选择第一个关联的面进行计算（可根据需求调整策略）
        if node_info["node_wall_faces"]:
            face = node_info["node_wall_faces"][0]
            try:
                node_info["march_vector"] = get_march_vector(grid, node_1based, face)
            except Exception as e:
                error(f"Error calculating vector for node {node_1based}: {e}")

    # 过滤无效向量
    valid_wall_nodes = [n for n in wall_nodes if n["march_vector"]]

    # 打印统计信息
    info(f"File: {file_path}")
    info(f"Total wall nodes: {len(wall_nodes)}")
    info(f"Valid vectors: {len(valid_wall_nodes)}")

    if visualize:
        visualize_wall_structure_2d(grid, valid_wall_nodes)

    return {
        "file_path": file_path,
        "grid": grid,
        "wall_faces": wall_faces,
        "wall_nodes": wall_nodes,
        "valid_wall_nodes": valid_wall_nodes,
    }


def batch_process_files(folder_path):
    """
    批量处理指定文件夹中的所有网格文件

    参数:
    folder_path (str): 文件夹路径

    返回:
    list: 每个文件的处理结果列表
    """
    results = []

    # 遍历文件夹中的所有.cas文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".cas"):
            file_path = os.path.join(folder_path, file_name)
            result = process_single_file(file_path)
            if result:
                results.append(result)

    return results

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    file_path = current_dir / "sample_grids/training/NACA0310.cas"
    result = process_single_file(file_path, visualize=True)

    folder_path = current_dir / "sample_grids/training"
    # results = batch_process_files(folder_path)
    # info(f"成功加载 {len(results)} 个数据集")

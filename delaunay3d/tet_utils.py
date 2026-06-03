"""四面体网格工具函数

从 bowyer_watson.py 中提取的通用四面体拓扑和几何工具函数。
"""
import numpy as np
from itertools import combinations


def tet_faces(node_ids):
    """提取四面体的4个面（每个面用排序后的节点ID元组表示）

    Args:
        node_ids: (4,) 四面体的4个节点ID

    Returns:
        list[tuple]: 4个面，每个面是排序后的3个节点ID元组
    """
    a, b, c, d = node_ids
    return [
        tuple(sorted([a, b, c])),
        tuple(sorted([a, b, d])),
        tuple(sorted([a, c, d])),
        tuple(sorted([b, c, d])),
    ]


def tet_edges(node_ids):
    """提取四面体的6条边（每条边用排序后的节点ID元组表示）

    Args:
        node_ids: (4,) 四面体的4个节点ID

    Returns:
        list[tuple]: 6条边，每条边是排序后的2个节点ID元组
    """
    return [tuple(sorted(c)) for c in combinations(node_ids, 2)]


def tet_centroid(p1, p2, p3, p4):
    """计算四面体的形心

    Args:
        p1, p2, p3, p4: 四面体的4个顶点坐标

    Returns:
        list: 形心坐标 [x, y, z]
    """
    return [
        (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0,
        (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0,
        (p1[2] + p2[2] + p3[2] + p4[2]) / 4.0,
    ]


def in_circumsphere(node_coords, tet_coords, tolerance=1e-10):
    """检查节点是否在四面体的外接球内

    Args:
        node_coords: (3,) 查询点坐标
        tet_coords: list of 4 coordinates, 四面体的4个顶点坐标
        tolerance: 相对容差

    Returns:
        bool: True 表示节点在外接球内
    """
    from utils.geom_toolkit import circumsphere

    center, r2 = circumsphere(*tet_coords)
    if r2 < 1e-30:
        return False

    dist2 = sum((node_coords[i] - center[i]) ** 2 for i in range(3))
    return dist2 < r2 * (1.0 + tolerance)


def find_boundary_faces(tets):
    """找出所有边界面（只被一个四面体使用的面）

    Args:
        tets: list of Tetrahedron 对象（需有 node_ids 属性）

    Returns:
        list[tuple]: 边界面列表，每个面是排序后的3个节点ID元组
    """
    face_count = {}
    for tet in tets:
        for face in tet_faces(tet.node_ids):
            face_count[face] = face_count.get(face, 0) + 1

    return [fk for fk, cnt in face_count.items() if cnt == 1]


def find_boundary_face_set(tets):
    """找出所有边界面的集合

    Args:
        tets: list of Tetrahedron 对象（需有 node_ids 属性）

    Returns:
        set: 边界面集合
    """
    face_count = {}
    for tet in tets:
        for face in tet_faces(tet.node_ids):
            face_count[face] = face_count.get(face, 0) + 1

    return {fk for fk, cnt in face_count.items() if cnt == 1}


def build_face_to_tets(tets):
    """构建面到四面体的映射

    Args:
        tets: list of Tetrahedron 对象（需有 node_ids 属性）

    Returns:
        dict: face_key -> list of (tet_index, opposite_node_id)
    """
    face_to_tets = {}
    for ci, tet in enumerate(tets):
        ids = tet.node_ids
        faces_with_opp = [
            (tuple(sorted([ids[0], ids[1], ids[2]])), ids[3]),
            (tuple(sorted([ids[0], ids[1], ids[3]])), ids[2]),
            (tuple(sorted([ids[0], ids[2], ids[3]])), ids[1]),
            (tuple(sorted([ids[1], ids[2], ids[3]])), ids[0]),
        ]
        for face_key, opp in faces_with_opp:
            if face_key not in face_to_tets:
                face_to_tets[face_key] = []
            face_to_tets[face_key].append((ci, opp))
    return face_to_tets


def build_edge_to_tets(tets):
    """构建边到四面体的映射

    Args:
        tets: list of Tetrahedron 对象（需有 node_ids 属性）

    Returns:
        dict: edge_key -> list of tet_index
    """
    edge_to_tets = {}
    for ci, tet in enumerate(tets):
        for edge in tet_edges(tet.node_ids):
            if edge not in edge_to_tets:
                edge_to_tets[edge] = []
            edge_to_tets[edge].append(ci)
    return edge_to_tets


def create_super_tetrahedron_coords(node_coords, scale=2.0):
    """计算超级四面体的4个顶点坐标

    超级四面体包含所有节点，用于 Bowyer-Watson 初始化。

    Args:
        node_coords: list of coordinates, 所有节点坐标
        scale: 缩放因子（相对于包围盒大小）

    Returns:
        list[list]: 4个顶点坐标
    """
    coords = np.array(node_coords)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    size = np.max(maxs - mins) * scale

    return [
        [center[0] - size, center[1] - size, center[2] - size],
        [center[0] + size * 2, center[1] - size, center[2] - size],
        [center[0], center[1] + size * 2, center[2] - size],
        [center[0], center[1], center[2] + size * 2],
    ]


def compute_max_edge_length(tet):
    """计算四面体的最大边长

    Args:
        tet: Tetrahedron 对象（需有 p1, p2, p3, p4 属性）

    Returns:
        float: 最大边长
    """
    coords = [tet.p1, tet.p2, tet.p3, tet.p4]
    max_edge = 0.0
    for i, j in combinations(range(4), 2):
        dx = coords[i][0] - coords[j][0]
        dy = coords[i][1] - coords[j][1]
        dz = coords[i][2] - coords[j][2]
        elen = (dx * dx + dy * dy + dz * dz) ** 0.5
        if elen > max_edge:
            max_edge = elen
    return max_edge


def node_hash(coords):
    """计算节点坐标的哈希值（用于去重）

    Args:
        coords: 坐标列表或数组

    Returns:
        int: 哈希值
    """
    return hash(tuple(f"{c:.6f}" for c in coords))

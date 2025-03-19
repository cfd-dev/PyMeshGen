from math import sqrt


def normal_vector(front):
    """计算二维平面阵面的法向量"""
    node1, node2 = front.nodes_coords
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]
    return dy, -dx


def calculate_distance(p1, p2):
    """计算二维/三维点间距"""
    return sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calculate_distance2(p1, p2):
    """计算二维/三维点间距"""
    return sum((a - b) ** 2 for a, b in zip(p1, p2))

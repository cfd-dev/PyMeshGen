import numpy as np


def denormalize_point(point, local_range):
    # 反归一化
    point = np.squeeze(point)
    x_min, x_max, y_min, y_max = local_range
    ref_d = max(x_max - x_min, y_max - y_min) + 1e-8  # 防止除零
    new_point = point * ref_d + np.array([x_min, y_min])

    return np.squeeze(new_point)

def normalize_point(point, local_range):
    # 归一化
    point = np.squeeze(point)
    x_min, x_max, y_min, y_max = local_range
    ref_d = max(x_max - x_min, y_max - y_min) + 1e-8  # 防止除零
    new_point = (point - np.array([x_min, y_min])) / ref_d

    return np.squeeze(new_point)


def normalize_polygon(polygon_coords):
    # 计算坐标范围
    x_min, x_max = np.min(polygon_coords[:, 0]), np.max(polygon_coords[:, 0])
    y_min, y_max = np.min(polygon_coords[:, 1]), np.max(polygon_coords[:, 1])
    ref_d = max(x_max - x_min, y_max - y_min) + 1e-8  # 防止除零

    # 归一化处理
    normalized_coords = polygon_coords - np.array([[x_min, y_min]])
    normalized_coords /= ref_d

    # 断言：归一化后的坐标必须在 [0, 1] 范围内
    assert np.all(
        (normalized_coords >= 0) & (normalized_coords <= 1)
    ), f"Normalized coordinates out of [0,1]: min={np.min(normalized_coords)}, max={np.max(normalized_coords)}"

    local_range = (x_min, x_max, y_min, y_max)  # 保存范围用于逆变换

    return normalized_coords, local_range


# def normalize_polygon(polygon_coords):
#     if polygon_coords.shape[1] == 2:
#         polygon_coords = np.hstack([polygon_coords, np.zeros((len(polygon_coords), 1))])

#     # 计算三维坐标范围
#     x_min, x_max = np.min(polygon_coords[:, 0]), np.max(polygon_coords[:, 0])
#     y_min, y_max = np.min(polygon_coords[:, 1]), np.max(polygon_coords[:, 1])
#     z_min, z_max = np.min(polygon_coords[:, 2]), np.max(polygon_coords[:, 2])
#     ref_d = max(x_max - x_min, y_max - y_min, z_max - z_min) + 1e-8  # 三维最大跨度

#     # 三维归一化处理
#     normalized_coords = polygon_coords - np.array([[x_min, y_min, z_min]])
#     normalized_coords /= ref_d

#     # 三维坐标验证
#     assert np.all(
#         (normalized_coords >= 0) & (normalized_coords <= 1)
#     ), f"三维坐标超出范围: min={np.min(normalized_coords)}, max={np.max(normalized_coords)}"

#     local_range = (x_min, x_max, y_min, y_max, z_min, z_max)  # 保存三维范围

#     return normalized_coords, local_range

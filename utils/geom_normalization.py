import numpy as np


def denormalize_point(point, local_range):
    # 反归一化
    point = np.squeeze(point)
    x_min, x_max, y_min, y_max = local_range
    ref_d = max(x_max - x_min, y_max - y_min) + 1e-8  # 防止除零
    new_point = point * ref_d + np.array([x_min, y_min])

    return np.squeeze(new_point)


def normalize_ploygon(polygon_coords):
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

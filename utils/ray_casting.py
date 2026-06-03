"""射线投射法判断点是否在封闭网格内部

基于 Möller-Trumbore 射线-三角形求交算法 + DDA 空间网格遍历。
沿 x/y/z 三个轴各发射射线，奇数次穿过 = 内部，多数投票决定结果。
"""
import numpy as np


def build_spatial_grid(tri_verts, grid_n=None):
    """构建 3D 空间网格加速结构

    Args:
        tri_verts: (N, 3, 3) 三角形顶点坐标
        grid_n: 网格分辨率（每轴），None 则自动计算

    Returns:
        dict: 包含 tri_verts, edge1, edge2, grid, grid_n, mins_g, inv_cell
    """
    n_tris = len(tri_verts)
    if n_tris == 0:
        return None

    edge1 = tri_verts[:, 1] - tri_verts[:, 0]
    edge2 = tri_verts[:, 2] - tri_verts[:, 0]

    mins = tri_verts.reshape(-1, 3).min(axis=0)
    maxs = tri_verts.reshape(-1, 3).max(axis=0)

    if grid_n is None:
        grid_n = max(16, int(np.sqrt(n_tris)))

    extent = maxs - mins
    margin = max(extent) * 0.01
    mins_g = mins - margin
    maxs_g = maxs + margin
    cell_size = (maxs_g - mins_g) / grid_n
    inv_cell = np.where(cell_size > 0, 1.0 / cell_size, 1.0)

    grid = {}
    for i in range(n_tris):
        tri_mins = tri_verts[i].min(axis=0)
        tri_maxs = tri_verts[i].max(axis=0)
        ijk_min = np.maximum(0, ((tri_mins - mins_g) * inv_cell).astype(int))
        ijk_max = np.minimum(grid_n - 1, ((tri_maxs - mins_g) * inv_cell).astype(int))
        for ix in range(ijk_min[0], ijk_max[0] + 1):
            for iy in range(ijk_min[1], ijk_max[1] + 1):
                for iz in range(ijk_min[2], ijk_max[2] + 1):
                    key = (ix, iy, iz)
                    if key not in grid:
                        grid[key] = []
                    grid[key].append(i)

    return {
        'tri_verts': tri_verts,
        'edge1': edge1,
        'edge2': edge2,
        'grid': grid,
        'grid_n': grid_n,
        'mins_g': mins_g,
        'inv_cell': inv_cell,
    }


def ray_triangle_intersect(origin, ray_dir, p0, edge1, edge2):
    """Möller-Trumbore 射线-三角形求交

    Args:
        origin: (3,) 射线起点
        ray_dir: (3,) 射线方向（不需要归一化）
        p0: (3,) 三角形第一个顶点
        edge1: (3,) 三角形边向量 v1 - v0
        edge2: (3,) 三角形边向量 v2 - v0

    Returns:
        float or None: 交点参数 t（>0 为正向），None 表示不相交
    """
    h = np.cross(ray_dir, edge2)
    det = np.dot(edge1, h)
    if abs(det) < 1e-30:
        return None

    inv_det = 1.0 / det
    s = origin - p0
    u = inv_det * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = inv_det * np.dot(ray_dir, q)
    if v < 0.0 or u + v > 1.0:
        return None

    t = inv_det * np.dot(edge2, q)
    return t


def count_ray_intersections(origin, ray_dir, grid_data):
    """沿射线遍历空间网格，统计与表面三角形的正向交点数

    使用 DDA 算法遍历射线经过的所有网格单元。

    Args:
        origin: (3,) 射线起点
        ray_dir: (3,) 射线方向
        grid_data: build_spatial_grid 返回的字典

    Returns:
        int: 正向交点数（t > 0）
    """
    tv = grid_data['tri_verts']
    e1 = grid_data['edge1']
    e2 = grid_data['edge2']
    grid = grid_data['grid']
    grid_n = grid_data['grid_n']
    mins_g = grid_data['mins_g']
    inv_cell = grid_data['inv_cell']

    cell = ((origin - mins_g) * inv_cell).astype(int)
    cell = np.clip(cell, 0, grid_n - 1)

    step = np.zeros(3, dtype=int)
    t_max = np.full(3, np.inf)
    t_delta = np.full(3, np.inf)

    for d in range(3):
        if abs(ray_dir[d]) > 1e-30:
            if ray_dir[d] > 0:
                step[d] = 1
                next_b = mins_g[d] + (cell[d] + 1) / inv_cell[d]
                t_max[d] = (next_b - origin[d]) / ray_dir[d]
            else:
                step[d] = -1
                next_b = mins_g[d] + cell[d] / inv_cell[d]
                t_max[d] = (next_b - origin[d]) / ray_dir[d]
            t_delta[d] = abs(1.0 / (ray_dir[d] * inv_cell[d]))

    t_hits = []
    visited = set()
    max_steps = grid_n * 3

    for _ in range(max_steps):
        cell_key = tuple(cell)
        if cell_key not in visited:
            visited.add(cell_key)
            for tri_idx in grid.get(cell_key, []):
                t = ray_triangle_intersect(
                    origin, ray_dir,
                    tv[tri_idx, 0], e1[tri_idx], e2[tri_idx],
                )
                if t is not None and t > 1e-10:
                    t_hits.append(t)

        if t_max[0] < t_max[1] and t_max[0] < t_max[2]:
            cell[0] += step[0]
            t_max[0] += t_delta[0]
        elif t_max[1] < t_max[2]:
            cell[1] += step[1]
            t_max[1] += t_delta[1]
        else:
            cell[2] += step[2]
            t_max[2] += t_delta[2]

        if not (0 <= cell[0] < grid_n and
                0 <= cell[1] < grid_n and
                0 <= cell[2] < grid_n):
            break

    # 去重：同一距离的多次命中视为同一面（如一个面被两个三角形共享）
    if not t_hits:
        return 0
    t_hits.sort()
    n_faces = 1
    for i in range(1, len(t_hits)):
        if t_hits[i] - t_hits[i - 1] > 1e-6:
            n_faces += 1
    return n_faces


def classify_points_inside(points, grid_data):
    """批量判断点是否在封闭网格内部

    沿 x/y/z 三个轴各发射射线，统计正向交点数。
    奇数次 = 内部，三个轴多数投票（>=2）决定最终结果。

    Args:
        points: (N, 3) 查询点坐标
        grid_data: build_spatial_grid 返回的字典

    Returns:
        np.ndarray: (N,) bool，True 表示内部
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    n_pts = len(pts)

    if grid_data is None:
        return np.ones(n_pts, dtype=bool)

    ray_dirs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    votes = np.zeros(n_pts, dtype=int)

    for ray_dir in ray_dirs:
        for pt_idx in range(n_pts):
            n_pos = count_ray_intersections(pts[pt_idx], ray_dir, grid_data)
            if n_pos > 0 and (n_pos % 2) != 0:
                votes[pt_idx] += 1

    return votes >= 2

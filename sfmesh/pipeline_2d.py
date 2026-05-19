"""
2D 阵面推进流水线

提供从边界离散化到 AFM 网格生成再到 3D 坐标映射的完整 2D 网格流水线。
复用 data_structure.front2d、meshsize.QuadtreeSizing、adfront2.Adfont2 等模块。
"""
import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_structure.front2d import Front
from data_structure.basic_elements import NodeElementALM
from .surface_front import SurfaceTriangle, NodeElement3D


def _discretize_edge_2d(
    start_2d: Tuple[float, float],
    end_2d: Tuple[float, float],
    spacing: float,
) -> List[Tuple[float, float]]:
    """将 2D 线段均匀离散化为点列表"""
    s = np.array(start_2d)
    e = np.array(end_2d)
    length = np.linalg.norm(e - s)
    if length < 1e-14:
        return [start_2d]
    n = max(2, round(length / spacing) + 1)
    points = []
    for i in range(n + 1):
        t = i / n
        pt = s + t * (e - s)
        points.append((float(pt[0]), float(pt[1])))
    return points


def _create_fronts_from_2d_edges(
    edge_points_2d: List[List[Tuple[float, float]]],
    face_name: str = "face",
) -> List[Front]:
    """从 2D 边界点列表创建 Front 对象列表"""
    all_fronts = []
    bc_type = "BCWall"
    for pts in edge_points_2d:
        for i in range(len(pts) - 1):
            node1 = NodeElementALM(
                coords=(pts[i][0], pts[i][1], 0.0),
                idx=-1, bc_type=bc_type, part_name=face_name,
            )
            node2 = NodeElementALM(
                coords=(pts[i + 1][0], pts[i + 1][1], 0.0),
                idx=-1, bc_type=bc_type, part_name=face_name,
            )
            front = Front(node1, node2, idx=-1, bc_type=bc_type, part_name=face_name)
            all_fronts.append(front)
    return all_fronts


def _run_afm_2d_pipeline(
    all_fronts: List[Front],
    spacing: float,
    face_size: float,
):
    """
    运行 2D AFM 核心流水线：QuadtreeSizing → Adfront2 → 边交换 → Laplacian 光滑

    Args:
        all_fronts: 边界阵面列表（CCW 排列）
        spacing: 网格尺寸
        face_size: 面的特征尺寸（用于计算边界扩展和迭代上限）

    Returns:
        优化后的 Unstructured_Grid
    """
    from meshsize.meshsize import QuadtreeSizing
    from adfront2.adfront2 import Adfront2
    from optimize.optimize import edge_swap_delaunay, laplacian_smooth
    import heapq as _heapq

    class _DummyVisual:
        ax = None

    _extra_pad = max(face_size * 0.5, 5.0 * spacing) / max(face_size, 1e-12)

    class _PaddedSizingField(QuadtreeSizing):
        def compute_global_parameters(self):
            super().compute_global_parameters()
            x0, y0, x1, y1 = self.bg_bounds
            dx, dy = x1 - x0, y1 - y0
            self.bg_bounds = (
                x0 - dx * _extra_pad, y0 - dy * _extra_pad,
                x1 + dx * _extra_pad, y1 + dy * _extra_pad,
            )

        def spacing_at(self, point):
            try:
                return super().spacing_at(point)
            except ValueError:
                return self.global_spacing

    sizing_system = _PaddedSizingField(
        initial_front=all_fronts,
        max_size=spacing * 10,
        resolution=0.1,
        decay=1.2,
        visual_obj=_DummyVisual(),
    )

    class _ParamObj:
        debug_level = 0
        mesh_type = 1

    front_heap = list(all_fronts)
    _heapq.heapify(front_heap)

    adfront = Adfront2(
        boundary_front=front_heap,
        sizing_system=sizing_system,
        node_coords=None,
        param_obj=_ParamObj(),
        visual_obj=_DummyVisual(),
    )

    max_steps = max(50000, int((face_size / spacing) ** 2 * 15))
    step = 0
    while adfront.front_list and step < max_steps:
        step += 1
        adfront.base_front = _heapq.heappop(adfront.front_list)
        sp = sizing_system.spacing_at(adfront.base_front.center)
        adfront.add_new_point(sp)
        adfront.search_candidates(adfront.base_front.al * sp)
        adfront.select_point()
        adfront.update_data()
    adfront.construct_unstr_grid()

    unstr_grid = adfront.unstr_grid
    edge_swap_delaunay(unstr_grid)
    laplacian_smooth(unstr_grid, num_iter=3)
    return unstr_grid


def _unstr_grid_to_3d(
    unstr_grid,
    map_to_3d,
    normal_3d: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    normal_func=None,
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    将 2D Unstructured_Grid 映射到 3D 坐标并生成 SurfaceTriangle 列表

    Args:
        unstr_grid: AFM 生成的 2D 网格
        map_to_3d: 坐标映射函数 (x2d, y2d) → (x3d, y3d, z3d)
        normal_3d: 默认法向量（当 normal_func 为 None 时使用）
        normal_func: 可选的法向量计算函数 (x2d, y2d) → (nx, ny, nz)

    Returns:
        (triangles, nodes_3d)
    """
    grid_nodes = unstr_grid.node_coords
    grid_cells = unstr_grid.cell_container

    nodes_3d = []
    for idx, coords_2d in enumerate(grid_nodes):
        x2d, y2d = float(coords_2d[0]), float(coords_2d[1])
        coord_3d = map_to_3d(x2d, y2d)
        n3d = normal_func(x2d, y2d) if normal_func else normal_3d
        node = NodeElement3D(
            coords=coord_3d, idx=idx,
            surface=None, uv_params=(0.0, 0.0),
            normal=n3d,
        )
        nodes_3d.append(node)

    triangles = []
    for cell in grid_cells:
        nids = cell.node_ids
        if len(nids) >= 3:
            tri = SurfaceTriangle(
                nodes_3d[nids[0]], nodes_3d[nids[1]], nodes_3d[nids[2]],
                surface=None, idx=len(triangles),
            )
            triangles.append(tri)

    return triangles, nodes_3d


def _mesh_face_2d_pipeline(
    corners_3d: List[Tuple[float, float, float]],
    spacing: float,
    face_name: str = "face",
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    对单个平面四边形面使用 2D 阵面推进流水线生成网格

    将 3D 平面四边形变换到 2D XY 空间，使用 Front → QuadtreeSizing →
    Adfront2 流水线生成网格，再经边交换和 Laplacian 光滑优化，
    最后变换回 3D 坐标。

    Args:
        corners_3d: 四个角点的 3D 坐标（按顺序排列，构成四边形）
        spacing: 网格尺寸
        face_name: 面名称（用于日志）

    Returns:
        (triangles, nodes_3d)
    """
    # 检测平面：确定常量轴和活跃轴
    c0, c1, c2, c3 = [np.array(p) for p in corners_3d]
    e0 = c1 - c0
    e1 = c3 - c0
    normal = np.cross(e0, e1)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-14:
        return [], []
    normal = normal / norm_len

    max_axis = int(np.argmax(np.abs(normal)))
    ax0, ax1 = [i for i in range(3) if i != max_axis]
    const_val = float(c0[max_axis])

    # 变换到 2D
    c2d = []
    for p in corners_3d:
        coord = [p[0], p[1], p[2]]
        c2d.append((coord[ax0], coord[ax1]))

    # 离散化四条边
    edge_points_2d = []
    for i in range(4):
        pts = _discretize_edge_2d(c2d[i], c2d[(i + 1) % 4], spacing)
        edge_points_2d.append(pts)

    # 创建 Front 对象并运行 AFM
    all_fronts = _create_fronts_from_2d_edges(edge_points_2d, face_name)
    _face_sz = max(
        max(c2d[i][0] for i in range(4)) - min(c2d[i][0] for i in range(4)),
        max(c2d[i][1] for i in range(4)) - min(c2d[i][1] for i in range(4)),
    )
    unstr_grid = _run_afm_2d_pipeline(all_fronts, spacing, _face_sz)

    # 变换回 3D 坐标
    normal_3d = tuple(float(x) for x in normal)

    def _map_to_3d(x2d, y2d):
        coord_3d = [0.0, 0.0, 0.0]
        coord_3d[ax0] = x2d
        coord_3d[ax1] = y2d
        coord_3d[max_axis] = const_val
        return tuple(coord_3d)

    return _unstr_grid_to_3d(unstr_grid, _map_to_3d, normal_3d)


def _discretize_circle_2d(
    center: Tuple[float, float],
    radius: float,
    n_segments: int,
) -> List[Tuple[float, float]]:
    """将圆离散化为 2D 点列表（CCW 排列）"""
    points = []
    for i in range(n_segments):
        angle = 2.0 * math.pi * i / n_segments
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points


def _mesh_disk_2d(
    center_xy: Tuple[float, float],
    radius: float,
    z: float,
    spacing: float,
    face_name: str = "disk",
    normal_z: float = 1.0,
    boundary_pts_2d: list = None,
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    对圆形平面使用 2D 阵面推进流水线生成网格

    Args:
        center_xy: 圆心 (x, y) 坐标
        radius: 圆半径
        z: 圆平面的 z 坐标
        spacing: 网格尺寸
        face_name: 面名称
        normal_z: 法向量 z 分量 (+1 或 -1)
        boundary_pts_2d: 预定义的边界点列表（用于共享边界节点）

    Returns:
        (triangles, nodes_3d)
    """
    if boundary_pts_2d is not None:
        circle_pts = boundary_pts_2d
        n_segments = len(circle_pts)
    else:
        n_segments = max(12, round(2 * math.pi * radius / spacing))
        circle_pts = _discretize_circle_2d(center_xy, radius, n_segments)

    # 圆形边界：将点序列拆分为多条边（每条弧一段）
    # 每段弧作为一个 edge，用直线段近似
    edge_points_2d = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        edge_points_2d.append([circle_pts[i], circle_pts[j]])

    all_fronts = _create_fronts_from_2d_edges(edge_points_2d, face_name)
    face_size = 2 * radius
    unstr_grid = _run_afm_2d_pipeline(all_fronts, spacing, face_size)

    normal_3d = (0.0, 0.0, normal_z)

    def _map_to_3d(x2d, y2d):
        return (x2d, y2d, z)

    return _unstr_grid_to_3d(unstr_grid, _map_to_3d, normal_3d)


def _mesh_lateral_cylinder_2d(
    base_center: Tuple[float, float, float],
    radius: float,
    height: float,
    spacing: float,
    face_name: str = "lateral",
    boundary_pts_2d: list = None,
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    对圆柱侧面使用 2D 阵面推进流水线生成网格

    将侧面展开为 (s, z) 平面上的矩形 [0, L] × [z0, z1]，
    其中 L = 2πr 为周长。使用 AFM 生成网格后映射回 3D 圆柱坐标。
    左右边界 (s=0 和 s=L) 代表同一条物理母线，映射回 3D 后坐标相同，
    由调用方的节点去重自动处理。

    Args:
        base_center: 底面圆心 (x, y, z)
        radius: 圆柱半径
        height: 圆柱高度
        spacing: 网格尺寸
        face_name: 面名称
        boundary_pts_2d: 预定义的边界点 (s, z) 列表（用于共享边界节点）

    Returns:
        (triangles, nodes_3d)
    """
    cx, cy, z0 = base_center
    z1 = z0 + height
    L = 2.0 * math.pi * radius

    if boundary_pts_2d is not None:
        # 使用预定义的边界点
        edge_points_2d = boundary_pts_2d
    else:
        # 展开矩形的四个角 (s, z)，CCW 排列
        corners_2d = [(0.0, z0), (L, z0), (L, z1), (0.0, z1)]

        # 离散化四条边
        edge_points_2d = []
        for i in range(4):
            pts = _discretize_edge_2d(corners_2d[i], corners_2d[(i + 1) % 4], spacing)
            edge_points_2d.append(pts)

    all_fronts = _create_fronts_from_2d_edges(edge_points_2d, face_name)
    face_size = max(L, height)
    unstr_grid = _run_afm_2d_pipeline(all_fronts, spacing, face_size)

    # 映射回 3D：(s, z) → (cx + r*cos(s/r), cy + r*sin(s/r), z)
    def _map_to_3d(s, z):
        theta = s / radius
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        return (x, y, z)

    # 法向量为径向向外
    def _normal_func(s, z):
        theta = s / radius
        return (math.cos(theta), math.sin(theta), 0.0)

    return _unstr_grid_to_3d(unstr_grid, _map_to_3d, normal_func=_normal_func)


def _mesh_cylinder_unified(
    base_center: Tuple[float, float, float],
    radius: float,
    height: float,
    spacing: float,
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    统一圆柱体网格生成：端面和柱面共享边界节点

    统一离散化圆边界，确保端面和柱面使用完全相同的边界节点。

    Args:
        base_center: 底面圆心 (x, y, z)
        radius: 圆柱半径
        height: 圆柱高度
        spacing: 网格尺寸

    Returns:
        (triangles, nodes)
    """
    cx, cy, z0 = base_center
    z1 = z0 + height

    # 统一离散化圆边界（与 primitives 分支一致，使用 max(6, ...)）
    L = 2.0 * math.pi * radius
    n_segments = max(6, round(L / spacing))
    circle_pts = _discretize_circle_2d((cx, cy), radius, n_segments)

    # 2D 圆边界点（x, y）用于端面
    circle_pts_2d = [(pt[0], pt[1]) for pt in circle_pts]

    # 转换为侧面展开坐标 (s, z)
    # 直接用均匀角度计算 s，避免 atan2 的 -0.0 问题
    lateral_bottom = []
    lateral_top = []
    for i in range(n_segments):
        s = radius * 2.0 * math.pi * i / n_segments
        lateral_bottom.append((s, z0))
        lateral_top.append((s, z1))

    # 添加接缝绕回点 (s=L)，hash 归一化后与起点 (s=0) 去重
    lateral_bottom.append((L, z0))
    lateral_top.append((L, z1))

    # 构建侧面闭合边界（四条边，首尾相连）
    # 底边: circle pts at z0 + 绕回点 (L, z0)
    # 右边: (L, z0) → (L, z1)，均匀离散化
    # 顶边: circle pts at z1 + 绕回点 (L, z1)（反向）
    # 左边: (0, z1) → (0, z0)，均匀离散化
    bottom_edge = list(lateral_bottom)
    right_edge = _discretize_edge_2d((L, z0), (L, z1), spacing)
    top_edge = list(reversed(lateral_top))
    left_edge = _discretize_edge_2d((0.0, z1), (0.0, z0), spacing)

    lateral_edge_pts = [bottom_edge, right_edge, top_edge, left_edge]

    # --- 底面网格（使用统一的圆边界）---
    bottom_tris, bottom_nodes = _mesh_disk_2d(
        center_xy=(cx, cy), radius=radius, z=z0,
        spacing=spacing, face_name="bottom", normal_z=-1.0,
        boundary_pts_2d=circle_pts_2d,
    )

    # --- 顶面网格 ---
    top_tris, top_nodes = _mesh_disk_2d(
        center_xy=(cx, cy), radius=radius, z=z1,
        spacing=spacing, face_name="top", normal_z=1.0,
        boundary_pts_2d=circle_pts_2d,
    )

    # --- 侧面网格（使用统一的边界）---
    lateral_tris, lateral_nodes = _mesh_lateral_cylinder_2d(
        base_center=base_center, radius=radius, height=height,
        spacing=spacing, face_name="lateral",
        boundary_pts_2d=lateral_edge_pts,
    )

    # --- 合并节点 ---
    node_hash_to_idx = {}
    all_nodes = []
    all_triangles = []
    global_idx = 0

    for face_nodes in [bottom_nodes, top_nodes, lateral_nodes]:
        node_local_to_global = {}
        for node in face_nodes:
            h = node.hash
            if h not in node_hash_to_idx:
                node_hash_to_idx[h] = global_idx
                node.idx = global_idx
                all_nodes.append(node)
                global_idx += 1
            node_local_to_global[id(node)] = node_hash_to_idx[h]

        face_tris = bottom_tris if face_nodes is bottom_nodes else (
            top_tris if face_nodes is top_nodes else lateral_tris
        )
        for tri in face_tris:
            new_tri = SurfaceTriangle(
                all_nodes[node_local_to_global[id(tri.nodes[0])]],
                all_nodes[node_local_to_global[id(tri.nodes[1])]],
                all_nodes[node_local_to_global[id(tri.nodes[2])]],
                idx=len(all_triangles),
            )
            all_triangles.append(new_tri)

    return all_triangles, all_nodes

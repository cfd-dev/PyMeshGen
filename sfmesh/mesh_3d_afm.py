"""
三维曲面阵面推进法（3D AFM）模块

基于阵面推进法在三维曲面上生成三角形网格，包含：
- SurfaceMeshGenerator: 类接口，支持相交检测、曲率自适应、line_mesh 边界

修复记录:
1. _check_intersection: 重写 shared_count==1 的边-边/边-面检测逻辑，避免误判
2. _select_best_node: 理想节点必须通过与候选节点相同的几何验证
3. _update_mesh: 新阵面创建前检查边饱和度，防止重复阵面和重复三角形
4. generate: 主循环增加陈旧阵面过滤（edge_count >= 2 直接丢弃）
5. _search_candidates: 自适应搜索半径，防止尺寸过渡区丢失候选
6. _create_ideal_node: 投影失败时增加 debug 日志
7. _check_intersection + _triangle_intersects_existing: 共享边蝴蝶形(bowtie)相交检测，
   防止两个共享一条边的三角形因非共享边交叉而产生几何相交
8. _optimize_mesh: 后处理优化（边交换 Delaunay + Laplacian 光滑），保持边界不动
"""
import heapq
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from OCC.Core.TopoDS import TopoDS_Face

from .surface_front import (
    SurfaceFront,
    NodeElement3D,
    SurfaceTriangle,
    create_initial_fronts_from_surface
)
from .surface_geometry import SurfaceGeometry
from .sizing_field import SurfaceSizingField
from .mesh_quality import SurfaceMeshQuality
from .geom_utils import (
    point_in_triangle_3d,
    check_triangle_intersection,
    check_triangle_vs_existing,
    check_edge_triangle_intersection,
    triangle_quality_from_coords,
    check_triangle_degenerate,
    segment_segment_distance_3d,
    validate_mesh_topology,
)

from utils.message import info, debug, warning
from utils.timer import TimeSpan
from data_structure.rtree_space import (
    build_space_index_3d_with_RTree,
    get_candidate_elements_id_3d,
    add_elems_to_space_index_3d_with_RTree
)


class SurfaceMeshGenerator:
    """
    曲面网格生成器

    使用阵面推进法在三维曲面上生成三角形网格
    """

    def __init__(
        self,
        surface: TopoDS_Face,
        global_spacing: float = 1.0,
        min_spacing: float = 0.01,
        max_spacing: float = 100.0,
        curvature_adaptation: bool = True,
        quality_threshold: float = 0.3,
        max_iterations: int = 100000,
        debug_level: int = 0,
        line_mesh: dict = None,
    ):
        """
        初始化曲面网格生成器

        Args:
            surface: OCC 曲面对象
            global_spacing: 全局网格尺寸
            min_spacing: 最小网格尺寸
            max_spacing: 最大网格尺寸
            curvature_adaptation: 是否启用曲率自适应
            quality_threshold: 质量阈值
            max_iterations: 最大迭代次数
            debug_level: 调试级别
            line_mesh: 预离散线网格（discretize_shape_edges 的输出），用于跨面共享边界
        """
        self.surface = surface
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.debug_level = debug_level
        self._line_mesh = line_mesh

        self.geometry = SurfaceGeometry()
        self.sizing_field = SurfaceSizingField(
            global_spacing=global_spacing,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            curvature_adaptation=curvature_adaptation,
            geometry_handler=self.geometry
        )

        self.front_list: List[SurfaceFront] = []
        self.node_list: List[NodeElement3D] = []
        self.triangle_list: List[SurfaceTriangle] = []

        self.node_coords: List[Tuple[float, float, float]] = []
        self.node_dict: Dict[int, NodeElement3D] = {}
        self.node_hash_set: Set[int] = set()
        self.node_hash_map: Dict[int, NodeElement3D] = {}  # hash → node
        self._used_node_idx: Set[int] = set()  # 已使用的 node.idx 集合
        self.triangle_set: Set[frozenset] = set()

        self.space_index_node = None
        self.space_index_front = None
        self.space_index_triangle = None
        self._triangle_dict: Dict[int, SurfaceTriangle] = {}
        self.edge_count: Dict[frozenset, int] = {}
        self._init_boundary_hashes: Set[int] = set()  # 初始化时的边界节点 hash（永不更新）

        self.num_nodes = 0
        self.num_triangles = 0

        self.search_radius_factor = 3.0
        self.quality_discount = 0.8

        # 曲面参数域边界（用于 UV 越界检测）
        self._surface_bounds = self._get_surface_bounds()

        self._initialize()

    def _initialize(self):
        """初始化网格生成器（3D AFM）"""
        info("初始化曲面网格生成器...")

        if self._line_mesh is not None:
            from .surface_front import create_fronts_from_line_mesh
            self.front_list = create_fronts_from_line_mesh(
                self.surface, self.geometry, self._line_mesh
            )
        else:
            self.front_list = create_initial_fronts_from_surface(
                self.surface,
                self.geometry,
                self.sizing_field
            )

        heapq.heapify(self.front_list)

        # 注册边界阵面到尺寸场（用于边界驱动尺寸场）
        self.sizing_field.register_boundary_fronts(self.front_list)

        for front in self.front_list:
            for node in front.node_elems:
                if node.hash not in self.node_hash_set:
                    self.node_hash_set.add(node.hash)
                    self.node_list.append(node)
                    self.node_coords.append(node.coords)
                    self.node_dict[node.idx] = node
                    self.node_hash_map[node.hash] = node
                    self._used_node_idx.add(node.idx)
                    self.num_nodes = max(self.num_nodes, node.idx + 1)

            # 初始化边计数：每个初始阵面对应一条边界边
            n0h = front.node_elems[0].hash
            n1h = front.node_elems[1].hash
            eh = frozenset([n0h, n1h])
            self.edge_count[eh] = self.edge_count.get(eh, 0) + 1
            # 记录初始边界节点（此时 edge_count 刚初始化，count=1 的边即为边界边）
            self._init_boundary_hashes.add(n0h)
            self._init_boundary_hashes.add(n1h)

        self._build_space_index()

        # 基于曲率的最大边长限制：防止跨越高曲率区域的大三角形
        # 对于球面，限制边长使平面三角形偏离曲面 < 0.15
        self._max_edge_len = self._compute_max_edge_len()

        info(f"初始阵面数量: {len(self.front_list)}")
        info(f"初始节点数量: {len(self.node_list)}")
        info(f"最大边长限制: {self._max_edge_len:.4f}")

    def _compute_max_edge_len(self) -> float:
        """
        基于曲面最大曲率计算最大允许边长。

        对于半径 R 的球面，边长 L 的平面三角形偏离曲面约:
            δ ≈ R * (1 - sqrt(1 - (L/2R)²))
        限制 δ < max_deviation → L < 2R * sqrt(1-(1-δ/R)²)

        Returns:
            最大允许边长
        """
        max_deviation = 0.15  # 最大允许偏离量
        max_edge_by_spacing = self.sizing_field.global_spacing * 3.0

        try:
            # 在参数域中心采样最大曲率
            u_min, u_max, v_min, v_max = self._get_surface_bounds()
            u_mid = (u_min + u_max) / 2
            v_mid = (v_min + v_max) / 2
            _, _, max_curv = self.geometry.get_surface_curvature(
                u_mid, v_mid, self.surface
            )
            if max_curv > 1e-12:
                radius = 1.0 / max_curv
                if radius > max_deviation:
                    max_edge_curv = 2.0 * radius * math.sqrt(
                        1.0 - (1.0 - max_deviation / radius) ** 2
                    )
                    return min(max_edge_curv, max_edge_by_spacing)
        except Exception:
            pass

        return max_edge_by_spacing

    def _get_surface_bounds(self):
        """获取曲面参数域边界"""
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        adaptor = BRepAdaptor_Surface(self.surface)
        return (
            adaptor.FirstUParameter(),
            adaptor.LastUParameter(),
            adaptor.FirstVParameter(),
            adaptor.LastVParameter(),
        )

    def _uv_out_of_bounds(self, uv, margin=0.1):
        """检查 UV 是否超出面的参数域"""
        u_min, u_max, v_min, v_max = self._surface_bounds
        u_range = u_max - u_min
        v_range = v_max - v_min
        u_margin = max(u_range * margin, 1e-6)
        v_margin = max(v_range * margin, 1e-6)
        u, v = uv
        return (u < u_min - u_margin or u > u_max + u_margin or
                v < v_min - v_margin or v > v_max + v_margin)

    def _build_space_index(self):
        """构建空间索引"""
        if self.node_list:
            self.node_dict, self.space_index_node = build_space_index_3d_with_RTree(
                self.node_list
            )

        if self.front_list:
            _, self.space_index_front = build_space_index_3d_with_RTree(
                list(self.front_list)
            )

        if self.triangle_list:
            self._triangle_dict, self.space_index_triangle = build_space_index_3d_with_RTree(
                self.triangle_list
            )

    def _trace_boundary_loop(self) -> List[int]:
        """
        从 edge_count 中追踪一个边界闭环，返回有序节点 hash 列表。

        Returns:
            边界节点 hash 的有序列表（闭环，首尾不重复），
            如果无法构成闭环则返回空列表。
        """
        adj: Dict[int, List[int]] = {}
        for eh, cnt in self.edge_count.items():
            if cnt == 1:
                n0, n1 = tuple(eh)
                adj.setdefault(n0, []).append(n1)
                adj.setdefault(n1, []).append(n0)

        start = None
        for n, neighbors in adj.items():
            if len(neighbors) == 2:
                start = n
                break
        if start is None:
            return []

        loop = [start]
        prev = None
        current = start
        max_steps = len(adj) + 1
        for _ in range(max_steps):
            neighbors = adj.get(current, [])
            nexts = [n for n in neighbors if n != prev]
            if not nexts:
                break
            next_node = nexts[0]
            if next_node == start:
                break
            loop.append(next_node)
            prev, current = current, next_node

        return loop

    def _trace_all_boundary_loops(self) -> List[List[int]]:
        """
        追踪所有边界闭环。

        Returns:
            所有边界闭环的列表，每个闭环是有序节点 hash 列表。
        """
        adj: Dict[int, List[int]] = {}
        for eh, cnt in self.edge_count.items():
            if cnt == 1:
                n0, n1 = tuple(eh)
                adj.setdefault(n0, []).append(n1)
                adj.setdefault(n1, []).append(n0)

        visited: Set[int] = set()
        loops = []

        for start in adj:
            if start in visited:
                continue
            # 从 start 开始追踪闭环
            loop = [start]
            visited.add(start)
            prev = None
            current = start
            max_steps = len(adj) + 1
            for _ in range(max_steps):
                neighbors = adj.get(current, [])
                nexts = [n for n in neighbors if n != prev]
                if not nexts:
                    break
                next_node = nexts[0]
                if next_node == start:
                    break
                if next_node in visited:
                    break
                loop.append(next_node)
                visited.add(next_node)
                prev, current = current, next_node
            loops.append(loop)

        return loops

    def _process_boundary_loop(self) -> int:
        """
        边界闭环塌缩：遍历所有边界环，检测角点并创建三角形，将边界逐层向内推进。

        逐环处理：对每个边界环尝试耳朵塌缩，失败则跳过该环处理下一个。
        使用每节点局部曲面法向判断绕序。
        增加边饱和检查和相交检查，防止非流形和自交。

        Returns:
            创建的三角形数量
        """
        created = 0
        max_outer = 200

        for _ in range(max_outer):
            loops = self._trace_all_boundary_loops()
            valid_loops = [lp for lp in loops if len(lp) >= 3]
            if not valid_loops:
                break

            any_success = False
            for loop in valid_loops:
                n_loop = len(loop)

                # 对于长环，使用更严格的角点阈值
                corner_threshold = 0.3 if n_loop > 20 else 0.5
                nodes_to_collapse = []

                for i in range(n_loop):
                    node_h = loop[i]
                    prev_h = loop[(i - 1) % n_loop]
                    next_h = loop[(i + 1) % n_loop]

                    tri_key = frozenset([prev_h, node_h, next_h])
                    if tri_key in self.triangle_set:
                        continue

                    if (prev_h not in self.node_hash_map or
                        node_h not in self.node_hash_map or
                        next_h not in self.node_hash_map):
                        continue

                    p_prev = np.array(self.node_hash_map[prev_h].coords)
                    p_node = np.array(self.node_hash_map[node_h].coords)
                    p_next = np.array(self.node_hash_map[next_h].coords)

                    e1 = p_node - p_prev
                    e2 = p_next - p_node
                    cross = np.cross(e1, e2)
                    cross_mag = np.linalg.norm(cross)
                    e1_len = np.linalg.norm(e1)
                    e2_len = np.linalg.norm(e2)

                    if e1_len < 1e-12 or e2_len < 1e-12:
                        continue

                    edge_len_max = max(e1_len, e2_len)
                    normalized_cross = cross_mag / (edge_len_max * edge_len_max)

                    if normalized_cross > corner_threshold:
                        nodes_to_collapse.append(i)

                # 小闭环松弛：≤6 节点时尝试所有节点作为塌缩候选
                if not nodes_to_collapse and n_loop <= 6:
                    for i in range(n_loop):
                        node_h = loop[i]
                        prev_h = loop[(i - 1) % n_loop]
                        next_h = loop[(i + 1) % n_loop]
                        if (prev_h not in self.node_hash_map or
                            node_h not in self.node_hash_map or
                            next_h not in self.node_hash_map):
                            continue
                        tri_key = frozenset([prev_h, node_h, next_h])
                        if tri_key not in self.triangle_set:
                            nodes_to_collapse.append(i)

                if not nodes_to_collapse:
                    continue

                # 尝试所有候选节点，找到第一个可以成功塌缩的
                for i in nodes_to_collapse:
                    node_h = loop[i]
                    prev_h = loop[(i - 1) % n_loop]
                    next_h = loop[(i + 1) % n_loop]

                    tri_key = frozenset([prev_h, node_h, next_h])
                    if tri_key in self.triangle_set:
                        continue

                    prev_node = self.node_hash_map.get(prev_h)
                    node_node = self.node_hash_map.get(node_h)
                    next_node = self.node_hash_map.get(next_h)
                    if not prev_node or not node_node or not next_node:
                        continue

                    if prev_h == node_h or prev_h == next_h or node_h == next_h:
                        continue

                    # 三角形大小检查：防止创建过大的三角形
                    p_prev = np.array(prev_node.coords)
                    p_node = np.array(node_node.coords)
                    p_next = np.array(next_node.coords)
                    edge1_len = np.linalg.norm(p_node - p_prev)
                    edge2_len = np.linalg.norm(p_next - p_node)
                    edge3_len = np.linalg.norm(p_prev - p_next)
                    max_edge = max(edge1_len, edge2_len, edge3_len)
                    if max_edge > self._max_edge_len * 2.0:
                        continue

                    # 边饱和检查
                    skip = False
                    for eh in [frozenset([prev_h, node_h]),
                               frozenset([node_h, next_h]),
                               frozenset([prev_h, next_h])]:
                        if self.edge_count.get(eh, 0) >= 2:
                            skip = True
                            break
                    if skip:
                        continue

                    # 相交检查
                    if self._ear_penetrates_existing(prev_node, node_node, next_node):
                        continue
                    if self._ear_too_close(prev_node, node_node, next_node,
                                            min_dist=self.sizing_field.global_spacing * 0.08):
                        continue

                    # 使用局部法向判断绕序
                    n_prev = self._get_local_surface_normal(prev_node)
                    n_node = self._get_local_surface_normal(node_node)
                    n_next = self._get_local_surface_normal(next_node)
                    avg_normal = (n_prev + n_node + n_next) / 3.0
                    an_len = np.linalg.norm(avg_normal)
                    if an_len > 1e-12:
                        avg_normal /= an_len

                    tri_normal = np.cross(p_next - p_prev, p_node - p_prev)

                    if np.dot(tri_normal, avg_normal) >= 0:
                        tri = SurfaceTriangle(prev_node, next_node, node_node,
                                              surface=self.surface, idx=self.num_triangles)
                    else:
                        tri = SurfaceTriangle(prev_node, node_node, next_node,
                                              surface=self.surface, idx=self.num_triangles)

                    self.triangle_list.append(tri)
                    self.triangle_set.add(tri_key)
                    self.num_triangles += 1
                    created += 1
                    any_success = True

                    for eh in [frozenset([prev_h, node_h]),
                               frozenset([node_h, next_h]),
                               frozenset([prev_h, next_h])]:
                        self.edge_count[eh] = self.edge_count.get(eh, 0) + 1

                    if self.space_index_triangle is None:
                        self._triangle_dict, self.space_index_triangle = build_space_index_3d_with_RTree([tri])
                    else:
                        self.space_index_triangle, self._triangle_dict = add_elems_to_space_index_3d_with_RTree(
                            [tri], self.space_index_triangle, self._triangle_dict,
                        )

                    break  # 成功创建三角形，跳出当前环的候选循环，重新追踪所有环

                if any_success:
                    break  # 成功创建一个三角形，重新追踪所有环

            if not any_success:
                break  # 所有环的所有候选都失败了

        return created

    def _refine_boundary_loop(self, max_edge_ratio: float = 1.2) -> int:
        """
        细化边界环：将过长的边界边拆分为更短的边。

        在边界塌缩之前调用，确保边界边长度不超过 max_edge_ratio * global_spacing。
        新节点投影到曲面上，并加入节点和边界数据结构。

        Args:
            max_edge_ratio: 最大边长与 global_spacing 的比值

        Returns:
            新增节点数量
        """
        max_edge_len = self.sizing_field.global_spacing * max_edge_ratio
        added = 0
        max_rounds = 10

        for _ in range(max_rounds):
            loop = self._trace_boundary_loop()
            n_loop = len(loop)
            if n_loop < 3:
                break

            split_done = False
            for i in range(n_loop):
                h0 = loop[i]
                h1 = loop[(i + 1) % n_loop]
                if h0 not in self.node_hash_map or h1 not in self.node_hash_map:
                    continue

                p0 = np.array(self.node_hash_map[h0].coords)
                p1 = np.array(self.node_hash_map[h1].coords)
                edge_len = np.linalg.norm(p1 - p0)

                if edge_len <= max_edge_len:
                    continue

                # 在边中点插入新节点
                mid = (p0 + p1) / 2.0
                try:
                    uv = self.geometry.project_point_to_surface(tuple(mid), self.surface)
                    mid_coords = self.geometry.evaluate_point(uv[0], uv[1], self.surface)
                    normal = self.geometry.get_surface_normal(uv[0], uv[1], self.surface)
                except Exception:
                    continue

                mid_node = NodeElement3D(
                    coords=mid_coords,
                    idx=self.num_nodes,
                    surface=self.surface,
                    uv_params=uv,
                    normal=normal
                )

                # 注册新节点
                self.node_hash_set.add(mid_node.hash)
                self.node_list.append(mid_node)
                self.node_coords.append(mid_node.coords)
                self.node_dict[mid_node.idx] = mid_node
                self.node_hash_map[mid_node.hash] = mid_node
                self._used_node_idx.add(mid_node.idx)
                self.num_nodes += 1

                # 更新边计数：移除旧边 h0-h1，添加 h0-mid 和 mid-h1
                old_edge = frozenset([h0, h1])
                old_count = self.edge_count.get(old_edge, 0)
                if old_count > 1:
                    self.edge_count[old_edge] = old_count - 1
                else:
                    self.edge_count.pop(old_edge, None)

                self.edge_count[frozenset([h0, mid_node.hash])] = 1
                self.edge_count[frozenset([mid_node.hash, h1])] = 1

                added += 1
                split_done = True
                break  # 重新追踪边界环

            if not split_done:
                break

        if added > 0:
            info(f"边界环细化: 新增 {added} 个节点")
        return added

    def _ear_penetrates_existing(self, n0, n1, n2) -> bool:
        """
        检查耳朵三角形是否穿透已有三角形（节点包含检查）。

        与 _triangle_intersects_existing_loose 的区别：
        - 只检查节点包含，不检查边相交
        - 跳过共享节点的三角形（它们是相邻的，不是穿透）
        - 用于边界闭合，允许曲面上的三角形接近

        Returns:
            True 表示穿透（应拒绝）
        """
        if self.space_index_triangle is None or not self.triangle_list:
            return False
        new_hashes = {n0.hash, n1.hash, n2.hash}
        new_coords = np.array([n0.coords, n1.coords, n2.coords])
        padding = self.sizing_field.global_spacing * 0.5
        bbox = (
            new_coords[:, 0].min() - padding, new_coords[:, 1].min() - padding, new_coords[:, 2].min() - padding,
            new_coords[:, 0].max() + padding, new_coords[:, 1].max() + padding, new_coords[:, 2].max() + padding,
        )

        for tri_id in self.space_index_triangle.intersection(bbox):
            if tri_id not in self._triangle_dict:
                continue
            existing = self._triangle_dict[tri_id]
            ex_hash_list = [nd.hash for nd in existing.nodes]
            shared = set(ex_hash_list) & new_hashes
            # 跳过共享节点的三角形
            if shared:
                continue
            ex_coords = np.array([nd.coords for nd in existing.nodes])
            # 检查新三角形的节点是否在已有三角形内部
            for pt in new_coords:
                if point_in_triangle_3d(pt, ex_coords[0], ex_coords[1], ex_coords[2]):
                    return True
            # 检查已有三角形的节点是否在新三角形内部
            for pt in ex_coords:
                if point_in_triangle_3d(pt, new_coords[0], new_coords[1], new_coords[2]):
                    return True
        return False

    def _ear_too_close(self, n0, n1, n2, min_dist: float) -> bool:
        """
        检查耳朵三角形与已有三角形的最小边距离是否过小。

        与 _check_min_edge_distance 类似，但跳过共享节点的三角形。

        Returns:
            True 表示距离过小（应拒绝）
        """
        if self.space_index_triangle is None or not self.triangle_list:
            return False
        new_hashes = {n0.hash, n1.hash, n2.hash}
        new_coords = np.array([n0.coords, n1.coords, n2.coords])
        padding = self.sizing_field.global_spacing * 0.5
        bbox = (
            new_coords[:, 0].min() - padding, new_coords[:, 1].min() - padding, new_coords[:, 2].min() - padding,
            new_coords[:, 0].max() + padding, new_coords[:, 1].max() + padding, new_coords[:, 2].max() + padding,
        )

        edges_new = [
            (new_coords[0], new_coords[1]),
            (new_coords[1], new_coords[2]),
            (new_coords[2], new_coords[0])
        ]

        for tri_id in self.space_index_triangle.intersection(bbox):
            if tri_id not in self._triangle_dict:
                continue
            existing = self._triangle_dict[tri_id]
            ex_hash_list = [nd.hash for nd in existing.nodes]
            shared = set(ex_hash_list) & new_hashes
            if shared:
                continue
            ex_coords = np.array([nd.coords for nd in existing.nodes])
            edges_ex = [
                (ex_coords[0], ex_coords[1]),
                (ex_coords[1], ex_coords[2]),
                (ex_coords[2], ex_coords[0])
            ]
            for a1, a2 in edges_new:
                for b1, b2 in edges_ex:
                    dist = segment_segment_distance_3d(a1, a2, b1, b2)
                    if dist < min_dist:
                        return True
        return False

    def _process_boundary_loop_conservative(self) -> int:
        """
        保守的边界闭环塌缩：遍历所有边界环，尝试将每个节点作为"耳朵"塌缩。

        使用几何准则（尺寸 + 法向一致性）而非相交检测来判断耳朵有效性。
        逐环处理：对每个边界环找最佳耳朵候选，失败则跳过处理下一个环。

        Returns:
            创建的三角形数量
        """
        created = 0
        max_outer = 200

        for _ in range(max_outer):
            loops = self._trace_all_boundary_loops()
            valid_loops = [lp for lp in loops if len(lp) >= 3]
            if not valid_loops:
                break

            any_success = False
            for loop in valid_loops:
                n_loop = len(loop)

                # 尝试所有节点作为耳朵候选，按"耳朵大小"排序
                candidates = []
                for i in range(n_loop):
                    node_h = loop[i]
                    prev_h = loop[(i - 1) % n_loop]
                    next_h = loop[(i + 1) % n_loop]

                    tri_key = frozenset([prev_h, node_h, next_h])
                    if tri_key in self.triangle_set:
                        continue

                    if (prev_h not in self.node_hash_map or
                        node_h not in self.node_hash_map or
                        next_h not in self.node_hash_map):
                        continue

                    if prev_h == node_h or prev_h == next_h or node_h == next_h:
                        continue

                    prev_node = self.node_hash_map[prev_h]
                    node_node = self.node_hash_map[node_h]
                    next_node = self.node_hash_map[next_h]

                    p_prev = np.array(prev_node.coords)
                    p_node = np.array(node_node.coords)
                    p_next = np.array(next_node.coords)

                    edge1_len = np.linalg.norm(p_node - p_prev)
                    edge2_len = np.linalg.norm(p_next - p_node)
                    edge3_len = np.linalg.norm(p_prev - p_next)
                    max_edge = max(edge1_len, edge2_len, edge3_len)

                    # 三角形太大则跳过
                    if max_edge > self._max_edge_len * 1.5:
                        continue

                    # 边饱和检查
                    skip = False
                    for eh in [frozenset([prev_h, node_h]),
                               frozenset([node_h, next_h]),
                               frozenset([prev_h, next_h])]:
                        if self.edge_count.get(eh, 0) >= 2:
                            skip = True
                            break
                    if skip:
                        continue

                    # 法向一致性检查
                    n_prev = self._get_local_surface_normal(prev_node)
                    n_node = self._get_local_surface_normal(node_node)
                    n_next = self._get_local_surface_normal(next_node)
                    avg_normal = (n_prev + n_node + n_next) / 3.0
                    an_len = np.linalg.norm(avg_normal)
                    if an_len > 1e-12:
                        avg_normal /= an_len

                    tri_normal = np.cross(p_next - p_prev, p_node - p_prev)
                    tn_len = np.linalg.norm(tri_normal)
                    if tn_len < 1e-12:
                        continue

                    if np.dot(tri_normal, avg_normal) <= 0:
                        continue

                    # 三角形质量检查
                    quality = triangle_quality_from_coords(p_prev, p_node, p_next)
                    if quality < 0.1:
                        continue

                    # 节点包含检查
                    if self._ear_penetrates_existing(prev_node, node_node, next_node):
                        continue

                    score = edge1_len + edge2_len + edge3_len
                    candidates.append((score, i, prev_h, node_h, next_h, tri_key))

                if not candidates:
                    continue

                # 优先塌缩小耳朵
                candidates.sort(key=lambda x: x[0])
                _, i, prev_h, node_h, next_h, tri_key = candidates[0]

                prev_node = self.node_hash_map[prev_h]
                node_node = self.node_hash_map[node_h]
                next_node = self.node_hash_map[next_h]

                # 最终边饱和检查（防止多轮处理导致的竞态）
                skip = False
                for eh in [frozenset([prev_h, node_h]),
                           frozenset([node_h, next_h]),
                           frozenset([prev_h, next_h])]:
                    if self.edge_count.get(eh, 0) >= 2:
                        skip = True
                        break
                if skip:
                    continue

                # 使用局部法向判断绕序
                n_prev = self._get_local_surface_normal(prev_node)
                n_node = self._get_local_surface_normal(node_node)
                n_next = self._get_local_surface_normal(next_node)
                avg_normal = (n_prev + n_node + n_next) / 3.0
                an_len = np.linalg.norm(avg_normal)
                if an_len > 1e-12:
                    avg_normal /= an_len

                p_prev = np.array(prev_node.coords)
                p_node = np.array(node_node.coords)
                p_next = np.array(next_node.coords)
                tri_normal = np.cross(p_next - p_prev, p_node - p_prev)

                if np.dot(tri_normal, avg_normal) >= 0:
                    tri = SurfaceTriangle(prev_node, next_node, node_node,
                                          surface=self.surface, idx=self.num_triangles)
                else:
                    tri = SurfaceTriangle(prev_node, node_node, next_node,
                                          surface=self.surface, idx=self.num_triangles)

                self.triangle_list.append(tri)
                self.triangle_set.add(tri_key)
                self.num_triangles += 1
                created += 1
                any_success = True

                for eh in [frozenset([prev_h, node_h]),
                           frozenset([node_h, next_h]),
                           frozenset([prev_h, next_h])]:
                    self.edge_count[eh] = self.edge_count.get(eh, 0) + 1

                if self.space_index_triangle is None:
                    self._triangle_dict, self.space_index_triangle = build_space_index_3d_with_RTree([tri])
                else:
                    self.space_index_triangle, self._triangle_dict = add_elems_to_space_index_3d_with_RTree(
                        [tri], self.space_index_triangle, self._triangle_dict,
                    )

                break  # 成功创建一个三角形，重新追踪所有环

            if not any_success:
                break

        return created

    def _resolve_boundary_junctions(self) -> int:
        """
        解决边界结点（度 > 2 的边界节点）：在结点处创建三角形以降低度数到 2。

        对于每个度 > 2 的边界节点，按角度排序其边界邻居，
        对每对相邻邻居尝试创建三角形（结点, 邻居i, 邻居i+1）。

        Returns:
            创建的三角形数量
        """
        created = 0
        max_rounds = 50

        for _ in range(max_rounds):
            # 构建边界邻接表
            boundary_adj: Dict[int, List[int]] = {}
            for eh, cnt in self.edge_count.items():
                if cnt == 1:
                    n0, n1 = tuple(eh)
                    boundary_adj.setdefault(n0, []).append(n1)
                    boundary_adj.setdefault(n1, []).append(n0)

            # 找到度 > 2 的结点
            junction_nodes = {n: nbrs for n, nbrs in boundary_adj.items() if len(nbrs) > 2}
            if not junction_nodes:
                break

            any_success = False
            for node_h, neighbors in junction_nodes.items():
                if node_h not in self.node_hash_map:
                    continue
                if len(neighbors) <= 2:
                    continue  # 可能被前面的处理降低了度数

                node = self.node_hash_map[node_h]
                p_node = np.array(node.coords)

                # 获取曲面法向用于排序
                normal = self._get_local_surface_normal(node)

                # 按角度排序邻居
                def _angle_to(neighbor_h):
                    if neighbor_h not in self.node_hash_map:
                        return 0.0
                    p_nbr = np.array(self.node_hash_map[neighbor_h].coords)
                    vec = p_nbr - p_node
                    # 投影到切平面
                    vec_proj = vec - np.dot(vec, normal) * normal
                    vec_len = np.linalg.norm(vec_proj)
                    if vec_len < 1e-12:
                        return 0.0
                    vec_proj /= vec_len
                    # 使用 atan2 计算角度
                    # 需要一个参考方向
                    ref = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(ref, normal)) > 0.9:
                        ref = np.array([0.0, 1.0, 0.0])
                    ref_proj = ref - np.dot(ref, normal) * normal
                    ref_len = np.linalg.norm(ref_proj)
                    if ref_len > 1e-12:
                        ref_proj /= ref_len
                    else:
                        return 0.0
                    cos_a = np.clip(np.dot(vec_proj, ref_proj), -1.0, 1.0)
                    cross = np.cross(ref_proj, vec_proj)
                    angle = math.atan2(np.dot(cross, normal), cos_a)
                    return angle

                sorted_nbrs = sorted(neighbors, key=_angle_to)
                n_nbrs = len(sorted_nbrs)

                # 对每对相邻邻居尝试创建三角形
                for i in range(n_nbrs):
                    if len(boundary_adj.get(node_h, [])) <= 2:
                        break  # 度数已降到 2

                    nbr1_h = sorted_nbrs[i]
                    nbr2_h = sorted_nbrs[(i + 1) % n_nbrs]

                    if nbr1_h == nbr2_h:
                        continue
                    if nbr1_h not in self.node_hash_map or nbr2_h not in self.node_hash_map:
                        continue

                    tri_key = frozenset([node_h, nbr1_h, nbr2_h])
                    if tri_key in self.triangle_set:
                        continue

                    # 边饱和检查
                    skip = False
                    for eh in [frozenset([node_h, nbr1_h]),
                               frozenset([node_h, nbr2_h]),
                               frozenset([nbr1_h, nbr2_h])]:
                        if self.edge_count.get(eh, 0) >= 2:
                            skip = True
                            break
                    if skip:
                        continue

                    nbr1 = self.node_hash_map[nbr1_h]
                    nbr2 = self.node_hash_map[nbr2_h]

                    p_nbr1 = np.array(nbr1.coords)
                    p_nbr2 = np.array(nbr2.coords)

                    # 退化检查
                    tri_normal = np.cross(p_nbr1 - p_node, p_nbr2 - p_node)
                    if np.linalg.norm(tri_normal) < 1e-12:
                        continue

                    # 法向一致性
                    n_nbr1 = self._get_local_surface_normal(nbr1)
                    n_nbr2 = self._get_local_surface_normal(nbr2)
                    avg_normal = (normal + n_nbr1 + n_nbr2) / 3.0
                    an_len = np.linalg.norm(avg_normal)
                    if an_len > 1e-12:
                        avg_normal /= an_len

                    if np.dot(tri_normal, avg_normal) <= 0:
                        continue

                    # 质量检查
                    quality = triangle_quality_from_coords(p_node, p_nbr1, p_nbr2)
                    if quality < 0.05:
                        continue

                    # 尺寸检查（结点解决允许更大的三角形）
                    max_edge = max(
                        np.linalg.norm(p_nbr1 - p_node),
                        np.linalg.norm(p_nbr2 - p_nbr1),
                        np.linalg.norm(p_node - p_nbr2),
                    )
                    if max_edge > self._max_edge_len * 3.0:
                        continue

                    # 相交检查（宽松：只检查节点穿透）
                    if self._ear_penetrates_existing(node, nbr1, nbr2):
                        continue

                    # 创建三角形
                    if np.dot(tri_normal, avg_normal) >= 0:
                        tri = SurfaceTriangle(node, nbr2, nbr1,
                                              surface=self.surface, idx=self.num_triangles)
                    else:
                        tri = SurfaceTriangle(node, nbr1, nbr2,
                                              surface=self.surface, idx=self.num_triangles)

                    self.triangle_list.append(tri)
                    self.triangle_set.add(tri_key)
                    self.num_triangles += 1
                    created += 1
                    any_success = True

                    for eh in [frozenset([node_h, nbr1_h]),
                               frozenset([node_h, nbr2_h]),
                               frozenset([nbr1_h, nbr2_h])]:
                        self.edge_count[eh] = self.edge_count.get(eh, 0) + 1

                    if self.space_index_triangle is None:
                        self._triangle_dict, self.space_index_triangle = build_space_index_3d_with_RTree([tri])
                    else:
                        self.space_index_triangle, self._triangle_dict = add_elems_to_space_index_3d_with_RTree(
                            [tri], self.space_index_triangle, self._triangle_dict,
                        )

                    break  # 成功创建一个三角形，重新追踪边界

                if any_success:
                    break  # 重新追踪所有边界

            if not any_success:
                break

        return created

    def _close_remaining_triangles(self) -> int:
        """闭合所有剩余边界闭环：
        - 移除退化环（1-2 节点，无法形成三角形）
        - 闭合所有 3 节点环
        用局部法向定绕序，含相交检查。
        """
        loops = self._trace_all_boundary_loops()
        if not loops:
            return 0

        created = 0

        # 1. 清理退化环（1-2 节点）：移除其边界边
        for loop in loops:
            if len(loop) >= 3:
                continue
            for i in range(len(loop)):
                h0 = loop[i]
                h1 = loop[(i + 1) % len(loop)]
                eh = frozenset([h0, h1])
                cnt = self.edge_count.get(eh, 0)
                if cnt <= 1:
                    self.edge_count.pop(eh, None)
                else:
                    self.edge_count[eh] = cnt - 1

        # 2. 闭合所有 3 节点环
        for loop in loops:
            if len(loop) != 3:
                continue

            a_h, b_h, c_h = loop
            if (a_h not in self.node_hash_map or b_h not in self.node_hash_map or
                    c_h not in self.node_hash_map):
                continue

            tri_key = frozenset([a_h, b_h, c_h])
            if tri_key in self.triangle_set:
                continue

            a_node = self.node_hash_map[a_h]
            b_node = self.node_hash_map[b_h]
            c_node = self.node_hash_map[c_h]

            # 边饱和检查
            skip = False
            for eh in [frozenset([a_h, b_h]), frozenset([b_h, c_h]), frozenset([c_h, a_h])]:
                if self.edge_count.get(eh, 0) >= 2:
                    skip = True
                    break
            if skip:
                continue

            # 相交检查（闭合三角形时不检查最小距离，允许边界闭合）
            if self._triangle_intersects_existing(a_node, b_node, c_node, check_min_dist=False):
                continue

            # 使用局部法向
            n_a = self._get_local_surface_normal(a_node)
            n_b = self._get_local_surface_normal(b_node)
            n_c = self._get_local_surface_normal(c_node)
            avg_normal = (n_a + n_b + n_c) / 3.0
            an_len = np.linalg.norm(avg_normal)
            if an_len > 1e-12:
                avg_normal /= an_len

            p_a = np.array(a_node.coords)
            p_b = np.array(b_node.coords)
            p_c = np.array(c_node.coords)
            tri_normal = np.cross(p_b - p_a, p_c - p_a)

            if np.dot(tri_normal, avg_normal) >= 0:
                tri = SurfaceTriangle(a_node, c_node, b_node,
                                      surface=self.surface, idx=self.num_triangles)
            else:
                tri = SurfaceTriangle(a_node, b_node, c_node,
                                      surface=self.surface, idx=self.num_triangles)

            self.triangle_list.append(tri)
            self.triangle_set.add(tri_key)
            self.num_triangles += 1
            created += 1

            for eh in [frozenset([a_h, b_h]), frozenset([b_h, c_h]), frozenset([c_h, a_h])]:
                self.edge_count[eh] = self.edge_count.get(eh, 0) + 1

            if self.space_index_triangle is None:
                self._triangle_dict, self.space_index_triangle = build_space_index_3d_with_RTree([tri])
            else:
                self.space_index_triangle, self._triangle_dict = add_elems_to_space_index_3d_with_RTree(
                    [tri], self.space_index_triangle, self._triangle_dict,
                )

        return created

    def _bridge_boundary_gaps(self) -> int:
        """
        桥接边界间隙：找到不同边界环上的近邻节点，创建三角形连接它们。

        策略：
        1. 追踪所有边界环
        2. 对每对环，找到最近的节点对
        3. 如果距离在阈值内，尝试创建三角形连接

        Returns:
            创建的三角形数量
        """
        created = 0
        max_rounds = 100
        bridge_dist = self._max_edge_len * 1.5  # 桥接距离阈值

        for _ in range(max_rounds):
            loops = self._trace_all_boundary_loops()
            valid_loops = [lp for lp in loops if len(lp) >= 3]
            if len(valid_loops) < 2:
                break

            any_success = False

            # 对每对环找最近节点对
            best_pair = None
            best_dist = bridge_dist

            for i in range(len(valid_loops)):
                for j in range(i + 1, len(valid_loops)):
                    loop_a = valid_loops[i]
                    loop_b = valid_loops[j]

                    for ha in loop_a:
                        if ha not in self.node_hash_map:
                            continue
                        pa = np.array(self.node_hash_map[ha].coords)
                        for hb in loop_b:
                            if hb not in self.node_hash_map:
                                continue
                            pb = np.array(self.node_hash_map[hb].coords)
                            d = np.linalg.norm(pb - pa)
                            if d < best_dist:
                                best_dist = d
                                best_pair = (ha, hb)

                    if best_dist < self.sizing_field.global_spacing * 0.5:
                        break  # 足够近了，不用继续搜索
                if best_dist < self.sizing_field.global_spacing * 0.5:
                    break

            if best_pair is None:
                break

            ha, hb = best_pair
            node_a = self.node_hash_map[ha]
            node_b = self.node_hash_map[hb]
            pa = np.array(node_a.coords)
            pb = np.array(node_b.coords)

            # 找一个共享邻居节点来形成三角形
            # 优先找 ha 的边界邻居
            boundary_adj: Dict[int, List[int]] = {}
            for eh, cnt in self.edge_count.items():
                if cnt == 1:
                    n0, n1 = tuple(eh)
                    boundary_adj.setdefault(n0, []).append(n1)
                    boundary_adj.setdefault(n1, []).append(n0)

            best_third = None
            best_quality = 0.0

            # 尝试 ha 的邻居
            for hc in boundary_adj.get(ha, []):
                if hc == hb or hc not in self.node_hash_map:
                    continue
                node_c = self.node_hash_map[hc]
                pc = np.array(node_c.coords)

                tri_key = frozenset([ha, hb, hc])
                if tri_key in self.triangle_set:
                    continue

                # 边饱和检查
                skip = False
                for eh in [frozenset([ha, hb]), frozenset([ha, hc]), frozenset([hb, hc])]:
                    if self.edge_count.get(eh, 0) >= 2:
                        skip = True
                        break
                if skip:
                    continue

                if not self._can_create_triangle(node_a, node_b, node_c):
                    continue

                quality = triangle_quality_from_coords(pa, pb, pc)
                if quality > best_quality:
                    best_quality = quality
                    best_third = hc

            # 尝试 hb 的邻居
            for hc in boundary_adj.get(hb, []):
                if hc == ha or hc not in self.node_hash_map:
                    continue
                node_c = self.node_hash_map[hc]
                pc = np.array(node_c.coords)

                tri_key = frozenset([ha, hb, hc])
                if tri_key in self.triangle_set:
                    continue

                skip = False
                for eh in [frozenset([ha, hb]), frozenset([ha, hc]), frozenset([hb, hc])]:
                    if self.edge_count.get(eh, 0) >= 2:
                        skip = True
                        break
                if skip:
                    continue

                if not self._can_create_triangle(node_a, node_b, node_c):
                    continue

                quality = triangle_quality_from_coords(pa, pb, pc)
                if quality > best_quality:
                    best_quality = quality
                    best_third = hc

            if best_third is not None and best_quality > 0.1:
                node_c = self.node_hash_map[best_third]
                if self._create_triangle_from_nodes(node_a, node_b, node_c):
                    created += 1
                    any_success = True

            if not any_success:
                break

        return created

    def _greedy_boundary_closure(self) -> int:
        """
        贪心边界闭合：直接处理边界边，不追踪环。

        对每个边界节点，尝试与其两个边界邻居形成三角形。
        优先闭合"耳朵"（第三个边也是边界边），然后处理非相邻邻居。

        Returns:
            创建的三角形数量
        """
        created = 0
        max_rounds = 500

        for _ in range(max_rounds):
            # 构建边界邻接表
            boundary_adj: Dict[int, List[int]] = {}
            for eh, cnt in self.edge_count.items():
                if cnt == 1:
                    n0, n1 = tuple(eh)
                    boundary_adj.setdefault(n0, []).append(n1)
                    boundary_adj.setdefault(n1, []).append(n0)

            if not boundary_adj:
                break

            any_success = False

            # 收集所有可能的耳朵候选
            ear_candidates = []
            gap_candidates = []

            for node_h, neighbors in boundary_adj.items():
                if len(neighbors) < 2:
                    continue
                n_nbrs = len(neighbors)
                for i in range(n_nbrs):
                    for j in range(i + 1, n_nbrs):
                        nbr1_h = neighbors[i]
                        nbr2_h = neighbors[j]

                        tri_key = frozenset([node_h, nbr1_h, nbr2_h])
                        if tri_key in self.triangle_set:
                            continue

                        # 边饱和检查
                        skip = False
                        for eh in [frozenset([node_h, nbr1_h]),
                                   frozenset([node_h, nbr2_h]),
                                   frozenset([nbr1_h, nbr2_h])]:
                            if self.edge_count.get(eh, 0) >= 2:
                                skip = True
                                break
                        if skip:
                            continue

                        if node_h not in self.node_hash_map or nbr1_h not in self.node_hash_map or nbr2_h not in self.node_hash_map:
                            continue

                        node = self.node_hash_map[node_h]
                        nbr1 = self.node_hash_map[nbr1_h]
                        nbr2 = self.node_hash_map[nbr2_h]

                        if not self._can_create_triangle(node, nbr1, nbr2):
                            continue

                        # 计算三角形周长作为排序依据
                        p0 = np.array(node.coords)
                        p1 = np.array(nbr1.coords)
                        p2 = np.array(nbr2.coords)
                        perimeter = (np.linalg.norm(p1 - p0) +
                                     np.linalg.norm(p2 - p1) +
                                     np.linalg.norm(p0 - p2))

                        # 检查是否是耳朵（第三个边也是边界边）
                        edge_12 = frozenset([nbr1_h, nbr2_h])
                        is_ear = self.edge_count.get(edge_12, 0) == 1

                        if is_ear:
                            ear_candidates.append((perimeter, node_h, nbr1_h, nbr2_h))
                        else:
                            gap_candidates.append((perimeter, node_h, nbr1_h, nbr2_h))

            # 优先闭合耳朵（消除整个环）
            if ear_candidates:
                ear_candidates.sort(key=lambda x: x[0])
                _, node_h, nbr1_h, nbr2_h = ear_candidates[0]
                node = self.node_hash_map[node_h]
                nbr1 = self.node_hash_map[nbr1_h]
                nbr2 = self.node_hash_map[nbr2_h]
                if self._create_triangle_from_nodes(node, nbr1, nbr2):
                    created += 1
                    any_success = True

            # 如果没有耳朵，尝试闭合间隙
            if not any_success and gap_candidates:
                gap_candidates.sort(key=lambda x: x[0])
                _, node_h, nbr1_h, nbr2_h = gap_candidates[0]
                node = self.node_hash_map[node_h]
                nbr1 = self.node_hash_map[nbr1_h]
                nbr2 = self.node_hash_map[nbr2_h]
                if self._create_triangle_from_nodes(node, nbr1, nbr2):
                    created += 1
                    any_success = True

            if not any_success:
                break

        return created

    def _close_boundary_at_junctions(self) -> int:
        """
        在边界结点处闭合三角形：处理度 > 2 的边界节点。

        策略：
        1. 优先闭合完整 3 边环（3 条边都是 count==1），消除整个环
        2. 对于非相邻邻居，创建三角形以降低结点度数

        Returns:
            创建的三角形数量
        """
        created = 0
        max_rounds = 200

        for _ in range(max_rounds):
            # 构建边界邻接表
            boundary_adj: Dict[int, List[int]] = {}
            for eh, cnt in self.edge_count.items():
                if cnt == 1:
                    n0, n1 = tuple(eh)
                    boundary_adj.setdefault(n0, []).append(n1)
                    boundary_adj.setdefault(n1, []).append(n0)

            # 找到度 > 2 的结点
            junction_nodes = {n: nbrs for n, nbrs in boundary_adj.items() if len(nbrs) > 2}
            if not junction_nodes:
                break

            any_success = False

            # 策略1: 优先闭合完整 3 边环
            for node_h, neighbors in junction_nodes.items():
                if len(neighbors) < 2:
                    continue
                n_nbrs = len(neighbors)
                for i in range(n_nbrs):
                    for j in range(i + 1, n_nbrs):
                        nbr1_h = neighbors[i]
                        nbr2_h = neighbors[j]
                        edge_12 = frozenset([nbr1_h, nbr2_h])
                        if self.edge_count.get(edge_12, 0) != 1:
                            continue

                        tri_key = frozenset([node_h, nbr1_h, nbr2_h])
                        if tri_key in self.triangle_set:
                            continue

                        if node_h not in self.node_hash_map or nbr1_h not in self.node_hash_map or nbr2_h not in self.node_hash_map:
                            continue

                        node = self.node_hash_map[node_h]
                        nbr1 = self.node_hash_map[nbr1_h]
                        nbr2 = self.node_hash_map[nbr2_h]

                        if not self._can_create_triangle(node, nbr1, nbr2):
                            continue

                        if self._create_triangle_from_nodes(node, nbr1, nbr2):
                            created += 1
                            any_success = True
                            break
                    if any_success:
                        break
                if any_success:
                    break

            if any_success:
                continue

            # 策略2: 创建三角形降低结点度数（非相邻邻居）
            for node_h, neighbors in junction_nodes.items():
                if len(neighbors) <= 2:
                    continue
                if node_h not in self.node_hash_map:
                    continue

                node = self.node_hash_map[node_h]
                p_node = np.array(node.coords)
                normal = self._get_local_surface_normal(node)

                # 按角度排序邻居
                def _angle_to(neighbor_h):
                    if neighbor_h not in self.node_hash_map:
                        return 0.0
                    p_nbr = np.array(self.node_hash_map[neighbor_h].coords)
                    vec = p_nbr - p_node
                    vec_proj = vec - np.dot(vec, normal) * normal
                    vec_len = np.linalg.norm(vec_proj)
                    if vec_len < 1e-12:
                        return 0.0
                    vec_proj /= vec_len
                    ref = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(ref, normal)) > 0.9:
                        ref = np.array([0.0, 1.0, 0.0])
                    ref_proj = ref - np.dot(ref, normal) * normal
                    ref_len = np.linalg.norm(ref_proj)
                    if ref_len > 1e-12:
                        ref_proj /= ref_len
                    else:
                        return 0.0
                    cos_a = np.clip(np.dot(vec_proj, ref_proj), -1.0, 1.0)
                    cross = np.cross(ref_proj, vec_proj)
                    angle = math.atan2(np.dot(cross, normal), cos_a)
                    return angle

                sorted_nbrs = sorted(neighbors, key=_angle_to)
                n_nbrs = len(sorted_nbrs)

                for i in range(n_nbrs):
                    nbr1_h = sorted_nbrs[i]
                    nbr2_h = sorted_nbrs[(i + 1) % n_nbrs]
                    if nbr1_h == nbr2_h:
                        continue
                    if nbr1_h not in self.node_hash_map or nbr2_h not in self.node_hash_map:
                        continue

                    tri_key = frozenset([node_h, nbr1_h, nbr2_h])
                    if tri_key in self.triangle_set:
                        continue

                    # 边饱和检查
                    skip = False
                    for eh in [frozenset([node_h, nbr1_h]),
                               frozenset([node_h, nbr2_h]),
                               frozenset([nbr1_h, nbr2_h])]:
                        if self.edge_count.get(eh, 0) >= 2:
                            skip = True
                            break
                    if skip:
                        continue

                    nbr1 = self.node_hash_map[nbr1_h]
                    nbr2 = self.node_hash_map[nbr2_h]

                    if not self._can_create_triangle(node, nbr1, nbr2):
                        continue

                    if self._create_triangle_from_nodes(node, nbr1, nbr2):
                        created += 1
                        any_success = True
                        break

                if any_success:
                    break

            if not any_success:
                break

        return created

    def _can_create_triangle(self, n0: NodeElement3D, n1: NodeElement3D, n2: NodeElement3D,
                             quality_threshold: float = 0.05,
                             max_edge_multiplier: float = 3.0) -> bool:
        """检查三个节点是否可以形成有效的三角形"""
        p0 = np.array(n0.coords)
        p1 = np.array(n1.coords)
        p2 = np.array(n2.coords)

        tri_normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(tri_normal) < 1e-12:
            return False

        n0_normal = self._get_local_surface_normal(n0)
        n1_normal = self._get_local_surface_normal(n1)
        n2_normal = self._get_local_surface_normal(n2)
        avg_normal = (n0_normal + n1_normal + n2_normal) / 3.0
        an_len = np.linalg.norm(avg_normal)
        if an_len > 1e-12:
            avg_normal /= an_len

        if np.dot(tri_normal, avg_normal) <= 0:
            return False

        quality = triangle_quality_from_coords(p0, p1, p2)
        if quality < quality_threshold:
            return False

        max_edge = max(
            np.linalg.norm(p1 - p0),
            np.linalg.norm(p2 - p1),
            np.linalg.norm(p0 - p2),
        )
        if max_edge > self._max_edge_len * max_edge_multiplier:
            return False

        if self._ear_penetrates_existing(n0, n1, n2):
            return False

        return True

    def _create_triangle_from_nodes(self, n0: NodeElement3D, n1: NodeElement3D, n2: NodeElement3D) -> bool:
        """从三个节点创建三角形并更新数据结构"""
        n0h = n0.hash
        n1h = n1.hash
        n2h = n2.hash

        tri_key = frozenset([n0h, n1h, n2h])
        if tri_key in self.triangle_set:
            return False

        for eh in [frozenset([n0h, n1h]), frozenset([n1h, n2h]), frozenset([n2h, n0h])]:
            if self.edge_count.get(eh, 0) >= 2:
                return False

        # 法向判断绕序
        p0 = np.array(n0.coords)
        p1 = np.array(n1.coords)
        p2 = np.array(n2.coords)
        tri_normal = np.cross(p1 - p0, p2 - p0)

        n0_normal = self._get_local_surface_normal(n0)
        n1_normal = self._get_local_surface_normal(n1)
        n2_normal = self._get_local_surface_normal(n2)
        avg_normal = (n0_normal + n1_normal + n2_normal) / 3.0
        an_len = np.linalg.norm(avg_normal)
        if an_len > 1e-12:
            avg_normal /= an_len

        if np.dot(tri_normal, avg_normal) >= 0:
            tri = SurfaceTriangle(n0, n2, n1, surface=self.surface, idx=self.num_triangles)
        else:
            tri = SurfaceTriangle(n0, n1, n2, surface=self.surface, idx=self.num_triangles)

        self.triangle_list.append(tri)
        self.triangle_set.add(tri_key)
        self.num_triangles += 1

        for eh in [frozenset([n0h, n1h]), frozenset([n1h, n2h]), frozenset([n2h, n0h])]:
            self.edge_count[eh] = self.edge_count.get(eh, 0) + 1

        if self.space_index_triangle is None:
            self._triangle_dict, self.space_index_triangle = build_space_index_3d_with_RTree([tri])
        else:
            self.space_index_triangle, self._triangle_dict = add_elems_to_space_index_3d_with_RTree(
                [tri], self.space_index_triangle, self._triangle_dict,
            )

        return True

    def _fallback_connect_boundary(self, front: SurfaceFront) -> Optional[NodeElement3D]:
        """
        回退策略：当阵面卡住时，尝试连接到最近的边界节点

        Args:
            front: 当前阵面

        Returns:
            最近的有效边界节点，如果没有合适的返回 None
        """
        n0 = front.node_elems[0]
        n1 = front.node_elems[1]
        p0 = np.array(n0.coords)
        p1 = np.array(n1.coords)
        front_mid = (p0 + p1) / 2.0

        # 收集所有边界节点（至少有一条边的 count == 1）
        boundary_hashes = set()
        for e, cnt in self.edge_count.items():
            if cnt == 1:
                boundary_hashes.update(e)

        # 排除阵面自身的两个节点
        boundary_hashes.discard(n0.hash)
        boundary_hashes.discard(n1.hash)

        if not boundary_hashes:
            return None

        # 按距离排序，找最近的边界节点
        candidates = []
        for bh in boundary_hashes:
            node = self.node_hash_map.get(bh)
            if node is None:
                continue
            p = np.array(node.coords)
            dist = np.linalg.norm(p - front_mid)
            candidates.append((dist, node))

        candidates.sort(key=lambda x: x[0])

        # 尝试最近的几个候选（最多10个）
        front_len = np.linalg.norm(p1 - p0)
        max_edge = self._max_edge_len * 2.0
        for _, node in candidates[:10]:
            p = np.array(node.coords)

            # 距离检查
            if np.linalg.norm(p - p0) > max_edge or np.linalg.norm(p - p1) > max_edge:
                continue

            # 退化检查
            tri_normal = np.cross(p1 - p0, p - p0)
            if np.linalg.norm(tri_normal) < 1e-12:
                continue

            # 质量检查
            quality = triangle_quality_from_coords(p0, p1, p)
            if quality < 0.05:
                continue

            # 边饱和检查
            n2h = node.hash
            skip = False
            for eh in [frozenset([n0.hash, n1.hash]),
                       frozenset([n0.hash, n2h]),
                       frozenset([n1.hash, n2h])]:
                if self.edge_count.get(eh, 0) >= 2:
                    skip = True
                    break
            if skip:
                continue

            return node

        return None

    def generate(self) -> List[SurfaceTriangle]:
        """
        使用 3D 阵面推进法（AFM）生成曲面网格（与 2D AFM 等价的主循环）

        Returns:
            生成的三角形列表
        """
        timer = TimeSpan("开始曲面网格生成...")

        iteration = 0
        while self.front_list and iteration < self.max_iterations:
            iteration += 1

            base_front = heapq.heappop(self.front_list)

            # 陈旧阵面跳过
            n0h = base_front.node_elems[0].hash
            n1h = base_front.node_elems[1].hash
            edge_hash = frozenset([n0h, n1h])
            if self.edge_count.get(edge_hash, 0) >= 2:
                continue

            spacing = self.sizing_field.compute_front_spacing(base_front, self.surface)
            ideal_point, ideal_uv = self._compute_ideal_point(base_front, spacing)
            candidates = self._search_candidates(ideal_point, base_front.al * spacing)
            selected_node = self._select_best_node(base_front, ideal_point, candidates)

            if selected_node is None:
                # 确保 al 至少为 2.0，避免搜索半径过小
                if base_front.al < 2.0:
                    base_front.al = 2.0
                base_front.al *= 1.2
                # 多策略回退：当阵面卡住时尝试不同策略
                if base_front.al >= 5 and base_front.al < 20:
                    # 策略1: 对边界阵面，尝试连接到最近的边界节点
                    if base_front.bc_type == "wall":
                        fallback = self._fallback_connect_boundary(base_front)
                        if fallback is not None and self._update_mesh(base_front, fallback):
                            continue
                if base_front.al < 200:
                    heapq.heappush(self.front_list, base_front)
                continue

            if not self._update_mesh(base_front, selected_node):
                base_front.al *= 1.2
                if base_front.al < 200:
                    heapq.heappush(self.front_list, base_front)
                continue

            if iteration % 100 == 0:
                info(f"迭代 {iteration}: 阵面数={len(self.front_list)}, "
                     f"节点数={len(self.node_list)}, 三角形数={len(self.triangle_list)}")

        timer.show_to_console("曲面网格生成完成")

        # 处理所有剩余边界环
        info("开始处理剩余边界环...")
        total_boundary_created = 0

        # 多轮细化 + 塌缩：先细化长边，再塌缩耳朵，重复直到稳定
        for refine_round in range(5):
            refined = self._refine_boundary_loop(max_edge_ratio=0.8)
            if refined == 0:
                break
            created = 0
            for _ in range(50):
                c = self._process_boundary_loop()
                created += c
                if c == 0:
                    break
            total_boundary_created += created
            if created == 0 and refined == 0:
                break

        # 保守塌缩处理所有边界环（逐环处理，失败则跳过）
        for _ in range(20):
            c = self._process_boundary_loop_conservative()
            total_boundary_created += c
            if c == 0:
                break

        info(f"边界环处理完成: 共创建 {total_boundary_created} 个三角形")

        # 桥接边界间隙：找到不同边界环上的近邻节点，创建三角形连接
        bridge_created = self._bridge_boundary_gaps()
        if bridge_created > 0:
            info(f"桥接边界间隙: {bridge_created} 个三角形")

        # 在边界结点处闭合完整 3 边环（优先消除整个环）
        junction_close_created = self._close_boundary_at_junctions()
        if junction_close_created > 0:
            info(f"边界结点闭合: {junction_close_created} 个三角形")

        # 解决边界结点（度 > 2）：在结点处创建三角形以降低度数到 2
        junction_created = self._resolve_boundary_junctions()
        if junction_created > 0:
            info(f"解决边界结点: {junction_created} 个三角形")

        # 闭合剩余边界环（清理退化环 + 闭合3节点环）
        close_created = 0
        for _ in range(10):
            c = self._close_remaining_triangles()
            close_created += c
            if c == 0:
                break
        if close_created > 0:
            info(f"闭合剩余三角形: {close_created} 个")

        self._print_statistics()

        # 后处理优化：边交换 + Laplacian 光滑（暂时关闭，调试用）
        self._optimize_mesh()

        return self.triangle_list

    def _compute_ideal_point(
        self,
        front: SurfaceFront,
        spacing: float
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float]]:
        """
        计算理想点：从阵面中点沿切平面垂直方向前进，投影到曲面。
        对于边界阵面，尝试两个方向（正/反），选择留在参数域内的方向。

        Args:
            front: 当前阵面
            spacing: 网格尺寸

        Returns:
            (理想点坐标, 参数坐标)
        """
        distance = self.sizing_field.compute_ideal_point_distance(front, self.surface)
        u_min, u_max, v_min, v_max = self._get_surface_bounds()

        # 正方向
        ideal_point, ideal_uv = self.geometry.compute_ideal_point_on_surface(
            front.center, front.tangent_normal, distance, self.surface,
        )

        # 对于边界阵面，检查正方向是否越界，如果越界则尝试反方向
        if front.bc_type == "wall":
            u, v = ideal_uv
            is_out_of_bounds = (u < u_min or u > u_max or v < v_min or v > v_max)

            if is_out_of_bounds:
                # 尝试反方向
                reverse_normal = tuple(-x for x in front.tangent_normal)
                try:
                    rev_point, rev_uv = self.geometry.compute_ideal_point_on_surface(
                        front.center, reverse_normal, distance, self.surface,
                    )
                    ru, rv = rev_uv
                    rev_in_bounds = (ru >= u_min and ru <= u_max and rv >= v_min and rv <= v_max)
                    if rev_in_bounds:
                        return (rev_point, rev_uv)
                except Exception:
                    pass

                # 反方向也越界，投影到参数域内部
                margin = 0.05
                u_clamped = max(u_min + margin, min(u_max - margin, u))
                v_clamped = max(v_min + margin, min(v_max - margin, v))

                # 向参数域中心偏移
                u_mid = (u_min + u_max) / 2
                v_mid = (v_min + v_max) / 2
                if abs(u_clamped - u_min) < margin or abs(u_clamped - u_max) < margin:
                    u_clamped = u_clamped * 0.7 + u_mid * 0.3
                if abs(v_clamped - v_min) < margin or abs(v_clamped - v_max) < margin:
                    v_clamped = v_clamped * 0.7 + v_mid * 0.3

                interior_point = self.geometry.evaluate_point(u_clamped, v_clamped, self.surface)
                return (interior_point, (u_clamped, v_clamped))

        return (ideal_point, ideal_uv)

    def _compute_reverse_ideal_point(
        self,
        front: SurfaceFront,
        spacing: float
    ) -> Optional[Tuple[float, float, float]]:
        """
        计算反方向理想点：沿切平面垂直反方向前进，投影到曲面

        Args:
            front: 当前阵面
            spacing: 网格尺寸

        Returns:
            理想点坐标，如果失败返回 None
        """
        try:
            distance = self.sizing_field.compute_ideal_point_distance(front, self.surface)
            # 反方向：翻转切平面法向
            reverse_normal = tuple(-x for x in front.tangent_normal)
            ideal_point, ideal_uv = self.geometry.compute_ideal_point_on_surface(
                front.center, reverse_normal, distance, self.surface,
            )

            # 检查UV是否在参数域内
            u_min, u_max, v_min, v_max = self._get_surface_bounds()
            u, v = ideal_uv
            if u < u_min or u > u_max or v < v_min or v > v_max:
                return None

            return ideal_point
        except Exception:
            return None

    def _search_candidates(
        self,
        center: Tuple[float, float, float],
        search_radius: float,
    ) -> List[NodeElement3D]:
        """
        在中心点周围搜索候选节点

        Args:
            center: 搜索中心点坐标
            search_radius: 搜索半径

        Returns:
            候选节点列表
        """
        if self.space_index_node is None:
            return []

        px, py, pz = center
        query_bbox = (
            px - search_radius, py - search_radius, pz - search_radius,
            px + search_radius, py + search_radius, pz + search_radius,
        )
        candidate_ids = list(self.space_index_node.intersection(query_bbox))

        r_sq = search_radius * search_radius
        candidates = []
        for nid in candidate_ids:
            if nid not in self.node_dict:
                continue
            node = self.node_dict[nid]
            dx = node.coords[0] - px
            dy = node.coords[1] - py
            dz = node.coords[2] - pz
            if dx * dx + dy * dy + dz * dz <= r_sq:
                candidates.append(node)

        return candidates

    def _is_boundary_node(self, node_hash: int) -> bool:
        """检查节点是否在网格边界上（有且仅有 count=1 的边）"""
        for eh, cnt in self.edge_count.items():
            if node_hash in eh and cnt == 1:
                return True
        return False

    def _boundary_neighbors(self, node_hash: int) -> set:
        """获取节点在边界上的邻居集合"""
        neighbors = set()
        for eh, cnt in self.edge_count.items():
            if cnt == 1 and node_hash in eh:
                for h in eh:
                    if h != node_hash:
                        neighbors.add(h)
        return neighbors

    def _get_local_surface_normal(self, node) -> np.ndarray:
        """获取节点处的局部曲面法向（用于边界塌缩的绕序判断）"""
        try:
            uv = self.geometry.project_point_to_surface(node.coords, self.surface)
            sn = np.array(self.geometry.get_surface_normal(uv[0], uv[1], self.surface))
            sn_len = np.linalg.norm(sn)
            if sn_len > 1e-12:
                return sn / sn_len
        except Exception:
            pass
        return np.array([0.0, 0.0, 1.0])

    def _triangle_intersects_existing(self, n0, n1, n2, check_min_dist: bool = True) -> bool:
        """检查三角形 (n0,n1,n2) 是否与已有三角形相交（跳过共享边/顶点）"""
        if self.space_index_triangle is None or not self.triangle_list:
            return False
        new_hashes = {n0.hash, n1.hash, n2.hash}
        new_hash_list = [n0.hash, n1.hash, n2.hash]
        new_coords = np.array([n0.coords, n1.coords, n2.coords])
        padding = self.sizing_field.global_spacing * 0.5
        bbox = (
            new_coords[:, 0].min() - padding, new_coords[:, 1].min() - padding, new_coords[:, 2].min() - padding,
            new_coords[:, 0].max() + padding, new_coords[:, 1].max() + padding, new_coords[:, 2].max() + padding,
        )
        # 最小边-边距离阈值（防止曲面上三角形过于接近）
        # 对于球面等曲面，三角形天然较近，使用更小的阈值
        min_edge_dist = self.sizing_field.global_spacing * 0.02

        for tri_id in self.space_index_triangle.intersection(bbox):
            if tri_id not in self._triangle_dict:
                continue
            existing = self._triangle_dict[tri_id]
            ex_hash_list = [nd.hash for nd in existing.nodes]
            shared = set(ex_hash_list) & new_hashes
            shared_count = len(shared)
            if shared_count == 0:
                if check_triangle_intersection(new_coords, existing):
                    return True
                # 检查最小边-边距离（仅在启用时检查）
                if check_min_dist and self._check_min_edge_distance(new_coords, existing, min_edge_dist):
                    return True
            else:
                new_shared_idx = [i for i, h in enumerate(new_hash_list) if h in shared]
                ex_shared_idx = [i for i, h in enumerate(ex_hash_list) if h in shared]
                ex_coords = np.array([nd.coords for nd in existing.nodes])
                if check_triangle_vs_existing(
                    new_coords, ex_coords, shared_count,
                    new_shared_idx, ex_shared_idx,
                ):
                    return True
        return False

    def _triangle_intersects_existing_loose(self, n0, n1, n2) -> bool:
        """
        宽松的相交检查：只检查节点包含，不检查边相交。
        用于边界环塌缩，允许边界闭合。
        """
        if self.space_index_triangle is None or not self.triangle_list:
            return False
        new_hashes = {n0.hash, n1.hash, n2.hash}
        new_coords = np.array([n0.coords, n1.coords, n2.coords])
        padding = self.sizing_field.global_spacing * 0.5
        bbox = (
            new_coords[:, 0].min() - padding, new_coords[:, 1].min() - padding, new_coords[:, 2].min() - padding,
            new_coords[:, 0].max() + padding, new_coords[:, 1].max() + padding, new_coords[:, 2].max() + padding,
        )

        for tri_id in self.space_index_triangle.intersection(bbox):
            if tri_id not in self._triangle_dict:
                continue
            existing = self._triangle_dict[tri_id]
            ex_hash_list = [nd.hash for nd in existing.nodes]
            shared = set(ex_hash_list) & new_hashes
            # 跳过共享边的三角形
            if len(shared) >= 2:
                continue
            # 跳过共享1个节点的三角形
            if len(shared) == 1:
                continue
            # 无共享节点：检查是否有节点在对方内部
            ex_coords = np.array([nd.coords for nd in existing.nodes])
            from .geom_utils import point_in_triangle_3d
            # 检查新三角形的节点是否在已有三角形内部
            for pt in new_coords:
                if point_in_triangle_3d(pt, ex_coords[0], ex_coords[1], ex_coords[2]):
                    return True
            # 检查已有三角形的节点是否在新三角形内部
            for pt in ex_coords:
                if point_in_triangle_3d(pt, new_coords[0], new_coords[1], new_coords[2]):
                    return True
        return False

    def _check_min_edge_distance(
        self,
        new_coords: np.ndarray,
        existing_tri,
        min_dist: float
    ) -> bool:
        """
        检查新三角形与已有三角形的边-边距离是否过小

        Args:
            new_coords: 新三角形顶点坐标 (3, 3)
            existing_tri: 已有三角形对象
            min_dist: 最小允许距离

        Returns:
            True 表示距离过小（应拒绝）
        """
        ex_coords = np.array([nd.coords for nd in existing_tri.nodes])
        edges_new = [
            (new_coords[0], new_coords[1]),
            (new_coords[1], new_coords[2]),
            (new_coords[2], new_coords[0])
        ]
        edges_ex = [
            (ex_coords[0], ex_coords[1]),
            (ex_coords[1], ex_coords[2]),
            (ex_coords[2], ex_coords[0])
        ]

        for a1, a2 in edges_new:
            for b1, b2 in edges_ex:
                dist = segment_segment_distance_3d(a1, a2, b1, b2)
                if dist < min_dist:
                    return True
        return False

    def _is_valid_candidate(
        self,
        front: SurfaceFront,
        p0: np.ndarray,
        p1: np.ndarray,
        node: NodeElement3D,
        min_height: float,
        min_edge_len: float,
    ) -> bool:
        """检查候选节点是否满足几何约束（与 2D AFM select_point 等价）"""
        if node.hash == front.node_elems[0].hash or node.hash == front.node_elems[1].hash:
            return False

        p2 = np.array(node.coords)

        tri_key = frozenset([front.node_elems[0].hash, front.node_elems[1].hash, node.hash])
        if tri_key in self.triangle_set:
            return False

        # 最大边长限制：防止跨越球面直径的大三角形
        max_edge = self._max_edge_len
        if (np.linalg.norm(p2 - p0) > max_edge or
            np.linalg.norm(p2 - p1) > max_edge):
            return False

        if check_triangle_degenerate(p0, p1, p2, min_height, min_edge_len):
            return False

        # 相交检查（2D: is_cross + is_cross_rtree）
        if self._check_intersection(front, node):
            return False

        return True

    def _select_best_node(
        self,
        front: SurfaceFront,
        ideal_point: Tuple[float, float, float],
        candidates: List[NodeElement3D]
    ) -> Optional[NodeElement3D]:
        """
        选择最佳节点：理想节点与候选节点统一验证，理想节点带质量折扣。
        当理想节点和已有候选都失败时，尝试在多个距离处创建试探节点。

        Args:
            front: 当前阵面
            ideal_point: 理想点
            candidates: 候选节点列表

        Returns:
            最佳节点，如果没有合适的返回None
        """
        p0 = np.array(front.node_elems[0].coords)
        p1 = np.array(front.node_elems[1].coords)

        front_len = np.linalg.norm(p1 - p0)
        min_height = front_len * 0.01
        min_edge_len = self.sizing_field.global_spacing * 0.3

        # 创建理想节点，加入候选列表统一验证
        is_boundary = front.bc_type == "wall"
        ideal_node = self._create_ideal_node(ideal_point, is_boundary_front=is_boundary)
        if ideal_node is not None:
            candidates = list(candidates) + [ideal_node]

        scored_candidates = []
        for node in candidates:
            if not self._is_valid_candidate(front, p0, p1, node, min_height, min_edge_len):
                continue

            quality = triangle_quality_from_coords(p0, p1, np.array(node.coords))
            if quality <= 0.1:
                continue

            # 理想节点带质量折扣，倾向选择已有节点
            if node is ideal_node:
                quality *= self.quality_discount

            scored_candidates.append((quality, node))

        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            return scored_candidates[0][1]

        return None

    def _create_ideal_node(
        self,
        point: Tuple[float, float, float],
        is_boundary_front: bool = False
    ) -> Optional[NodeElement3D]:
        """
        创建理想点节点（修复6：投影失败时增加 debug 日志）

        Args:
            point: 点坐标
            is_boundary_front: 是否是边界阵面

        Returns:
            节点对象，如果UV越界或投影失败则返回None
        """
        try:
            uv = self.geometry.project_point_to_surface(point, self.surface)
        except Exception as e:
            debug(f"理想点投影失败: {point}, 错误: {e}")
            return None

        # 检查UV是否在参数域内
        u_min, u_max, v_min, v_max = self._get_surface_bounds()
        
        # 对于边界阵面，允许在参数域边界上创建节点（将UV限制在边界上）
        if is_boundary_front:
            # 将UV限制在参数域内
            u = max(u_min, min(u_max, uv[0]))
            v = max(v_min, min(v_max, uv[1]))
            if u != uv[0] or v != uv[1]:
                # 重新计算边界上的点
                uv = (u, v)
                point = self.geometry.evaluate_point(u, v, self.surface)
        else:
            margin = 0.05  # 允许的小边界裕度
            if (uv[0] < u_min - margin or uv[0] > u_max + margin or
                uv[1] < v_min - margin or uv[1] > v_max + margin):
                return None

        try:
            normal = self.geometry.get_surface_normal(uv[0], uv[1], self.surface)
        except Exception:
            normal = (0.0, 0.0, 1.0)

        # idx 延迟到 _update_mesh 节点真正加入时分配，避免未选中节点造成编号间隙
        node = NodeElement3D(
            coords=point,
            idx=-1,
            surface=self.surface,
            uv_params=uv,
            normal=normal
        )

        return node

    def _check_intersection(
        self,
        front: SurfaceFront,
        node: NodeElement3D
    ) -> bool:
        """
        检查新三角形是否与已有网格相交（修复1：重写共享顶点检测逻辑）

        3D 混合策略：
        1. 边饱和检查（manifold 约束）
        2. 退化三角形检测
        3. RTree 3D 候选查询
        4. 按共享节点数分级检测：
           - shared_count >= 2: 共享边，合法邻接，跳过（除非完全重复）
           - shared_count == 1: 仅检查非共享边之间的交叉 + 非共享边穿透对方三角形
           - shared_count == 0: 无共享节点，完整三角形相交检测

        Args:
            front: 当前阵面
            node: 候选节点

        Returns:
            是否相交
        """
        n0 = front.node_elems[0]
        n1 = front.node_elems[1]
        n2 = node

        # 边饱和检查：每条边最多 2 个三角形
        for eh in [frozenset([n0.hash, n1.hash]),
                    frozenset([n0.hash, n2.hash]),
                    frozenset([n1.hash, n2.hash])]:
            if self.edge_count.get(eh, 0) >= 2:
                return True

        if self.space_index_triangle is None or len(self.triangle_list) == 0:
            return False

        p0 = np.array(n0.coords)
        p1 = np.array(n1.coords)
        p2 = np.array(n2.coords)

        # 退化三角形检测
        new_tri_normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(new_tri_normal) < 1e-12:
            return True

        # RTree 查询候选三角形
        all_pts = np.array([p0, p1, p2])
        padding = self.sizing_field.global_spacing * 0.5
        query_bbox = (
            all_pts[:, 0].min() - padding,
            all_pts[:, 1].min() - padding,
            all_pts[:, 2].min() - padding,
            all_pts[:, 0].max() + padding,
            all_pts[:, 1].max() + padding,
            all_pts[:, 2].max() + padding,
        )
        candidate_ids = list(self.space_index_triangle.intersection(query_bbox))

        # 使用 node.hash（基于坐标）而不是 node.idx 来识别共享节点
        new_hash_list = [n0.hash, n1.hash, n2.hash]
        new_node_hashes = set(new_hash_list)

        # 最小边-边距离阈值（防止曲面上三角形过于接近）
        # 对于球面等曲面，三角形天然较近，使用更小的阈值
        min_edge_dist = self.sizing_field.global_spacing * 0.02

        for tri_id in candidate_ids:
            if tri_id not in self._triangle_dict:
                continue
            existing_tri = self._triangle_dict[tri_id]
            ex_hash_list = [nd.hash for nd in existing_tri.nodes]
            shared = set(ex_hash_list) & new_node_hashes
            shared_count = len(shared)

            if shared_count == 0:
                if check_triangle_intersection(
                    np.array([p0, p1, p2]), existing_tri
                ):
                    return True
                # 检查最小边-边距离
                if self._check_min_edge_distance(
                    np.array([p0, p1, p2]), existing_tri, min_edge_dist
                ):
                    return True
            else:
                new_shared_idx = [i for i, h in enumerate(new_hash_list) if h in shared]
                ex_shared_idx = [i for i, h in enumerate(ex_hash_list) if h in shared]
                ex_coords = np.array([nd.coords for nd in existing_tri.nodes])
                if check_triangle_vs_existing(
                    np.array([p0, p1, p2]), ex_coords, shared_count,
                    new_shared_idx, ex_shared_idx,
                ):
                    return True

        return False

    def _boundary_degree(self, node_hash: int) -> int:
        """计算节点的边界度数（连接的边界边数量）"""
        count = 0
        for eh, cnt in self.edge_count.items():
            if node_hash in eh and cnt == 1:
                count += 1
        return count

    def _update_mesh(
        self,
        front: SurfaceFront,
        node: NodeElement3D,
        bc_type: str = "interior"
    ) -> bool:
        """
        更新网格数据（修复3：新阵面创建前检查边饱和度，防止重复阵面）
        （修复7：检查边界节点度数，防止非流形边界）

        Args:
            front: 当前阵面
            node: 选中的节点
            bc_type: 新阵面的边界类型（"wall" 或 "interior"）

        Returns:
            是否成功（False 表示节点 UV 严重越界被拒绝）
        """
        # UV 越界检查：严重越界则拒绝（防止环绕），不做 clamp
        if node.uv_params:
            if self._uv_out_of_bounds(node.uv_params, margin=0.2):
                return False

        n0h = front.node_elems[0].hash
        n1h = front.node_elems[1].hash
        n2h = node.hash

        # 退化检查：不允许重复节点（hash 或 idx）
        if n2h == n0h or n2h == n1h:
            return False

        # 边饱和检查：任一边已满则拒绝
        for eh in [frozenset([n0h, n1h]), frozenset([n0h, n2h]), frozenset([n1h, n2h])]:
            if self.edge_count.get(eh, 0) >= 2:
                return False

        new_node_added = False
        if node.hash not in self.node_hash_set:
            # 分配连续 idx（与 2D AFM 一致，避免编号间隙）
            node.idx = self.num_nodes
            self.node_hash_set.add(node.hash)
            self.node_list.append(node)
            self.node_coords.append(node.coords)
            self.node_dict[node.idx] = node
            self.node_hash_map[node.hash] = node
            self._used_node_idx.add(node.idx)
            self.num_nodes += 1
            new_node_added = True
        else:
            # 节点已存在：理想节点 idx=-1 时需回填已有节点的 idx
            existing = self.node_hash_map.get(node.hash)
            if existing is not None and node.idx < 0:
                node.idx = existing.idx

        # 最终退化检查：三个节点必须互不相同（hash 和 idx 双重检查）
        n0 = front.node_elems[0]
        n1 = front.node_elems[1]
        if node is n0 or node is n1 or n0 is n1:
            return False
        if n0.idx == n1.idx or n0.idx == node.idx or n1.idx == node.idx:
            return False

        # 确保三角形法向与曲面法向一致
        # 计算当前节点顺序的法向
        p0 = np.array(front.node_elems[0].coords)
        p1 = np.array(front.node_elems[1].coords)
        p2 = np.array(node.coords)
        normal = np.cross(p1 - p0, p2 - p0)
        
        # 获取曲面法向（使用阵面中点）
        try:
            mid_uv = self.geometry.project_point_to_surface(front.center, self.surface)
            surface_normal = np.array(self.geometry.get_surface_normal(mid_uv[0], mid_uv[1], self.surface))
        except Exception:
            surface_normal = np.array([0.0, 0.0, 1.0])
        
        # 如果法向方向相反，交换节点顺序
        if np.dot(normal, surface_normal) < 0:
            triangle = SurfaceTriangle(
                front.node_elems[1],
                front.node_elems[0],
                node,
                surface=self.surface,
                idx=self.num_triangles
            )
        else:
            triangle = SurfaceTriangle(
                front.node_elems[0],
                front.node_elems[1],
                node,
                surface=self.surface,
                idx=self.num_triangles
            )
        self.triangle_list.append(triangle)
        self.triangle_set.add(frozenset([
            front.node_elems[0].hash, front.node_elems[1].hash, node.hash
        ]))
        self.num_triangles += 1

        # 更新边计数
        for eh in [frozenset([n0h, n1h]), frozenset([n0h, n2h]), frozenset([n1h, n2h])]:
            self.edge_count[eh] = self.edge_count.get(eh, 0) + 1

        # 更新空间索引
        if new_node_added:
            if self.space_index_node is not None:
                self.space_index_node, self.node_dict = add_elems_to_space_index_3d_with_RTree(
                    [node], self.space_index_node, self.node_dict
                )

        if self.space_index_triangle is None:
            self._triangle_dict, self.space_index_triangle = build_space_index_3d_with_RTree([triangle])
        else:
            self.space_index_triangle, self._triangle_dict = add_elems_to_space_index_3d_with_RTree(
                [triangle], self.space_index_triangle, self._triangle_dict,
            )

        # 【修复3】新阵面去重：只有当边尚未被两个三角形共享时才创建新阵面
        min_front_len = self.sizing_field.global_spacing * 0.1

        def _should_add_front(na: NodeElement3D, nb: NodeElement3D) -> bool:
            """检查是否应该为该边创建新阵面"""
            edge_hash = frozenset([na.hash, nb.hash])
            # 如果该边已有2个三角形，则不再需要阵面
            if self.edge_count.get(edge_hash, 0) >= 2:
                return False
            return True

        if _should_add_front(front.node_elems[0], node):
            new_front1 = SurfaceFront(
                front.node_elems[0],
                node,
                surface=self.surface,
                idx=len(self.front_list) + 1,
                bc_type=bc_type
            )
            # 边界阵面设置 al=0 确保优先处理，维护边界连贯性
            if bc_type != "interior":
                new_front1.al = 0.0
            if new_front1.length > min_front_len:
                heapq.heappush(self.front_list, new_front1)

        if _should_add_front(node, front.node_elems[1]):
            new_front2 = SurfaceFront(
                node,
                front.node_elems[1],
                surface=self.surface,
                idx=len(self.front_list) + 2,
                bc_type=bc_type
            )
            if bc_type != "interior":
                new_front2.al = 0.0
            if new_front2.length > min_front_len:
                heapq.heappush(self.front_list, new_front2)

        return True

    def _optimize_mesh(self, swap_iterations: int = 3, smooth_iterations: int = 3):
        """
        网格后处理优化：边交换（Delaunay 准则）+ Laplacian 光滑

        保持边界节点和边界边不动，光滑后的节点投影回曲面。

        Args:
            swap_iterations: 边交换迭代轮数
            smooth_iterations: Laplacian 光滑迭代次数
        """
        if len(self.triangle_list) < 2:
            return

        # 构建完整的 node.idx → NodeElement3D 映射（确保包含所有节点）
        self._node_idx_map: Dict[int, NodeElement3D] = {}
        for node in self.node_list:
            self._node_idx_map[node.idx] = node
        for idx, node in self.node_dict.items():
            if idx not in self._node_idx_map:
                self._node_idx_map[idx] = node

        # 使用初始化时记录的边界节点 hash（此时 edge_count 已被三角形更新，不可靠）
        self._boundary_hashes = set(self._init_boundary_hashes)
        # 同时记录边界节点的 idx 集合（双重保护）
        self._boundary_idx = set()
        for node in self.node_list:
            if node.hash in self._boundary_hashes:
                self._boundary_idx.add(node.idx)

        info(f"[优化] 边界节点数: {len(self._boundary_idx)}, 总节点数: {len(self.node_list)}")

        # 记录优化前边界节点坐标，用于验证
        boundary_coords_before = {
            node.idx: tuple(node.coords) for node in self.node_list
            if node.idx in self._boundary_idx
        }

        # ---- 1. 边交换优化 ----
        for _ in range(swap_iterations):
            swapped = self._edge_swap_pass()
            if swapped == 0:
                break

        # ---- 2. Laplacian 光滑 ----
        for _ in range(smooth_iterations):
            self._laplacian_smooth_pass()

        # 验证边界节点坐标未变化
        boundary_moved = 0
        for node in self.node_list:
            if node.idx in boundary_coords_before:
                before = boundary_coords_before[node.idx]
                after = tuple(node.coords)
                if before != after:
                    boundary_moved += 1
                    warning(
                        f"边界节点 {node.idx} 坐标变化: "
                        f"({before[0]:.6f},{before[1]:.6f},{before[2]:.6f}) → "
                        f"({after[0]:.6f},{after[1]:.6f},{after[2]:.6f})"
                    )
        if boundary_moved > 0:
            warning(f"边界节点校验：{boundary_moved}/{len(boundary_coords_before)} 个边界节点坐标发生了变化！")
        else:
            info(f"边界节点校验通过：{len(boundary_coords_before)} 个边界节点坐标均未变化")

        # 清理临时数据
        del self._node_idx_map
        del self._boundary_hashes
        del self._boundary_idx

        # 刷新统计
        self._print_statistics()

    def _build_edge_triangle_map(self) -> Dict[frozenset, List[int]]:
        """构建边 → 三角形索引列表 的映射"""
        edge_map: Dict[frozenset, List[int]] = {}
        for tri_idx, tri in enumerate(self.triangle_list):
            ids = tri.node_ids
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                edge = frozenset([ids[a], ids[b]])
                edge_map.setdefault(edge, []).append(tri_idx)
        return edge_map

    def _swap_causes_intersection(self, new_tri, skip_indices: Set[int]) -> bool:
        """
        检查新三角形是否与已有三角形相交（用于边交换验证）。

        跳过 skip_indices 中的三角形（正在被替换的三角形）和共享节点的三角形。

        Args:
            new_tri: 新三角形
            skip_indices: 要跳过的三角形索引集合

        Returns:
            True 表示存在相交
        """
        if self.space_index_triangle is None:
            return False

        new_coords = np.array([nd.coords for nd in new_tri.nodes])
        new_hashes = {nd.hash for nd in new_tri.nodes}
        padding = self.sizing_field.global_spacing * 0.5
        bbox = (
            new_coords[:, 0].min() - padding, new_coords[:, 1].min() - padding, new_coords[:, 2].min() - padding,
            new_coords[:, 0].max() + padding, new_coords[:, 1].max() + padding, new_coords[:, 2].max() + padding,
        )

        for tri_id in self.space_index_triangle.intersection(bbox):
            if tri_id in skip_indices:
                continue
            if tri_id not in self._triangle_dict:
                continue
            existing = self._triangle_dict[tri_id]
            ex_hash_list = [nd.hash for nd in existing.nodes]
            shared = set(ex_hash_list) & new_hashes
            if shared:
                continue  # 共享节点，跳过
            ex_coords = np.array([nd.coords for nd in existing.nodes])
            if check_triangle_intersection(new_coords, existing):
                return True
        return False

    def _edge_swap_pass(self) -> int:
        """
        一轮边交换：遍历所有内部边，若交换后最小角增大则执行交换。

        Returns:
            本轮交换次数
        """
        edge_map = self._build_edge_triangle_map()
        boundary_hashes = self._boundary_hashes
        swapped = 0
        # 跟踪本轮已修改的三角形索引，避免读取被先前交换污染的三角形
        dirty_indices: Set[int] = set()

        for edge, tri_indices in list(edge_map.items()):
            if len(tri_indices) != 2:
                continue

            idx0, idx1 = tri_indices
            # 跳过涉及已被本轮交换修改的三角形的边
            if idx0 in dirty_indices or idx1 in dirty_indices:
                continue

            # 跳过任何涉及边界节点的边（保护边界不动）
            if edge & boundary_hashes:
                continue
            # 双重保护：也用 node.idx 检查
            edge_ids = set(edge)
            if edge_ids & self._boundary_idx:
                continue

            tri0 = self.triangle_list[idx0]
            tri1 = self.triangle_list[idx1]

            ids0 = set(tri0.node_ids)
            ids1 = set(tri1.node_ids)
            common = ids0 & ids1
            if len(common) != 2:
                continue

            # a-b 是公共边，c 属于 tri0 独有，d 属于 tri1 独有
            a, b = sorted(common)
            c = (ids0 - common).pop()
            d = (ids1 - common).pop()

            # 当前最小角
            angles_before = self._triangle_min_angle(tri0.node_ids)
            angles_before = min(angles_before, self._triangle_min_angle(tri1.node_ids))

            # 交换后：a-c-d 和 b-c-d
            new_ids0 = self._orient_ccw([a, c, d])
            new_ids1 = self._orient_ccw([b, c, d])
            if new_ids0 is None or new_ids1 is None:
                continue

            angles_after = self._triangle_min_angle(new_ids0)
            angles_after = min(angles_after, self._triangle_min_angle(new_ids1))

            if angles_after <= angles_before:
                continue

            # 执行交换：创建新的 SurfaceTriangle
            node_map = self._node_idx_map
            new_tri0 = SurfaceTriangle(
                node_map[new_ids0[0]], node_map[new_ids0[1]], node_map[new_ids0[2]],
                surface=self.surface, idx=tri0.idx,
            )
            new_tri1 = SurfaceTriangle(
                node_map[new_ids1[0]], node_map[new_ids1[1]], node_map[new_ids1[2]],
                surface=self.surface, idx=tri1.idx,
            )

            # 检查新三角形退化
            if new_tri0.area < 1e-16 or new_tri1.area < 1e-16:
                continue

            # 检查新三角形是否与已有三角形重复
            new_key0 = frozenset(nd.hash for nd in new_tri0.nodes)
            new_key1 = frozenset(nd.hash for nd in new_tri1.nodes)
            old_key0 = frozenset(nd.hash for nd in tri0.nodes)
            old_key1 = frozenset(nd.hash for nd in tri1.nodes)
            # 先从 triangle_set 移除旧条目，再检查新条目
            self.triangle_set.discard(old_key0)
            self.triangle_set.discard(old_key1)
            if new_key0 in self.triangle_set or new_key1 in self.triangle_set:
                # 会产生重复三角形，回退
                self.triangle_set.add(old_key0)
                self.triangle_set.add(old_key1)
                continue

            # 在曲面上检查新三角形是否与已有三角形自相交
            # 排除正在被替换的两个三角形
            skip_ids = {idx0, idx1}
            if self._swap_causes_intersection(new_tri0, skip_ids) or \
               self._swap_causes_intersection(new_tri1, skip_ids):
                self.triangle_set.add(old_key0)
                self.triangle_set.add(old_key1)
                continue

            # 更新 edge_count：只更新被交换的公共边，其余边保持不变
            old_shared_edge = frozenset([a, b])
            new_shared_edge = frozenset([c, d])
            cnt = self.edge_count.get(old_shared_edge, 0) - 2
            if cnt <= 0:
                self.edge_count.pop(old_shared_edge, None)
            else:
                self.edge_count[old_shared_edge] = cnt
            self.edge_count[new_shared_edge] = self.edge_count.get(new_shared_edge, 0) + 2

            self.triangle_list[idx0] = new_tri0
            self.triangle_list[idx1] = new_tri1
            self._triangle_dict[new_tri0.hash] = new_tri0
            self._triangle_dict[new_tri1.hash] = new_tri1
            # 更新 triangle_set
            self.triangle_set.add(new_key0)
            self.triangle_set.add(new_key1)
            dirty_indices.add(idx0)
            dirty_indices.add(idx1)
            swapped += 1

        return swapped

    def _laplacian_smooth_pass(self):
        """
        一轮 Laplacian 光滑：将内部节点移向邻居平均位置，投影回曲面。
        边界节点不动。移动后检查法向一致性和自相交，若导致问题则回退。
        """
        boundary_hashes = self._boundary_hashes
        boundary_idx = self._boundary_idx

        # 构建节点邻居映射 (node.idx → set of neighbor idx)
        # 以及节点 → 所属三角形索引映射
        neighbors: Dict[int, Set[int]] = {}
        node_tri_indices: Dict[int, Set[int]] = {}
        for tri_idx, tri in enumerate(self.triangle_list):
            ids = tri.node_ids
            for i in range(3):
                node_tri_indices.setdefault(ids[i], set()).add(tri_idx)
                for j in range(3):
                    if i != j:
                        neighbors.setdefault(ids[i], set()).add(ids[j])

        # 获取曲面法向用于法向一致性检查
        def _surf_normal_at(coords):
            try:
                uv = self.geometry.project_point_to_surface(coords, self.surface)
                return np.array(self.geometry.get_surface_normal(uv[0], uv[1], self.surface))
            except Exception:
                return np.array([0.0, 0.0, 1.0])

        relax = 0.1  # 松弛因子（曲面上小步移动防止自交）

        for node in self.node_list:
            # 双重保护：hash 和 idx 都检查
            if node.hash in boundary_hashes or node.idx in boundary_idx:
                continue
            nbrs = neighbors.get(node.idx)
            if not nbrs:
                continue

            # 计算邻居平均坐标
            avg = np.zeros(3)
            for nid in nbrs:
                avg += np.array(self._node_idx_map[nid].coords)
            avg /= len(nbrs)

            # 松弛：只移动一部分
            old_coords = np.array(node.coords)
            relaxed = old_coords + relax * (avg - old_coords)

            # 投影回曲面
            try:
                uv = self.geometry.project_point_to_surface(tuple(relaxed), self.surface)
                new_coords_tuple = self.geometry.evaluate_point(uv[0], uv[1], self.surface)
            except Exception:
                continue

            new_coords = np.array(new_coords_tuple)

            # 临时更新节点坐标
            node.coords = new_coords_tuple
            node.uv_params = uv

            # 检查所属三角形的法向一致性和质量
            surf_norm = _surf_normal_at(new_coords_tuple)
            ok = True
            my_tri_ids = node_tri_indices.get(node.idx, set())

            for tri_idx in my_tri_ids:
                tri = self.triangle_list[tri_idx]
                pts = np.array([nd.coords for nd in tri.nodes])
                tri_norm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                tn_len = np.linalg.norm(tri_norm)
                if tn_len < 1e-16:
                    ok = False
                    break
                if np.dot(tri_norm / tn_len, surf_norm) < 0.1:
                    ok = False
                    break
                q = triangle_quality_from_coords(pts[0], pts[1], pts[2])
                if q < 0.05:
                    ok = False
                    break

            # 检查所属三角形是否与非相邻三角形相交
            if ok and self.space_index_triangle is not None:
                # 收集所有相邻三角形的节点 hash（共享节点的三角形跳过）
                adjacent_hashes: Set[int] = set()
                for tri_idx in my_tri_ids:
                    tri = self.triangle_list[tri_idx]
                    for nd in tri.nodes:
                        adjacent_hashes.add(nd.hash)

                for tri_idx in my_tri_ids:
                    tri = self.triangle_list[tri_idx]
                    tri_coords = np.array([nd.coords for nd in tri.nodes])
                    padding = self.sizing_field.global_spacing * 0.3
                    bbox = (
                        tri_coords[:, 0].min() - padding, tri_coords[:, 1].min() - padding, tri_coords[:, 2].min() - padding,
                        tri_coords[:, 0].max() + padding, tri_coords[:, 1].max() + padding, tri_coords[:, 2].max() + padding,
                    )
                    for other_id in self.space_index_triangle.intersection(bbox):
                        if other_id in my_tri_ids:
                            continue
                        if other_id not in self._triangle_dict:
                            continue
                        other = self._triangle_dict[other_id]
                        other_hashes = {nd.hash for nd in other.nodes}
                        if other_hashes & adjacent_hashes:
                            continue  # 共享节点，跳过
                        if check_triangle_intersection(tri_coords, other):
                            ok = False
                            break
                    if not ok:
                        break

            if not ok:
                # 回退
                node.coords = tuple(old_coords)
                continue

        # 刷新 node_coords 和三角形属性
        self.node_coords = [node.coords for node in self.node_list]
        for tri in self.triangle_list:
            tri.normal = tri._compute_normal()
            tri.area = tri._compute_area()
            tri.quality = tri._compute_quality()
            tri.bbox = tri._compute_bbox()

    def _get_node_coords_by_idx(self, idx: int) -> np.ndarray:
        """通过 node.idx 获取节点坐标"""
        node = self._node_idx_map.get(idx)
        if node is not None:
            return np.array(node.coords)
        raise KeyError(f"Node idx={idx} not found in _node_idx_map")

    def _triangle_min_angle(self, node_ids: list) -> float:
        """计算三角形最小角（度）"""
        p0 = self._get_node_coords_by_idx(node_ids[0])
        p1 = self._get_node_coords_by_idx(node_ids[1])
        p2 = self._get_node_coords_by_idx(node_ids[2])
        angles = []
        for apex, a, b in [(p0, p1, p2), (p1, p0, p2), (p2, p0, p1)]:
            va = a - apex
            vb = b - apex
            la = np.linalg.norm(va)
            lb = np.linalg.norm(vb)
            if la < 1e-15 or lb < 1e-15:
                return 0.0
            cos_a = np.clip(np.dot(va, vb) / (la * lb), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_a)))
        return min(angles)

    def _orient_ccw(self, node_ids: list):
        """确保三角形节点在 3D 中保持一致的绕序（返回 node_ids 或重排版本，退化时返回 None）"""
        p0 = self._get_node_coords_by_idx(node_ids[0])
        p1 = self._get_node_coords_by_idx(node_ids[1])
        p2 = self._get_node_coords_by_idx(node_ids[2])
        cross = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(cross) < 1e-16:
            return None
        # 使用曲面法向判断方向
        try:
            uv = self.geometry.project_point_to_surface(
                tuple((p0 + p1 + p2) / 3.0), self.surface
            )
            sn = np.array(self.geometry.get_surface_normal(uv[0], uv[1], self.surface))
            if np.dot(cross, sn) < 0:
                return [node_ids[0], node_ids[2], node_ids[1]]
        except Exception:
            pass
        return node_ids

    def _print_statistics(self):
        """打印统计信息（质量 + 拓扑）"""
        if self.triangle_list:
            validate_mesh_topology(self.triangle_list, verbose=True)

    def export_to_vtk(self, filename: str):
        """
        导出为 Legacy ASCII VTK 格式

        Args:
            filename: 输出文件名（.vtk）
        """
        self._export_to_vtk_simple(filename)

    def _export_to_vtk_simple(self, filename: str):
        """
        简单VTK导出（不依赖VTK库）

        Args:
            filename: 输出文件名
        """
        # 构建 node.hash -> index 映射，确保 VTK 索引正确
        node_hash_to_idx = {}
        for idx, node in enumerate(self.node_list):
            node_hash_to_idx[node.hash] = idx

        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Surface Mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            f.write(f"POINTS {len(self.node_list)} float\n")
            for node in self.node_list:
                f.write(f"{node.coords[0]} {node.coords[1]} {node.coords[2]}\n")

            f.write(f"\nCELLS {len(self.triangle_list)} {4 * len(self.triangle_list)}\n")
            for tri in self.triangle_list:
                # 使用 hash 映射获取正确的 VTK 索引
                idx0 = node_hash_to_idx[tri.nodes[0].hash]
                idx1 = node_hash_to_idx[tri.nodes[1].hash]
                idx2 = node_hash_to_idx[tri.nodes[2].hash]
                f.write(f"3 {idx0} {idx1} {idx2}\n")

            f.write(f"\nCELL_TYPES {len(self.triangle_list)}\n")
            for _ in self.triangle_list:
                f.write("5\n")

        info(f"网格已导出到: {filename}")
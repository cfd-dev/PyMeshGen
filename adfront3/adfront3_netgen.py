"""三维阵面推进法 - 参考 Netgen adfront3.cpp 实现

核心策略对齐 Netgen：
1. qualclass + frontnr 决定面优先级（SelectBaseElement）
2. 失败面 IncrementClass 而非延迟/丢弃
3. frontnr 世代追踪：边界面节点=0，内部节点逐层递增
"""
import heapq
import numpy as np
from typing import List, Optional, Dict

from data_structure.basic_elements import NodeElement, Tetrahedron
from data_structure.unstructured_grid import Unstructured_Grid
from data_structure.rtree_space import (
    build_space_index_3d_with_RTree,
    get_candidate_elements_id_3d,
    add_elems_to_space_index_3d_with_RTree,
)
from data_structure.vtk_types import VTKCellType
from utils.geom_toolkit import tetrahedron_volume
from optimize.mesh_quality import tetrahedron_shape_quality, compute_mesh_quality_stats
from utils.ray_casting import build_spatial_grid, classify_points_inside
from utils.timer import TimeSpan
from utils.message import info, warning
from fileIO.vtk_io import write_vtk
from adfront3.front3d import Front3D
from adfront3.geom3d import (
    point_in_tet, intersect_tet_triangle, intersect_tet_triangle2,
    intersect_triangle_triangle
)


class AdFront3Netgen:
    """三维阵面推进法 - 参考 Netgen adfront3.cpp 实现

    核心策略：
    1. 面优先级 = qualclass + frontnr_sum（Netgen SelectBaseElement）
    2. 失败时 IncrementClass（增大 qualclass 降低优先级）
    3. frontnr 世代机制：边界点=0，内部点逐层+1
    4. 推进到活跃阵面为空自然终止
    """

    def __init__(self, surface_triangles, sizing_system=None, debug_level=0):
        self.debug_level = debug_level
        self.surface_triangles = surface_triangles

        # ——— 推进数据结构 ———
        self.front_list: List[Front3D] = []          # heapq 优先队列
        self.front_dict: Dict[int, Front3D] = {}     # id → Front3D
        self.space_index_front = None                 # RTree 空间索引
        self.front_id_by_facekey: Dict[tuple, int] = {}  # face_key → id(front)

        # ——— 节点数据 ———
        self.node_coords: List[list] = []
        self.node_hash_list: set = set()
        self.node_elem_by_hash: Dict[int, NodeElement] = {}
        self.node_frontnr: Dict[int, int] = {}       # node_hash → frontnr (Netgen 世代)
        self.front_node_list: List[NodeElement] = []
        self.boundary_nodes: set = set()             # 边界节点集
        self.space_index_node = None
        self.node_dict: Dict[int, NodeElement] = {}
        self.num_nodes: int = 0

        # ——— 单元数据 ———
        self.cell_container: list = []
        self.cell_hash_list: set = set()
        self.space_index_cell = None
        self.cell_dict: Dict[int, Tetrahedron] = {}
        self.num_cells: int = 0

        # ——— 搜索状态 ———
        self.base_front: Optional[Front3D] = None
        self.pselected: Optional[NodeElement] = None
        self.node_candidates: List[NodeElement] = []
        self.front_candidates: List[Front3D] = []
        self.cell_candidates: List[Tetrahedron] = []

        # ——— 面生命周期 ———
        self.deleted_facekeys: set = set()

        # ——— 尺寸场 ———
        if sizing_system is None:
            from adfront3.sizing3d import UniformSizing3D
            spacing = self._compute_mean_edge(surface_triangles)
            sizing_system = UniformSizing3D(spacing)
        self.sizing_system = sizing_system

        # ——— 射线投射（内外判定） ———
        self._raycast_data = None

        # ——— 初始化 ———
        self._initialize()

    # ============ 初始化 ============

    def _initialize(self):
        """将曲面三角形转换为 Front3D 并初始化所有数据结构"""
        # 射线投射数据
        tri_verts = []
        for tri in self.surface_triangles:
            verts = [list(n.coords) for n in tri.nodes]
            tri_verts.append(verts)
        self._raycast_data = build_spatial_grid(np.array(tri_verts))

        # 曲面质心（用于判断法向量方向）
        all_coords = []
        for tri in self.surface_triangles:
            for node3d in tri.nodes:
                all_coords.append(node3d.coords)
        centroid = np.mean(all_coords, axis=0) if all_coords else np.zeros(3)

        # 构建节点和阵面
        node_cache: Dict[tuple, NodeElement] = {}

        for tri in self.surface_triangles:
            tri_nodes: List[NodeElement] = []
            for node3d in tri.nodes:
                coord_key = tuple(round(c, 10) for c in node3d.coords)
                if coord_key not in node_cache:
                    ne = NodeElement(list(node3d.coords), self.num_nodes)
                    c = ne.coords
                    ne.bbox = [c[0], c[1], c[2], c[0], c[1], c[2]]
                    self.node_elem_by_hash[ne.hash] = ne
                    self.node_hash_list.add(ne.hash)
                    self.front_node_list.append(ne)
                    self.boundary_nodes.add(ne)
                    self.node_coords.append(list(ne.coords))
                    self.node_frontnr[ne.hash] = 1000  # 初始大值，SetStartFront 后归零
                    node_cache[coord_key] = ne
                    self.num_nodes += 1
                tri_nodes.append(node_cache[coord_key])

            # 判断法向量方向，确保阵面法向量指向内部
            tri_center = np.mean([n.coords for n in tri.nodes], axis=0)
            to_centroid = centroid - tri_center
            tri_normal = np.array(tri.normal)
            dot = np.dot(tri_normal, to_centroid)

            if dot > 0:
                front = Front3D(tri_nodes[0], tri_nodes[1], tri_nodes[2],
                                bc_type="wall", part_name="wall")
            else:
                front = Front3D(tri_nodes[0], tri_nodes[2], tri_nodes[1],
                                bc_type="wall", part_name="wall")
            front.qualclass = 1
            front.frontnr_sum = 0  # SetStartFront 后计算
            self.front_list.append(front)

        # —— SetStartFront: 边界点 frontnr 归零 ——
        for ne in self.boundary_nodes:
            self.node_frontnr[ne.hash] = 0

        # 计算边界面 frontnr_sum（全零）
        for front in self.front_list:
            front.frontnr_sum = sum(
                self.node_frontnr.get(n.hash, 0) for n in front.node_elems)

        heapq.heapify(self.front_list)

        # 空间索引
        self._build_spatial_index()

        if self.debug_level >= 1:
            info(f"初始化完成: 节点={self.num_nodes}, 阵面={len(self.front_list)}")

    def _build_spatial_index(self):
        """构建 RTree 空间索引"""
        from rtree import index as rtree_index

        # 阵面索引
        if self.front_list:
            self.front_dict, self.space_index_front = \
                build_space_index_3d_with_RTree(self.front_list)
            for front in self.front_list:
                self.front_id_by_facekey[front.face_key] = id(front)

        # 节点索引
        p = rtree_index.Property()
        p.dimension = 3
        self.space_index_node = rtree_index.Index(properties=p)
        for ne in self.front_node_list:
            add_elems_to_space_index_3d_with_RTree(
                [ne], self.space_index_node, self.node_dict)

        # 单元索引（初始为空）
        self.space_index_cell = rtree_index.Index(properties=p)

    # ============ 主推进循环 ============

    def generate(self, max_steps=None) -> Unstructured_Grid:
        """主推进循环 — Netgen Meshing3 对齐

        核心流程：
        1. heapq 弹出最小 (qualclass + frontnr_sum) 面
        2. 尝试创建四面体
        3. 成功 → 更新数据，继续
        4. 失败 → qualclass+=1 放回堆中（IncrementClass）
        5. 当堆为空或所有面 qualclass 超标时自然终止
        """
        timer = TimeSpan("开始三维四面体网格生成...")
        step = 0
        giveuptol = 20               # qualclass 上限（内部面）
        wall_giveuptol = 50          # qualclass 上限（边界面）
        face_retry: Dict[tuple, int] = {}  # per-face retry counter
        consecutive_fails = 0
        max_consecutive_fails = 500
        bbox_volume = self._estimate_volume()

        if max_steps is None:
            est_tet_vol = (self.sizing_system.global_spacing ** 3) / (6 * 2 ** 0.5)
            est_cells = max(1, int(bbox_volume / est_tet_vol))
            max_steps = min(int(est_cells * 5), 100000)
            info(f"预估步数上限: {max_steps} (初始阵面={len(self.front_list)}, "
                 f"估计单元数={est_cells})")

        info(f"开始推进: 初始阵面={len(self.front_list)}, 初始节点={self.num_nodes}, "
             f"包围盒体积={bbox_volume:.2f}")

        while self.front_list and step < max_steps:
            step += 1

            # —— 1. 弹出优先级最高的面 ——
            self.base_front = heapq.heappop(self.front_list)

            # 跳过已消耗面
            if (self.base_front.face_key in self.deleted_facekeys or
                    self.base_front.is_satisfied):
                continue

            # —— 2. 搜索候选 ——
            base_fk = self.base_front.face_key
            spacing = self.sizing_system.spacing_at(self.base_front.center)
            # 搜索半径随 qualclass 增长（Netgen hinner/houter 概念）
            self.search_candidates(spacing * self.base_front.al)

            # —— 3. 选择点 ——
            selected = self.select_point()

            # —— 4. 成功 ——
            if selected is not None:
                consecutive_fails = 0
                self.update_data()

                if self.debug_level >= 2:
                    info(f"步骤{step}: 创建四面体 {self.base_front.node_ids} "
                         f"+ {selected.idx}, 阵面={len(self.front_list)}, "
                         f"单元={self.num_cells}")

                # 定期清理堆中过期面
                if step % 50 == 0:
                    self._rebuild_heap()
            else:
                # —— 5. 失败 → IncrementClass ——
                consecutive_fails += 1
                face_retry[base_fk] = face_retry.get(base_fk, 0) + 1
                self.base_front.qualclass += 1

                is_wall = self.base_front.bc_type == "wall"
                tol = wall_giveuptol if is_wall else giveuptol

                if self.base_front.qualclass >= tol:
                    if is_wall and face_retry[base_fk] < 100:
                        # 边界面在硬限制内重置继续
                        if self.debug_level >= 2:
                            info(f"步骤{step}: 边界面对{self.base_front.node_ids} "
                                 f"qualclass={self.base_front.qualclass} 重置, "
                                 f"retry={face_retry[base_fk]}")
                        self.base_front.qualclass = tol // 2
                    else:
                        # 内部面或超限边界面：放弃
                        if self.debug_level >= 1:
                            info(f"步骤{step}: {'边界' if is_wall else '内部'}面"
                                 f"{self.base_front.node_ids} "
                                 f"qualclass={self.base_front.qualclass}, "
                                 f"retry={face_retry.get(base_fk, 0)} 放弃")
                        self.deleted_facekeys.add(base_fk)
                        continue

                # 适度增大搜索半径
                self.base_front.al = min(self.base_front.al * 1.1, 2.5)

                if self.debug_level >= 2:
                    info(f"步骤{step}: 阵面{self.base_front.node_ids}选择点失败, "
                         f"qualclass={self.base_front.qualclass}, "
                         f"候选节点={len(self.node_candidates)}")

                heapq.heappush(self.front_list, self.base_front)

                # 堆空且有高 qualclass 边界面：重置继续
                if not self.front_list:
                    self._rebuild_heap()
                    if not self.front_list:
                        break

            # —— 6. 进度与体积检查 ——
            if self.debug_level >= 1 and step % 500 == 0:
                info(f"步骤{step}: 阵面={len(self.front_list)}, "
                     f"节点={self.num_nodes}, 单元={self.num_cells}")
            elif step % 1000 == 0:
                info(f"步骤{step}: 阵面={len(self.front_list)}, "
                     f"节点={self.num_nodes}, 单元={self.num_cells}")

            if step % 100 == 0:
                total_vol = sum(
                    tetrahedron_volume(c.p1, c.p2, c.p3, c.p4)
                    for c in self.cell_container
                    if isinstance(c, Tetrahedron)
                )
                if total_vol > bbox_volume * 2.0:
                    if self.debug_level >= 1:
                        info(f"步骤{step}: 体积超过包围盒2倍，提前终止")
                    break

            if consecutive_fails >= max_consecutive_fails:
                warning(f"步骤{step}: 连续失败{consecutive_fails}次, "
                        f"当前阵面={len(self.front_list)}, "
                        f"候选节点={len(self.node_candidates)}")
                # 重置所有面的 qualclass 尝试打破僵局
                for f in self.front_list:
                    if f.qualclass > 5 and f.bc_type == "wall":
                        f.qualclass = max(1, f.qualclass - 3)
                heapq.heapify(self.front_list)
                consecutive_fails = 0

        # —— 结束 ——
        end_reason = "完成"
        if step >= max_steps:
            end_reason = f"达到步数上限({max_steps})"
        elif not self.front_list:
            end_reason = "阵面列表为空"

        info(f"推进结束: {end_reason}, 总步数={step}, "
             f"节点={self.num_nodes}, 四面体={self.num_cells}, "
             f"剩余阵面={len(self.front_list)}, "
             f"删除面={len(self.deleted_facekeys)}")

        # 最终清理并重建堆（可能残留陈旧面）
        self._rebuild_heap()

        self.construct_unstr_grid()
        timer.show_to_console("三维四面体网格生成完成.")
        return self.unstr_grid

    def _rebuild_heap(self):
        """移除堆中的过期阵面并重建"""
        n_before = len(self.front_list)
        self.front_list = [f for f in self.front_list
                           if f.face_key not in self.deleted_facekeys
                           and not f.is_satisfied]
        heapq.heapify(self.front_list)
        if self.debug_level >= 2 and len(self.front_list) < n_before:
            info(f"堆重建: {n_before} → {len(self.front_list)} (移除{n_before - len(self.front_list)}个过期面)")

    # ============ 候选搜索 ============

    def search_candidates(self, search_radius: float):
        """搜索候选节点、阵面、单元（RTree 空间查询）"""
        self.node_candidates = []
        self.front_candidates = []
        self.cell_candidates = []

        if self.space_index_node is not None:
            node_ids = get_candidate_elements_id_3d(
                self.base_front, self.space_index_node, search_radius)
            self.node_candidates = [
                self.node_dict[nid] for nid in node_ids if nid in self.node_dict]

        if self.space_index_front is not None:
            front_ids = get_candidate_elements_id_3d(
                self.base_front, self.space_index_front, search_radius)
            self.front_candidates = [
                self.front_dict[fid] for fid in front_ids
                if fid in self.front_dict]

        if self.space_index_cell is not None:
            cell_ids = get_candidate_elements_id_3d(
                self.base_front, self.space_index_cell, search_radius)
            self.cell_candidates = [
                self.cell_dict[cid] for cid in cell_ids
                if cid in self.cell_dict]

    # ============ 点选择 ============

    def select_point(self):
        """选择推进点 — 参考 Netgen 的点选择策略"""
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords
        p2 = self.base_front.node_elems[2].coords

        self.pselected = None

        # 评分候选节点
        scored_candidates = []
        low_quality_candidates = []
        for node_elem in self.node_candidates:
            if node_elem in self.base_front.node_elems:
                continue
            quality = tetrahedron_shape_quality(p0, p1, p2, node_elem.coords)
            if quality > 0.1:
                scored_candidates.append((quality, node_elem))
            elif quality > 0.01:
                low_quality_candidates.append((quality, node_elem))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        low_quality_candidates.sort(key=lambda x: x[0], reverse=True)

        def _check_node_candidate(node_elem):
            if not self._is_correct_side(node_elem):
                return False
            if node_elem not in self.boundary_nodes:
                if not self._is_inside_boundary(node_elem.coords):
                    return False
            if self._is_cross(node_elem):
                return False
            self.pselected = node_elem
            return True

        # 1. 尝试已有候选节点（前10个高质量）
        for quality, node_elem in scored_candidates[:10]:
            if self.debug_level >= 3:
                info(f"  候选点 {node_elem.idx}: quality={quality:.3f}")
            if _check_node_candidate(node_elem):
                return self.pselected

        # 2. 尝试创建新点（理想正四面体高度 ≈ 0.816 * edge_length）
        center = self.base_front.center
        normal = self.base_front.normal
        spacing = self.sizing_system.spacing_at(center)

        def _try_new_point(coords):
            candidate = NodeElement(
                coords, self.num_nodes,
                part_name="interior-node", bc_type="interior")
            candidate.bbox = [coords[0], coords[1], coords[2],
                              coords[0], coords[1], coords[2]]
            quality = tetrahedron_shape_quality(p0, p1, p2, coords)
            if quality <= 0.1:
                return None
            if not self._is_correct_side(candidate):
                return None
            if not self._is_inside_boundary(coords):
                return None
            if self._is_cross(candidate):
                return None
            self.pselected = candidate
            return candidate

        for dist_factor in [0.82, 0.60, 0.40]:
            dist = spacing * dist_factor
            candidate_coords = [
                center[0] + normal[0] * dist,
                center[1] + normal[1] * dist,
                center[2] + normal[2] * dist,
            ]
            if _try_new_point(candidate_coords):
                return self.pselected

        # 3. 边界面额外策略（随 qualclass 增长自动重试）
        if self.base_front.bc_type == "wall":
            for dist_factor in [0.70, 0.50, 0.35]:
                dist = spacing * dist_factor
                for ne in self.base_front.node_elems:
                    v = ne.coords
                    candidate_coords = [
                        v[0] + normal[0] * dist,
                        v[1] + normal[1] * dist,
                        v[2] + normal[2] * dist,
                    ]
                    if _try_new_point(candidate_coords):
                        return self.pselected

            dist = spacing * 0.25
            candidate_coords = [
                center[0] + normal[0] * dist,
                center[1] + normal[1] * dist,
                center[2] + normal[2] * dist,
            ]
            if _try_new_point(candidate_coords):
                return self.pselected

            # 放宽到更多已有节点
            for quality, node_elem in scored_candidates[10:50]:
                if _check_node_candidate(node_elem):
                    return self.pselected

            # 低质量候选
            for quality, node_elem in low_quality_candidates[:30]:
                if _check_node_candidate(node_elem):
                    if self.debug_level >= 2:
                        info(f"  使用低质量候选点 {node_elem.idx}: quality={quality:.3f}")
                    return self.pselected

            # 借用相邻单元节点
            front_node_set = set(self.base_front.node_ids)
            for cell in self.cell_candidates:
                if not isinstance(cell, Tetrahedron):
                    continue
                cell_ids = list(cell.node_ids)
                shared = front_node_set.intersection(cell_ids)
                if len(shared) >= 2:
                    for nid in cell_ids:
                        if nid in shared:
                            continue
                        if nid < len(self.node_coords):
                            nc = self.node_coords[nid]
                            candidate = NodeElement(
                                nc, nid,
                                part_name="interior-node", bc_type="interior")
                            candidate.bbox = [nc[0], nc[1], nc[2],
                                              nc[0], nc[1], nc[2]]
                            quality = tetrahedron_shape_quality(p0, p1, p2, nc)
                            if quality > 0.01:
                                if _check_node_candidate(candidate):
                                    if self.debug_level >= 2:
                                        info(f"  借用相邻单元节点 {nid}: quality={quality:.3f}")
                                    return self.pselected

        return self.pselected

    # ============ 几何检查 ============

    def _is_correct_side(self, node_elem) -> bool:
        """检查点是否在阵面法向量一侧"""
        p0 = np.array(self.base_front.node_elems[0].coords)
        normal = np.array(self.base_front.normal)
        v = np.array(node_elem.coords) - p0
        return np.dot(v, normal) > 1e-12

    def _is_inside_boundary(self, point) -> bool:
        """射线法检查点是否在边界内部"""
        if self._raycast_data is None:
            return True
        return classify_points_inside([np.array(point)], self._raycast_data)[0]

    @staticmethod
    def _bbox_intersect(b1, b2):
        """两个包围盒是否相交 (xmin,ymin,zmin, xmax,ymax,zmax)"""
        return not (b1[3] < b2[0] or b2[3] < b1[0] or
                    b1[4] < b2[1] or b2[4] < b1[1] or
                    b1[5] < b2[2] or b2[5] < b1[2])

    @staticmethod
    def _compute_bbox(coords):
        """从坐标列表计算包围盒"""
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def _is_cross(self, node_elem) -> bool:
        """检查新四面体是否与现有元素相交"""
        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]
        p2 = self.base_front.node_elems[2]
        p3 = node_elem

        new_tet_coords = [p0.coords, p1.coords, p2.coords, p3.coords]
        new_tet_ids = [p0.idx, p1.idx, p2.idx, p3.idx]
        new_tet_id_set = set(new_tet_ids)
        new_tet_bbox = self._compute_bbox(new_tet_coords)

        # 检查与现有单元的相交
        for cell in self.cell_candidates:
            if not isinstance(cell, Tetrahedron):
                continue
            cell_coords = [cell.p1, cell.p2, cell.p3, cell.p4]
            cell_ids = cell.node_ids

            if not self._bbox_intersect(new_tet_bbox, cell.bbox):
                continue

            shared_nodes = new_tet_id_set.intersection(set(cell_ids))
            # 完全相同
            if len(shared_nodes) == 4:
                return True
            # 共享3节点（面相邻），检查第4节点是否在面的同侧
            if len(shared_nodes) == 3:
                shared_ids = sorted(shared_nodes)
                new_other = [new_tet_coords[i] for i, nid in enumerate(new_tet_ids)
                             if nid not in shared_ids][0]
                cell_other = [cell_coords[i] for i, cid in enumerate(cell_ids)
                              if cid not in shared_ids][0]
                s0 = np.array(new_tet_coords[new_tet_ids.index(shared_ids[0])])
                s1 = np.array(new_tet_coords[new_tet_ids.index(shared_ids[1])])
                s2 = np.array(new_tet_coords[new_tet_ids.index(shared_ids[2])])
                face_normal = np.cross(s1 - s0, s2 - s0)
                d1 = np.dot(np.array(new_other) - s0, face_normal)
                d2 = np.dot(np.array(cell_other) - s0, face_normal)
                if d1 * d2 > 0:
                    return True
                continue

            # 顶点包含检查
            for i, coord in enumerate(new_tet_coords):
                if new_tet_ids[i] in shared_nodes:
                    continue
                if point_in_tet(coord, cell_coords):
                    if self.debug_level >= 3:
                        info(f"  点{new_tet_ids[i]}在单元{cell_ids}内部")
                    return True

            for i, coord in enumerate(cell_coords):
                if cell_ids[i] in new_tet_id_set:
                    continue
                if point_in_tet(coord, new_tet_coords):
                    if self.debug_level >= 3:
                        info(f"  点{cell_ids[i]}在新四面体内部")
                    return True

            # 面-面相交检测
            if len(shared_nodes) < 3:
                cell_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
                new_tet_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

                for cfi, cfj, cfk in cell_faces:
                    cell_face_ids = [cell_ids[cfi], cell_ids[cfj], cell_ids[cfk]]
                    cell_face_set = set(cell_face_ids)
                    cell_face_coords = [cell_coords[cfi], cell_coords[cfj], cell_coords[cfk]]
                    face1_bbox = self._compute_bbox(cell_face_coords)

                    for nfi, nfj, nfk in new_tet_faces:
                        new_face_ids = [new_tet_ids[nfi], new_tet_ids[nfj], new_tet_ids[nfk]]
                        if not cell_face_set.isdisjoint(new_face_ids):
                            continue
                        new_face_coords = [new_tet_coords[nfi], new_tet_coords[nfj], new_tet_coords[nfk]]
                        if not self._bbox_intersect(face1_bbox, self._compute_bbox(new_face_coords)):
                            continue
                        if intersect_triangle_triangle(cell_face_coords, new_face_coords,
                                                       cell_face_ids, new_face_ids):
                            if self.debug_level >= 3:
                                info(f"  面相交: 单元面{cell_face_ids} vs 新面{new_face_ids}")
                            return True

        # 检查与非邻接阵面的相交
        for front in self.front_candidates:
            if front is self.base_front:
                continue
            if front.is_satisfied:
                continue
            if front.bc_type == "wall":
                continue

            if not self._bbox_intersect(new_tet_bbox, front.bbox):
                continue

            front_ids = front.node_ids
            if not new_tet_id_set.isdisjoint(front_ids):
                continue

            tri_coords = [n.coords for n in front.node_elems]
            tri_ids = front.node_ids
            if intersect_tet_triangle(new_tet_coords, tri_coords, new_tet_ids, tri_ids):
                return True
            if intersect_tet_triangle2(new_tet_coords, tri_coords, new_tet_ids, tri_ids):
                return True

        return False

    # ============ 数据更新 ============

    def update_data(self):
        """创建四面体并更新阵面列表"""
        if self.pselected is None:
            return

        self._update_nodes()

        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]
        p2 = self.base_front.node_elems[2]
        p3 = self.pselected

        # 创建新阵面，确保法向量指向四面体外部（远离第4节点）
        def _make_face(na, nb, nc, fourth):
            a = np.array(na.coords)
            b = np.array(nb.coords)
            c = np.array(nc.coords)
            f = np.array(fourth.coords)
            fnormal = np.cross(b - a, c - a)
            if np.dot(f - a, fnormal) > 0:
                return Front3D(na, nc, nb, bc_type="interior", part_name="interior")
            else:
                return Front3D(na, nb, nc, bc_type="interior", part_name="interior")

        new_faces = [
            _make_face(p2, p1, p3, p0),
            _make_face(p0, p3, p2, p1),
            _make_face(p3, p1, p0, p2),
        ]

        # 标记基准面已消耗
        base_face_key = self.base_front.face_key
        self.base_front.left = 1
        self.base_front.right = 1
        self.deleted_facekeys.add(base_face_key)

        # 从空间索引移除基准面
        base_fid = self.front_id_by_facekey.pop(base_face_key, None)
        if base_fid is not None and base_fid in self.front_dict:
            del self.front_dict[base_fid]

        self._update_fronts(new_faces)

        # 创建四面体
        new_tet = Tetrahedron(
            p0, p1, p2, p3,
            part_name='interior-tetrahedron',
            idx=self.num_cells,
        )
        all_coords = [new_tet.p1, new_tet.p2, new_tet.p3, new_tet.p4]
        new_tet.bbox = [
            min(c[0] for c in all_coords),
            min(c[1] for c in all_coords),
            min(c[2] for c in all_coords),
            max(c[0] for c in all_coords),
            max(c[1] for c in all_coords),
            max(c[2] for c in all_coords),
        ]
        self._update_cells(new_tet)

    def _update_nodes(self):
        """更新节点 — 注册新节点并计算 frontnr"""
        node = self.pselected
        node_hash = node.hash

        if node_hash not in self.node_hash_list:
            self.node_hash_list.add(node_hash)
            node.idx = self.num_nodes
            self.node_coords.append(node.coords)
            self.node_elem_by_hash[node_hash] = node
            self.front_node_list.append(node)
            add_elems_to_space_index_3d_with_RTree(
                [node], self.space_index_node, self.node_dict)

            # Netgen DecFrontNr: 新节点 frontnr = min(面节点frontnr) + 1
            base_nodes = self.base_front.node_elems
            min_fn = min(self.node_frontnr.get(n.hash, 1000) for n in base_nodes)
            self.node_frontnr[node_hash] = min_fn + 1
            self.num_nodes += 1
        else:
            existing = self.node_elem_by_hash.get(node_hash)
            if existing is not None:
                self.pselected = existing

    def _update_fronts(self, new_fronts):
        """更新阵面 — 匹配已有面或添加新面，含 qualclass/frontnr 初始化"""
        for front in new_fronts:
            face_key = front.face_key

            if face_key in self.deleted_facekeys:
                continue

            if face_key in self.front_id_by_facekey:
                # 面已存在 → 内部面，标记消耗
                existing_fid = self.front_id_by_facekey.get(face_key)
                existing = self.front_dict.get(existing_fid)
                if existing is not None:
                    existing.left = 1
                    existing.right = 1
                self.deleted_facekeys.add(face_key)
                self.front_id_by_facekey.pop(face_key, None)
                if existing_fid is not None and existing_fid in self.front_dict:
                    del self.front_dict[existing_fid]
            else:
                # 新面 → 初始化 qualclass 和 frontnr
                front.qualclass = 1
                fn_sum = sum(self.node_frontnr.get(n.hash, 0)
                             for n in front.node_elems)
                front.frontnr_sum = fn_sum

                # Netgen DecFrontNr: 面的节点降低 frontnr
                min_fn = min(self.node_frontnr.get(n.hash, 1000)
                            for n in front.node_elems)
                for n in front.node_elems:
                    cur = self.node_frontnr.get(n.hash, 1000)
                    if cur > min_fn + 1:
                        self.node_frontnr[n.hash] = min_fn + 1

                front.right = 1  # 一侧被占据
                heapq.heappush(self.front_list, front)
                add_elems_to_space_index_3d_with_RTree(
                    [front], self.space_index_front, self.front_dict)
                self.front_id_by_facekey[face_key] = id(front)

    def _update_cells(self, new_cell):
        """更新单元"""
        cell_hash = new_cell.hash
        if cell_hash not in self.cell_hash_list:
            self.cell_hash_list.add(cell_hash)
            self.cell_container.append(new_cell)
            add_elems_to_space_index_3d_with_RTree(
                [new_cell], self.space_index_cell, self.cell_dict)
            self.num_cells += 1
        else:
            warning(f"发现重复单元：{new_cell.node_ids}")

    # ============ 后处理 ============

    def construct_unstr_grid(self):
        """构建非结构网格"""
        self.unstr_grid = Unstructured_Grid(
            self.cell_container,
            self.node_coords,
            list(self.boundary_nodes),
            grid_dimension=3,
        )

    def export_to_vtk(self, filename: str):
        """导出 VTK"""
        cell_idx = [list(c.node_ids) for c in self.cell_container]
        cell_types = [VTKCellType.TETRA] * len(self.cell_container)
        write_vtk(filename, self.node_coords, cell_idx, [], cell_types)

    def get_quality_stats(self) -> dict:
        """获取质量统计"""
        return compute_mesh_quality_stats(self.cell_container)

    # ============ 工具方法 ============

    @staticmethod
    def _compute_mean_edge(surface_triangles) -> float:
        """从表面三角形计算平均边长作为均匀间距"""
        edges = []
        for tri in surface_triangles:
            for i in range(3):
                for j in range(i + 1, 3):
                    dx = tri.nodes[i].coords[0] - tri.nodes[j].coords[0]
                    dy = tri.nodes[i].coords[1] - tri.nodes[j].coords[1]
                    dz = tri.nodes[i].coords[2] - tri.nodes[j].coords[2]
                    edges.append((dx * dx + dy * dy + dz * dz) ** 0.5)
        return float(np.mean(edges)) if edges else 1.0

    def _estimate_volume(self) -> float:
        """估算体积"""
        if not self.node_coords:
            return 1.0
        coords = np.array(self.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        return float(np.prod(maxs - mins))

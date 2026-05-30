import heapq
import numpy as np

from utils.geom_toolkit import tetrahedron_volume, calculate_distance
from optimize.mesh_quality import tetrahedron_shape_quality
from data_structure.basic_elements import NodeElement, Tetrahedron, is_node_element
from data_structure.unstructured_grid import Unstructured_Grid
from data_structure.rtree_space import (
    build_space_index_3d_with_RTree,
    get_candidate_elements_id_3d,
    add_elems_to_space_index_3d_with_RTree,
)
from utils.timer import TimeSpan
from utils.message import info, warning, error
from adfront3.front3d import Front3D
from adfront3.sizing3d import UniformSizing3D


class Adfront3:
    """三维阵面推进法，生成四面体网格"""

    def __init__(self, surface_triangles, sizing_system=None, debug_level=0):
        self.debug_level = debug_level
        self.al = 3.0
        self.discount = 0.8
        self.progress_interval = 100

        self.surface_triangles = surface_triangles

        # 根据边界面网格自动确定最大间距
        max_spacing = self._compute_max_edge_length(surface_triangles)
        if sizing_system is None:
            sizing_system = UniformSizing3D(max_spacing)
        self.sizing_system = sizing_system

        self.front_list = []
        self.base_front = None
        self.pbest = None
        self.pselected = None
        self.best_flag = False

        self.search_radius = None
        self.node_candidates = []
        self.front_candidates = []
        self.cell_candidates = []

        self.space_index_node = None
        self.node_dict = {}
        self.space_index_front = None
        self.front_dict = {}
        self.front_id_by_facekey = {}  # face_key -> id(front_obj) for correct deletion
        self.space_index_cell = None
        self.cell_dict = {}

        self.num_cells = 0
        self.num_nodes = 0
        self.cell_container = []
        self.node_coords = []
        self.front_node_list = []

        self.boundary_nodes = set()
        self.unstr_grid = None

        self.node_hash_list = set()
        self.node_elem_by_hash = {}
        self.cell_hash_list = set()
        self.face_usage = {}

        self.boundary_halfspaces = []  # (inward_normal, ref_point) for containment check
        self.base_face_used = set()  # 已作为基准面使用过的 face_key
        self.initialize()

    def initialize(self):
        """将曲面三角形转换为 Front3D 阵面和 NodeElement 节点"""
        # 计算曲面质心用于法向量方向判断
        all_coords = []
        for tri in self.surface_triangles:
            for node3d in tri.nodes:
                all_coords.append(node3d.coords)
        centroid = np.mean(all_coords, axis=0) if all_coords else np.zeros(3)

        front_idx = 0
        for tri in self.surface_triangles:
            node_elems = []
            for node3d in tri.nodes:
                node_hash = hash(tuple(f"{c:.6f}" for c in node3d.coords))
                if node_hash in self.node_hash_list:
                    existing = self.node_elem_by_hash[node_hash]
                    node_elems.append(existing)
                else:
                    new_node = NodeElement(
                        list(node3d.coords),
                        self.num_nodes,
                        part_name=getattr(node3d, 'part_name', None) or "wall",
                        bc_type="wall",
                    )
                    c = new_node.coords
                    new_node.bbox = [c[0], c[1], c[2], c[0], c[1], c[2]]
                    self.node_hash_list.add(node_hash)
                    self.node_elem_by_hash[node_hash] = new_node
                    self.node_coords.append(new_node.coords)
                    self.front_node_list.append(new_node)
                    self.boundary_nodes.add(new_node)
                    node_elems.append(new_node)
                    self.num_nodes += 1

            # 判断法向量方向：确保 Front3D 法向量指向体积内部
            # 通过质心方向判断：法向量应指向质心一侧
            tri_center = np.mean([n.coords for n in tri.nodes], axis=0)
            to_centroid = centroid - tri_center
            tri_normal = np.array(tri.normal)
            dot = np.dot(tri_normal, to_centroid)

            if dot > 0:
                # SurfaceTriangle 法向量已指向内部，保持原始绕向
                front = Front3D(
                    node_elems[0], node_elems[1], node_elems[2],
                    idx=front_idx,
                    bc_type="wall",
                    part_name="wall",
                    al=self.al,
                )
            else:
                # SurfaceTriangle 法向量指向外部，反转绕向
                front = Front3D(
                    node_elems[0], node_elems[2], node_elems[1],
                    idx=front_idx,
                    bc_type="wall",
                    part_name="wall",
                    al=self.al,
                )

            self.front_list.append(front)
            front_idx += 1

            # 记录边界半空间用于包含检测
            if front.area > 1e-30:
                self.boundary_halfspaces.append(
                    (list(front.normal), list(front.node_elems[0].coords))
                )

        heapq.heapify(self.front_list)
        self._build_spatial_index()

        if self.debug_level >= 1:
            info(f"初始化完成: 节点={self.num_nodes}, 阵面={len(self.front_list)}")

    def _build_spatial_index(self):
        """构建三维 RTree 空间索引"""
        # NodeElement.bbox 是 [x,y,x,y,z,z]，需修正为 RTree 3D 格式 [x,y,z,x,y,z]
        for node in self.front_node_list:
            c = node.coords
            node.bbox = [c[0], c[1], c[2], c[0], c[1], c[2]]

        if self.front_node_list:
            self.node_dict, self.space_index_node = build_space_index_3d_with_RTree(
                self.front_node_list
            )
        if self.front_list:
            self.front_dict, self.space_index_front = build_space_index_3d_with_RTree(
                self.front_list
            )
            # 建立 face_key -> id 映射
            for front in self.front_list:
                self.front_id_by_facekey[front.face_key] = id(front)

    def generate(self, max_steps=None):
        """主推进循环，生成四面体网格，返回 Unstructured_Grid"""
        timer = TimeSpan("开始三维四面体网格生成...")
        step = 0

        if max_steps is None:
            # 根据体积和间距估算最大步数
            volume = self._estimate_volume()
            spacing = self.sizing_system.global_spacing
            tet_volume = (spacing ** 3) / (6 * 2 ** 0.5)
            max_steps = int(volume / tet_volume * 3)  # 留3倍余量

        while self.front_list and step < max_steps:
            step += 1
            self.base_front = heapq.heappop(self.front_list)

            # 跳过已使用过的基准面或已被消耗的阵面
            base_fk = self.base_front.face_key
            if base_fk in self.base_face_used or self.face_usage.get(base_fk, 0) >= 2:
                continue

            spacing = self.sizing_system.spacing_at(self.base_front.center)

            self.add_new_point(spacing)
            self.search_candidates(self.base_front.al * spacing)

            selected = self.select_point()

            if selected is None:
                self.base_front.al *= 1.2
                if self.base_front.al > 20:
                    if self.debug_level >= 1:
                        warning(f"阵面{self.base_front.node_ids}搜索半径超过20，放弃该阵面")
                    continue
                heapq.heappush(self.front_list, self.base_front)
                continue

            self.update_data()

            if step % self.progress_interval == 0:
                info(f"步骤{step}: 阵面={len(self.front_list)}, "
                     f"节点={self.num_nodes}, 单元={self.num_cells}")

        self.construct_unstr_grid()
        timer.show_to_console("三维四面体网格生成完成.")
        info(f"最终: 节点={self.num_nodes}, 四面体={self.num_cells}")
        return self.unstr_grid

    @staticmethod
    def _compute_max_edge_length(surface_triangles):
        """从边界面网格计算最大边长，作为体积网格间距参考"""
        max_len = 0.0
        for tri in surface_triangles:
            nodes = tri.nodes
            for i in range(3):
                for j in range(i + 1, 3):
                    dx = nodes[i].coords[0] - nodes[j].coords[0]
                    dy = nodes[i].coords[1] - nodes[j].coords[1]
                    dz = nodes[i].coords[2] - nodes[j].coords[2]
                    edge_len = (dx * dx + dy * dy + dz * dz) ** 0.5
                    if edge_len > max_len:
                        max_len = edge_len
        return max_len if max_len > 0 else 1.0

    def _estimate_volume(self):
        """估算封闭体积（使用包围盒）"""
        if not self.node_coords:
            return 1.0
        coords = np.array(self.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        return float(np.prod(maxs - mins))

    def add_new_point(self, spacing):
        """沿法向量方向推进生成理想点 pbest"""
        center = self.base_front.center
        normal = self.base_front.normal

        # 尝试不同的推进距离，选择第一个在边界内部的
        for factor in [1.0, 0.5, 0.25, 0.1, 0.05]:
            advance = spacing * factor
            candidate = [
                center[0] + normal[0] * advance,
                center[1] + normal[1] * advance,
                center[2] + normal[2] * advance,
            ]
            if self._is_inside_boundary(candidate):
                self.pbest = NodeElement(
                    candidate, self.num_nodes,
                    part_name="interior-node", bc_type="interior",
                )
                self.pbest.bbox = [candidate[0], candidate[1], candidate[2],
                                   candidate[0], candidate[1], candidate[2]]
                return self.pbest

        # 所有推进距离都在边界外，使用微小偏移
        eps = spacing * 0.01
        pbest = [
            center[0] + normal[0] * eps,
            center[1] + normal[1] * eps,
            center[2] + normal[2] * eps,
        ]
        self.pbest = NodeElement(
            pbest, self.num_nodes,
            part_name="interior-node", bc_type="interior",
        )
        self.pbest.bbox = [pbest[0], pbest[1], pbest[2], pbest[0], pbest[1], pbest[2]]
        return self.pbest

    def search_candidates(self, search_radius):
        """使用 3D RTree 搜索候选节点、阵面和单元"""
        self.search_radius = search_radius
        self.node_candidates = []
        self.front_candidates = []
        self.cell_candidates = []

        if self.space_index_node is not None:
            node_ids = get_candidate_elements_id_3d(
                self.base_front, self.space_index_node, search_radius
            )
            self.node_candidates = [
                self.node_dict[nid] for nid in node_ids if nid in self.node_dict
            ]

        if self.space_index_front is not None:
            front_ids = get_candidate_elements_id_3d(
                self.base_front, self.space_index_front, search_radius
            )
            self.front_candidates = [
                self.front_dict[fid] for fid in front_ids if fid in self.front_dict
            ]

        if self.space_index_cell is not None:
            cell_ids = get_candidate_elements_id_3d(
                self.base_front, self.space_index_cell, search_radius
            )
            self.cell_candidates = [
                self.cell_dict[cid] for cid in cell_ids if cid in self.cell_dict
            ]

    def select_point(self):
        """从候选点中选择最佳推进点，优先使用已有节点"""
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords
        p2 = self.base_front.node_elems[2].coords

        scored_candidates = []
        for node_elem in self.node_candidates:
            # 跳过基准面自身的节点
            if node_elem in self.base_front.node_elems:
                continue
            quality = tetrahedron_shape_quality(p0, p1, p2, node_elem.coords)
            if quality > 0:
                scored_candidates.append((quality, node_elem))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        self.pselected = None
        self.best_flag = False
        for quality, node_elem in scored_candidates:
            # 包含性检查（对凸域，_is_inside_boundary 已隐含 _is_correct_side）
            if not self._is_inside_boundary(node_elem.coords):
                continue
            if self._is_cross(node_elem):
                continue
            self.pselected = node_elem
            break

        if self.pselected is None:
            if self.debug_level >= 2:
                warning(f"阵面{self.base_front.node_ids}未找到合适推进点，扩大搜索范围")

        return self.pselected

    def _is_correct_side(self, node_elem):
        """检查候选点是否在阵面内侧（法向量方向）"""
        p0 = np.array(self.base_front.node_elems[0].coords)
        normal = np.array(self.base_front.normal)
        v = np.array(node_elem.coords) - p0
        return np.dot(v, normal) > 1e-12

    def _is_inside_boundary(self, point):
        """检查点是否在封闭体积内部（所有边界半空间内侧）"""
        pt = np.array(point)
        for normal, ref in self.boundary_halfspaces:
            v = pt - np.array(ref)
            if np.dot(v, normal) < -1e-6:
                return False
        return True

    def _is_cross(self, node_elem):
        """检查新四面体是否与现有单元相交"""
        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]
        p2 = self.base_front.node_elems[2]
        p3 = node_elem

        new_tet_nodes = {p0.idx, p1.idx, p2.idx, p3.idx}
        new_tet_coords = [p0.coords, p1.coords, p2.coords, p3.coords]

        # 检查新四面体的所有面是否已饱和（每个面最多被2个四面体共享）
        new_tet_hashes = [p0.hash, p1.hash, p2.hash, p3.hash]
        for i in range(4):
            for j in range(i+1, 4):
                for k in range(j+1, 4):
                    face_key = tuple(sorted([new_tet_hashes[i], new_tet_hashes[j], new_tet_hashes[k]]))
                    if self.face_usage.get(face_key, 0) >= 2:
                        return True

        # 检查与现有四面体的相交
        for cell in self.cell_candidates:
            if not isinstance(cell, Tetrahedron):
                continue
            shared = new_tet_nodes & set(cell.node_ids)
            if len(shared) >= 3:
                continue
            if len(shared) >= 2:
                continue
            # 无共享或1个共享节点：检查包围盒和点包含
            if self._bbox_overlap_3d(new_tet_coords, [cell.p1, cell.p2, cell.p3, cell.p4]):
                if self._tets_intersect(new_tet_coords, [cell.p1, cell.p2, cell.p3, cell.p4]):
                    return True

        return False

    def _bbox_overlap_3d(self, coords1, coords2):
        """检查两个点集的包围盒是否重叠"""
        eps = 1e-10
        min1 = [min(c[i] for c in coords1) - eps for i in range(3)]
        max1 = [max(c[i] for c in coords1) + eps for i in range(3)]
        min2 = [min(c[i] for c in coords2) - eps for i in range(3)]
        max2 = [max(c[i] for c in coords2) + eps for i in range(3)]
        return all(min1[i] <= max2[i] and max1[i] >= min2[i] for i in range(3))

    def _point_in_tet(self, p, tet_coords):
        """检查点是否在四面体内部（使用有符号体积法）"""
        p0, p1, p2, p3 = [np.array(c) for c in tet_coords]
        pt = np.array(p)

        v0 = np.dot(pt - p0, np.cross(p1 - p0, p2 - p0))
        v1 = np.dot(pt - p0, np.cross(p2 - p0, p3 - p0))
        v2 = np.dot(pt - p1, np.cross(p3 - p1, p0 - p1))
        v3 = np.dot(pt - p2, np.cross(p0 - p2, p3 - p2))

        ref = np.dot(p3 - p0, np.cross(p1 - p0, p2 - p0))
        if abs(ref) < 1e-30:
            return False

        if ref > 0:
            return v0 > 1e-10 and v1 > 1e-10 and v2 > 1e-10 and v3 > 1e-10
        else:
            return v0 < -1e-10 and v1 < -1e-10 and v2 < -1e-10 and v3 < -1e-10

    def _tets_intersect(self, coords1, coords2):
        """检查两个四面体是否相交"""
        for c in coords1:
            if self._point_in_tet(c, coords2):
                return True
        for c in coords2:
            if self._point_in_tet(c, coords1):
                return True
        return False

    def _edge_intersects_triangle(self, edge, tri_coords):
        """检查线段是否与三角形相交"""
        p0 = np.array(edge[0])
        p1 = np.array(edge[1])
        a, b, c = [np.array(x) for x in tri_coords]

        edge_vec = p1 - p0
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            return False

        normal = np.cross(b - a, c - a)
        normal_len = np.linalg.norm(normal)
        if normal_len < 1e-30:
            return False
        normal = normal / normal_len

        denom = np.dot(normal, edge_vec)
        if abs(denom) < 1e-12:
            return False

        t = np.dot(normal, a - p0) / denom
        if t < 1e-8 or t > 1 - 1e-8:
            return False

        hit = p0 + t * edge_vec

        v0 = c - a
        v1 = b - a
        v2 = hit - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-30)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        return u >= -1e-8 and v >= -1e-8 and (u + v) <= 1 + 1e-8

    def update_data(self):
        """创建四面体，更新节点、阵面和单元"""
        if self.pselected is None:
            return

        self._update_nodes()

        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]
        p2 = self.base_front.node_elems[2]
        p3 = self.pselected

        # 三个新阵面（不含基准面）
        new_faces = [
            Front3D(p1, p2, p3, idx=-1, bc_type="interior", part_name="interior"),
            Front3D(p2, p0, p3, idx=-1, bc_type="interior", part_name="interior"),
            Front3D(p0, p1, p3, idx=-1, bc_type="interior", part_name="interior"),
        ]

        # 先记录基准面已被消耗（在处理新面之前，确保匹配检测正确）
        base_face_key = self.base_front.face_key
        self.face_usage[base_face_key] = self.face_usage.get(base_face_key, 0) + 1
        self.base_face_used.add(base_face_key)

        self._update_fronts(new_faces)

        new_tet = Tetrahedron(
            p0, p1, p2, p3,
            part_name='interior-tetrahedron',
            idx=self.num_cells,
        )
        self._update_cells(new_tet)

        # 移除基准阵面（已从堆中弹出，需从索引中清除）
        base_fid = self.front_id_by_facekey.pop(base_face_key, None)
        if base_fid is not None and base_fid in self.front_dict:
            del self.front_dict[base_fid]

        heapq.heapify(self.front_list)

    def _update_nodes(self):
        """更新节点列表"""
        node = self.pselected
        node_hash = node.hash
        if node_hash not in self.node_hash_list:
            self.node_hash_list.add(node_hash)
            node.idx = self.num_nodes
            self.node_coords.append(node.coords)
            self.node_elem_by_hash[node_hash] = node
            self.front_node_list.append(node)
            self.add_elems_to_space_index([node], self.space_index_node, self.node_dict)
            self.num_nodes += 1
        else:
            existing = self.node_elem_by_hash.get(node_hash)
            if existing is not None:
                self.pselected.idx = existing.idx

    def _update_fronts(self, new_fronts):
        """更新阵面列表：新面若已存在则移除（内部面），否则添加"""
        for front in new_fronts:
            face_key = front.face_key
            if face_key in self.face_usage:
                # 第二次使用，移除已有阵面（变为内部面）
                self.face_usage[face_key] += 1
                self._remove_front_by_hash(face_key)
                if self.debug_level >= 2:
                    print(f'  CONSUMED face {face_key}, remaining: {len(self.front_list)}')
            else:
                # 首次使用，添加到阵面列表
                self.face_usage[face_key] = 1
                self.front_list.append(front)
                self.add_elems_to_space_index(
                    [front], self.space_index_front, self.front_dict
                )
                self.front_id_by_facekey[face_key] = id(front)

    def _remove_front_by_hash(self, face_key):
        """从阵面列表中移除指定 face_key 的阵面"""
        self.front_list = [f for f in self.front_list if f.face_key != face_key]
        # 使用 face_key -> id 映射从 front_dict 中移除
        fid = self.front_id_by_facekey.pop(face_key, None)
        if fid is not None and fid in self.front_dict:
            del self.front_dict[fid]

    def _update_cells(self, new_cell):
        """更新单元列表"""
        cell_hash = new_cell.hash
        if cell_hash not in self.cell_hash_list:
            self.cell_hash_list.add(cell_hash)
            self.cell_container.append(new_cell)
            self.add_elems_to_space_index(
                [new_cell], self.space_index_cell, self.cell_dict
            )
            self.num_cells += 1
        else:
            warning(f"发现重复单元：{new_cell.node_ids}")

    def add_elems_to_space_index(self, elems, space_index, elem_dict):
        """向 RTree 空间索引添加元素"""
        if space_index is not None:
            add_elems_to_space_index_3d_with_RTree(elems, space_index, elem_dict)

    def construct_unstr_grid(self):
        """构建非结构网格对象"""
        self.unstr_grid = Unstructured_Grid(
            self.cell_container,
            self.node_coords,
            list(self.boundary_nodes),
            grid_dimension=3,
        )

    def export_to_vtk(self, filename):
        """导出四面体网格为 VTK 文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Tetrahedral Mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            f.write(f"POINTS {len(self.node_coords)} float\n")
            for coord in self.node_coords:
                f.write(f"{coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")

            n_cells = len(self.cell_container)
            f.write(f"\nCELLS {n_cells} {5 * n_cells}\n")
            for cell in self.cell_container:
                ids = cell.node_ids
                f.write(f"4 {ids[0]} {ids[1]} {ids[2]} {ids[3]}\n")

            f.write(f"\nCELL_TYPES {n_cells}\n")
            for _ in range(n_cells):
                f.write("10\n")

    def get_quality_stats(self):
        """计算网格质量统计"""
        if not self.cell_container:
            return {}

        qualities = []
        volumes = []
        for cell in self.cell_container:
            if isinstance(cell, Tetrahedron):
                q = tetrahedron_shape_quality(cell.p1, cell.p2, cell.p3, cell.p4)
                v = tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
                qualities.append(q)
                volumes.append(v)

        if not qualities:
            return {}

        return {
            'num_cells': len(self.cell_container),
            'num_nodes': self.num_nodes,
            'quality_mean': sum(qualities) / len(qualities),
            'quality_min': min(qualities),
            'quality_max': max(qualities),
            'volume_total': sum(volumes),
            'volume_min': min(volumes),
            'volume_max': max(volumes),
        }

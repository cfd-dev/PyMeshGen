import heapq
import numpy as np

from utils.geom_toolkit import tetrahedron_volume, calculate_distance
from optimize.mesh_quality import tetrahedron_shape_quality, compute_mesh_quality_stats
from data_structure.basic_elements import NodeElement, Tetrahedron, is_node_element
from data_structure.unstructured_grid import Unstructured_Grid
from data_structure.rtree_space import (
    build_space_index_3d_with_RTree,
    get_candidate_elements_id_3d,
    add_elems_to_space_index_3d_with_RTree,
)
from data_structure.vtk_types import VTKCellType
from utils.ray_casting import build_spatial_grid, classify_points_inside
from utils.timer import TimeSpan
from utils.message import info, warning, error
from fileIO.vtk_io import write_vtk
from adfront3.front3d import Front3D
from adfront3.sizing3d import UniformSizing3D
from adfront3.geom3d import point_in_tet, intersect_tet_triangle, intersect_tet_triangle2, intersect_triangle_line
from meshsize.size_field_3d import SizeField3D


class Adfront3:
    """三维阵面推进法，生成四面体网格"""

    def __init__(self, surface_triangles, sizing_system=None, debug_level=0):
        self.debug_level = debug_level
        self.al = 6.0
        self.discount = 0.8
        self.progress_interval = 100

        self.surface_triangles = surface_triangles

        # 根据边界面网格自动确定最大间距
        max_spacing = self._compute_max_edge_length(surface_triangles)
        if sizing_system is None:
            from adfront3.sizing3d import SurfaceSizing3D
            sizing_system = SurfaceSizing3D(surface_triangles)
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

        self.deleted_facekeys = set()  # 已消耗的阵面 face_key（惰性删除）
        self._raycast_data = None  # ray casting 数据
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

        heapq.heapify(self.front_list)
        self._build_spatial_index()
        self._init_raycast_data()

        if self.debug_level >= 1:
            info(f"初始化完成: 节点={self.num_nodes}, 阵面={len(self.front_list)}")

    def _init_raycast_data(self):
        """初始化射线投射数据用于边界包含检测"""
        tri_verts = []
        for tri in self.surface_triangles:
            verts = [list(n.coords) for n in tri.nodes]
            tri_verts.append(verts)
        self._raycast_data = build_spatial_grid(np.array(tri_verts))

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

        # 初始化空的单元空间索引
        from rtree import index as rtree_index
        p = rtree_index.Property()
        p.dimension = 3
        self.space_index_cell = rtree_index.Index(properties=p)

    def generate(self, max_steps=None):
        """主推进循环，生成四面体网格，返回 Unstructured_Grid"""
        timer = TimeSpan("开始三维四面体网格生成...")
        step = 0
        retry_count = 0  # 当前阵面连续失败次数

        if max_steps is None:
            # 根据体积和间距估算最大步数
            volume = self._estimate_volume()
            spacing = self.sizing_system.global_spacing
            tet_volume = (spacing ** 3) / (6 * 2 ** 0.5)
            max_steps = int(volume / tet_volume * 3)  # 留3倍余量

        while self.front_list and step < max_steps:
            step += 1
            self.base_front = heapq.heappop(self.front_list)

            # 跳过已消耗的阵面（惰性删除）
            base_fk = self.base_front.face_key
            if base_fk in self.deleted_facekeys:
                continue

            spacing = self.sizing_system.spacing_at(self.base_front.center)

            self.search_candidates(self.base_front.al * spacing)

            selected = self.select_point()

            if selected is None:
                retry_count += 1
                self.base_front.al *= 1.2

                # 连续失败 3 次后反转阵面法向量（参考 Netgen cnt1=4）
                if retry_count >= 3:
                    ne = self.base_front.node_elems
                    ne[0], ne[2] = ne[2], ne[0]
                    self.base_front.normal = [
                        -c for c in self.base_front.normal
                    ]
                    retry_count = 0

                if self.base_front.al > 20:
                    if self.debug_level >= 1:
                        warning(f"阵面{self.base_front.node_ids}搜索半径超过20，放弃该阵面")
                    retry_count = 0
                    continue
                heapq.heappush(self.front_list, self.base_front)
                continue

            retry_count = 0
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
        """沿法向量方向推进生成理想点 pbest

        参考公式: sp = sqrt(6)/3 * avg_edge_length（正四面体高）
        spacing 已是局部目标边长，直接用作推进距离。
        """
        center = self.base_front.center
        normal = self.base_front.normal

        candidate = [
            center[0] + normal[0] * spacing,
            center[1] + normal[1] * spacing,
            center[2] + normal[2] * spacing,
        ]
        self.pbest = NodeElement(
            candidate, self.num_nodes,
            part_name="interior-node", bc_type="interior",
        )
        self.pbest.bbox = [candidate[0], candidate[1], candidate[2],
                           candidate[0], candidate[1], candidate[2]]
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
        """从候选点中选择最佳推进点

        策略：
        1. 先尝试已有候选节点（不创建新几何，按质量降序）
        2. 尝试 pbest（理想推进距离）
        3. 如果 pbest 失败，尝试更近的新点（spacing/2, spacing/4）
        """
        p0 = self.base_front.node_elems[0].coords
        p1 = self.base_front.node_elems[1].coords
        p2 = self.base_front.node_elems[2].coords

        self.pselected = None
        self.best_flag = False

        # 1. 先尝试已有候选节点，按质量降序排列
        scored_candidates = []
        for node_elem in self.node_candidates:
            if node_elem in self.base_front.node_elems:
                continue
            quality = tetrahedron_shape_quality(p0, p1, p2, node_elem.coords)
            if quality > 0.1:
                scored_candidates.append((quality, node_elem))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        for quality, node_elem in scored_candidates:
            if not self._is_correct_side(node_elem):
                continue
            if not self._is_inside_boundary(node_elem.coords):
                continue
            if self._is_cross(node_elem):
                continue
            self.pselected = node_elem
            return self.pselected

        # 2. 尝试 pbest 及更近的新点
        center = self.base_front.center
        normal = self.base_front.normal
        spacing = self.sizing_system.spacing_at(center)

        for dist_factor in [1.0, 0.5, 0.25]:
            dist = spacing * dist_factor
            candidate_coords = [
                center[0] + normal[0] * dist,
                center[1] + normal[1] * dist,
                center[2] + normal[2] * dist,
            ]
            candidate = NodeElement(
                candidate_coords, self.num_nodes,
                part_name="interior-node", bc_type="interior",
            )
            candidate.bbox = [candidate_coords[0], candidate_coords[1], candidate_coords[2],
                              candidate_coords[0], candidate_coords[1], candidate_coords[2]]

            quality = tetrahedron_shape_quality(p0, p1, p2, candidate_coords)
            if quality <= 0.1:
                continue
            if not self._is_correct_side(candidate):
                continue
            if not self._is_inside_boundary(candidate_coords):
                continue
            if self._is_cross(candidate):
                continue
            self.pselected = candidate
            self.best_flag = True
            if dist_factor < 1.0:
                self.pbest = candidate
            return self.pselected

        return self.pselected

    def _is_correct_side(self, node_elem):
        """检查候选点是否在阵面内侧（法向量方向）"""
        p0 = np.array(self.base_front.node_elems[0].coords)
        normal = np.array(self.base_front.normal)
        v = np.array(node_elem.coords) - p0
        return np.dot(v, normal) > 1e-12

    def _is_inside_boundary(self, point):
        """检查点是否在封闭体积内部（射线投射法，适用于任意封闭曲面）"""
        if self._raycast_data is None:
            return True
        return classify_points_inside([np.array(point)], self._raycast_data)[0]

    def _point_in_tet_strict(self, point, tet_coords):
        """检查点是否严格在四面体内部（使用有符号体积法）

        与 geom3d.point_in_tet 相同逻辑，作为快速早期排除。
        """
        pts = [np.array(c) for c in tet_coords]
        pt = np.array(point)

        ref = np.dot(pts[1] - pts[0], np.cross(pts[2] - pts[0], pts[3] - pts[0]))
        if abs(ref) < 1e-30:
            return False

        eps = 1e-10 * abs(ref)
        for i in range(4):
            j = (i + 2) % 4
            k = (i + 3) % 4
            v = float(np.dot(pt - pts[i], np.cross(pts[j] - pts[i], pts[k] - pts[i])))
            signed_v = ((-1) ** i) * v
            if signed_v < eps:
                return False
        return True

    def _is_cross(self, node_elem):
        """检查候选四面体是否与现有阵面或单元相交

        参考 Netgen 策略：对每个候选面/单元调用 IntersectTetTriangle + IntersectTetTriangle2。
        """
        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]
        p2 = self.base_front.node_elems[2]
        p3 = node_elem

        new_tet_coords = [p0.coords, p1.coords, p2.coords, p3.coords]
        new_tet_ids = [p0.idx, p1.idx, p2.idx, p3.idx]
        new_tet_id_set = set(new_tet_ids)

        # 1. 检查候选点是否严格在现有四面体内（排除边界情况）
        for cell in self.cell_candidates:
            if not isinstance(cell, Tetrahedron):
                continue
            cell_coords = [cell.p1, cell.p2, cell.p3, cell.p4]
            if self._point_in_tet_strict(p3.coords, cell_coords):
                return True

        # 2. 检查与现有单元的相交（参考 Netgen: IntersectTetTriangle + IntersectTetTriangle2）
        for cell in self.cell_candidates:
            if not isinstance(cell, Tetrahedron):
                continue
            cell_coords = [cell.p1, cell.p2, cell.p3, cell.p4]
            cell_ids = cell.node_ids
            # 检查每个面
            faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
            for i0, i1, i2 in faces:
                tri_coords = [cell_coords[i0], cell_coords[i1], cell_coords[i2]]
                tri_ids = [cell_ids[i0], cell_ids[i1], cell_ids[i2]]
                if intersect_tet_triangle(new_tet_coords, tri_coords, new_tet_ids, tri_ids):
                    return True
                if intersect_tet_triangle2(new_tet_coords, tri_coords, new_tet_ids, tri_ids):
                    return True

        # 3. 检查与非邻接阵面的相交
        for front in self.front_candidates:
            if front is self.base_front:
                continue
            if front.is_satisfied:
                continue
            # 跳过边界面（固定几何，不参与相交判断）
            if front.bc_type == "wall":
                continue
            tri_coords = [n.coords for n in front.node_elems]
            tri_ids = front.node_ids
            if intersect_tet_triangle(new_tet_coords, tri_coords, new_tet_ids, tri_ids):
                return True
            if intersect_tet_triangle2(new_tet_coords, tri_coords, new_tet_ids, tri_ids):
                return True

        return False

    def update_data(self):
        """创建四面体，更新节点、阵面和单元

        使用 left/right 机制跟踪阵面消耗状态：
        - 边界面(wall)创建时即标记为两侧已占据
        - 内界面每被一个四面体使用调用 set_left_right_auto()
        - 两侧都被占据后标记删除
        """
        if self.pselected is None:
            return

        self._update_nodes()

        p0 = self.base_front.node_elems[0]
        p1 = self.base_front.node_elems[1]
        p2 = self.base_front.node_elems[2]
        p3 = self.pselected

        # 三个新阵面（不含基准面），法向量指向四面体外侧（未填充区域）
        # 绕向规则：每个面的法向量必须远离对顶点
        #   面(p2,p1,p3) 对顶点p0,  面(p0,p3,p2) 对顶点p1,  面(p3,p1,p0) 对顶点p2
        new_faces = [
            Front3D(p2, p1, p3, idx=-1, bc_type="interior", part_name="interior"),
            Front3D(p0, p3, p2, idx=-1, bc_type="interior", part_name="interior"),
            Front3D(p3, p1, p0, idx=-1, bc_type="interior", part_name="interior"),
        ]

        # 标记基准面已消耗
        base_face_key = self.base_front.face_key
        if self.base_front.bc_type == "wall":
            # 边界面只能在一侧创建四面体，直接标记为两侧已占据
            self.base_front.left = 1
            self.base_front.right = 1
        else:
            self.base_front.set_left_right_auto()
        self.deleted_facekeys.add(base_face_key)

        # 从空间索引中移除基准面
        base_fid = self.front_id_by_facekey.pop(base_face_key, None)
        if base_fid is not None and base_fid in self.front_dict:
            del self.front_dict[base_fid]

        self._update_fronts(new_faces)

        new_tet = Tetrahedron(
            p0, p1, p2, p3,
            part_name='interior-tetrahedron',
            idx=self.num_cells,
        )
        # 设置 bbox 用于 RTree 空间索引
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
        """更新阵面列表：新面若已存在则标记删除（内部面），否则添加

        使用 left/right 机制：若面已存在（第二次使用），标记其 left 侧
        已被占据。两侧都被占据后删除。
        """
        for front in new_fronts:
            face_key = front.face_key
            if face_key in self.front_id_by_facekey:
                # 面已存在（第二次使用），标记 left 侧已被占据
                existing_fid = self.front_id_by_facekey.get(face_key)
                existing = self.front_dict.get(existing_fid)
                if existing is not None:
                    existing.left = 1
                # 标记为已消耗（惰性删除）
                self.deleted_facekeys.add(face_key)
                self.front_id_by_facekey.pop(face_key, None)
                if existing_fid is not None and existing_fid in self.front_dict:
                    del self.front_dict[existing_fid]
            else:
                # 首次使用，标记 right 侧已被占据，添加到阵面列表
                front.right = 1
                self.front_list.append(front)
                self.add_elems_to_space_index(
                    [front], self.space_index_front, self.front_dict
                )
                self.front_id_by_facekey[face_key] = id(front)

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
        """导出四面体网格为 VTK 文件（委托给 fileIO.vtk_io）"""
        cell_idx = [list(c.node_ids) for c in self.cell_container]
        cell_types = [VTKCellType.TETRA] * len(self.cell_container)
        write_vtk(filename, self.node_coords, cell_idx, [], cell_types)

    def get_quality_stats(self):
        """计算网格质量统计（委托给 optimize.mesh_quality）"""
        return compute_mesh_quality_stats(self.cell_container)

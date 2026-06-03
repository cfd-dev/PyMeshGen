"""基于 Bowyer-Watson Delaunay 三角剖分的四面体网格生成算法（第5节）"""
import numpy as np

from utils.geom_toolkit import (
    tetrahedron_volume, tetrahedron_signed_volume,
    calculate_distance, circumsphere,
)
from optimize.mesh_quality import tetrahedron_shape_quality, tetrahedron_shape_quality_v2
from data_structure.basic_elements import NodeElement, Tetrahedron
from data_structure.unstructured_grid import Unstructured_Grid
from utils.timer import TimeSpan
from utils.message import info, warning, error
from delaunay3d.sizing import UniformSizing3D


class BowyerWatsonTetGen:
    """基于 Bowyer-Watson Delaunay 三角剖分的四面体网格生成"""

    def __init__(self, surface_triangles, sizing_system=None, debug_level=0):
        self.debug_level = debug_level
        self.surface_triangles = surface_triangles

        # 根据边界面网格自动确定间距
        max_spacing = self._compute_max_edge_length(surface_triangles)
        if sizing_system is None:
            sizing_system = UniformSizing3D(max_spacing)
        self.sizing_system = sizing_system

        # 节点和单元
        self.node_coords = []  # List[List[float]]
        self.node_elem_by_hash = {}
        self.node_hash_list = set()
        self.num_nodes = 0

        self.cell_container = []  # List[Tetrahedron]
        self.num_cells = 0

        self.boundary_nodes = set()
        self.unstr_grid = None

        # 边界平面
        self.boundary_halfspaces = []
        self._boundary_planes = []

        # 统计
        self._surface_node_count = 0
        self._interior_node_count = 0

    @staticmethod
    def _compute_max_edge_length(surface_triangles):
        """从边界面网格计算最大边长"""
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

    def _extract_surface_nodes(self):
        """从曲面三角形提取唯一节点"""
        nodes = []
        node_hashes = set()

        for tri in self.surface_triangles:
            for node3d in tri.nodes:
                node_hash = hash(tuple(f"{c:.6f}" for c in node3d.coords))
                if node_hash not in node_hashes:
                    node_hashes.add(node_hash)
                    coords = list(node3d.coords)
                    node_elem = NodeElement(
                        coords, self.num_nodes,
                        part_name="wall", bc_type="wall",
                    )
                    nodes.append(node_elem)
                    self.node_coords.append(coords)
                    self.node_elem_by_hash[node_hash] = node_elem
                    self.node_hash_list.add(node_hash)
                    self.boundary_nodes.add(node_elem)
                    self.num_nodes += 1

        self._surface_node_count = len(nodes)
        return nodes

    def _compute_boundary_planes(self):
        """从曲面三角形计算合并后的边界平面"""
        if not self.boundary_halfspaces:
            return []

        planes = {}
        for normal, ref in self.boundary_halfspaces:
            n = np.array(normal)
            ref_pt = np.array(ref)
            d = np.dot(n, ref_pt)

            # 量化法向量到主轴方向
            abs_n = np.abs(n)
            axis = int(np.argmax(abs_n))
            quantized = np.zeros(3)
            quantized[axis] = 1.0 if n[axis] > 0 else -1.0
            key = tuple(round(x, 1) for x in quantized)

            if key not in planes:
                planes[key] = (n, d)
            else:
                existing_n, existing_d = planes[key]
                planes[key] = (existing_n, max(existing_d, d))

        return list(planes.values())

    def _is_inside_boundary(self, point):
        """检查点是否在封闭体积内部"""
        if not self._boundary_planes:
            return True

        pt = np.array(point)
        tol = self.sizing_system.global_spacing * 0.01
        for normal, d_val in self._boundary_planes:
            if np.dot(normal, pt) < d_val - tol:
                return False
        return True

    def _generate_interior_nodes(self):
        """在边界内部生成规则网格节点"""
        # 计算包围盒
        if not self.node_coords:
            return []

        coords = np.array(self.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        # 使用间距生成规则网格
        spacing = self.sizing_system.global_spacing
        interior_nodes = []

        # 在包围盒内生成规则网格点
        x_range = np.arange(mins[0] + spacing * 0.5, maxs[0], spacing)
        y_range = np.arange(mins[1] + spacing * 0.5, maxs[1], spacing)
        z_range = np.arange(mins[2] + spacing * 0.5, maxs[2], spacing)

        for x in x_range:
            for y in y_range:
                for z in z_range:
                    point = [x, y, z]
                    if self._is_inside_boundary(point):
                        node_hash = hash(tuple(f"{c:.6f}" for c in point))
                        if node_hash not in self.node_hash_list:
                            node_elem = NodeElement(
                                point, self.num_nodes,
                                part_name="interior-node", bc_type="interior",
                            )
                            interior_nodes.append(node_elem)
                            self.node_coords.append(point)
                            self.node_elem_by_hash[node_hash] = node_elem
                            self.node_hash_list.add(node_hash)
                            self.num_nodes += 1

        self._interior_node_count = len(interior_nodes)
        return interior_nodes

    def _create_super_tetrahedron(self):
        """创建超级四面体，包含所有节点（4.2节）"""
        coords = np.array(self.node_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        center = (mins + maxs) / 2.0
        size = np.max(maxs - mins) * 2.0  # 足够大以包含所有节点

        # 创建4个顶点，形成一个大的四面体
        super_nodes = []
        super_coords = [
            [center[0] - size, center[1] - size, center[2] - size],
            [center[0] + size * 2, center[1] - size, center[2] - size],
            [center[0], center[1] + size * 2, center[2] - size],
            [center[0], center[1], center[2] + size * 2],
        ]

        for i, coords in enumerate(super_coords):
            node = NodeElement(
                coords, self.num_nodes + i,
                part_name="super-tet", bc_type="super",
            )
            super_nodes.append(node)

        return super_nodes

    def _in_circumsphere(self, node_coords, tet_coords):
        """检查节点是否在四面体的外接球内（5.4.1节）"""
        center, r2 = circumsphere(*tet_coords)
        if r2 < 1e-30:
            return False

        dist2 = sum((node_coords[i] - center[i]) ** 2 for i in range(3))
        # 使用容差避免数值问题
        return dist2 < r2 * (1.0 + 1e-10)

    def _bowyer_watson(self, all_nodes):
        """Bowyer-Watson Delaunay 三角剖分算法（第5节）

        Args:
            all_nodes: List[NodeElement] 所有节点（表面 + 内部 + 超级四面体顶点）

        Returns:
            List[Tetrahedron] 三角剖分结果
        """
        # 创建超级四面体
        super_nodes = self._create_super_tetrahedron()
        super_ids = set(n.idx for n in super_nodes)

        # 初始四面体 = 超级四面体
        initial_tet = Tetrahedron(
            super_nodes[0], super_nodes[1], super_nodes[2], super_nodes[3],
            part_name='super-tetrahedron', idx=0,
        )
        tets = [initial_tet]

        # 构建节点列表（超级四面体节点 + 表面节点 + 内部节点）
        insert_nodes = list(all_nodes)

        # 逐个插入节点
        for step, node in enumerate(insert_nodes):
            if step % 500 == 0 and step > 0:
                info(f"  Delaunay插入进度: {step}/{len(insert_nodes)}, 四面体={len(tets)}")

            node_coords = node.coords

            # 1. 找到所有外接球包含新节点的四面体（腔体）
            cavity = []
            non_cavity = []

            for tet in tets:
                tet_coords = [tet.p1, tet.p2, tet.p3, tet.p4]
                if self._in_circumsphere(node_coords, tet_coords):
                    cavity.append(tet)
                else:
                    non_cavity.append(tet)

            if not cavity:
                continue

            # 2. 找到腔体的边界面
            # 边界面 = 腔体中只被一个腔体四面体使用的面
            face_count = {}  # face_key -> count

            for tet in cavity:
                tet_node_ids = tet.node_ids
                # 四面体的4个面
                faces = [
                    tuple(sorted([tet_node_ids[0], tet_node_ids[1], tet_node_ids[2]])),
                    tuple(sorted([tet_node_ids[0], tet_node_ids[1], tet_node_ids[3]])),
                    tuple(sorted([tet_node_ids[0], tet_node_ids[2], tet_node_ids[3]])),
                    tuple(sorted([tet_node_ids[1], tet_node_ids[2], tet_node_ids[3]])),
                ]
                for face_key in faces:
                    face_count[face_key] = face_count.get(face_key, 0) + 1

            # 边界面 = 只出现一次的面
            boundary_faces = [fk for fk, cnt in face_count.items() if cnt == 1]

            # 3. 为每个边界面创建新四面体
            new_tets = []
            for face_key in boundary_faces:
                # 获取面的3个节点
                face_nodes = []
                for nid in face_key:
                    # 查找节点对象
                    for n in super_nodes:
                        if n.idx == nid:
                            face_nodes.append(n)
                            break
                    else:
                        # 在 all_nodes 中查找
                        for n in all_nodes:
                            if n.idx == nid:
                                face_nodes.append(n)
                                break

                if len(face_nodes) != 3:
                    continue

                # 创建新四面体：face_nodes + node
                new_tet = Tetrahedron(
                    face_nodes[0], face_nodes[1], face_nodes[2], node,
                    part_name='interior-tetrahedron', idx=len(tets) + len(new_tets),
                )

                # 检查有符号体积（确保绕向正确）
                sv = tetrahedron_signed_volume(
                    new_tet.p1, new_tet.p2, new_tet.p3, new_tet.p4
                )
                if sv < 0:
                    # 反转绕向
                    new_tet = Tetrahedron(
                        face_nodes[0], face_nodes[2], face_nodes[1], node,
                        part_name='interior-tetrahedron', idx=new_tet.idx,
                    )

                new_tets.append(new_tet)

            # 4. 更新四面体列表
            tets = non_cavity + new_tets

        return tets, super_ids

    def _remove_exterior_tets(self, tets, super_ids):
        """移除外部四面体

        移除：
        1. 使用超级四面体顶点的四面体
        2. 形心在边界外的四面体
        3. 退化四面体（零体积或负体积）
        """
        valid_tets = []

        for tet in tets:
            # 检查是否使用超级四面体节点
            if any(nid in super_ids for nid in tet.node_ids):
                continue

            # 检查有符号体积
            sv = tetrahedron_signed_volume(tet.p1, tet.p2, tet.p3, tet.p4)
            if sv <= 1e-15:
                continue

            # 检查形心是否在边界内
            centroid = [
                (tet.p1[0] + tet.p2[0] + tet.p3[0] + tet.p4[0]) / 4.0,
                (tet.p1[1] + tet.p2[1] + tet.p3[1] + tet.p4[1]) / 4.0,
                (tet.p1[2] + tet.p2[2] + tet.p3[2] + tet.p4[2]) / 4.0,
            ]
            if not self._is_inside_boundary(centroid):
                continue

            valid_tets.append(tet)

        return valid_tets

    def _laplacian_smooth(self, iterations=3):
        """Laplacian 光滑内部节点（8.2.1节）"""
        # 构建节点-单元邻接
        node_cells = {}  # node_idx -> list of cell indices
        for ci, cell in enumerate(self.cell_container):
            if not isinstance(cell, Tetrahedron):
                continue
            for nid in cell.node_ids:
                if nid not in node_cells:
                    node_cells[nid] = []
                node_cells[nid].append(ci)

        for _ in range(iterations):
            for nid, cell_indices in node_cells.items():
                # 只光滑内部节点
                coords = self.node_coords[nid]
                node_hash = hash(tuple(f"{c:.6f}" for c in coords))
                node = self.node_elem_by_hash.get(node_hash)
                if node is None or node.bc_type != "interior":
                    continue

                if not cell_indices:
                    continue

                # 计算邻域单元形心的加权平均
                weighted_sum = np.zeros(3)
                total_weight = 0.0
                for ci in cell_indices:
                    cell = self.cell_container[ci]
                    if not isinstance(cell, Tetrahedron):
                        continue
                    centroid = (
                        np.array(cell.p1) + np.array(cell.p2)
                        + np.array(cell.p3) + np.array(cell.p4)
                    ) / 4.0
                    vol = tetrahedron_volume(cell.p1, cell.p2, cell.p3, cell.p4)
                    if vol > 1e-30:
                        weighted_sum += centroid * vol
                        total_weight += vol

                if total_weight > 1e-30:
                    new_pos = weighted_sum / total_weight

                    # 限制移动幅度
                    old_pos = np.array(self.node_coords[nid])
                    max_move = self.sizing_system.global_spacing * 0.5
                    delta = new_pos - old_pos
                    dist = np.linalg.norm(delta)
                    if dist > max_move:
                        delta = delta * (max_move / dist)
                        new_pos = old_pos + delta

                    # 检查新位置是否在边界内
                    if not self._is_inside_boundary(new_pos):
                        continue

                    # 检查移动后所有关联单元是否仍有效
                    valid = True
                    for ci in cell_indices:
                        cell = self.cell_container[ci]
                        if not isinstance(cell, Tetrahedron):
                            continue
                        coords_list = [cell.p1, cell.p2, cell.p3, cell.p4]
                        for idx, attr in enumerate(['p1', 'p2', 'p3', 'p4']):
                            if cell.node_ids[idx] == nid:
                                coords_list[idx] = new_pos.tolist()
                                break
                        sv = tetrahedron_signed_volume(*coords_list)
                        if sv <= 1e-15:
                            valid = False
                            break

                    if not valid:
                        continue

                    new_pos_list = new_pos.tolist()
                    self.node_coords[nid] = new_pos_list

                    # 更新关联单元的坐标
                    for ci in cell_indices:
                        cell = self.cell_container[ci]
                        if not isinstance(cell, Tetrahedron):
                            continue
                        for idx, attr in enumerate(['p1', 'p2', 'p3', 'p4']):
                            if cell.node_ids[idx] == nid:
                                setattr(cell, attr, new_pos_list)
                                break

    def generate(self, max_steps=None):
        """生成四面体网格

        使用 Bowyer-Watson Delaunay 三角剖分算法

        Returns:
            Unstructured_Grid 生成的网格
        """
        timer = TimeSpan("开始 Bowyer-Watson 四面体网格生成...")

        # 1. 提取表面节点
        surface_nodes = self._extract_surface_nodes()
        info(f"表面节点: {self._surface_node_count}")

        # 计算边界半空间
        for tri in self.surface_triangles:
            # 计算三角形法向量（指向体积内部）
            coords = [n.coords for n in tri.nodes]
            v1 = np.array(coords[1]) - np.array(coords[0])
            v2 = np.array(coords[2]) - np.array(coords[0])
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-30:
                normal = normal / norm_len

            # 使用曲面质心判断法向量方向
            all_coords = []
            for t in self.surface_triangles:
                for n in t.nodes:
                    all_coords.append(n.coords)
            centroid = np.mean(all_coords, axis=0)
            tri_center = np.mean(coords, axis=0)
            to_centroid = centroid - tri_center
            if np.dot(normal, to_centroid) < 0:
                normal = -normal

            self.boundary_halfspaces.append(
                (list(normal), list(coords[0]))
            )

        self._boundary_planes = self._compute_boundary_planes()
        info(f"边界平面: {len(self._boundary_planes)}")

        # 2. 生成内部节点
        interior_nodes = self._generate_interior_nodes()
        info(f"内部节点: {self._interior_node_count}")

        # 3. 合并所有节点
        all_nodes = surface_nodes + interior_nodes
        info(f"总节点: {len(all_nodes)}")

        # 4. Bowyer-Watson Delaunay 三角剖分
        info("开始 Bowyer-Watson Delaunay 三角剖分...")
        tets, super_ids = self._bowyer_watson(all_nodes)
        info(f"Delaunay 三角剖分完成: {len(tets)} 个四面体")

        # 5. 移除外部四面体
        valid_tets = self._remove_exterior_tets(tets, super_ids)
        info(f"移除外部四面体后: {len(valid_tets)} 个有效四面体")

        # 6. 存储结果
        self.cell_container = valid_tets
        self.num_cells = len(valid_tets)

        # 7. 重新编号单元
        for i, cell in enumerate(self.cell_container):
            cell.idx = i

        # 8. Laplacian 光滑
        info("开始 Laplacian 光滑...")
        self._laplacian_smooth(iterations=3)

        # 9. 构建网格
        self.construct_unstr_grid()

        timer.show_to_console("Bowyer-Watson 四面体网格生成完成.")
        info(f"最终结果: 节点={self.num_nodes}, 四面体={self.num_cells}")
        info(f"  表面节点={self._surface_node_count}, 内部节点={self._interior_node_count}")

        return self.unstr_grid

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

"""
曲面 Delaunay 三角剖分（Bowyer-Watson 增量节点插入）
- 在 3D 曲面上进行 Bowyer-Watson 增量节点插入
- 使用 2D 参数空间定位 + 3D 外接圆测试
- 支持边界约束和表面恢复

核心算法：
1. 构建虚拟网格（2 个超级三角形覆盖所有边界节点）
2. 逐个插入边界节点（Delaunay 腔体构建）
3. 移除虚拟三角形
4. 可选插入内部节点
5. 边交换恢复表面贴合
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import deque, defaultdict
import numpy as np
from OCC.Core.TopoDS import TopoDS_Face

from utils.message import info
from .surface_geometry import SurfaceGeometry


# ============================================================================
# 数据结构
# ============================================================================

class DelaunayNode:
    """Delaunay 三角剖分节点（2D 参数空间 + 3D 曲面坐标）"""
    __slots__ = ['idx', 'uv', 'xyz']
    def __init__(self, idx: int, uv: Tuple[float, float],
                 xyz: Tuple[float, float, float]):
        self.idx = idx
        self.uv = uv
        self.xyz = xyz


class DelaunayTriangle:
    """Delaunay 三角形"""
    __slots__ = ['idx', 'nodes', 'node_ids', 'neighbors', 'removed', 'virtual']
    def __init__(self, idx: int, n0: DelaunayNode, n1: DelaunayNode,
                 n2: DelaunayNode):
        self.idx = idx
        self.nodes = [n0, n1, n2]
        self.node_ids = [n0.idx, n1.idx, n2.idx]
        self.neighbors = [None, None, None]  # 三条边的邻接三角形
        self.removed = False
        self.virtual = False  # 是否为虚拟三角形

    @property
    def coords_2d(self) -> np.ndarray:
        return np.array([n.uv for n in self.nodes])

    @property
    def coords_3d(self) -> np.ndarray:
        return np.array([n.xyz for n in self.nodes])


# ============================================================================
# Bowyer-Watson 表面 Delaunay 三角剖分器
# ============================================================================

class SurfaceDelaunay:
    """
    曲面 Delaunay 三角剖分器。

    Args:
        surface: OCC 曲面对象
        geometry_handler: 曲面几何操作对象
        normal_flip: 法向翻转因子（来自 SurfaceMeshGenerator._normal_flip）
    """

    def __init__(self, surface: TopoDS_Face,
                 geometry_handler: SurfaceGeometry = None,
                 normal_flip: float = 1.0):
        self.surface = surface
        self.geometry = geometry_handler or SurfaceGeometry()
        self._normal_flip = normal_flip

        self.nodes: List[DelaunayNode] = []
        self.triangles: List[DelaunayTriangle] = []
        self._edge_map: Dict[frozenset, List[int]] = defaultdict(list)

        # 虚拟节点（超级三角形的顶点）
        self._virtual_nodes: List[DelaunayNode] = []
        self._next_node_idx = 0
        self._next_tri_idx = 0

        # 容差
        self._circumsphere_tol = 1e-8   # 外接圆容差
        self._node_tolerance = 1e-6     # 节点重复容差

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _add_node(self, uv: Tuple[float, float],
                  xyz: Tuple[float, float, float]) -> DelaunayNode:
        """创建节点"""
        node = DelaunayNode(self._next_node_idx, uv, xyz)
        self._next_node_idx += 1
        self.nodes.append(node)
        return node

    def _add_triangle(self, n0: DelaunayNode, n1: DelaunayNode,
                      n2: DelaunayNode, virtual: bool = False) -> DelaunayTriangle:
        """创建三角形并注册边映射（自动跳过重复三角形）"""
        # 检查重复
        node_set = frozenset([n0.idx, n1.idx, n2.idx])
        for tri in self.triangles:
            if tri.removed:
                continue
            if frozenset(tri.node_ids) == node_set:
                return tri  # 已存在，返回已有三角形

        tri = DelaunayTriangle(self._next_tri_idx, n0, n1, n2)
        tri.virtual = virtual
        self._next_tri_idx += 1
        self.triangles.append(tri)

        for k in range(3):
            a, b = tri.node_ids[k], tri.node_ids[(k + 1) % 3]
            self._edge_map[frozenset([a, b])].append(tri.idx)
        return tri

    def _remove_triangle(self, tri: DelaunayTriangle):
        """标记三角形为已删除，从边映射中移除"""
        tri.removed = True
        for k in range(3):
            a, b = tri.node_ids[k], tri.node_ids[(k + 1) % 3]
            e = frozenset([a, b])
            lst = self._edge_map.get(e, [])
            if tri.idx in lst:
                lst.remove(tri.idx)
            if not lst:
                self._edge_map.pop(e, None)

    # ------------------------------------------------------------------
    # 3D 外接圆测试
    # ------------------------------------------------------------------

    def _in_circumcircle_2d(self, node: DelaunayNode,
                            tri: DelaunayTriangle) -> bool:
        """
        2D 外接圆测试：在 UV 参数空间中检查节点是否在三角形的外接圆内。

        虚拟节点的 3D 坐标在曲面外无意义，所以使用 UV 空间进行
        外接圆测试。对于平面区域和非退化曲面，UV 空间中的 Delaunay
        三角剖分在映射到 3D 后仍然保持 Delaunay 性质。
        """
        u, v = node.uv
        (u0, v0), (u1, v1), (u2, v2) = tri.coords_2d

        # 2D 外接圆行列式测试
        # 构建矩阵 [x²+y², x, y, 1] 的行列式
        m = np.array([
            [u0*u0 + v0*v0, u0, v0, 1.0],
            [u1*u1 + v1*v1, u1, v1, 1.0],
            [u2*u2 + v2*v2, u2, v2, 1.0],
            [u*u + v*v,     u,  v,  1.0],
        ])
        det = np.linalg.det(m)
        # 根据三角形顶点绕序确定符号
        orient = (u1-u0)*(v2-v0) - (v1-v0)*(u2-u0)
        if orient > 0:
            return det > 1e-10
        else:
            return det < -1e-10

    # ------------------------------------------------------------------
    # 虚拟网格
    # ------------------------------------------------------------------

    def _create_virtual_mesh(self, boundary_nodes: List[DelaunayNode]):
        """
        创建虚拟网格：2 个超级三角形包围所有边界节点。

        计算边界节点的 UV 包围盒，在包围盒外侧创建 4 个虚拟节点，
        用于形成 2 个覆盖整个域的超级三角形。
        """
        uvs = np.array([n.uv for n in boundary_nodes])
        u_min, u_max = uvs[:, 0].min(), uvs[:, 0].max()
        v_min, v_max = uvs[:, 1].min(), uvs[:, 1].max()

        # 扩展包围盒
        margin = max(u_max - u_min, v_max - v_min) * 0.5 + 1.0
        u_min -= margin; u_max += margin
        v_min -= margin; v_max += margin

        # 4 个虚拟节点（包围盒四角）
        virtual_uvs = [
            (u_min, v_min), (u_max, v_min),
            (u_max, v_max), (u_min, v_max),
        ]
        for uv in virtual_uvs:
            xyz = self._uv_to_xyz(uv)
            vn = self._add_node(uv, xyz)
            vn.idx = self._next_node_idx - 1  # keep idx
            self._virtual_nodes.append(vn)

        v = self._virtual_nodes
        # 2 个超级三角形
        tri0 = self._add_triangle(v[0], v[1], v[2], virtual=True)
        tri1 = self._add_triangle(v[0], v[2], v[3], virtual=True)

        # 设置邻接关系
        tri0.neighbors[1] = tri1
        tri1.neighbors[1] = tri0

    def _uv_to_xyz(self, uv: Tuple[float, float]
                   ) -> Tuple[float, float, float]:
        """将 UV 参数坐标映射到 3D 曲面坐标"""
        return self.geometry.evaluate_point(uv[0], uv[1], self.surface)

    def _project_to_uv(self, xyz: Tuple[float, float, float]
                       ) -> Tuple[float, float]:
        """将 3D 坐标投影到 UV 参数空间"""
        return self.geometry.project_point_to_surface(xyz, self.surface)

    def _find_seed_triangle(self, uv: Tuple[float, float]
                            ) -> DelaunayTriangle:
        """
        在 2D 参数空间中找到包含给定 UV 点的三角形（含边界和虚拟三角形）。
        """
        u, v = uv
        for tri in self.triangles:
            if tri.removed:
                continue
            pts = tri.coords_2d
            denom = ((pts[1][1] - pts[2][1]) * (pts[0][0] - pts[2][0]) +
                     (pts[2][0] - pts[1][0]) * (pts[0][1] - pts[2][1]))
            if abs(denom) < 1e-20:
                continue
            w0 = ((pts[1][1] - pts[2][1]) * (u - pts[2][0]) +
                  (pts[2][0] - pts[1][0]) * (v - pts[2][1])) / denom
            w1 = ((pts[2][1] - pts[0][1]) * (u - pts[2][0]) +
                  (pts[0][0] - pts[2][0]) * (v - pts[2][1])) / denom
            w2 = 1.0 - w0 - w1
            if w0 >= -1e-9 and w1 >= -1e-9 and w2 >= -1e-9:
                return tri
        # 回退：返回任意未删除三角形（含虚拟）
        for tri in self.triangles:
            if not tri.removed:
                return tri
        return None

    # ------------------------------------------------------------------
    # Delaunay 腔体构建与节点插入
    # ------------------------------------------------------------------

    def _build_cavity(self, node: DelaunayNode,
                      seed: DelaunayTriangle
                      ) -> Tuple[Set[DelaunayTriangle], Set[frozenset]]:
        """
        BFS 构建 Delaunay 腔体（使用 edge_map 遍历邻接三角形）。

        Returns:
            (cavity_triangles, cavity_boundary_edges)
        """
        cavity: Set[DelaunayTriangle] = set()
        queue = deque([seed])
        visited: Set[int] = set()

        while queue:
            tri = queue.popleft()
            if tri.idx in visited:
                continue
            visited.add(tri.idx)

            if not self._in_circumcircle_2d(node, tri):
                continue

            cavity.add(tri)

            for k in range(3):
                a, b = tri.node_ids[k], tri.node_ids[(k + 1) % 3]
                e = frozenset([a, b])
                for neighbor_idx in self._edge_map.get(e, []):
                    if neighbor_idx not in visited:
                        neighbor = self.triangles[neighbor_idx]
                        if not neighbor.removed:
                            queue.append(neighbor)

        # 腔体边界边（仅被 1 个腔体三角形使用）
        edge_use: Dict[frozenset, int] = defaultdict(int)
        for tri in cavity:
            for k in range(3):
                a, b = tri.node_ids[k], tri.node_ids[(k + 1) % 3]
                edge_use[frozenset([a, b])] += 1

        boundary_edges = {e for e, cnt in edge_use.items() if cnt == 1}
        return cavity, boundary_edges

    def _insert_node(self, node: DelaunayNode) -> bool:
        """
        使用 Bowyer-Watson 算法插入节点到 Delaunay 三角剖分。

        Returns:
            True 表示成功插入
        """
        # 1. 检查重复节点（排除自身）
        for existing in self.nodes:
            if existing.idx == node.idx:
                continue
            d2 = (node.uv[0] - existing.uv[0])**2 + (node.uv[1] - existing.uv[1])**2
            if d2 < self._node_tolerance ** 2:
                return True  # 视为成功（重复节点已在网格中）

        # 2. 找到包含该节点的种子三角形
        seed = self._find_seed_triangle(node.uv)
        if seed is None:
            return False

        # 3. 构建 Delaunay 腔体
        cavity, cavity_boundary = self._build_cavity(node, seed)
        if not cavity:
            return False

        # 4. 构建腔体边界边的有序环
        edge_adj: Dict[int, List[int]] = defaultdict(list)
        for e in cavity_boundary:
            a, b = tuple(e)
            edge_adj[a].append(b)
            edge_adj[b].append(a)

        # 追踪边界环
        if not cavity_boundary:
            return False

        # 按角度排序边界节点
        boundary_nodes = set()
        for e in cavity_boundary:
            boundary_nodes.update(e)
        bn_list = list(boundary_nodes)

        # 计算腔体质心
        bn_uvs = np.array([self.nodes[n].uv for n in bn_list])
        centroid = bn_uvs.mean(axis=0)

        # 按绕质心角度排序
        angles = []
        for n in bn_list:
            uv = self.nodes[n].uv
            angle = np.arctan2(uv[1] - centroid[1], uv[0] - centroid[0])
            angles.append((angle, n))
        angles.sort()
        sorted_bn = [n for _, n in angles]

        # 5. 移除腔体三角形
        for tri in cavity:
            self._remove_triangle(tri)

        # 6. 创建新三角形（从新节点连接到腔体边界环的每条边）
        new_tris = []
        n_bn = len(sorted_bn)
        for i in range(n_bn):
            n0 = node
            n1 = self.nodes[sorted_bn[i]]
            n2 = self.nodes[sorted_bn[(i + 1) % n_bn]]
            if n0.idx == n1.idx or n1.idx == n2.idx or n0.idx == n2.idx:
                continue
            tri = self._add_triangle(n0, n1, n2)
            new_tris.append(tri)

        # 7. 更新邻接关系
        self._update_neighbors(cavity, new_tris, sorted_bn, node)

        return True

    def _update_neighbors(self, cavity: Set[DelaunayTriangle],
                          new_tris: List[DelaunayTriangle],
                          sorted_bn: List[int], node: DelaunayNode):
        """更新新三角形的邻接关系"""
        n_bn = len(sorted_bn)
        n_new = len(new_tris)

        # 新三角形之间的邻接
        for i in range(n_new):
            new_tris[i].neighbors[0] = new_tris[(i - 1) % n_new]
            new_tris[i].neighbors[2] = new_tris[(i + 1) % n_new]

        # 新三角形与外部三角形的邻接
        # 遍历腔体边界边，找到外部三角形
        for i in range(n_bn):
            a, b = sorted_bn[i], sorted_bn[(i + 1) % n_bn]
            e = frozenset([a, b])
            # 在全局边映射中查找共享此边的外部三角形
            ext_tris = [ti for ti in self._edge_map.get(e, [])
                       if ti not in {t.idx for t in cavity}]
            # 实际上腔体三角形已被移除，edge_map 中只有新三角形和外部三角形
            ext_tris_after = [ti for ti in self._edge_map.get(e, [])
                             if ti not in {t.idx for t in new_tris}]

    # ------------------------------------------------------------------
    # 边界边恢复
    # ------------------------------------------------------------------

    def _recover_boundary_edges(self, boundary_edges: List[frozenset]) -> int:
        """
        通过边交换恢复缺失的边界边。

        对每个缺失的边界边 (a,b)：
        1. 找到所有与该线段 UV 投影相交的三角形
        2. 如果恰好 2 个三角形相交：单次边交换恢复
        3. 如果 >2 个：移除相交三角形，用边界边 (a,b) 为约束重新三角化

        Returns:
            恢复的边数
        """
        recovered = 0
        max_passes = 20  # 增加迭代次数

        for _pass in range(max_passes):
            missing = []
            for be in boundary_edges:
                if be not in self._edge_map or len(self._edge_map[be]) == 0:
                    missing.append(be)

            if not missing:
                break

            any_recovered = False
            for be in missing:
                a, b = tuple(be)
                uva = np.array(self.nodes[a].uv)
                uvb = np.array(self.nodes[b].uv)

                # 尝试1：单次边交换（2 交叉）
                intersecting = self._find_tris_crossing_edge(a, b, uva, uvb)
                if len(intersecting) == 2:
                    if self._swap_to_recover_edge(a, b, intersecting):
                        recovered += 1; any_recovered = True; continue

                # 尝试2：顺序交换（>2 交叉：沿着线段逐步推进）
                if len(intersecting) > 2:
                    if self._sequential_swap_recover_edge(a, b):
                        recovered += 1; any_recovered = True; continue

                # 尝试3：直接插入（0 交叉：凸包边界外）
                if len(intersecting) == 0:
                    if self._insert_boundary_edge_direct(a, b):
                        recovered += 1; any_recovered = True; continue

                # 尝试4：删除+重三角化（兜底）
                if self._retriangulate_to_recover_edge(a, b):
                    recovered += 1; any_recovered = True

            if not any_recovered:
                break

        return recovered

    def _sequential_swap_recover_edge(self, a: int, b: int) -> bool:
        """
        通过顺序边交换恢复边界边 (a,b)（处理 >2 三角形交叉）。

        逐步交换与线段 (a,b) 相交的三角形边，每次交换减少相交
        三角形数量，最终收敛到 2 三角形配置后用单次交换完成。

        Returns:
            True 表示成功恢复
        """
        uva = np.array(self.nodes[a].uv)
        uvb = np.array(self.nodes[b].uv)
        target = frozenset([a, b])

        for _ in range(50):
            if target in self._edge_map and len(self._edge_map[target]) > 0:
                return True

            intersecting = self._find_tris_crossing_edge(a, b, uva, uvb)
            if len(intersecting) <= 1:
                return False

            # 尝试单次交换（恰好 2 个三角形）
            if len(intersecting) == 2:
                if self._swap_to_recover_edge(a, b, intersecting):
                    continue  # 成功，检查是否需要更多交换
                return False  # 2 三角形但交换失败

            # >2 三角形：找到最靠近 a 的相交边并交换它
            swapped = False
            for ti in intersecting:
                tri = self.triangles[ti]
                if tri.removed:
                    continue
                pts = tri.coords_2d
                for k in range(3):
                    p0, p1 = pts[k], pts[(k + 1) % 3]
                    if not self._segments_intersect_2d(uva, uvb, p0, p1):
                        continue
                    n0, n1 = tri.node_ids[k], tri.node_ids[(k + 1) % 3]
                    if a in (n0, n1) or b in (n0, n1):
                        continue
                    e = frozenset([n0, n1])
                    neighbors = [idx for idx in self._edge_map.get(e, [])
                                if idx != ti and not self.triangles[idx].removed]
                    if len(neighbors) != 1:
                        continue
                    # 通用交换（不检查是否直接产生 (a,b)）
                    if self._try_flip_edge(ti, neighbors[0]):
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                return False
        return False

    def _try_flip_edge(self, idx0: int, idx1: int) -> bool:
        """
        交换两个三角形的共享边（通用边交换，用于减少边界边交叉数量）。

        检查凸性后执行交换，不要求交换产生特定边。
        """
        tri0 = self.triangles[idx0]
        tri1 = self.triangles[idx1]
        if tri0.removed or tri1.removed:
            return False

        ids0 = set(tri0.node_ids)
        ids1 = set(tri1.node_ids)
        common = ids0 & ids1
        if len(common) != 2:
            return False

        c, d = sorted(common)
        a = (ids0 - common).pop()
        b = (ids1 - common).pop()

        # 凸性检查
        uvc = np.array(self.nodes[c].uv)
        uvd = np.array(self.nodes[d].uv)
        uva = np.array(self.nodes[a].uv)
        uvb = np.array(self.nodes[b].uv)
        if not self._is_quad_convex_2d(uvc, uva, uvd, uvb):
            return False

        self._remove_triangle(tri0)
        self._remove_triangle(tri1)

        self._add_triangle(self.nodes[a], self.nodes[c], self.nodes[d])
        self._add_triangle(self.nodes[b], self.nodes[c], self.nodes[d])
        return True

    def _retriangulate_to_recover_edge(self, a: int, b: int) -> bool:
        """
        删除+重三角化恢复边界边 (a,b)。

        找到线段 (a,b) 穿过的所有三角形（或包含 a 或 b 作为顶点的三角形），
        移除它们，构建剩余区域的边界多边形，沿 (a,b) 分割后对两半分别
        做耳切法三角化。
        """
        uva = np.array(self.nodes[a].uv)
        uvb = np.array(self.nodes[b].uv)
        mid = (uva + uvb) / 2.0

        # 收集需要移除的三角形：与线段相交，或包含中点
        to_remove: Set[int] = set()
        for tri in self.triangles:
            if tri.removed or tri.virtual:
                continue
            ids = set(tri.node_ids)
            # 检查三角形边是否与线段相交
            pts = tri.coords_2d
            crosses = False
            for k in range(3):
                p0, p1 = pts[k], pts[(k + 1) % 3]
                if self._segments_intersect_2d(uva, uvb, p0, p1):
                    crosses = True
                    break
            # 或者三角形包含线段中点
            if crosses or self._point_in_triangle_2d(
                mid, pts[0], pts[1], pts[2]):
                to_remove.add(tri.idx)

        if not to_remove:
            return False

        # 收集已移除区域的边界边：被恰好 1 个移除三角形使用的边
        edge_use: Dict[frozenset, int] = defaultdict(int)
        for ti in to_remove:
            tri = self.triangles[ti]
            for k in range(3):
                n0, n1 = tri.node_ids[k], tri.node_ids[(k + 1) % 3]
                edge_use[frozenset([n0, n1])] += 1

        region_boundary = [e for e, cnt in edge_use.items() if cnt == 1]

        # 构建边界节点并排序
        boundary_set: Set[int] = set()
        for e in region_boundary:
            boundary_set.update(e)

        # 移除三角形
        for ti in to_remove:
            tri = self.triangles[ti]
            if not tri.removed:
                self._remove_triangle(tri)

        # 构建多边形：从 a 出发沿边界走到 b，再从 b 沿另一侧走回 a
        adj: Dict[int, List[int]] = defaultdict(list)
        for e in region_boundary:
            n0, n1 = tuple(e)
            adj[n0].append(n1)
            adj[n1].append(n0)

        # 从 a 出发沿邻接走到 b（沿一侧）
        poly_left = self._trace_path(adj, a, b)
        # 从 b 沿邻接走回 a（沿另一侧，排除 poly_left 已用的边）
        # 简单做法：从 b 出发，排除通向 a 的第一条边
        poly_right = self._trace_path_other(adj, b, a, poly_left)

        # 耳切法三角化两个子多边形
        for poly in [poly_left, poly_right]:
            if len(poly) >= 3:
                self._ear_clip_triangulate(poly)

        return len(poly_left) >= 2 or len(poly_right) >= 2

    def _trace_path(self, adj: Dict[int, List[int]],
                    start: int, end: int) -> List[int]:
        """从 start 沿邻接走到 end（贪心：总是走向 end 最近的邻居）"""
        path = [start]
        current = start
        prev = None
        uvb = np.array(self.nodes[end].uv)
        while current != end and len(path) < len(adj) + 1:
            neighbors = [n for n in adj.get(current, []) if n != prev]
            if not neighbors:
                break
            # 选择 UV 距离 end 最近的邻居
            best = min(neighbors, key=lambda n:
                np.linalg.norm(np.array(self.nodes[n].uv) - uvb))
            path.append(best)
            prev, current = current, best
        return path

    def _trace_path_other(self, adj: Dict[int, List[int]],
                           start: int, end: int,
                           exclude_path: List[int]) -> List[int]:
        """从 start 走到 end，不使用 exclude_path 中相邻节点间的边"""
        exclude_edges = set()
        for i in range(len(exclude_path) - 1):
            exclude_edges.add(frozenset([exclude_path[i], exclude_path[i+1]]))

        path = [start]
        current = start
        prev = None
        uvb = np.array(self.nodes[end].uv)
        while current != end and len(path) < len(adj) + 1:
            neighbors = [n for n in adj.get(current, [])
                        if n != prev
                        and frozenset([current, n]) not in exclude_edges]
            if not neighbors:
                break
            best = min(neighbors, key=lambda n:
                np.linalg.norm(np.array(self.nodes[n].uv) - uvb))
            path.append(best)
            prev, current = current, best
        return path

    def _ear_clip_triangulate(self, polygon: List[int]):
        """
        对给定的 UV 空间多边形做耳切法三角化。

        多边形顶点按绕序排列（顺时针或逆时针）。
        """
        n = len(polygon)
        if n < 3:
            return
        if n == 3:
            self._add_triangle(
                self.nodes[polygon[0]],
                self.nodes[polygon[1]],
                self.nodes[polygon[2]])
            return

        # 迭代耳切法
        remaining = list(polygon)
        while len(remaining) > 2:
            m = len(remaining)
            clipped = False
            for i in range(m):
                prev = remaining[(i - 1) % m]
                curr = remaining[i]
                next_n = remaining[(i + 1) % m]

                # 检查耳朵：prev-curr-next 是否形成有效三角形
                uva = np.array(self.nodes[prev].uv)
                uvb = np.array(self.nodes[curr].uv)
                uvc = np.array(self.nodes[next_n].uv)

                # 凸性检查
                cross = (uvb[0]-uva[0])*(uvc[1]-uva[1]) - (uvb[1]-uva[1])*(uvc[0]-uva[0])
                if cross <= 0:  # 凹角，跳过
                    continue

                # 检查是否有其他顶点在三角形内
                has_interior = False
                for j in range(m):
                    if j in [(i-1)%m, i, (i+1)%m]:
                        continue
                    pt = np.array(self.nodes[remaining[j]].uv)
                    if self._point_in_triangle_2d(pt, uva, uvb, uvc):
                        has_interior = True
                        break
                if has_interior:
                    continue

                # 创建三角形
                self._add_triangle(
                    self.nodes[prev],
                    self.nodes[curr],
                    self.nodes[next_n])
                remaining.pop(i)
                clipped = True
                break
            if not clipped:
                break  # 无法继续，丢弃剩余多边形

    @staticmethod
    def _point_in_triangle_2d(p, a, b, c):
        """检查 2D 点是否在三角形内（含边界）"""
        d1 = (p[0]-b[0])*(a[1]-b[1]) - (p[1]-b[1])*(a[0]-b[0])
        d2 = (p[0]-c[0])*(b[1]-c[1]) - (p[1]-c[1])*(b[0]-c[0])
        d3 = (p[0]-a[0])*(c[1]-a[1]) - (p[1]-a[1])*(c[0]-a[0])
        has_neg = d1 < -1e-12 or d2 < -1e-12 or d3 < -1e-12
        has_pos = d1 > 1e-12 or d2 > 1e-12 or d3 > 1e-12
        return not (has_neg and has_pos)

    def _find_tris_crossing_edge(self, a: int, b: int,
                                  uva: np.ndarray, uvb: np.ndarray
                                  ) -> List[int]:
        """找到与线段 (a,b) 在 UV 空间相交的三角形"""
        result = []
        for tri in self.triangles:
            if tri.removed:
                continue
            ids = set(tri.node_ids)
            # 三角形包含 a 或 b → 邻接，不是交叉
            if a in ids or b in ids:
                continue
            # 检查三角形边是否与线段 (a,b) 相交
            pts = tri.coords_2d
            for k in range(3):
                p0, p1 = pts[k], pts[(k + 1) % 3]
                if self._segments_intersect_2d(uva, uvb, p0, p1):
                    result.append(tri.idx)
                    break
        return result

    def _swap_to_recover_edge(self, a: int, b: int,
                               tris: List[int]) -> bool:
        """
        通过交换两个三角形来恢复边界边 (a,b)。

        两个三角形共享一条边 (c,d)，交换后产生边 (a,b)。
        前提：两个三角形构成凸四边形 a-c-b-d。
        """
        if len(tris) != 2:
            return False

        t0 = self.triangles[tris[0]]
        t1 = self.triangles[tris[1]]

        # 找到共享边
        ids0 = set(t0.node_ids)
        ids1 = set(t1.node_ids)
        common = ids0 & ids1
        if len(common) != 2:
            return False

        c, d = sorted(common)
        other0 = (ids0 - common).pop()
        other1 = (ids1 - common).pop()

        # 验证：交换后应该产生 (a,b)
        new_edge = frozenset([other0, other1])
        if new_edge != frozenset([a, b]):
            # 尝试交换角色：a,b 可能不是 other0, other1
            # 重新识别：与 (a,b) 相交的两个三角形，交换后产生 (a,b)
            # 交换前：t0=(a, c, d的某个), t1=(b, c, d的某个)
            # 需要确保 other0, other1 分别是 a 和 b
            pass

        # 需要更精确的逻辑：找到两个三角形的配置
        # 两个三角形共享边 (c,d)，分别包含 a 和 b
        if a not in ids0 and a not in ids1:
            return False
        if b not in ids0 and b not in ids1:
            return False

        # 交换
        uva = np.array(self.nodes[a].uv)
        uvb = np.array(self.nodes[b].uv)
        uvc = np.array(self.nodes[c].uv)
        uvd = np.array(self.nodes[d].uv)

        # 凸性检查
        if not self._is_quad_convex_2d(uvc, uva, uvd, uvb):
            return False

        self._remove_triangle(t0)
        self._remove_triangle(t1)

        self._add_triangle(self.nodes[a], self.nodes[c], self.nodes[d])
        self._add_triangle(self.nodes[b], self.nodes[c], self.nodes[d])
        return True

    @staticmethod
    def _is_quad_convex_2d(p0, p1, p2, p3):
        """检查四边形是否凸（2D 叉积符号一致）"""
        pts = [p0, p1, p2, p3]
        signs = []
        for i in range(4):
            a = pts[i]
            b = pts[(i + 1) % 4]
            c = pts[(i + 2) % 4]
            cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            if abs(cross) > 1e-12:
                signs.append(cross > 0)
        return len(set(signs)) == 1 if signs else False

    def _insert_boundary_edge_direct(self, a: int, b: int) -> bool:
        """
        直接插入边界边 (a,b)：找到 a 在三角剖分中的邻居，选择
        能使三角形 (a,b,c) 不与其他三角形相交的邻居 c。

        用于恢复被虚拟网格清理移除的凸包边界边（0-intersecting 情况）。
        """
        # 收集 a 和 b 的邻居
        neighbors_a = set()
        neighbors_b = set()
        for tri in self.triangles:
            if tri.removed:
                continue
            ids = tri.node_ids
            if a in ids:
                neighbors_a.update(ids)
            if b in ids:
                neighbors_b.update(ids)
        neighbors_a.discard(a); neighbors_b.discard(b)

        uva = np.array(self.nodes[a].uv)
        uvb = np.array(self.nodes[b].uv)

        # 候选节点：a 邻居 ∩ b 邻居，或 a 的邻居
        candidates = list(neighbors_a & neighbors_b)
        if not candidates:
            candidates = list(neighbors_a)

        for c in candidates:
            # 检查三角形 (a,b,c) 是否与已有三角形相交（除共享 n 的三角形外）
            uvc = np.array(self.nodes[c].uv)
            # 退化检查
            if np.linalg.norm(np.cross(uvb - uva, uvc - uva)) < 1e-16:
                continue
            # 检查是否与已有三角形重叠
            if self._triangle_intersects_existing(a, b, c):
                continue
            self._add_triangle(self.nodes[a], self.nodes[b], self.nodes[c])
            return True
        return False

    def _triangle_intersects_existing(self, a: int, b: int, c: int) -> bool:
        """检查三角形 (a,b,c) 是否与已有三角形相交"""
        new_ids = {a, b, c}
        uvs_new = np.array([self.nodes[n].uv for n in (a, b, c)])
        for tri in self.triangles:
            if tri.removed:
                continue
            ids = set(tri.node_ids)
            shared = ids & new_ids
            if len(shared) >= 2:  # 共享边，跳过
                continue
            if len(shared) == 1:  # 共享顶点，检查边是否交叉
                continue  # 共享顶点的三角形不视为相交
            # 检查三角形重叠
            uvs_ex = tri.coords_2d
            for k in range(3):
                p0, p1 = uvs_new[k], uvs_new[(k+1)%3]
                for j in range(3):
                    q0, q1 = uvs_ex[j], uvs_ex[(j+1)%3]
                    if self._segments_intersect_2d(p0, p1, q0, q1):
                        return True
        return False

    def _insert_node_with_boundary_protection(
        self, node: DelaunayNode, boundary_edges: Set[frozenset]
    ) -> bool:
        """
        插入节点时保护已有边界边——腔体不跨越边界边。

        用于内部节点插入阶段：边界边已在三角剖分中，内部节点插入
        不应破坏它们。
        """
        # 1. 重复检查
        for existing in self.nodes:
            if existing.idx == node.idx:
                continue
            d2 = (node.uv[0]-existing.uv[0])**2 + (node.uv[1]-existing.uv[1])**2
            if d2 < self._node_tolerance**2:
                return True

        # 2. 种子三角形
        seed = self._find_seed_triangle(node.uv)
        if seed is None:
            return False

        # 3. 构建腔体（不跨越边界边）
        cavity, cavity_boundary = self._build_cavity_protected(
            node, seed, boundary_edges)
        if not cavity:
            return False

        if not cavity_boundary:
            return False

        # 4-6. 腔体边界排序、移除、重建（与 _insert_node 相同）
        boundary_nodes_set = set()
        for e in cavity_boundary:
            boundary_nodes_set.update(e)
        bn_list = list(boundary_nodes_set)
        bn_uvs = np.array([self.nodes[n].uv for n in bn_list])
        centroid = bn_uvs.mean(axis=0)
        angles = np.arctan2(bn_uvs[:,1]-centroid[1], bn_uvs[:,0]-centroid[0])
        sorted_bn = [n for _, n in sorted(zip(angles, bn_list))]

        for tri in cavity:
            self._remove_triangle(tri)

        n_bn = len(sorted_bn)
        for i in range(n_bn):
            n0 = node
            n1 = self.nodes[sorted_bn[i]]
            n2 = self.nodes[sorted_bn[(i+1)%n_bn]]
            if n0.idx == n1.idx or n1.idx == n2.idx or n0.idx == n2.idx:
                continue
            self._add_triangle(n0, n1, n2)

        return True

    def _build_cavity_protected(
        self, node: DelaunayNode, seed: DelaunayTriangle,
        protected: Set[frozenset]
    ) -> Tuple[Set[DelaunayTriangle], Set[frozenset]]:
        """
        构建受保护腔体：三角形可以包含受保护边（加入腔体），
        但 BFS 遍历时不跨越受保护边。这确保边界边在周围三角形
        被移除后仍由相邻的非腔体三角形保留。
        """
        cavity: Set[DelaunayTriangle] = set()
        queue = deque([seed])
        visited: Set[int] = set()

        while queue:
            tri = queue.popleft()
            if tri.idx in visited:
                continue
            visited.add(tri.idx)

            if not self._in_circumcircle_2d(node, tri):
                continue

            cavity.add(tri)
            for k in range(3):
                e = frozenset([tri.node_ids[k], tri.node_ids[(k+1)%3]])
                # 不跨越受保护边
                if e in protected:
                    continue
                for ni in self._edge_map.get(e, []):
                    if ni not in visited:
                        nb = self.triangles[ni]
                        if not nb.removed:
                            queue.append(nb)

        edge_use: Dict[frozenset, int] = defaultdict(int)
        for tri in cavity:
            for k in range(3):
                a, b = tri.node_ids[k], tri.node_ids[(k+1)%3]
                edge_use[frozenset([a,b])] += 1
        boundary_edges = {e for e, c in edge_use.items() if c == 1}
        return cavity, boundary_edges

    @staticmethod
    def _segments_intersect_2d(a1, a2, b1, b2):
        """检查两个 2D 线段是否相交（不含端点接触）"""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(b1, b2, a1)
        d2 = cross(b1, b2, a2)
        d3 = cross(a1, a2, b1)
        d4 = cross(a1, a2, b2)
        eps = 1e-10
        return ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
               ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps))

    # ------------------------------------------------------------------
    # 内部节点生成
    # ------------------------------------------------------------------

    def generate_interior_nodes(
        self, boundary_nodes: List[DelaunayNode],
        spacing: float, uv_bounds: Tuple[float, float, float, float] = None
    ) -> List[DelaunayNode]:
        """
        基于曲面度量张量自适应生成内部节点。

        使用曲面的第一基本形式 (E, F, G) 计算 UV 参数空间到 3D
        的局部缩放因子，使得 3D 空间中的节点间距接近均匀。

        Args:
            boundary_nodes: 边界节点（用于构建 UV 边界多边形）
            spacing: 目标 3D 内部节点间距
            uv_bounds: (u_min, u_max, v_min, v_max) 参数域范围

        Returns:
            内部节点列表
        """
        if len(boundary_nodes) < 3:
            return []

        uvs = np.array([n.uv for n in boundary_nodes])
        centroid = uvs.mean(axis=0)
        angles = np.arctan2(uvs[:, 1] - centroid[1], uvs[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        polygon = uvs[sorted_idx]

        if uv_bounds is None:
            u_min, u_max = uvs[:, 0].min(), uvs[:, 0].max()
            v_min, v_max = uvs[:, 1].min(), uvs[:, 1].max()
        else:
            u_min, u_max, v_min, v_max = uv_bounds

        # 使用曲面度量计算 v 方向的采样线数（G 通常较均匀）
        n_v = max(3, int((v_max - v_min) / spacing * 2))
        dv = (v_max - v_min) / n_v

        interior = []
        for j in range(1, n_v):
            v = v_min + j * dv

            # 在 v 线中点处计算曲面度量 E = ||dS/du||²
            u_mid = (u_min + u_max) / 2.0
            E = self._compute_metric_E(u_mid, v)

            # 根据度量调整 u 方向的采样密度
            # 3D 间距 ds_u ≈ sqrt(E) * du, 目标 ds_u ≈ spacing
            if E > 1e-12:
                sqrtE = np.sqrt(E)
                n_u = max(3, int((u_max - u_min) * sqrtE / spacing))
            else:
                n_u = max(3, int((u_max - u_min) / spacing))

            du = (u_max - u_min) / n_u

            for i in range(1, n_u):
                uv = (u_min + i * du, v)

                if not self._point_in_polygon_2d(uv, polygon):
                    continue

                # 边界距离检查
                too_close = False
                for bn in boundary_nodes:
                    d2 = (uv[0] - bn.uv[0])**2 + (uv[1] - bn.uv[1])**2
                    if d2 < (spacing * 0.2)**2:
                        too_close = True
                        break
                if too_close:
                    continue

                try:
                    xyz = self._uv_to_xyz(uv)
                except Exception:
                    continue

                node = DelaunayNode(self._next_node_idx, uv, xyz)
                self._next_node_idx += 1
                self.nodes.append(node)
                interior.append(node)

        return interior

    def _compute_metric_E(self, u: float, v: float) -> float:
        """
        计算曲面第一基本形式的 E = ||dS/du||²。

        使用中心差分近似曲面偏导数。
        """
        eps = 1e-6
        try:
            p0 = np.array(self._uv_to_xyz((u - eps, v)))
            p1 = np.array(self._uv_to_xyz((u + eps, v)))
            dS_du = (p1 - p0) / (2.0 * eps)
            return float(np.dot(dS_du, dS_du))
        except Exception:
            return 1.0  # 回退到单位度量

    @staticmethod
    def _point_in_polygon_2d(uv: Tuple[float, float],
                             polygon: np.ndarray) -> bool:
        """射线法判断点是否在 2D 多边形内部"""
        u, v = uv
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            ui, vi = polygon[i]
            uj, vj = polygon[j]
            if ((vi > v) != (vj > v)) and \
               (u < (uj - ui) * (v - vi) / (vj - vi + 1e-30) + ui):
                inside = not inside
            j = i
        return inside

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def _cleanup_virtual(self):
        """移除所有与虚拟节点相连的三角形"""
        virtual_ids = {vn.idx for vn in self._virtual_nodes}
        to_remove = []
        for tri in self.triangles:
            if tri.removed:
                continue
            if any(nid in virtual_ids for nid in tri.node_ids):
                to_remove.append(tri)
        for tri in to_remove:
            self._remove_triangle(tri)

    def _cleanup_outside_boundary(self, boundary_nodes: List[DelaunayNode]):
        """
        删除边界外部的三角形。

        方法：从任意三角形开始 BFS，遇到 boundary_edges 中不存在的边时停止。
        同时处理 boundary_nodes 中可能不在 edge_map 中的情况。
        """
        if not self.triangles or len(boundary_nodes) < 3:
            return

        # 构建边界边集合（原始边界节点顺序的边）
        n_bn = len(boundary_nodes)
        boundary_edges = set()
        for i in range(n_bn):
            a = boundary_nodes[i].idx
            b = boundary_nodes[(i + 1) % n_bn].idx
            boundary_edges.add(frozenset([a, b]))

        # 找到种子三角形（重心在 UV 边界内部的三角形）
        polygon = np.array([n.uv for n in boundary_nodes])
        seed = None
        for tri in self.triangles:
            if tri.removed:
                continue
            tri_uvs = tri.coords_2d
            c = tri_uvs.mean(axis=0)
            if self._point_in_polygon_2d((c[0], c[1]), polygon):
                seed = tri
                break

        if seed is None:
            return

        # BFS：跨非边界边遍历
        visited: Set[int] = set()
        queue = deque([seed])
        while queue:
            tri = queue.popleft()
            if tri.idx in visited:
                continue
            visited.add(tri.idx)
            for k in range(3):
                a, b = tri.node_ids[k], tri.node_ids[(k + 1) % 3]
                e = frozenset([a, b])
                if e in boundary_edges:
                    continue
                for ni in self._edge_map.get(e, []):
                    if ni not in visited:
                        nb = self.triangles[ni]
                        if not nb.removed:
                            queue.append(nb)

        # 删除未访问的三角形
        removed = 0
        for tri in self.triangles:
            if tri.removed:
                continue
            if tri.idx not in visited:
                self._remove_triangle(tri)
                removed += 1
        if removed:
            info(f"清理边界外三角形: {removed}")

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def triangulate(self, boundary_nodes: List[DelaunayNode],
                    interior_nodes: List[DelaunayNode] = None,
                    interior_spacing: float = None,
                    uv_bounds: Tuple[float, float, float, float] = None,
                    boundary_edges_list: List[Tuple[int, int]] = None
                    ) -> List[DelaunayTriangle]:
        """
        对给定节点集进行 Delaunay 三角剖分（含边界约束恢复）。

        Args:
            boundary_nodes: 边界节点列表
            interior_nodes: 预生成的内部节点列表（可选）
            interior_spacing: 内部节点间距（自动生成内部节点）
            uv_bounds: UV 参数域 (u_min, u_max, v_min, v_max)
            boundary_edges_list: 边界边列表 [(idx_a, idx_b), ...]。
                如果不提供，则按 boundary_nodes 顺序构建。

        Returns:
            边界约束的三角形列表
        """
        info(f"Delaunay 三角剖分: {len(boundary_nodes)} 边界节点"
             + (f" + {len(interior_nodes)} 内部节点" if interior_nodes else ""))

        # 添加边界节点
        for bn in boundary_nodes:
            self._add_node(bn.uv, bn.xyz)

        # 构建边界边集合
        if boundary_edges_list is not None:
            boundary_edges = {frozenset([a, b])
                             for a, b in boundary_edges_list}
        else:
            n_bn = len(boundary_nodes)
            boundary_edges = set()
            for i in range(n_bn):
                a = boundary_nodes[i].idx
                b = boundary_nodes[(i + 1) % n_bn].idx
                boundary_edges.add(frozenset([a, b]))

        # 1. 创建虚拟网格
        self._create_virtual_mesh(boundary_nodes)

        # 2. 插入边界节点（无保护）
        for bn in boundary_nodes:
            self._insert_node(bn)

        # 3. 移除虚拟三角形
        self._cleanup_virtual()

        # 4. 边界边恢复（此时边界边还未受保护）
        recovered = self._recover_boundary_edges(list(boundary_edges))
        if recovered > 0:
            info(f"边界边恢复: {recovered} 条")

        # 5. 删除边界外部的三角形
        self._cleanup_outside_boundary(boundary_nodes)

        # 6. 生成内部节点
        if interior_spacing is not None and interior_spacing > 0:
            gen_interior = self.generate_interior_nodes(
                boundary_nodes, interior_spacing, uv_bounds)
            info(f"生成内部节点: {len(gen_interior)}")

            # 7. 插入内部节点（保护已恢复的边界边）
            for inode in gen_interior:
                self._insert_node_with_boundary_protection(inode, boundary_edges)

        # 8. 插入预生成的内部节点
        if interior_nodes:
            for inode in interior_nodes:
                self._insert_node_with_boundary_protection(inode, boundary_edges)

        # 9. 多轮边界边恢复（内部节点插入可能破坏多条边界边）
        for round_idx in range(5):
            recovered_n = self._recover_boundary_edges(list(boundary_edges))
            if recovered_n == 0:
                break
            info(f"边界边恢复(第{round_idx+2}轮): {recovered_n} 条")

        # 10. 再次清理边界外三角形
        self._cleanup_outside_boundary(boundary_nodes)

        # 11. 最终边界边检查与恢复
        self._recover_boundary_edges(list(boundary_edges))
        valid = [t for t in self.triangles if not t.removed]
        info(f"Delaunay 完成: {len(valid)} 三角形")
        return valid


# ============================================================================
# 与 SurfaceMeshGenerator 集成的辅助函数
# ============================================================================

def delaunay_triangulate_surface(
    surface: TopoDS_Face,
    boundary_nodes: list,      # List[NodeElement3D]
    interior_nodes: list = None,
    geometry: SurfaceGeometry = None,
    normal_flip: float = 1.0,
) -> List:
    """
    使用 Delaunay 三角剖分生成曲面网格。

    将 SurfaceMeshGenerator 的 NodeElement3D 转换为内部 DelaunayNode，
    执行 Bowyer-Watson 三角剖分，然后重建 SurfaceTriangle 列表。

    Args:
        surface: OCC 曲面
        boundary_nodes: 边界节点（NodeElement3D 列表）
        interior_nodes: 内部节点（可选）
        geometry: 曲面几何处理器
        normal_flip: 法向翻转因子

    Returns:
        SurfaceTriangle 列表
    """
    from .surface_front import SurfaceTriangle, NodeElement3D

    geo = geometry or SurfaceGeometry()

    # 转换为 DelaunayNode
    d_nodes = []
    for bn in boundary_nodes:
        uv = geo.project_point_to_surface(tuple(bn.coords), surface)
        d_nodes.append(DelaunayNode(bn.idx, uv, tuple(bn.coords)))

    # 创建 Delaunay 三角剖分器
    delaunay = SurfaceDelaunay(surface, geo, normal_flip)

    # 执行三角剖分
    d_triangles = delaunay.triangulate(d_nodes)

    # 转换为 SurfaceTriangle
    idx_to_node = {bn.idx: bn for bn in boundary_nodes}
    surface_tris = []
    for dt in d_triangles:
        n0 = idx_to_node[dt.node_ids[0]]
        n1 = idx_to_node[dt.node_ids[1]]
        n2 = idx_to_node[dt.node_ids[2]]
        tri = SurfaceTriangle(n0, n1, n2, surface=surface, idx=len(surface_tris))
        surface_tris.append(tri)

    return surface_tris

"""
曲面背景网格模块

在 3D 曲面上构建三角形背景网格，预计算曲率和边界衰减值，
查询时通过 UV 定位 + 重心坐标插值得到平滑的尺寸值。
"""
import numpy as np
from typing import Tuple, List, Optional
from OCC.Core.TopoDS import TopoDS_Face

from utils.geom_toolkit import point_to_segment_distance_3d
from meshsize.surface_sizing import decay_spacing


class SurfaceBackgroundMesh:
    """
    曲面背景网格

    在 UV 参数域上均匀采样，投影到 3D 曲面构建三角形背景网格。
    每个节点存储预计算的尺寸值（曲率 + 边界衰减），
    查询时通过 UV 定位 + 3D 重心坐标插值。
    """

    def __init__(
        self,
        surface: TopoDS_Face,
        geometry_handler,
        global_spacing: float = 1.0,
        min_spacing: float = 0.01,
        max_spacing: float = 100.0,
        curvature_factor: float = 0.2,
        boundary_decay: float = 1.2,
        resolution: int = 21,
    ):
        self.surface = surface
        self.geometry = geometry_handler
        self.global_spacing = global_spacing
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.curvature_factor = curvature_factor
        self.boundary_decay = boundary_decay
        self.resolution = resolution

        # UV 参数域边界
        self._u_min = self._u_max = self._v_min = self._v_max = 0.0
        self._du = self._dv = 0.0

        # 节点数据
        self.nodes_uv = None    # (N, 2)
        self.nodes_3d = None    # (N, 3)
        self.spacing_values = None  # (N,)
        self.triangles = None   # (M, 3)
        self.nu = 0
        self.nv = 0

    def build(self):
        """构建背景网格：采样 UV 域 → 计算 3D 坐标 → 计算曲率尺寸 → 三角化"""
        u_min, u_max, v_min, v_max = self.geometry.get_surface_bounds(self.surface)
        self._u_min, self._u_max = u_min, u_max
        self._v_min, self._v_max = v_min, v_max

        self.nu = self.resolution
        self.nv = self.resolution
        self._du = (u_max - u_min) / (self.nu - 1)
        self._dv = (v_max - v_min) / (self.nv - 1)

        n_nodes = self.nu * self.nv
        self.nodes_uv = np.zeros((n_nodes, 2))
        self.nodes_3d = np.zeros((n_nodes, 3))
        self.spacing_values = np.full(n_nodes, self.global_spacing)

        # 采样节点
        for j in range(self.nv):
            v = v_min + j * self._dv
            for i in range(self.nu):
                idx = j * self.nu + i
                u = u_min + i * self._du
                self.nodes_uv[idx] = [u, v]

                # 3D 坐标
                pt = self.geometry.evaluate_point(u, v, self.surface)
                self.nodes_3d[idx] = pt

                # 曲率尺寸
                self.spacing_values[idx] = self._compute_curvature_spacing(u, v)

        # 三角化：每个矩形网格单元拆分为 2 个三角形
        tri_list = []
        for j in range(self.nv - 1):
            for i in range(self.nu - 1):
                n00 = j * self.nu + i
                n10 = n00 + 1
                n01 = n00 + self.nu
                n11 = n01 + 1
                tri_list.append([n00, n10, n11])
                tri_list.append([n00, n11, n01])
        self.triangles = np.array(tri_list, dtype=np.int32)

    def _compute_curvature_spacing(self, u: float, v: float) -> float:
        """计算单点的曲率尺寸"""
        mean_curv, gauss_curv, max_curv = self.geometry.get_surface_curvature(
            u, v, self.surface
        )
        if max_curv < 1e-12:
            return self.global_spacing
        radius = 1.0 / max_curv
        spacing = self.curvature_factor * radius
        return np.clip(spacing, self.min_spacing, self.max_spacing)

    def apply_boundary_decay(self, fronts):
        """
        将边界衰减预计算到背景网格节点。

        对每个节点，计算到所有前沿边的 3D 距离，
        应用 decay_spacing() 公式，取最小值。

        Args:
            fronts: SurfaceFront 列表
        """
        front_list = list(fronts)
        if not front_list:
            return

        # 收集所有前沿边的 3D 端点和源尺寸
        edges_3d = []
        for front in front_list:
            p0 = np.array(front.node_elems[0].coords)
            p1 = np.array(front.node_elems[1].coords)
            source_size = np.linalg.norm(p1 - p0)
            if source_size > 1e-12:
                edges_3d.append((p0, p1, source_size))

        if not edges_3d:
            return

        decay = self.boundary_decay
        for idx in range(len(self.nodes_3d)):
            p = self.nodes_3d[idx]
            min_spacing = self.global_spacing
            for p0, p1, source_size in edges_3d:
                dist = point_to_segment_distance_3d(p, p0, p1)
                sp = decay_spacing(source_size, dist, decay)
                if sp < min_spacing:
                    min_spacing = sp
            # 与当前值取 min
            if min_spacing < self.spacing_values[idx]:
                self.spacing_values[idx] = min_spacing

    def spacing_at(self, point_3d: Tuple[float, float, float]) -> float:
        """
        查询 3D 点处的尺寸值。

        Args:
            point_3d: 三维坐标

        Returns:
            插值后的尺寸值
        """
        uv = self.geometry.project_point_to_surface(point_3d, self.surface)
        u, v = uv

        # O(1) 定位到网格单元
        fi = (u - self._u_min) / self._du
        fj = (v - self._v_min) / self._dv

        # clamp 到有效范围
        fi = np.clip(fi, 0, self.nu - 1 - 1e-10)
        fj = np.clip(fj, 0, self.nv - 1 - 1e-10)

        ci = int(fi)
        cj = int(fj)
        ci = min(ci, self.nu - 2)
        cj = min(cj, self.nv - 2)

        # 局部坐标 [0,1]
        s = fi - ci
        t = fj - cj

        # 四个角节点索引
        n00 = cj * self.nu + ci
        n10 = n00 + 1
        n01 = n00 + self.nu
        n11 = n01 + 1

        # 确定包含点的三角形 + 重心坐标插值
        # 对角线从 n00 到 n11
        if s + t <= 1.0:
            # 三角形 (n00, n10, n11)
            w0 = 1.0 - s - t
            w1 = s
            w2 = t
            return self._clamp(
                w0 * self.spacing_values[n00]
                + w1 * self.spacing_values[n10]
                + w2 * self.spacing_values[n11]
            )
        else:
            # 三角形 (n00, n11, n01)
            w0 = 1.0 - s
            w1 = s + t - 1.0
            w2 = 1.0 - t
            return self._clamp(
                w0 * self.spacing_values[n00]
                + w1 * self.spacing_values[n11]
                + w2 * self.spacing_values[n01]
            )

    def _clamp(self, value: float) -> float:
        return max(self.min_spacing, min(self.max_spacing, value))

    def enforce_gradient(self, max_gradient: float = 1.3):
        """
        梯度限制：确保相邻节点尺寸比不超过 max_gradient。

        参照 Octree.enforce_gradient() 设计。

        Args:
            max_gradient: 最大尺寸比（默认 1.3）
        """
        if self.triangles is None or self.spacing_values is None:
            return

        # 构建共享边
        edge_dict = {}
        for tri in self.triangles:
            for k in range(3):
                i0 = tri[k]
                i1 = tri[(k + 1) % 3]
                edge = (min(i0, i1), max(i0, i1))
                if edge not in edge_dict:
                    edge_dict[edge] = True

        # 迭代限制
        for _ in range(100):
            changed = False
            for (i0, i1) in edge_dict:
                s0 = self.spacing_values[i0]
                s1 = self.spacing_values[i1]
                if s0 < 1e-30 or s1 < 1e-30:
                    continue
                ratio = s0 / s1
                if ratio > max_gradient:
                    self.spacing_values[i0] = s1 * max_gradient
                    changed = True
                elif ratio < 1.0 / max_gradient:
                    self.spacing_values[i1] = s0 * max_gradient
                    changed = True
            if not changed:
                break

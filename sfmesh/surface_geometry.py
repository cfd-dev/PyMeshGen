"""
曲面几何操作模块

提供曲面投影、法向计算、曲率计算等几何操作
基于 OpenCASCADE (pythonocc-core)
"""
import numpy as np
from typing import Tuple, Optional, Any, List

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fileIO.occ_loader import ensure_occ_loaded
ensure_occ_loaded()

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Pnt2d
from OCC.Core.Geom import Geom_Surface
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface


class SurfaceGeometry:
    """
    曲面几何操作类

    封装 OpenCASCADE 的曲面几何操作，提供：
    - 点投影到曲面
    - 曲面法向计算
    - 曲面曲率计算
    - 参数域与三维空间转换
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        初始化曲面几何操作器

        Args:
            tolerance: 几何计算容差
        """
        self.tolerance = tolerance

    def get_surface_from_face(self, face: TopoDS_Face) -> Geom_Surface:
        """
        从 TopoDS_Face 获取几何曲面

        Args:
            face: OCC 拓扑面

        Returns:
            Geom_Surface: 几何曲面
        """
        surface_handle = BRep_Tool.Surface(face)
        return surface_handle

    def project_point_to_surface(
        self,
        point: Tuple[float, float, float],
        surface: TopoDS_Face,
        max_distance: float = 1000.0
    ) -> Tuple[float, float]:
        """
        将三维点投影到曲面上，返回参数坐标

        Args:
            point: 三维点坐标
            surface: OCC 拓扑面
            max_distance: 最大投影距离

        Returns:
            参数坐标
        """
        pnt = gp_Pnt(point[0], point[1], point[2])

        geom_surface = self.get_surface_from_face(surface)
        projector = GeomAPI_ProjectPointOnSurf(pnt, geom_surface)

        if projector.NbPoints() == 0:
            return self._estimate_uv(point, surface)

        u, v = projector.Parameters(1)
        return (u, v)

    def _estimate_uv(
        self,
        point: Tuple[float, float, float],
        surface: TopoDS_Face
    ) -> Tuple[float, float]:
        """
        估计点的参数坐标（当投影失败时）

        Args:
            point: 三维点坐标
            surface: OCC 拓扑面

        Returns:
            估计的参数坐标
        """
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        adaptor = BRepAdaptor_Surface(surface)
        u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()

        return ((u_min + u_max) / 2, (v_min + v_max) / 2)

    def evaluate_point(
        self,
        u: float,
        v: float,
        surface: TopoDS_Face
    ) -> Tuple[float, float, float]:
        """
        计算曲面在参数 处的三维坐标

        Args:
            u, v: 参数坐标
            surface: OCC 拓扑面

        Returns:
            三维坐标
        """
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        adaptor = BRepAdaptor_Surface(surface)
        pnt = adaptor.Value(u, v)
        return (pnt.X(), pnt.Y(), pnt.Z())

    def get_surface_normal(
        self,
        u: float,
        v: float,
        surface: TopoDS_Face
    ) -> Tuple[float, float, float]:
        """
        获取曲面在参数 处的法向量

        Args:
            u, v: 参数坐标
            surface: OCC 拓扑面

        Returns:
            法向量
        """
        geom_surface = self.get_surface_from_face(surface)
        props = GeomLProp_SLProps(geom_surface, u, v, 1, self.tolerance)

        if props.IsNormalDefined():
            normal = props.Normal()
            return (normal.X(), normal.Y(), normal.Z())
        else:
            return (0.0, 0.0, 1.0)

    def get_surface_curvature(
        self,
        u: float,
        v: float,
        surface: TopoDS_Face
    ) -> Tuple[float, float, float]:
        """
        获取曲面在参数 处的曲率信息

        Args:
            u, v: 参数坐标
            surface: OCC 拓扑面

        Returns:
            (平均曲率，高斯曲率，最大主曲率)
        """
        geom_surface = self.get_surface_from_face(surface)
        props = GeomLProp_SLProps(geom_surface, u, v, 2, self.tolerance)

        mean_curvature = 0.0
        gauss_curvature = 0.0
        max_curvature = 0.0

        try:
            if props.IsCurvatureDefined():
                mean_curvature = props.MeanCurvature()
                gauss_curvature = props.GaussianCurvature()
                k1 = props.MaxCurvature()
                k2 = props.MinCurvature()
                max_curvature = max(abs(k1), abs(k2))
        except Exception:
            pass

        return (mean_curvature, gauss_curvature, max_curvature)

    def get_surface_derivatives(
        self,
        u: float,
        v: float,
        surface: TopoDS_Face
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取曲面在参数 处的一阶导数（切向量）

        Args:
            u, v: 参数坐标
            surface: OCC 拓扑面

        Returns:
            (du 方向切向量，dv 方向切向量)
        """
        geom_surface = self.get_surface_from_face(surface)
        props = GeomLProp_SLProps(geom_surface, u, v, 1, self.tolerance)

        du = np.array([1.0, 0.0, 0.0])
        dv = np.array([0.0, 1.0, 0.0])

        try:
            du_vec = props.D1U()
            du = np.array([du_vec.X(), du_vec.Y(), du_vec.Z()])

            dv_vec = props.D1V()
            dv = np.array([dv_vec.X(), dv_vec.Y(), dv_vec.Z()])
        except Exception:
            pass

        return (du, dv)

    def compute_ideal_point_on_surface(
        self,
        start_point: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        distance: float,
        surface: TopoDS_Face,
        max_iterations: int = 10
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float]]:
        """
        在曲面上计算理想点

        算法步骤:
        1. 从起点沿方向向量前进指定距离，得到初始理想点（切平面内）
        2. 将初始理想点投影到目标曲面上
        3. 迭代投影直到收敛（处理曲面曲率导致的投影偏差）

        Args:
            start_point: 起点坐标（通常为阵面中点）
            direction: 推进方向（切平面内垂直于阵面的单位向量）
            distance: 推进距离
            surface: 目标曲面
            max_iterations: 最大迭代次数

        Returns:
            (理想点三维坐标, 参数坐标)
        """
        # 步骤 1: 沿切平面方向前进
        initial = np.array(start_point) + distance * np.array(direction)

        # 步骤 2: 投影到曲面
        uv = self.project_point_to_surface(tuple(initial), surface)
        point = np.array(self.evaluate_point(uv[0], uv[1], surface))

        # 步骤 3: 迭代投影直到收敛
        for _ in range(max_iterations):
            new_uv = self.project_point_to_surface(tuple(point), surface)
            new_point = np.array(self.evaluate_point(new_uv[0], new_uv[1], surface))
            if np.linalg.norm(new_point - point) < self.tolerance:
                return (tuple(new_point), new_uv)
            point, uv = new_point, new_uv

        return (tuple(point), uv)

    def get_surface_bounds(
        self,
        surface: TopoDS_Face
    ) -> Tuple[float, float, float, float]:
        """
        获取曲面的参数范围

        Args:
            surface: OCC 拓扑面

        Returns:
            (u_min, u_max, v_min, v_max)
        """
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        adaptor = BRepAdaptor_Surface(surface)
        return (
            adaptor.FirstUParameter(),
            adaptor.LastUParameter(),
            adaptor.FirstVParameter(),
            adaptor.LastVParameter()
        )

    def is_point_on_surface(
        self,
        point: Tuple[float, float, float],
        surface: TopoDS_Face,
        tolerance: float = None
    ) -> bool:
        """
        检查点是否在曲面上

        Args:
            point: 三维点坐标
            surface: OCC 拓扑面
            tolerance: 容差

        Returns:
            是否在曲面上
        """
        if tolerance is None:
            tolerance = self.tolerance

        pnt = gp_Pnt(point[0], point[1], point[2])

        geom_surface = self.get_surface_from_face(surface)
        projector = GeomAPI_ProjectPointOnSurf(pnt, geom_surface)

        if projector.NbPoints() == 0:
            return False
        distance = projector.Distance(1)
        return distance < tolerance

    def compute_geodesic_distance(
        self,
        point1: Tuple[float, float, float],
        point2: Tuple[float, float, float],
        surface: TopoDS_Face
    ) -> float:
        """
        计算曲面上两点间的测地距离（近似）

        使用直线距离作为近似，对于平坦曲面足够精确

        Args:
            point1, point2: 两点的三维坐标
            surface: OCC 拓扑面

        Returns:
            测地距离
        """
        uv1 = self.project_point_to_surface(point1, surface)
        uv2 = self.project_point_to_surface(point2, surface)

        p1_3d = self.evaluate_point(uv1[0], uv1[1], surface)
        p2_3d = self.evaluate_point(uv2[0], uv2[1], surface)

        return np.linalg.norm(np.array(p2_3d) - np.array(p1_3d))

    def get_surface_area(self, surface: TopoDS_Face, num_samples: int = 100) -> float:
        """
        计算曲面面积（数值积分近似）

        Args:
            surface: OCC 拓扑面
            num_samples: 采样点数

        Returns:
            曲面面积
        """
        bounds = self.get_surface_bounds(surface)
        u_min, u_max, v_min, v_max = bounds

        du = (u_max - u_min) / num_samples
        dv = (v_max - v_min) / num_samples

        total_area = 0.0

        for i in range(num_samples):
            for j in range(num_samples):
                u = u_min + (i + 0.5) * du
                v = v_min + (j + 0.5) * dv

                du_vec, dv_vec = self.get_surface_derivatives(u, v, surface)

                cross = np.cross(du_vec, dv_vec)
                area_element = np.linalg.norm(cross)

                total_area += area_element * du * dv

        return total_area

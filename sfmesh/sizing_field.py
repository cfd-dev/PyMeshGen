"""
曲面尺寸场模块

提供曲面网格生成的尺寸控制功能
支持均匀尺寸场和基于曲率的自适应尺寸场
"""
import numpy as np
from typing import Tuple, Optional, Any, Dict, List
from OCC.Core.TopoDS import TopoDS_Face

from .surface_geometry import SurfaceGeometry


class SurfaceSizingField:
    """
    曲面尺寸场
    
    控制曲面网格的单元尺寸
    支持：
    - 均匀尺寸场
    - 基于曲率的自适应尺寸场
    - 用户自定义尺寸场
    """
    
    def __init__(
        self,
        global_spacing: float = 1.0,
        min_spacing: float = 0.01,
        max_spacing: float = 100.0,
        curvature_adaptation: bool = True,
        curvature_factor: float = 0.2,
        geometry_handler: SurfaceGeometry = None,
        boundary_decay: float = 1.2,
        boundary_adaptation: bool = True,
    ):
        """
        初始化曲面尺寸场

        Args:
            global_spacing: 全局网格尺寸
            min_spacing: 最小网格尺寸
            max_spacing: 最大网格尺寸
            curvature_adaptation: 是否启用曲率自适应
            curvature_factor: 曲率自适应系数 (0~1)
            geometry_handler: 曲面几何操作对象
            boundary_decay: 边界尺寸衰减系数
            boundary_adaptation: 是否启用边界驱动尺寸场
        """
        self.global_spacing = global_spacing
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.curvature_adaptation = curvature_adaptation
        self.curvature_factor = curvature_factor
        self.geometry_handler = geometry_handler or SurfaceGeometry()
        self.boundary_decay = boundary_decay
        self.boundary_adaptation = boundary_adaptation

        self._boundary_rtree = None
        self._boundary_front_dict = None
        self._boundary_registered = False
    
    def spacing_at(
        self,
        point: Tuple[float, float, float],
        surface: TopoDS_Face = None,
        uv: Tuple[float, float] = None
    ) -> float:
        """
        获取指定位置的网格尺寸

        Args:
            point: 三维坐标
            surface: 所属曲面
            uv: 参数坐标

        Returns:
            网格尺寸
        """
        base_spacing = self.global_spacing

        # 边界驱动
        if self.boundary_adaptation and self._boundary_registered:
            boundary_spacing = self._compute_boundary_spacing(point)
            base_spacing = min(base_spacing, boundary_spacing)

        # 曲率驱动
        if self.curvature_adaptation and surface is not None:
            if uv is None:
                uv = self.geometry_handler.project_point_to_surface(point, surface)
            curvature_spacing = self._compute_curvature_based_spacing(uv[0], uv[1], surface)
            base_spacing = min(base_spacing, curvature_spacing)

        return np.clip(base_spacing, self.min_spacing, self.max_spacing)
    
    def _compute_curvature_based_spacing(
        self,
        u: float,
        v: float,
        surface: TopoDS_Face
    ) -> float:
        """
        基于曲率计算网格尺寸
        
        Args:
            u, v: 参数坐标
            surface: 曲面
        
        Returns:
            基于曲率的网格尺寸
        """
        mean_curv, gauss_curv, max_curv = self.geometry_handler.get_surface_curvature(
            u, v, surface
        )
        
        if max_curv < 1e-12:
            return self.max_spacing
        
        radius = 1.0 / max_curv
        
        spacing = self.curvature_factor * radius
        
        return np.clip(spacing, self.min_spacing, self.max_spacing)
    
    def register_boundary_fronts(self, fronts):
        """
        注册边界阵面，构建 RTree 空间索引用于边界驱动尺寸场。

        Args:
            fronts: SurfaceFront 列表
        """
        try:
            from data_structure.rtree_space import build_space_index_3d_with_RTree
        except ImportError:
            return

        front_list = list(fronts)
        if not front_list:
            return

        _, rtree = build_space_index_3d_with_RTree(front_list)
        self._boundary_rtree = rtree
        self._boundary_front_dict = {id(f): f for f in front_list}
        self._boundary_registered = True

    def _compute_boundary_spacing(self, point: Tuple[float, float, float]) -> float:
        """
        基于边界阵面计算点处的网格尺寸（指数衰减公式，同 QuadtreeSizing）。

        Args:
            point: 三维坐标

        Returns:
            边界驱动的网格尺寸
        """
        if not self._boundary_registered or self._boundary_rtree is None:
            return self.global_spacing

        search_radius = 6.0 * self.global_spacing
        px, py, pz = point
        bbox = (px - search_radius, py - search_radius, pz - search_radius,
                px + search_radius, py + search_radius, pz + search_radius)

        candidate_ids = list(self._boundary_rtree.intersection(bbox))
        if not candidate_ids:
            return self.global_spacing

        p = np.array(point)
        min_spacing = self.global_spacing
        decay = self.boundary_decay

        for fid in candidate_ids:
            front = self._boundary_front_dict.get(fid)
            if front is None:
                continue

            p0 = np.array(front.node_elems[0].coords)
            p1 = np.array(front.node_elems[1].coords)
            source_size = np.linalg.norm(p1 - p0)
            if source_size < 1e-12:
                continue

            dist = self._point_to_segment_distance_3d(p, p0, p1)
            exponent = 0.5 * dist * (decay - 1.0) / source_size
            exponent = min(exponent, 50.0)
            sp = source_size * np.exp(exponent)
            if sp < min_spacing:
                min_spacing = sp

        return min(min_spacing, self.global_spacing)

    @staticmethod
    def _point_to_segment_distance_3d(
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
    ) -> float:
        """计算 3D 点到线段的最短距离"""
        seg_vec = seg_end - seg_start
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-12:
            return np.linalg.norm(point - seg_start)
        seg_unit = seg_vec / seg_len
        t = np.dot(point - seg_start, seg_unit)
        t = np.clip(t, 0, seg_len)
        closest = seg_start + t * seg_unit
        return np.linalg.norm(point - closest)

    def compute_front_spacing(
        self,
        front,
        surface: TopoDS_Face
    ) -> float:
        """
        计算阵面处的网格尺寸
        
        Args:
            front: 曲面阵面对象
            surface: 曲面
        
        Returns:
            阵面处的网格尺寸
        """
        center = front.center
        return self.spacing_at(center, surface)
    
    def compute_ideal_point_distance(
        self,
        front,
        surface: TopoDS_Face,
    ) -> float:
        """
        计算理想点的推进距离（等于局部网格尺寸）

        Args:
            front: 曲面阵面对象
            surface: 曲面

        Returns:
            推进距离
        """
        return self.compute_front_spacing(front, surface)


class AdaptiveSizingField(SurfaceSizingField):
    """
    自适应曲面尺寸场
    
    支持更复杂的尺寸控制策略
    """
    
    def __init__(
        self,
        global_spacing: float = 1.0,
        min_spacing: float = 0.01,
        max_spacing: float = 100.0,
        curvature_adaptation: bool = True,
        curvature_factor: float = 0.2,
        proximity_adaptation: bool = True,
        proximity_factor: float = 0.5,
        geometry_handler: SurfaceGeometry = None
    ):
        """
        初始化自适应尺寸场
        
        Args:
            global_spacing: 全局网格尺寸
            min_spacing: 最小网格尺寸
            max_spacing: 最大网格尺寸
            curvature_adaptation: 是否启用曲率自适应
            curvature_factor: 曲率自适应系数
            proximity_adaptation: 是否启用邻近自适应
            proximity_factor: 邻近自适应系数
            geometry_handler: 曲面几何操作对象
        """
        super().__init__(
            global_spacing=global_spacing,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            curvature_adaptation=curvature_adaptation,
            curvature_factor=curvature_factor,
            geometry_handler=geometry_handler
        )
        
        self.proximity_adaptation = proximity_adaptation
        self.proximity_factor = proximity_factor
        
        self._feature_points: List[Tuple[float, float, float]] = []
        self._feature_lines: List[List[Tuple[float, float, float]]] = []
    
    def add_feature_point(
        self,
        point: Tuple[float, float, float],
        refinement_radius: float,
        refinement_ratio: float = 0.5
    ):
        """
        添加特征点（如角点、尖点等）
        
        Args:
            point: 特征点坐标
            refinement_radius: 细化半径
            refinement_ratio: 细化比例
        """
        self._feature_points.append((point, refinement_radius, refinement_ratio))
    
    def add_feature_line(
        self,
        points: List[Tuple[float, float, float]],
        refinement_radius: float,
        refinement_ratio: float = 0.5
    ):
        """
        添加特征线（如棱线、脊线等）
        
        Args:
            points: 特征线上的点序列
            refinement_radius: 细化半径
            refinement_ratio: 细化比例
        """
        self._feature_lines.append((points, refinement_radius, refinement_ratio))
    
    def spacing_at(
        self,
        point: Tuple[float, float, float],
        surface: TopoDS_Face = None,
        uv: Tuple[float, float] = None
    ) -> float:
        """
        获取指定位置的网格尺寸（考虑特征）
        
        Args:
            point: 三维坐标
            surface: 所属曲面
            uv: 参数坐标
        
        Returns:
            网格尺寸
        """
        base_spacing = super().spacing_at(point, surface, uv)
        
        if self.proximity_adaptation:
            feature_spacing = self._compute_feature_spacing(point)
            base_spacing = min(base_spacing, feature_spacing)
        
        return base_spacing
    
    def _compute_feature_spacing(
        self,
        point: Tuple[float, float, float]
    ) -> float:
        """
        计算基于特征的网格尺寸
        
        Args:
            point: 三维坐标
        
        Returns:
            基于特征的网格尺寸
        """
        min_spacing = self.max_spacing
        point_arr = np.array(point)
        
        for feat_point, radius, ratio in self._feature_points:
            dist = np.linalg.norm(point_arr - np.array(feat_point))
            if dist < radius:
                spacing = self.global_spacing * ratio * (dist / radius + 0.1)
                min_spacing = min(min_spacing, spacing)
        
        for feat_line, radius, ratio in self._feature_lines:
            for i in range(len(feat_line) - 1):
                p1 = np.array(feat_line[i])
                p2 = np.array(feat_line[i + 1])
                
                dist = self._point_to_segment_distance(point_arr, p1, p2)
                
                if dist < radius:
                    spacing = self.global_spacing * ratio * (dist / radius + 0.1)
                    min_spacing = min(min_spacing, spacing)
        
        return np.clip(min_spacing, self.min_spacing, self.max_spacing)
    
    def _point_to_segment_distance(
        self,
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray
    ) -> float:
        """
        计算点到线段的最短距离
        
        Args:
            point: 点坐标
            seg_start: 线段起点
            seg_end: 线段终点
        
        Returns:
            最短距离
        """
        seg_vec = seg_end - seg_start
        seg_len = np.linalg.norm(seg_vec)
        
        if seg_len < 1e-12:
            return np.linalg.norm(point - seg_start)
        
        seg_unit = seg_vec / seg_len
        point_vec = point - seg_start
        
        t = np.dot(point_vec, seg_unit)
        t = np.clip(t, 0, seg_len)
        
        closest = seg_start + t * seg_unit
        return np.linalg.norm(point - closest)

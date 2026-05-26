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
        geometry_handler: SurfaceGeometry = None
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
        """
        self.global_spacing = global_spacing
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.curvature_adaptation = curvature_adaptation
        self.curvature_factor = curvature_factor
        self.geometry_handler = geometry_handler or SurfaceGeometry()
        
        self._local_spacing: Dict[int, float] = {}
        self._surface_curvature_cache: Dict[int, Tuple[float, float, float]] = {}
    
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
    
    def set_local_spacing(
        self,
        point: Tuple[float, float, float],
        spacing: float
    ):
        """
        设置局部网格尺寸
        
        Args:
            point: 三维坐标
            spacing: 网格尺寸
        """
        point_key = hash(tuple(f"{c:.6f}" for c in point))
        self._local_spacing[point_key] = spacing
    
    def get_gradient_spacing(
        self,
        point: Tuple[float, float, float],
        surface: TopoDS_Face,
        gradient_factor: float = 0.3
    ) -> float:
        """
        考虑尺寸梯度的网格尺寸
        
        Args:
            point: 三维坐标
            surface: 曲面
            gradient_factor: 梯度限制因子
        
        Returns:
            考虑梯度限制的网格尺寸
        """
        base_spacing = self.spacing_at(point, surface)
        
        uv = self.geometry_handler.project_point_to_surface(point, surface)
        
        delta = base_spacing * 0.1
        du, dv = self.geometry_handler.get_surface_derivatives(uv[0], uv[1], surface)
        
        du_norm = np.linalg.norm(du)
        dv_norm = np.linalg.norm(dv)
        
        if du_norm > 1e-12:
            neighbor_u = self.geometry_handler.evaluate_point(
                uv[0] + delta / du_norm, uv[1], surface
            )
            spacing_u = self.spacing_at(neighbor_u, surface)
            base_spacing = min(base_spacing, spacing_u * (1 + gradient_factor))
        
        if dv_norm > 1e-12:
            neighbor_v = self.geometry_handler.evaluate_point(
                uv[0], uv[1] + delta / dv_norm, surface
            )
            spacing_v = self.spacing_at(neighbor_v, surface)
            base_spacing = min(base_spacing, spacing_v * (1 + gradient_factor))
        
        return base_spacing
    
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

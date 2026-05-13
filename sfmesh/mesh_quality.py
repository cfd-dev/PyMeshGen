"""
曲面网格质量评估模块

提供曲面三角形网格的质量评估功能
"""
import numpy as np
from typing import Tuple, List, Dict, Any

from .surface_front import SurfaceTriangle, NodeElement3D


class SurfaceMeshQuality:
    """
    曲面网格质量评估类
    
    提供多种质量指标：
    - 形状质量（Aspect Ratio）
    - 面积质量
    - 角度质量
    - 翘曲度（Warpage）
    - 雅可比质量
    """
    
    @staticmethod
    def triangle_quality(triangle: SurfaceTriangle) -> float:
        """
        计算三角形质量（形状因子）
        
        使用面积与边长平方和的比值
        质量范围: 0 ~ 1，1 表示等边三角形
        
        Args:
            triangle: 曲面三角形单元
        
        Returns:
            质量值
        """
        return triangle.quality
    
    @staticmethod
    def triangle_aspect_ratio(triangle: SurfaceTriangle) -> float:
        """
        计算三角形长宽比
        
        Args:
            triangle: 曲面三角形单元
        
        Returns:
            长宽比（越大越差）
        """
        p1 = np.array(triangle.nodes[0].coords)
        p2 = np.array(triangle.nodes[1].coords)
        p3 = np.array(triangle.nodes[2].coords)
        
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        
        max_edge = max(a, b, c)
        
        s = (a + b + c) / 2.0
        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
        
        if area < 1e-12:
            return float('inf')
        
        return max_edge / (2.0 * np.sqrt(3) * area / max_edge)
    
    @staticmethod
    def triangle_min_angle(triangle: SurfaceTriangle) -> float:
        """
        计算三角形最小内角（度）
        
        Args:
            triangle: 曲面三角形单元
        
        Returns:
            最小内角（度）
        """
        p1 = np.array(triangle.nodes[0].coords)
        p2 = np.array(triangle.nodes[1].coords)
        p3 = np.array(triangle.nodes[2].coords)
        
        def angle_at_vertex(v1, vertex, v2):
            a = v1 - vertex
            b = v2 - vertex
            
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a < 1e-12 or norm_b < 1e-12:
                return 0.0
            
            cos_angle = np.dot(a, b) / (norm_a * norm_b)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            return np.degrees(np.arccos(cos_angle))
        
        angle1 = angle_at_vertex(p2, p1, p3)
        angle2 = angle_at_vertex(p1, p2, p3)
        angle3 = angle_at_vertex(p1, p3, p2)
        
        return min(angle1, angle2, angle3)
    
    @staticmethod
    def triangle_max_angle(triangle: SurfaceTriangle) -> float:
        """
        计算三角形最大内角（度）
        
        Args:
            triangle: 曲面三角形单元
        
        Returns:
            最大内角（度）
        """
        p1 = np.array(triangle.nodes[0].coords)
        p2 = np.array(triangle.nodes[1].coords)
        p3 = np.array(triangle.nodes[2].coords)
        
        def angle_at_vertex(v1, vertex, v2):
            a = v1 - vertex
            b = v2 - vertex
            
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a < 1e-12 or norm_b < 1e-12:
                return 180.0
            
            cos_angle = np.dot(a, b) / (norm_a * norm_b)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            return np.degrees(np.arccos(cos_angle))
        
        angle1 = angle_at_vertex(p2, p1, p3)
        angle2 = angle_at_vertex(p1, p2, p3)
        angle3 = angle_at_vertex(p1, p3, p2)
        
        return max(angle1, angle2, angle3)
    
    @staticmethod
    def triangle_warpage(triangle: SurfaceTriangle) -> float:
        """
        计算三角形翘曲度（与最佳平面的偏差）
        
        对于平面三角形，翘曲度为0
        
        Args:
            triangle: 曲面三角形单元
        
        Returns:
            翘曲度
        """
        return 0.0
    
    @staticmethod
    def triangle_jacobian(
        triangle: SurfaceTriangle,
        uv_coords: List[Tuple[float, float]] = None
    ) -> float:
        """
        计算三角形雅可比质量
        
        Args:
            triangle: 曲面三角形单元
            uv_coords: 参数坐标（可选）
        
        Returns:
            雅可比质量
        """
        p1 = np.array(triangle.nodes[0].coords)
        p2 = np.array(triangle.nodes[1].coords)
        p3 = np.array(triangle.nodes[2].coords)
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        cross = np.cross(v1, v2)
        jacobian = np.linalg.norm(cross)
        
        return jacobian
    
    @staticmethod
    def evaluate_triangle(
        triangle: SurfaceTriangle,
        quality_threshold: float = 0.3,
        min_angle_threshold: float = 15.0,
        max_angle_threshold: float = 150.0
    ) -> Dict[str, Any]:
        """
        全面评估三角形质量
        
        Args:
            triangle: 曲面三角形单元
            quality_threshold: 质量阈值
            min_angle_threshold: 最小角度阈值
            max_angle_threshold: 最大角度阈值
        
        Returns:
            评估结果字典
        """
        quality = SurfaceMeshQuality.triangle_quality(triangle)
        aspect_ratio = SurfaceMeshQuality.triangle_aspect_ratio(triangle)
        min_angle = SurfaceMeshQuality.triangle_min_angle(triangle)
        max_angle = SurfaceMeshQuality.triangle_max_angle(triangle)
        jacobian = SurfaceMeshQuality.triangle_jacobian(triangle)
        
        is_good = (
            quality >= quality_threshold and
            min_angle >= min_angle_threshold and
            max_angle <= max_angle_threshold
        )
        
        return {
            'quality': quality,
            'aspect_ratio': aspect_ratio,
            'min_angle': min_angle,
            'max_angle': max_angle,
            'jacobian': jacobian,
            'is_good': is_good,
            'area': triangle.area
        }

    @staticmethod
    def _safe_histogram(data, bins=10):
        """安全的直方图计算，处理数据范围为零的情况"""
        try:
            return np.histogram(data, bins=bins)[0].tolist()
        except ValueError:
            return [len(data)]

    @staticmethod
    def evaluate_mesh(
        triangles: List[SurfaceTriangle],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        评估整个网格的质量
        
        Args:
            triangles: 三角形列表
            verbose: 是否输出详细信息
        
        Returns:
            评估结果字典
        """
        if not triangles:
            return {'error': 'Empty mesh'}
        
        qualities = []
        aspect_ratios = []
        min_angles = []
        max_angles = []
        areas = []
        
        for tri in triangles:
            qualities.append(SurfaceMeshQuality.triangle_quality(tri))
            aspect_ratios.append(SurfaceMeshQuality.triangle_aspect_ratio(tri))
            min_angles.append(SurfaceMeshQuality.triangle_min_angle(tri))
            max_angles.append(SurfaceMeshQuality.triangle_max_angle(tri))
            areas.append(tri.area)
        
        result = {
            'num_triangles': len(triangles),
            'quality_mean': np.mean(qualities),
            'quality_min': np.min(qualities),
            'quality_max': np.max(qualities),
            'quality_std': np.std(qualities),
            'aspect_ratio_mean': np.mean(aspect_ratios),
            'aspect_ratio_max': np.max(aspect_ratios),
            'min_angle_mean': np.mean(min_angles),
            'min_angle_min': np.min(min_angles),
            'max_angle_mean': np.mean(max_angles),
            'max_angle_max': np.max(max_angles),
            'total_area': np.sum(areas),
            'area_mean': np.mean(areas),
            'quality_histogram': SurfaceMeshQuality._safe_histogram(qualities, bins=10)
        }
        
        poor_quality_count = sum(1 for q in qualities if q < 0.3)
        result['poor_quality_count'] = poor_quality_count
        result['poor_quality_ratio'] = poor_quality_count / len(triangles)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"曲面网格质量评估报告")
            print(f"{'='*60}")
            print(f"三角形数量: {result['num_triangles']}")
            print(f"总表面积: {result['total_area']:.6f}")
            print(f"平均面积: {result['area_mean']:.6f}")
            print(f"--- 质量指标 ---")
            print(f"平均质量: {result['quality_mean']:.4f}")
            print(f"最小质量: {result['quality_min']:.4f}")
            print(f"最大质量: {result['quality_max']:.4f}")
            print(f"质量标准差: {result['quality_std']:.4f}")
            print(f"--- 长宽比 ---")
            print(f"平均长宽比: {result['aspect_ratio_mean']:.4f}")
            print(f"最大长宽比: {result['aspect_ratio_max']:.4f}")
            print(f"--- 角度 ---")
            print(f"最小角度均值: {result['min_angle_mean']:.2f}°")
            print(f"最小角度极值: {result['min_angle_min']:.2f}°")
            print(f"最大角度均值: {result['max_angle_mean']:.2f}°")
            print(f"最大角度极值: {result['max_angle_max']:.2f}°")
            print(f"--- 质量分布 ---")
            print(f"低质量三角形数量: {result['poor_quality_count']}")
            print(f"低质量比例: {result['poor_quality_ratio']*100:.2f}%")
            print(f"{'='*60}\n")
        
        return result


def check_triangle_intersection(
    tri1: SurfaceTriangle,
    tri2: SurfaceTriangle,
    tolerance: float = 1e-10
) -> bool:
    """
    检查两个三角形是否相交
    
    Args:
        tri1, tri2: 两个三角形
        tolerance: 容差
    
    Returns:
        是否相交
    """
    p1 = np.array(tri1.nodes[0].coords)
    p2 = np.array(tri1.nodes[1].coords)
    p3 = np.array(tri1.nodes[2].coords)
    
    q1 = np.array(tri2.nodes[0].coords)
    q2 = np.array(tri2.nodes[1].coords)
    q3 = np.array(tri2.nodes[2].coords)
    
    n1 = np.cross(p2 - p1, p3 - p1)
    n2 = np.cross(q2 - q1, q3 - q1)
    
    if np.linalg.norm(n1) < tolerance or np.linalg.norm(n2) < tolerance:
        return False
    
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    direction = np.cross(n1, n2)
    if np.linalg.norm(direction) < tolerance:
        return False
    
    def signed_distance(point, plane_point, plane_normal):
        return np.dot(point - plane_point, plane_normal)
    
    d1_q1 = signed_distance(q1, p1, n1)
    d1_q2 = signed_distance(q2, p1, n1)
    d1_q3 = signed_distance(q3, p1, n1)
    
    if (d1_q1 > tolerance and d1_q2 > tolerance and d1_q3 > tolerance):
        return False
    if (d1_q1 < -tolerance and d1_q2 < -tolerance and d1_q3 < -tolerance):
        return False
    
    d2_p1 = signed_distance(p1, q1, n2)
    d2_p2 = signed_distance(p2, q1, n2)
    d2_p3 = signed_distance(p3, q1, n2)
    
    if (d2_p1 > tolerance and d2_p2 > tolerance and d2_p3 > tolerance):
        return False
    if (d2_p1 < -tolerance and d2_p2 < -tolerance and d2_p3 < -tolerance):
        return False
    
    return True


def check_edge_triangle_intersection(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    triangle: SurfaceTriangle,
    tolerance: float = 1e-10
) -> bool:
    """
    检查边与三角形是否相交
    
    Args:
        edge_start, edge_end: 边的端点
        triangle: 三角形
        tolerance: 容差
    
    Returns:
        是否相交
    """
    p1 = np.array(triangle.nodes[0].coords)
    p2 = np.array(triangle.nodes[1].coords)
    p3 = np.array(triangle.nodes[2].coords)
    
    normal = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(normal)
    if norm < tolerance:
        return False
    normal = normal / norm
    
    d1 = np.dot(edge_start - p1, normal)
    d2 = np.dot(edge_end - p1, normal)
    
    if abs(d1) < tolerance and abs(d2) < tolerance:
        return False
    
    if d1 * d2 > tolerance:
        return False
    
    t = d1 / (d1 - d2 + 1e-20)
    intersection = edge_start + t * (edge_end - edge_start)
    
    v0 = p3 - p1
    v1 = p2 - p1
    v2 = intersection - p1
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < tolerance:
        return False
    
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= -tolerance) and (v >= -tolerance) and (u + v <= 1 + tolerance)

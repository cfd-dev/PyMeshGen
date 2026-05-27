"""
曲面网格质量评估模块

提供曲面三角形网格的质量评估功能，支持批量向量化计算。
"""
import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from .surface_front import SurfaceTriangle, NodeElement3D
from .geom_utils import check_triangle_intersection, check_edge_triangle_intersection

logger = logging.getLogger(__name__)


class SurfaceMeshQuality:
    """
    曲面网格质量评估类

    提供多种质量指标：
    - 形状质量（Shape Quality）: 4√3·A / Σl²，等边三角形=1
    - 长宽比（Aspect Ratio）: 最长边 / (2√3·A/l_max)
    - 最小/最大内角
    - 翘曲度（Warpage）: 三角形平面与曲面法向的偏差
    - 归一化雅可比（Normalized Jacobian）
    """

    # ------------------------------------------------------------------
    # 单三角形指标
    # ------------------------------------------------------------------

    @staticmethod
    def triangle_quality(triangle: SurfaceTriangle) -> float:
        """
        计算三角形形状质量因子

        公式: 4√3 · Area / (a² + b² + c²)
        范围: [0, 1]，1 = 等边三角形

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

        定义: l_max² / (4√3 · Area)
        等边三角形 = 1.0，退化三角形 → ∞

        Args:
            triangle: 曲面三角形单元

        Returns:
            长宽比（≥1.0，越大越差）
        """
        p1 = np.array(triangle.nodes[0].coords)
        p2 = np.array(triangle.nodes[1].coords)
        p3 = np.array(triangle.nodes[2].coords)

        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)

        max_edge_sq = max(a * a, b * b, c * c)
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)

        if area_sq <= 1e-30:
            return float('inf')

        return max_edge_sq / (4.0 * np.sqrt(3) * np.sqrt(area_sq))

    @staticmethod
    def _compute_angles(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float]:
        """
        计算三角形三个内角（度），内部复用避免重复数组构建

        Returns:
            (angle_at_p1, angle_at_p2, angle_at_p3) in degrees
        """
        edges = np.array([p2 - p1, p3 - p2, p1 - p3])
        norms = np.linalg.norm(edges, axis=1)

        if np.any(norms < 1e-12):
            return (0.0, 0.0, 180.0)

        # 三个顶点的夹角分别对应 edge[2]&edge[0], edge[0]&edge[1], edge[1]&edge[2]
        cos_angles = np.array([
            np.dot(-edges[2], edges[0]) / (norms[2] * norms[0]),
            np.dot(-edges[0], edges[1]) / (norms[0] * norms[1]),
            np.dot(-edges[1], edges[2]) / (norms[1] * norms[2]),
        ])
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angles))
        return tuple(angles)

    @staticmethod
    def triangle_min_angle(triangle: SurfaceTriangle) -> float:
        """计算三角形最小内角（度）"""
        p1 = np.array(triangle.nodes[0].coords)
        p2 = np.array(triangle.nodes[1].coords)
        p3 = np.array(triangle.nodes[2].coords)
        return min(SurfaceMeshQuality._compute_angles(p1, p2, p3))

    @staticmethod
    def triangle_max_angle(triangle: SurfaceTriangle) -> float:
        """计算三角形最大内角（度）"""
        p1 = np.array(triangle.nodes[0].coords)
        p2 = np.array(triangle.nodes[1].coords)
        p3 = np.array(triangle.nodes[2].coords)
        return max(SurfaceMeshQuality._compute_angles(p1, p2, p3))

    @staticmethod
    def triangle_warpage(triangle: SurfaceTriangle) -> float:
        """
        计算三角形翘曲度

        定义: 三角形法向与三个顶点处曲面法向的平均偏差角（度）。
        对于纯平面三角形或完美贴合曲面的三角形，翘曲度为 0。

        Args:
            triangle: 曲面三角形单元

        Returns:
            翘曲度（度），0 = 完美贴合
        """
        p1 = np.array(triangle.nodes[0].coords)
        p2 = np.array(triangle.nodes[1].coords)
        p3 = np.array(triangle.nodes[2].coords)

        # 三角形面法向
        face_normal = np.cross(p2 - p1, p3 - p1)
        fn_norm = np.linalg.norm(face_normal)
        if fn_norm < 1e-24:
            return 0.0
        face_normal /= fn_norm

        # 收集可用的节点法向
        node_normals = []
        for node in triangle.nodes:
            n = getattr(node, 'normal', None)
            if n is not None:
                nn = np.array(n, dtype=np.float64)
                nn_norm = np.linalg.norm(nn)
                if nn_norm > 1e-12:
                    node_normals.append(nn / nn_norm)

        if not node_normals:
            return 0.0

        # 计算面法向与各节点法向的偏差角均值
        deviations = []
        for nn in node_normals:
            cos_val = np.clip(np.dot(face_normal, nn), -1.0, 1.0)
            deviations.append(np.degrees(np.arccos(abs(cos_val))))

        return float(np.mean(deviations))

    @staticmethod
    def triangle_jacobian(
        triangle: SurfaceTriangle,
        uv_coords: Optional[List[Tuple[float, float]]] = None
    ) -> float:
        """
        计算归一化雅可比质量

        定义: 2·Area / (l_max²)，映射到 [0, 1]
        等边三角形 ≈ 0.866，退化三角形 → 0
        相比原始叉积模长，此指标与网格尺寸无关，可跨尺度比较。

        Args:
            triangle: 曲面三角形单元
            uv_coords: 参数坐标（保留接口，当前未使用）

        Returns:
            归一化雅可比质量 [0, ~0.866]
        """
        p1 = np.array(triangle.nodes[0].coords)
        p2 = np.array(triangle.nodes[1].coords)
        p3 = np.array(triangle.nodes[2].coords)

        v1 = p2 - p1
        v2 = p3 - p1
        cross_norm = np.linalg.norm(np.cross(v1, v2))

        a_sq = np.dot(v1, v1)
        b_sq = np.dot(v2, v2)
        c_sq = np.dot(p3 - p2, p3 - p2)
        max_edge_sq = max(a_sq, b_sq, c_sq)

        if max_edge_sq < 1e-30:
            return 0.0

        return float(cross_norm / max_edge_sq)

    @staticmethod
    def evaluate_triangle(
        triangle: SurfaceTriangle,
        quality_threshold: float = 0.3,
        min_angle_threshold: float = 15.0,
        max_angle_threshold: float = 150.0
    ) -> Dict[str, Any]:
        """
        全面评估单个三角形质量

        Args:
            triangle: 曲面三角形单元
            quality_threshold: 形状质量阈值
            min_angle_threshold: 最小角度阈值（度）
            max_angle_threshold: 最大角度阈值（度）

        Returns:
            评估结果字典
        """
        quality = SurfaceMeshQuality.triangle_quality(triangle)
        aspect_ratio = SurfaceMeshQuality.triangle_aspect_ratio(triangle)
        min_angle = SurfaceMeshQuality.triangle_min_angle(triangle)
        max_angle = SurfaceMeshQuality.triangle_max_angle(triangle)
        jacobian = SurfaceMeshQuality.triangle_jacobian(triangle)
        warpage = SurfaceMeshQuality.triangle_warpage(triangle)

        is_good = (
            quality >= quality_threshold and
            min_angle >= min_angle_threshold and
            max_angle <= max_angle_threshold and
            warpage < 30.0
        )

        return {
            'quality': quality,
            'aspect_ratio': aspect_ratio,
            'min_angle': min_angle,
            'max_angle': max_angle,
            'jacobian': jacobian,
            'warpage': warpage,
            'is_good': is_good,
            'area': triangle.area,
        }

    # ------------------------------------------------------------------
    # 全网格批量评估
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_histogram(data: np.ndarray, bins: int = 10) -> List[int]:
        """安全的直方图计算，处理空数据、NaN、Inf 和零范围"""
        clean = data[np.isfinite(data)]
        if len(clean) == 0:
            return [0] * bins
        try:
            return np.histogram(clean, bins=bins)[0].tolist()
        except ValueError:
            return [len(clean)] + [0] * (bins - 1)

    @staticmethod
    def evaluate_mesh(
        triangles: List[SurfaceTriangle],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        评估整个网格的质量（批量优化版本）

        一次性提取所有三角形数据，避免逐三角形重复构建数组。

        Args:
            triangles: 三角形列表
            verbose: 是否通过 logging 输出详细报告

        Returns:
            评估结果字典
        """
        if not triangles:
            return {'error': 'Empty mesh', 'num_triangles': 0}

        n = len(triangles)

        # 批量提取坐标，形状 (N, 3, 3)
        coords = np.array(
            [[tri.nodes[k].coords for k in range(3)] for tri in triangles],
            dtype=np.float64
        )

        # 批量计算边长
        e0 = coords[:, 1] - coords[:, 0]  # p2 - p1
        e1 = coords[:, 2] - coords[:, 1]  # p3 - p2
        e2 = coords[:, 0] - coords[:, 2]  # p1 - p3

        len0 = np.linalg.norm(e0, axis=1)
        len1 = np.linalg.norm(e1, axis=1)
        len2 = np.linalg.norm(e2, axis=1)

        # 批量面积 (Heron)
        semi = (len0 + len1 + len2) / 2.0
        area_sq = semi * (semi - len0) * (semi - len1) * (semi - len2)
        area_sq = np.maximum(area_sq, 0.0)
        areas = np.sqrt(area_sq)

        # 批量形状质量: 4√3·A / Σl²
        sum_len_sq = len0**2 + len1**2 + len2**2
        qualities = np.where(
            sum_len_sq > 1e-30,
            4.0 * np.sqrt(3) * areas / sum_len_sq,
            0.0
        )
        qualities = np.clip(qualities, 0.0, 1.0)

        # 批量长宽比: l_max² / (4√3·A)
        max_edge_sq = np.maximum.reduce([len0**2, len1**2, len2**2])
        denom_ar = 4.0 * np.sqrt(3) * areas
        aspect_ratios = np.where(denom_ar > 1e-30, max_edge_sq / denom_ar, np.inf)

        # 批量角度
        cos_a0 = np.clip(
            np.sum((-e2) * e0, axis=1) / (len2 * len0 + 1e-30), -1.0, 1.0
        )
        cos_a1 = np.clip(
            np.sum((-e0) * e1, axis=1) / (len0 * len1 + 1e-30), -1.0, 1.0
        )
        cos_a2 = np.clip(
            np.sum((-e1) * e2, axis=1) / (len1 * len2 + 1e-30), -1.0, 1.0
        )
        all_angles = np.degrees(np.arccos(np.stack([cos_a0, cos_a1, cos_a2], axis=1)))
        min_angles = np.min(all_angles, axis=1)
        max_angles = np.max(all_angles, axis=1)

        # 统计汇总
        poor_mask = qualities < 0.3
        poor_count = int(np.sum(poor_mask))

        result: Dict[str, Any] = {
            'num_triangles': n,
            'quality_mean': float(np.mean(qualities)),
            'quality_min': float(np.min(qualities)),
            'quality_max': float(np.max(qualities)),
            'quality_std': float(np.std(qualities)),
            'aspect_ratio_mean': float(np.mean(aspect_ratios[np.isfinite(aspect_ratios)])),
            'aspect_ratio_max': float(np.max(aspect_ratios[np.isfinite(aspect_ratios)])) if np.any(np.isfinite(aspect_ratios)) else float('inf'),
            'min_angle_mean': float(np.mean(min_angles)),
            'min_angle_min': float(np.min(min_angles)),
            'max_angle_mean': float(np.mean(max_angles)),
            'max_angle_max': float(np.max(max_angles)),
            'total_area': float(np.sum(areas)),
            'area_mean': float(np.mean(areas)),
            'quality_histogram': SurfaceMeshQuality._safe_histogram(qualities, bins=10),
            'poor_quality_count': poor_count,
            'poor_quality_ratio': poor_count / n,
        }

        if verbose:
            lines = [
                "",
                "=" * 60,
                "曲面网格质量评估报告",
                "=" * 60,
                f"三角形数量: {result['num_triangles']}",
                f"总表面积: {result['total_area']:.6f}",
                f"平均面积: {result['area_mean']:.6f}",
                "--- 质量指标 ---",
                f"平均质量: {result['quality_mean']:.4f}",
                f"最小质量: {result['quality_min']:.4f}",
                f"最大质量: {result['quality_max']:.4f}",
                f"质量标准差: {result['quality_std']:.4f}",
                "--- 长宽比 ---",
                f"平均长宽比: {result['aspect_ratio_mean']:.4f}",
                f"最大长宽比: {result['aspect_ratio_max']:.4f}",
                "--- 角度 ---",
                f"最小角度均值: {result['min_angle_mean']:.2f}°",
                f"最小角度极值: {result['min_angle_min']:.2f}°",
                f"最大角度均值: {result['max_angle_mean']:.2f}°",
                f"最大角度极值: {result['max_angle_max']:.2f}°",
                "--- 质量分布 ---",
                f"低质量三角形数量: {result['poor_quality_count']}",
                f"低质量比例: {result['poor_quality_ratio'] * 100:.2f}%",
                "=" * 60,
                "",
            ]
            logger.info("\n".join(lines))

        return result
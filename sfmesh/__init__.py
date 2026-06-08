"""
sfmesh - Surface Mesh Generation Module

基于阵面推进法(Advancing Front Method)的三维曲面网格生成模块
支持从 IGES/STEP 几何模型生成三角形曲面网格
"""

from .surface_front import (
    SurfaceFront, NodeElement3D, SurfaceTriangle,
    create_initial_fronts_from_surface,
)
from .surface_geometry import SurfaceGeometry
from meshsize.surface_sizing import SurfaceSizingField
from .mesh_3d_afm import SurfaceMeshGenerator
from .mesh_parametric import generate_sphere_mesh, generate_ellipsoid_mesh
from .shape_generators import (
    generate_cube_mesh, generate_cylinder_mesh, generate_rectangle_mesh,
    generate_ellipsoid_mesh_2d_afm,
    generate_surface_mesh_from_shape, generate_surface_mesh_from_file,
    _export_combined_mesh,
    PrimitiveMeshResult,
)

# 向后兼容：SurfaceMeshQuality 已迁移到 utils.mesh_topology / utils.geom_toolkit
import numpy as _np
from utils.mesh_topology import evaluate_mesh_quality as _evaluate_mesh_quality
from utils.geom_toolkit import triangle_min_angle_3d, triangle_quality_from_coords

class SurfaceMeshQuality:
    """向后兼容包装，委托给 utils 模块"""
    @staticmethod
    def triangle_quality(triangle):
        return triangle.quality

    @staticmethod
    def triangle_aspect_ratio(triangle):
        p = [_np.array(n.coords) for n in triangle.nodes]
        a = _np.linalg.norm(p[1] - p[0])
        b = _np.linalg.norm(p[2] - p[1])
        c = _np.linalg.norm(p[0] - p[2])
        max_sq = max(a*a, b*b, c*c)
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 1e-30: return float('inf')
        return max_sq / (4.0 * _np.sqrt(3) * _np.sqrt(area_sq))

    @staticmethod
    def triangle_min_angle(triangle):
        p = [_np.array(n.coords) for n in triangle.nodes]
        return triangle_min_angle_3d(p[0], p[1], p[2])

    @staticmethod
    def triangle_max_angle(triangle):
        p = [_np.array(n.coords) for n in triangle.nodes]
        return max(SurfaceMeshQuality._compute_angles(p[0], p[1], p[2]))

    @staticmethod
    def triangle_jacobian(triangle, uv_coords=None):
        p = [_np.array(n.coords) for n in triangle.nodes]
        v1, v2 = p[1] - p[0], p[2] - p[0]
        cross_norm = _np.linalg.norm(_np.cross(v1, v2))
        max_sq = max(_np.dot(v1, v1), _np.dot(v2, v2), _np.dot(p[2]-p[1], p[2]-p[1]))
        if max_sq < 1e-30: return 0.0
        return float(cross_norm / max_sq)

    @staticmethod
    def triangle_warpage(triangle):
        p = [_np.array(n.coords) for n in triangle.nodes]
        face_n = _np.cross(p[1] - p[0], p[2] - p[0])
        fn = _np.linalg.norm(face_n)
        if fn < 1e-24: return 0.0
        face_n /= fn
        node_ns = []
        for n in triangle.nodes:
            nn = getattr(n, 'normal', None)
            if nn is not None:
                nv = _np.array(nn, dtype=_np.float64)
                nv_n = _np.linalg.norm(nv)
                if nv_n > 1e-12: node_ns.append(nv / nv_n)
        if not node_ns: return 0.0
        devs = [_np.degrees(_np.arccos(abs(_np.clip(_np.dot(face_n, nn_), -1.0, 1.0)))) for nn_ in node_ns]
        return float(_np.mean(devs))

    @staticmethod
    def evaluate_triangle(triangle, quality_threshold=0.3,
                          min_angle_threshold=15.0, max_angle_threshold=150.0):
        q = SurfaceMeshQuality.triangle_quality(triangle)
        ar = SurfaceMeshQuality.triangle_aspect_ratio(triangle)
        ma = SurfaceMeshQuality.triangle_min_angle(triangle)
        xa = SurfaceMeshQuality.triangle_max_angle(triangle)
        j = SurfaceMeshQuality.triangle_jacobian(triangle)
        w = SurfaceMeshQuality.triangle_warpage(triangle)
        is_good = (q >= quality_threshold and ma >= min_angle_threshold
                   and xa <= max_angle_threshold and w < 30.0)
        return {'quality': q, 'aspect_ratio': ar, 'min_angle': ma,
                'max_angle': xa, 'jacobian': j, 'warpage': w, 'is_good': is_good,
                'area': triangle.area}

    @staticmethod
    def evaluate_mesh(triangles, verbose=True):
        return _evaluate_mesh_quality(triangles, verbose=verbose)

    @staticmethod
    def _compute_angles(p1, p2, p3):
        edges = _np.array([p2 - p1, p3 - p2, p1 - p3])
        norms = _np.linalg.norm(edges, axis=1)
        if _np.any(norms < 1e-12): return (0.0, 0.0, 180.0)
        cos_a = _np.array([
            _np.dot(-edges[2], edges[0]) / (norms[2] * norms[0]),
            _np.dot(-edges[0], edges[1]) / (norms[0] * norms[1]),
            _np.dot(-edges[1], edges[2]) / (norms[1] * norms[2]),
        ])
        cos_a = _np.clip(cos_a, -1.0, 1.0)
        return tuple(_np.degrees(_np.arccos(cos_a)))

__all__ = [
    'SurfaceFront',
    'NodeElement3D',
    'SurfaceTriangle',
    'create_initial_fronts_from_surface',
    'SurfaceGeometry',
    'SurfaceSizingField',
    'SurfaceMeshQuality',
    'SurfaceMeshGenerator',
    'generate_cube_mesh',
    'generate_cylinder_mesh',
    'generate_rectangle_mesh',
    'generate_sphere_mesh',
    'generate_ellipsoid_mesh',
    'generate_ellipsoid_mesh_2d_afm',
    'PrimitiveMeshResult',
]

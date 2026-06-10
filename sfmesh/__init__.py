"""
sfmesh - Surface Mesh Generation Module

共享数据结构（SurfaceFront / NodeElement3D / SurfaceTriangle）、
Delaunay 曲面三角剖分和几何操作。

AFM（阵面推进法）算法在 sfmesh_aft 包中。
"""

from .surface_front import (
    SurfaceFront, NodeElement3D, SurfaceTriangle,
    create_initial_fronts_from_surface,
)

# Lazy imports — 避免导入 sfmesh 时强制加载 OCC
def __getattr__(name):
    if name == 'SurfaceGeometry':
        from .surface_geometry import SurfaceGeometry
        return SurfaceGeometry
    if name == 'SurfaceSizingField':
        from meshsize.surface_sizing import SurfaceSizingField
        return SurfaceSizingField
    if name == 'SurfaceMeshQuality':
        from optimize.mesh_quality import SurfaceMeshQuality
        return SurfaceMeshQuality
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'SurfaceFront',
    'NodeElement3D',
    'SurfaceTriangle',
    'create_initial_fronts_from_surface',
    'SurfaceGeometry',
    'SurfaceSizingField',
    'SurfaceMeshQuality',
]

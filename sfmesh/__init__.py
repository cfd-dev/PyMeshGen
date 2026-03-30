"""
sfmesh - Surface Mesh Generation Module

基于阵面推进法(Advancing Front Method)的三维曲面网格生成模块
支持从 IGES/STEP 几何模型生成三角形曲面网格
"""

from .surface_front import SurfaceFront, NodeElement3D, SurfaceTriangle
from .surface_geometry import SurfaceGeometry
from .sizing_field import SurfaceSizingField
from .mesh_quality import SurfaceMeshQuality
from .surface_mesh import SurfaceMeshGenerator

__all__ = [
    'SurfaceFront',
    'NodeElement3D',
    'SurfaceTriangle',
    'SurfaceGeometry',
    'SurfaceSizingField',
    'SurfaceMeshQuality',
    'SurfaceMeshGenerator',
]

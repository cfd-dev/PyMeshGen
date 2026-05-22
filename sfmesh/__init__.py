"""
sfmesh - Surface Mesh Generation Module

基于阵面推进法(Advancing Front Method)的三维曲面网格生成模块
支持从 IGES/STEP 几何模型生成三角形曲面网格
"""

from .surface_front import (
    SurfaceFront, NodeElement3D, SurfaceTriangle,
    discretize_shape_edges, create_fronts_from_line_mesh,
)
from .surface_geometry import SurfaceGeometry
from .sizing_field import SurfaceSizingField
from .mesh_quality import SurfaceMeshQuality
from .mesh_3d_afm import SurfaceMeshGenerator
from .mesh_parametric import generate_sphere_mesh, generate_ellipsoid_mesh
from .shape_generators import (
    generate_cube_mesh, generate_cylinder_mesh, generate_rectangle_mesh,
    generate_ellipsoid_mesh_2d_afm,
)
from .dispatcher import PrimitiveMeshResult

__all__ = [
    'SurfaceFront',
    'NodeElement3D',
    'SurfaceTriangle',
    'SurfaceGeometry',
    'SurfaceSizingField',
    'SurfaceMeshQuality',
    'SurfaceMeshGenerator',
    'discretize_shape_edges',
    'create_fronts_from_line_mesh',
    'generate_cube_mesh',
    'generate_cylinder_mesh',
    'generate_rectangle_mesh',
    'generate_sphere_mesh',
    'generate_ellipsoid_mesh',
    'generate_ellipsoid_mesh_2d_afm',
    'PrimitiveMeshResult',
]

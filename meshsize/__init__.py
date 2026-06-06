"""
meshsize subpackage for PyMeshGen

尺寸场模块，包含：
- QuadtreeSizing: 2D 四叉树尺寸场
- SizeField3D: 3D 八叉树尺寸场
- SurfaceSizingField: 曲面尺寸场（曲率/边界自适应）
- AdaptiveSizingField: 自适应曲面尺寸场（含特征点/线）
"""

# Import the main classes/functions for easy access
from .meshsize import QuadtreeSizing
from .size_field_3d import SizeField3D
from .size_sources import PointSource, BoxSource, SphereSource
from .octree import Octree
from .surface_sizing import SurfaceSizingField, AdaptiveSizingField, decay_spacing
from .surface_bg_mesh import SurfaceBackgroundMesh
from utils.geom_toolkit import point_to_segment_distance_3d

# Make them available at the package level
__all__ = [
    'QuadtreeSizing',
    'SizeField3D',
    'PointSource',
    'BoxSource',
    'SphereSource',
    'Octree',
    'SurfaceSizingField',
    'AdaptiveSizingField',
    'SurfaceBackgroundMesh',
    'point_to_segment_distance_3d',
    'decay_spacing',
]
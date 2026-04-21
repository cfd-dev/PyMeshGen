"""Compatibility exports for the historical ``delaunay.bw_core`` module.

The active Bowyer-Watson implementations now live in:

- ``delaunay.bw_core_stable`` for the standard generator
- ``delaunay.bw_core_gmsh`` for the mesh_type=4 / Gmsh-style generator
"""

from delaunay.bw_core_gmsh import GmshBowyerWatsonMeshGenerator
from delaunay.bw_core_stable import (
    BowyerWatsonMeshGenerator,
    Triangle,
    mtri3_to_triangle,
    triangle_to_mtri3,
)

__all__ = [
    "BowyerWatsonMeshGenerator",
    "GmshBowyerWatsonMeshGenerator",
    "Triangle",
    "mtri3_to_triangle",
    "triangle_to_mtri3",
]

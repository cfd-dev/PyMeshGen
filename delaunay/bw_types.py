"""Legacy compatibility wrapper for Gmsh Bowyer-Watson types."""

from delaunay.backup_old.bw_types import (
    MTri3,
    EdgeXFace,
    TriangleComparator,
    build_adjacency_from_triangles,
    collect_cavity_shell,
    compute_cavity_volume,
)

__all__ = [
    "MTri3",
    "EdgeXFace",
    "TriangleComparator",
    "build_adjacency_from_triangles",
    "collect_cavity_shell",
    "compute_cavity_volume",
]

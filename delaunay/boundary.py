"""Compatibility exports for the legacy ``delaunay.boundary`` module.

The historical boundary-recovery helpers are kept under ``delaunay.backup_old``
so the package root only exposes a thin compatibility surface.
"""

from delaunay.backup_old.boundary import (
    find_cavity_boundary,
    find_isolated_boundary_points,
    find_path_in_boundary,
    flip_edge,
    recover_edge_by_boundary_path,
    recover_edge_by_splitting,
    recover_edge_by_swaps,
    retriangulate_with_constraint,
    segment_intersects_triangle,
)

__all__ = [
    "find_cavity_boundary",
    "find_isolated_boundary_points",
    "find_path_in_boundary",
    "flip_edge",
    "recover_edge_by_boundary_path",
    "recover_edge_by_splitting",
    "recover_edge_by_swaps",
    "retriangulate_with_constraint",
    "segment_intersects_triangle",
]

"""Compatibility exports for the legacy ``delaunay.predicates`` module."""

from delaunay.backup_old.predicates import (
    circumcenter_precise,
    incircle,
    orient2d,
    point_in_triangle,
    segments_intersect,
)

__all__ = [
    "circumcenter_precise",
    "incircle",
    "orient2d",
    "point_in_triangle",
    "segments_intersect",
]

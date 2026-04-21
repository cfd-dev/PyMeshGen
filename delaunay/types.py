"""Compatibility exports for the legacy ``delaunay.types`` module."""

from delaunay.backup_old.types import (
    EdgeXFace,
    MTri3,
    build_adjacency,
    collect_shell_edges,
    compute_cavity_area,
)

__all__ = [
    "EdgeXFace",
    "MTri3",
    "build_adjacency",
    "collect_shell_edges",
    "compute_cavity_area",
]

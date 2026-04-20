"""Legacy compatibility wrapper for Gmsh Bowyer-Watson cavity routines."""

from delaunay.backup_old.bw_cavity import (
    recur_find_cavity,
    find_cavity_iterative,
    validate_star_shaped,
    validate_edge_lengths,
    insert_vertex,
    restore_cavity,
)

__all__ = [
    "recur_find_cavity",
    "find_cavity_iterative",
    "validate_star_shaped",
    "validate_edge_lengths",
    "insert_vertex",
    "restore_cavity",
]

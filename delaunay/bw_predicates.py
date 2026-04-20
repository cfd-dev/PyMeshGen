"""Legacy compatibility wrapper for robust geometric predicates."""

from delaunay.backup_old.bw_predicates import (
    orient2d,
    orient2d_fast,
    incircle,
    incircle_fast,
    circumcenter_precise,
    compute_circumcircle,
    point_in_circumcircle_robust,
)

__all__ = [
    "orient2d",
    "orient2d_fast",
    "incircle",
    "incircle_fast",
    "circumcenter_precise",
    "compute_circumcircle",
    "point_in_circumcircle_robust",
]

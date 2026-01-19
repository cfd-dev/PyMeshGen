"""
optimize subpackage for PyMeshGen
"""

try:
    from .optimize import edge_swap, laplacian_smooth, hybrid_smooth, optimize_hybrid_grid
except ImportError:
    # Functions will be available when modules are properly loaded with path manipulation
    pass

try:
    from . import mesh_quality
except ImportError:
    pass

try:
    from .angle_based_smoothing import smooth_mesh_angle_based, smooth_mesh_getme, angle_based_smoothing, getme_method
except ImportError:
    # Functions will be available when modules are properly loaded with path manipulation
    pass

try:
    from .nn_smoothing import nn_smoothing_adam, smooth_mesh_drl, adam_optimization_smoothing, drl_smoothing
except ImportError:
    # Functions will be available when modules are properly loaded with path manipulation
    pass
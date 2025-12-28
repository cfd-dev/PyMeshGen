"""
optimize subpackage for PyMeshGen
"""

try:
    from .optimize import edge_swap, laplacian_smooth, merge_elements, hybrid_smooth, optimize_hybrid_grid
except ImportError:
    # Functions will be available when modules are properly loaded with path manipulation
    pass

try:
    from . import mesh_quality
except ImportError:
    pass
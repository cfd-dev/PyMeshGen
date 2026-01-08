"""
visualization subpackage for PyMeshGen
"""

try:
    from .mesh_visualization import Visualization
    from .mesh_visualization import visualize_unstr_grid_2d
except ImportError:
    # Classes will be available when modules are properly loaded with path manipulation
    pass
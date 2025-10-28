"""
visualization subpackage for PyMeshGen
"""

try:
    from .mesh_visualization import Visualization
except ImportError:
    # Classes will be available when modules are properly loaded with path manipulation
    pass
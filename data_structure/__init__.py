"""
data_structure subpackage for PyMeshGen
"""

try:
    from .front2d import construct_initial_front
    from .basic_elements import Connector, Part, NodeElement, Triangle, Quadrilateral
    from .unstructured_grid import Unstructured_Grid
except ImportError:
    # Classes will be available when modules are properly loaded with path manipulation
    pass
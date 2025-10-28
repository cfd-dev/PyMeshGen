"""
data_structure subpackage for PyMeshGen
"""

try:
    from .front2d import construct_initial_front
    from .basic_elements import Connector, Part, Unstructured_Grid, NodeElement, Triangle, Quadrilateral
except ImportError:
    # Classes will be available when modules are properly loaded with path manipulation
    pass
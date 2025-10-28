"""
fileIO subpackage for PyMeshGen
"""

try:
    from .read_cas import parse_fluent_msh
    from .vtk_io import write_vtk, parse_vtk_msh, VTK_ELEMENT_TYPE
    from .stl_io import parse_stl_msh
except ImportError:
    # Classes will be available when modules are properly loaded with path manipulation
    pass
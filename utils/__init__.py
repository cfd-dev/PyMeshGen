"""
utils subpackage for PyMeshGen
"""

try:
    from .timer import TimeSpan
    from .message import info, debug, verbose, warning, error, set_debug_level, gui_log, gui_progress
    from .mesh_utils import merge_triangles_to_quads
except ImportError:
    # Classes will be available when modules are properly loaded with path manipulation
    pass

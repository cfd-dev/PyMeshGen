"""
utils subpackage for PyMeshGen
"""

try:
    from .timer import TimeSpan
    from .message import info, debug, verbose, warning, error, set_debug_level
except ImportError:
    # Classes will be available when modules are properly loaded with path manipulation
    pass
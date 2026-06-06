"""
adfront3 subpackage for PyMeshGen - 3D advancing front method for tetrahedral mesh generation
"""

try:
    from .adfront3_netgen import AdFront3Netgen as Adfront3
except ImportError:
    try:
        from .adfront3 import Adfront3
    except ImportError:
        pass

"""
delaunay3d subpackage for PyMeshGen - Bowyer-Watson Delaunay 3D tetrahedral mesh generation
"""

try:
    from .bowyer_watson import BowyerWatsonTetGen
except ImportError:
    pass

"""Public package entrypoints for delaunay."""

import sys as _sys

from delaunay.bw_core import BowyerWatsonMeshGenerator
from delaunay.bw_utils import create_bowyer_watson_mesh
from delaunay.types import MTri3

# Backward compatibility: preserve `import delaunay.utils`
from delaunay import bw_utils as _bw_utils
_sys.modules.setdefault(__name__ + ".utils", _bw_utils)

# Backward compatibility: preserve `import delaunay.core`.
from delaunay import bw_core as _bw_core
_sys.modules.setdefault(__name__ + ".core", _bw_core)

__all__ = [
    "BowyerWatsonMeshGenerator",
    "MTri3",
    "create_bowyer_watson_mesh",
]

"""
向后兼容模块：delaunay.bowyer_watson

提供与旧版本相同的导入路径。
"""

from delaunay.backup_old.bw_core import BowyerWatsonMeshGenerator
from delaunay.bw_utils import create_bowyer_watson_mesh

__all__ = ["BowyerWatsonMeshGenerator", "create_bowyer_watson_mesh"]

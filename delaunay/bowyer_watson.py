"""Backward-compatible Bowyer-Watson package exports."""

from delaunay.bw_core import BowyerWatsonMeshGenerator
from delaunay.bw_utils import create_bowyer_watson_mesh

__all__ = ["BowyerWatsonMeshGenerator", "create_bowyer_watson_mesh"]

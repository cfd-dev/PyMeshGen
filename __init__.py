"""
PyMeshGen - A Python-based unstructured mesh generator

This package provides tools for generating unstructured meshes for computational fluid dynamics simulations.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add subdirectories to the Python path to ensure all modules can be imported
SUBDIRS = [
    "fileIO",
    "data_structure", 
    "meshsize",
    "visualization",
    "adfront2",
    "optimize",
    "utils",
    "gui"
]

for subdir in SUBDIRS:
    subdir_path = project_root / subdir
    if subdir_path.exists() and str(subdir_path) not in sys.path:
        sys.path.append(str(subdir_path))

# Import main functions for easy access
from .core import generate_mesh
from .data_structure.parameters import Parameters

__version__ = "1.0.0"
__author__ = "CFD_Dev"
__all__ = ["generate_mesh", "Parameters"]

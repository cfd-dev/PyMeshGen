# PyMeshGen [![License](https://img.shields.io/badge/License-GPLv2+-brightgreen.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)

![Mesh Example](./docs/images/demo_mesh.png)

## Project Overview
An open-source Python-based unstructured Mesh Generator(PyMeshGen) for CFD/FEA analysis, providing basic 2D mesh generation tools and study platform of widely used algorithms.

## Project：
- Nianhua Wang，nianhuawong@qq.com

## Key Features
- **Input/Output**
  - Import Fluent `.cas` mesh format
  - Import and export VTK visualization `.vtk` format
  - Import and export `.stl` tessellation format
- **Core Algorithms**
  - 2D Advancing Front Method
  - Boundary Layer Advancing Technique
  - Quadtree Background Mesh Sizing
- **Supported Elements**
  - Isotropic Triangular Meshes
  - Anisotropic Quadrilateral Boundary Layers
- **Advanced Mesh Optimization**
  - Neural Network-Based Mesh Smoothing(NN-Smoothing)
  - Deep Reinforcement Learning-Based Mesh Smoothing(DRL-Smoothing)
  - Mesh Optimization with Adam Optimizer(Adam-Smoothing)

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample mesh
python PyMeshGen.py --case "./config/30p30n.json"
```

## Contact
- **Author**: Nianhua Wang <nianhuawong@qq.com>
- **Maintainer**: [Nianhua Wang] <nianhuawong@qq.com>

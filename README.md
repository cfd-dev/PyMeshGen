# PyMeshGen [![License](https://img.shields.io/badge/License-GPLv2+-brightgreen.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)

![Mesh Example](./docs/images/demo_mesh.png)

## Project Overview
An open-source unstructured mesh generator for CFD/FEA analysis, providing user-friendly 2D mesh generation solutions.

## Project：
- Nianhua Wang，nianhuawong@qq.com
- 
## Key Features
- **Input/Output**
  - Import Fluent `.cas` format
  - Export VTK visualization format
- **Core Algorithms**
  - 2D Advancing Front Method
  - Boundary Layer Advancing Technique
  - Quadtree Background Mesh Sizing
- **Supported Elements**
  - Isotropic Triangular Meshes
  - Anisotropic Quadrilateral Boundary Layers

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

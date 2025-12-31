# PyMeshGen [![License](https://img.shields.io/badge/License-GPLv2+-brightgreen.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)

![Mesh Example](./docs/images/demo_mesh.png)

## Project Overview
An open-source Python-based unstructured Mesh Generator(PyMeshGen) for CFD/FEA analysis, providing basic 2D mesh generation tools and study platform of widely used algorithms.

## Project：
- cfd_dev，cfd_dev@126.com

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
- **GUI Interface**
  - Graphical user interface for parameter setting
  - Interactive mesh visualization
  - File import/export operations

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample mesh (command line)
python PyMeshGen.py --case "./config/30p30n.json"

# Launch GUI interface
python start_gui.py
```

## GUI Usage
The GUI provides a graphical interface for easier mesh generation with the following workflow:
1. **Import CAS Mesh**: Load Fluent `.cas` files through the File tab
2. **Extract Boundary**: Extract boundary mesh and part information using the Geometry tab
3. **Configure Parameters**: Set global and part-specific mesh parameters in the Configuration tab
4. **Generate Mesh**: Create mesh using the Mesh tab's Generate button
5. **Export Mesh**: Save the generated mesh in various formats

The Ribbon interface includes 6 tabs:
- **File**: Project management and file operations
- **Geometry**: Mesh import and boundary extraction
- **View**: Camera controls and rendering options
- **Configuration**: Global and part parameter settings
- **Mesh**: Mesh generation and optimization operations
- **Help**: Documentation and support

See `gui/README.md` for detailed GUI usage instructions.

## Contact
- **Author**: cfd_dev <cfd_dev@126.com>
- **Maintainer**: cfd_dev <cfd_dev@126.com>

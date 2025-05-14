# PyMeshGen User Guide v0.1.0

## Table of Contents
- Chapter 1: Introduction
- Chapter 2: Getting Started
- Chapter 3: Framework and Core Modules
- Chapter 4: Mesh Generation Algorithms
- Chapter 5: Mesh Post-Processing
- Chapter 6: Advanced Features
- Chapter 7: Examples
- Chapter 8: Troubleshooting

## Chapter 1: Introduction
PyMeshGen is an open-source Python-based unstructured Mesh Generator(PyMeshGen) for CFD/FEA (Computational Fluid Dynamics/ Finite Element) analysis, providing basic 2D mesh generation tools and a study platform of widely used meshing algorithms.
There are three main features of PyMeshGen:
1. Basic 2D mesh generation tools: 
   - CAD file import
   - 1D curve meshes
   - 2D Isotropic Triangular/Quadrilateral Meshes
   - 2D Anisotropic Quadrilateral Boundary Layers
   - CAE Mesh file export
2. Study platform of widely used algorithms: 
   - curve following method
   - advancing front method
   - boundary layer advancing technique
   - quadtree background mesh sizing
   - mesh optimization algorithms: laplace smoothing, adam optimizer
3. Study platform for AI-driven Mesh Generation
   - Mesh Sampling and Dataset Construction
   - Neural Network-Based Mesh Smoothing(NN-Smoothing)
   - Deep Reinforcement Learning-Based Mesh Smoothing(DRL-Smoothing)
   - Mesh Optimization with Adam Optimizer(Adam-Smoothing)
This guide will walk you through the installation, usage, and advanced features of PyMeshGen.

## Chapter 2: Getting Started
To get started with PyMeshGen, you need to install the library. You can do this using pip:
```bash
pip install pymeshgen
```
Once installed, you can import the library in your Python script:
```python
import pymeshgen as pmg
```

## Chapter 3: Framework and Core Modules
PyMeshGen is built around a modular architecture, allowing you to easily extend its functionality. The core modules include:
- **Mesh**: Represents a mesh with vertices, edges, and faces.
- **MeshGenerator**: Provides methods for generating meshes.
- **MeshPostProcessor**: Contains post-processing functions for refining and optimizing meshes.
- **MeshVisualizer**: Provides tools for visualizing meshes.
- **MeshIO**: Handles input and output operations for meshes.
- **MeshUtils**: Contains utility functions for mesh manipulation.
- **MeshQuality**: Provides quality metrics for evaluating mesh quality.
- **MeshOptimizer**: Contains optimization algorithms for improving mesh quality.
- **MeshRefiner**: Provides methods for refining meshes.
- **MeshSimplifier**: Contains methods for simplifying meshes.
- **MeshSampler**: Provides methods for sampling meshes.
- **MeshValidator**: Contains methods for validating meshes.

## Chapter 4: Mesh Generation Algorithms
PyMeshGen supports various mesh generation algorithms, including:
- **Delaunay Triangulation**: Generates a Delaunay triangulation of a set of points.
- **Quadrilateralization**: Generates a quadrilateral mesh from a set of points.
- **Triangulation**: Generates a triangulation of a set of points.
- **Voronoi Diagram**: Generates a Voronoi diagram from a set of points.

## Chapter 5: Mesh Post-Processing
PyMeshGen provides post-processing functions for refining and optimizing meshes. These include:
- **MeshRefiner**: Refines a mesh by adding vertices and edges.
- **MeshOptimizer**: Optimizes a mesh by removing redundant vertices and edges.
- **MeshSimplifier**: Simplifies a mesh by removing vertices and edges.

## Chapter 6: Advanced Features
PyMeshGen also includes advanced features, such as:
- **MeshSampler**: Samples a mesh at specified points.
- **MeshValidator**: Validates a mesh for correctness.
- **MeshQuality**: Provides quality metrics for evaluating mesh quality.

## Chapter 7: Examples
To illustrate the usage of PyMeshGen, we provide several examples:
```python
# Example 1: Generate a Delaunay triangulation
points = [[0, 0], [1, 0], [0, 1], [1, 1]]
mesh = pmg.MeshGenerator.delaunay_triangulation(points)
```
```python
# Example 2: Refine a mesh
mesh = pmg.MeshRefiner.refine(mesh)
```
```python
# Example 3: Optimize a mesh
mesh = pmg.MeshOptimizer.optimize(mesh)
```
```python
# Example 4: Simplify a mesh
mesh = pmg.MeshSimplifier.simplify(mesh)
```

## Chapter 8: Troubleshooting
If you encounter any issues while using PyMeshGen, please refer to the documentation or contact the developer for assistance.


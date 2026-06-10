"""
sfmesh_aft - 阵面推进法（Advancing Front Method）模块

包含 3D AFM、2D AFM 流水线、参数化网格生成和基础几何体生成器。
共享数据结构和 Delaunay 算法保留在 sfmesh/ 包中。
"""


def __getattr__(name):
    _lazy = {
        'SurfaceMeshGenerator': '.mesh_3d_afm',
        'generate_sphere_mesh': '.mesh_parametric',
        'generate_ellipsoid_mesh': '.mesh_parametric',
        'generate_cube_mesh': '.shape_generators',
        'generate_cylinder_mesh': '.shape_generators',
        'generate_rectangle_mesh': '.shape_generators',
        'generate_ellipsoid_mesh_2d_afm': '.shape_generators',
        'generate_surface_mesh_from_shape': '.shape_generators',
        'generate_surface_mesh_from_file': '.shape_generators',
        '_export_combined_mesh': '.shape_generators',
        'PrimitiveMeshResult': '.shape_generators',
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'SurfaceMeshGenerator',
    'generate_cube_mesh',
    'generate_cylinder_mesh',
    'generate_rectangle_mesh',
    'generate_sphere_mesh',
    'generate_ellipsoid_mesh',
    'generate_ellipsoid_mesh_2d_afm',
    'generate_surface_mesh_from_shape',
    'generate_surface_mesh_from_file',
    '_export_combined_mesh',
    'PrimitiveMeshResult',
]

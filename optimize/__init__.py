"""
optimize subpackage for PyMeshGen

使用延迟导入避免循环依赖问题
"""

def __getattr__(name):
    """延迟导入优化模块，避免循环依赖"""
    if name in ['edge_swap', 'laplacian_smooth', 'hybrid_smooth', 'optimize_hybrid_grid']:
        from .optimize import (
            edge_swap, laplacian_smooth, hybrid_smooth, optimize_hybrid_grid
        )
        return locals()[name]
    elif name == 'mesh_quality':
        from . import mesh_quality
        return mesh_quality
    elif name in ['smooth_mesh_angle_based', 'smooth_mesh_getme', 'angle_based_smoothing', 'getme_method']:
        from .angle_based_smoothing import (
            smooth_mesh_angle_based, smooth_mesh_getme, angle_based_smoothing, getme_method
        )
        return locals()[name]
    elif name in ['nn_smoothing_adam', 'smooth_mesh_drl', 'adam_optimization_smoothing', 'drl_smoothing']:
        try:
            from .nn_smoothing import (
                nn_smoothing_adam, smooth_mesh_drl, adam_optimization_smoothing, drl_smoothing
            )
            return locals()[name]
        except ImportError as e:
            print(f"Warning: Failed to import from nn_smoothing: {e}")
            return None
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
"""
optimize subpackage for PyMeshGen

使用延迟导入避免循环依赖问题，并缓存已导入的模块以提高性能
"""

import importlib

# 直接导入 mesh_quality 模块，因为它不会导致循环导入
from . import mesh_quality

# 缓存已导入的模块
_module_cache = {}

# 导入映射表：函数名 -> (模块名, 函数名)
_IMPORT_MAP = {
    # 从 optimize.py 导入的函数
    'edge_swap': ('optimize', 'edge_swap'),
    'laplacian_smooth': ('optimize', 'laplacian_smooth'),
    'hybrid_smooth': ('optimize', 'hybrid_smooth'),
    'optimize_hybrid_grid': ('optimize', 'optimize_hybrid_grid'),
    'node_perturbation': ('optimize', 'node_perturbation'),
    
    # 从 angle_based_smoothing 导入的函数
    'smooth_mesh_angle_based': ('angle_based_smoothing', 'smooth_mesh_angle_based'),
    'smooth_mesh_getme': ('angle_based_smoothing', 'smooth_mesh_getme'),
    'angle_based_smoothing': ('angle_based_smoothing', 'angle_based_smoothing'),
    'getme_method': ('angle_based_smoothing', 'getme_method'),
    
    # 从 nn_smoothing 导入的函数（可选依赖）
    'nn_smoothing_adam': ('nn_smoothing', 'nn_smoothing_adam'),
    'nn_based_smoothing': ('nn_smoothing', 'nn_based_smoothing'),
    'smooth_mesh_nn': ('nn_smoothing', 'smooth_mesh_nn'),
    'smooth_mesh_drl': ('nn_smoothing', 'smooth_mesh_drl'),
    'adam_optimization_smoothing': ('nn_smoothing', 'adam_optimization_smoothing'),
    'drl_smoothing': ('nn_smoothing', 'drl_smoothing'),
}


def _import_module(module_name):
    """
    导入指定的模块并缓存结果
    
    Args:
        module_name: 要导入的模块名（不带包前缀）
        
    Returns:
        导入的模块对象，如果导入失败则返回 None
    """
    if module_name in _module_cache:
        return _module_cache[module_name]
    
    try:
        full_module_name = f'{__name__}.{module_name}'
        module = importlib.import_module(full_module_name)
        _module_cache[module_name] = module
        return module
    except ImportError as e:
        print(f"Warning: Failed to import module '{module_name}': {e}")
        return None


def __getattr__(name):
    """
    延迟导入优化模块，避免循环依赖
    
    Args:
        name: 要访问的属性名
        
    Returns:
        对应的函数或模块
        
    Raises:
        AttributeError: 如果属性不存在
    """
    if name in _IMPORT_MAP:
        module_name, attr_name = _IMPORT_MAP[name]
        module = _import_module(module_name)
        
        if module is None:
            raise AttributeError(f"Failed to import module '{module_name}' for attribute '{name}'")
        
        # 如果 attr_name 为 None，返回整个模块
        if attr_name is None:
            return module
        
        # 返回模块中的指定属性
        if hasattr(module, attr_name):
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"Module '{module_name}' has no attribute '{attr_name}'")
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    返回模块的公共属性列表
    
    Returns:
        可用属性名的列表
    """
    attributes = list(_IMPORT_MAP.keys())
    # 添加直接导入的模块
    attributes.append('mesh_quality')
    return attributes

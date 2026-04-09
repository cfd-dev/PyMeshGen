"""
delaunay 包 - Bowyer-Watson Delaunay 网格生成

目录结构:
    bw_core.py      - BowyerWatsonMeshGenerator 和 Triangle 核心实现
    helpers.py      - 辅助函数 (边界环提取, create_bowyer_watson_mesh 接口)
    bowyer_watson.py - 向后兼容模块 (重新导出上述内容)
    ref/            - Gmsh C++ 参考实现
    __init__.py     - 包入口

公共接口:
    from delaunay import create_bowyer_watson_mesh
    from delaunay.bw_core import BowyerWatsonMeshGenerator, Triangle
"""

from delaunay.bw_core import BowyerWatsonMeshGenerator, Triangle
from delaunay.helpers import create_bowyer_watson_mesh

__all__ = [
    "BowyerWatsonMeshGenerator",
    "Triangle",
    "create_bowyer_watson_mesh",
]

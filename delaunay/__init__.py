"""
delaunay 包 - Bowyer-Watson Delaunay 网格生成

简化后的目录结构：
    types.py       - 数据类型定义 (MTri3, EdgeXFace)
    predicates.py  - 几何谓词 (orient2d, incircle)
    core.py        - BowyerWatsonMeshGenerator 核心类
    boundary.py    - 边界恢复功能
    cavity.py      - Cavity 搜索和点插入
    utils.py       - 辅助函数 (边界环提取等)
    __init__.py    - 包入口

公共接口：
    from delaunay import create_bowyer_watson_mesh
    from delaunay.core import BowyerWatsonMeshGenerator
    from delaunay.types import MTri3
"""

from delaunay.core import BowyerWatsonMeshGenerator
from delaunay.utils import create_bowyer_watson_mesh
from delaunay.types import MTri3

__all__ = [
    "BowyerWatsonMeshGenerator",
    "MTri3",
    "create_bowyer_watson_mesh",
]

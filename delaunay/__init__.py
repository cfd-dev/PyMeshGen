"""
delaunay 包 - Bowyer-Watson Delaunay 网格生成

简化后的目录结构：
    types.py       - 数据类型定义 (MTri3, EdgeXFace)
    predicates.py  - 几何谓词 (orient2d, incircle)
    bw_core.py     - BowyerWatsonMeshGenerator 核心类
    boundary.py    - 边界恢复功能
    cavity.py      - Cavity 搜索和点插入
    bw_utils.py    - 辅助函数 (边界环提取等)
    __init__.py    - 包入口

公共接口：
    from delaunay import create_bowyer_watson_mesh
    from delaunay.bw_core import BowyerWatsonMeshGenerator
    from delaunay.types import MTri3
"""

import sys as _sys

from delaunay.backup_old.bw_core import BowyerWatsonMeshGenerator
from delaunay.bw_utils import create_bowyer_watson_mesh
from delaunay.types import MTri3

# Backward compatibility: preserve `import delaunay.utils`
from delaunay import bw_utils as _bw_utils
_sys.modules.setdefault(__name__ + ".utils", _bw_utils)

# Backward compatibility: preserve `import delaunay.core`
from delaunay.backup_old import bw_core as _bw_core
_sys.modules.setdefault(__name__ + ".core", _bw_core)

__all__ = [
    "BowyerWatsonMeshGenerator",
    "MTri3",
    "create_bowyer_watson_mesh",
]

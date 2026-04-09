"""
Bowyer-Watson Delaunay 网格生成器 - 向后兼容模块

此模块已将实现迁移到 delaunay.core 和 delaunay.utils。
所有公共 API 保持不变，直接导入即可使用。

推荐的新导入方式:
    from delaunay import create_bowyer_watson_mesh
    from delaunay.core import BowyerWatsonMeshGenerator, Triangle
"""

# 向后兼容：从新模块重新导出
from delaunay.bw_core import (
    Triangle,
    BowyerWatsonMeshGenerator,
)
from delaunay.helpers import (
    _extract_boundary_loops_from_fronts,
    create_bowyer_watson_mesh,
)

__all__ = [
    "Triangle",
    "BowyerWatsonMeshGenerator",
    "_extract_boundary_loops_from_fronts",
    "create_bowyer_watson_mesh",
]

"""
面网格调度模块

根据面的类型（平面、闭合曲面、一般曲面）调度到对应的网格生成方法：
- 平面 → 参数化方法
- 闭合曲面 → 带极点处理的参数化方法
- 一般曲面 → 3D 阵面推进法（AFM）
"""
from typing import List, Dict

from OCC.Core.TopoDS import TopoDS_Face

from .mesh_3d_afm import _mesh_face_afm
from .mesh_parametric import _mesh_face_parametric, _mesh_face_closed_surface
from .surface_front import SurfaceTriangle, NodeElement3D

from utils.message import info, warning

from .occ_utils import _is_planar_face, _is_closed_surface


class PrimitiveMeshResult:
    """
    基础几何体网格生成结果

    Attributes:
        triangles: 所有三角形列表
        nodes: 所有节点列表（去重）
        face_map: 面索引到三角形列表的映射
        num_faces: 面的总数
        face_types: 每个面的类型描述
    """

    def __init__(self):
        self.triangles: List[SurfaceTriangle] = []
        self.nodes: List[NodeElement3D] = []
        self.face_map: Dict[int, List[SurfaceTriangle]] = {}
        self.face_types: Dict[int, str] = {}
        self.num_faces: int = 0


def _mesh_faces(
    faces: List[TopoDS_Face],
    face_types: Dict[int, str],
    spacing: float = 1.0,
) -> PrimitiveMeshResult:
    """
    对一组面逐个生成面网格

    - 平面使用参数化网格方法（高效、质量稳定）
    - 闭合曲面（球面、椭球面等）使用带极点处理的参数化方法
    - 其他曲面使用阵面推进法（AFM，从几何边界出发生成高质量网格）
    """
    result = PrimitiveMeshResult()
    result.num_faces = len(faces)
    result.face_types = face_types

    node_hash_set = set()
    node_id_offset = 0

    for i, face in enumerate(faces):
        ftype = face_types.get(i, "unknown")

        if _is_planar_face(face):
            info(f"生成面 {i + 1}/{len(faces)} ({ftype}) 网格 [参数化]...")
            triangles, face_nodes = _mesh_face_parametric(face, spacing, node_id_offset)
        elif _is_closed_surface(face):
            info(f"生成面 {i + 1}/{len(faces)} ({ftype}) 网格 [闭合曲面参数化]...")
            triangles, face_nodes = _mesh_face_closed_surface(face, spacing, node_id_offset)
        else:
            info(f"生成面 {i + 1}/{len(faces)} ({ftype}) 网格 [AFM]...")
            try:
                triangles, face_nodes = _mesh_face_afm(face, spacing, node_id_offset)
            except Exception as e:
                warning(f"AFM 失败 (面 {i}, {ftype})，回退到参数化方法: {e}")
                triangles, face_nodes = _mesh_face_parametric(face, spacing, node_id_offset)

        # 去重并收集节点
        unique_face_nodes = []
        for node in face_nodes:
            if node.hash not in node_hash_set:
                node_hash_set.add(node.hash)
                unique_face_nodes.append(node)

        result.face_map[i] = triangles
        result.triangles.extend(triangles)
        result.nodes.extend(unique_face_nodes)
        node_id_offset += len(face_nodes)

    return result

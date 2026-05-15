"""
OCC 几何内核辅助函数

提供基于 OpenCASCADE 的几何操作：面提取、包围盒、面分类、点在面内判断等。
"""
from typing import List, Dict, Tuple

from fileIO.occ_loader import ensure_occ_loaded
ensure_occ_loaded()

from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCC.Core.TopoDS import TopoDS_Face, topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepClass import BRepClass_FaceClassifier


def _extract_faces(shape) -> List[TopoDS_Face]:
    """从 OCC 形状中提取所有面"""
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while explorer.More():
        face = topods.Face(explorer.Current())
        faces.append(face)
        explorer.Next()
    return faces


def _get_face_bbox(face: TopoDS_Face):
    """获取面的包围盒，返回 (xmin, ymin, zmin, xmax, ymax, zmax)"""
    bbox = Bnd_Box()
    brepbndlib.Add(face, bbox)
    return bbox.Get()


def _get_face_bbox_center(face: TopoDS_Face) -> Tuple[float, float, float]:
    """获取面的包围盒中心"""
    xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
    return ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)


def _classify_cube_faces(
    faces: List[TopoDS_Face],
    corner1: Tuple[float, float, float],
    corner2: Tuple[float, float, float]
) -> Dict[int, str]:
    """根据包围盒中心坐标判断长方体各面类型"""
    x_min, x_max = min(corner1[0], corner2[0]), max(corner1[0], corner2[0])
    y_min, y_max = min(corner1[1], corner2[1]), max(corner1[1], corner2[1])
    z_min, z_max = min(corner1[2], corner2[2]), max(corner1[2], corner2[2])
    tol = 1e-6
    face_types = {}
    for i, face in enumerate(faces):
        cx, cy, cz = _get_face_bbox_center(face)
        if abs(cz - z_min) < tol:
            face_types[i] = "bottom"
        elif abs(cz - z_max) < tol:
            face_types[i] = "top"
        elif abs(cy - y_min) < tol:
            face_types[i] = "front"
        elif abs(cy - y_max) < tol:
            face_types[i] = "back"
        elif abs(cx - x_min) < tol:
            face_types[i] = "left"
        elif abs(cx - x_max) < tol:
            face_types[i] = "right"
        else:
            face_types[i] = f"face_{i}"
    return face_types


def _classify_cylinder_faces(
    faces: List[TopoDS_Face],
    base_z: float,
    top_z: float
) -> Dict[int, str]:
    """根据曲面类型和Z坐标判断圆柱体各面类型"""
    tol = 1e-6
    face_types = {}
    for i, face in enumerate(faces):
        surface = BRep_Tool.Surface(face)
        is_planar = surface.DynamicType().Name() == "Geom_Plane"
        if is_planar:
            _, _, cz = _get_face_bbox_center(face)
            if abs(cz - base_z) < tol:
                face_types[i] = "bottom"
            elif abs(cz - top_z) < tol:
                face_types[i] = "top"
            else:
                face_types[i] = f"disk_{i}"
        else:
            face_types[i] = "lateral"
    return face_types


def _is_point_in_face(u: float, v: float, face: TopoDS_Face) -> bool:
    """检查参数坐标 (u, v) 是否在面的边界内"""
    classifier = BRepClass_FaceClassifier()
    classifier.Perform(face, gp_Pnt2d(u, v), 1e-6)
    state = classifier.State()
    return state == TopAbs_IN or state == TopAbs_ON


def _is_planar_face(face: TopoDS_Face) -> bool:
    """判断面是否为平面"""
    surface = BRep_Tool.Surface(face)
    return surface.DynamicType().Name() == "Geom_Plane"

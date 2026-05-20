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


def _is_disk_face(face: TopoDS_Face) -> bool:
    """判断平面是否为圆形面（圆盘）"""
    if not _is_planar_face(face):
        return False

    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE

    # 检查边界是否为单个圆形边
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while wire_explorer.More():
        edge_explorer = TopExp_Explorer(wire_explorer.Current(), TopAbs_EDGE)
        n_edges = 0
        has_circle = False
        while edge_explorer.More():
            edge = edge_explorer.Current()
            curve, first, last = BRep_Tool.Curve(edge)
            if curve is not None:
                type_name = curve.DynamicType().Name()
                if type_name == "Geom_Circle":
                    has_circle = True
                elif type_name == "Geom_BSplineCurve":
                    # 检查是否为圆形 B-spline（首尾重合）
                    p1 = curve.Value(first)
                    p2 = curve.Value(last)
                    chord = ((p2.X()-p1.X())**2 + (p2.Y()-p1.Y())**2 + (p2.Z()-p1.Z())**2)**0.5
                    if chord < 1e-6:
                        has_circle = True
            n_edges += 1
            edge_explorer.Next()
        if n_edges == 1 and has_circle:
            return True
        wire_explorer.Next()

    return False


def _extract_disk_params(face: TopoDS_Face) -> dict:
    """
    从圆形平面提取几何参数

    Returns:
        dict: {
            'center': (x, y, z),
            'radius': float,
            'normal': (nx, ny, nz),
        }
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
    from OCC.Core.gp import gp_Dir

    adaptor = BRepAdaptor_Surface(face)
    plane = adaptor.Plane()
    normal = plane.Axis().Direction()
    loc = plane.Location()

    # 从边界边获取圆心和半径
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while wire_explorer.More():
        edge_explorer = TopExp_Explorer(wire_explorer.Current(), TopAbs_EDGE)
        while edge_explorer.More():
            edge = edge_explorer.Current()
            curve, first, last = BRep_Tool.Curve(edge)
            if curve is not None:
                type_name = curve.DynamicType().Name()
                if type_name == "Geom_Circle":
                    from OCC.Core.Geom import Geom_Circle
                    circle = Geom_Circle.DownCast(curve)
                    center = circle.Location()
                    radius = circle.Radius()
                    return {
                        'center': (center.X(), center.Y(), center.Z()),
                        'radius': radius,
                        'normal': (normal.X(), normal.Y(), normal.Z()),
                    }
                elif type_name == "Geom_BSplineCurve":
                    # 估算圆心和半径
                    p1 = curve.Value(first)
                    p2 = curve.Value(last)
                    mid = curve.Value((first + last) / 2)
                    # 圆心是首尾和中点的外心
                    # 简化：用包围盒中心
                    xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    cz = (zmin + zmax) / 2
                    radius = max(xmax - xmin, ymax - ymin) / 2
                    return {
                        'center': (cx, cy, cz),
                        'radius': radius,
                        'normal': (normal.X(), normal.Y(), normal.Z()),
                    }
            edge_explorer.Next()
        wire_explorer.Next()

    # 回退：从包围盒估算
    xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    radius = max(xmax - xmin, ymax - ymin) / 2
    return {
        'center': (cx, cy, cz),
        'radius': radius,
        'normal': (normal.X(), normal.Y(), normal.Z()),
    }


def _is_cylinder_face(face: TopoDS_Face) -> bool:
    """判断面是否为圆柱面（包括 B-spline 近似的圆柱面）"""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_BSplineSurface
    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE

    adaptor = BRepAdaptor_Surface(face)
    surf_type = adaptor.GetType()

    if surf_type == GeomAbs_Cylinder:
        return True

    # B-spline 近似的圆柱面：U 闭合，有两条圆形边界边
    if surf_type == GeomAbs_BSplineSurface and adaptor.IsUClosed():
        n_circles = 0
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            edge_explorer = TopExp_Explorer(wire_explorer.Current(), TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                curve, first, last = BRep_Tool.Curve(edge)
                if curve is not None:
                    type_name = curve.DynamicType().Name()
                    if type_name in ("Geom_Circle", "Geom_BSplineCurve"):
                        # 检查是否为圆形曲线（首尾重合且中点在圆弧上）
                        p1 = curve.Value(first)
                        p2 = curve.Value(last)
                        chord = ((p2.X()-p1.X())**2 + (p2.Y()-p1.Y())**2 + (p2.Z()-p1.Z())**2)**0.5
                        if chord < 1e-6:
                            n_circles += 1
                edge_explorer.Next()
            wire_explorer.Next()
        if n_circles >= 2:
            return True

    return False


def _extract_cylinder_params(face: TopoDS_Face) -> dict:
    """
    从圆柱面提取几何参数

    Returns:
        dict: {
            'base_center': (x, y, z),
            'radius': float,
            'height': float,
            'axis_dir': (dx, dy, dz),
        }
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cylinder
    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE

    adaptor = BRepAdaptor_Surface(face)
    surf_type = adaptor.GetType()

    if surf_type == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        axis = cyl.Axis()
        loc = axis.Location()
        dir = axis.Direction()
        radius = cyl.Radius()
        base_center = (loc.X(), loc.Y(), loc.Z())
        axis_dir = (dir.X(), dir.Y(), dir.Z())

        # 从边界边获取高度范围
        z_min, z_max = float('inf'), float('-inf')
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            edge_explorer = TopExp_Explorer(wire_explorer.Current(), TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                curve, first, last = BRep_Tool.Curve(edge)
                if curve is not None:
                    p1 = curve.Value(first)
                    p2 = curve.Value(last)
                    chord = ((p2.X()-p1.X())**2 + (p2.Y()-p1.Y())**2 + (p2.Z()-p1.Z())**2)**0.5
                    if chord < 1e-6:
                        # 圆形边界边
                        z_mid = (p1.Z() + p2.Z()) / 2
                        z_min = min(z_min, z_mid)
                        z_max = max(z_max, z_mid)
                edge_explorer.Next()
            wire_explorer.Next()

        if z_min < z_max:
            height = z_max - z_min
        else:
            # 从包围盒获取高度
            xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
            height = zmax - zmin

        return {
            'base_center': base_center,
            'radius': radius,
            'height': height,
            'axis_dir': axis_dir,
        }

    # B-spline 近似：从包围盒和边界边估算
    xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    radius = max(xmax - xmin, ymax - ymin) / 2
    height = zmax - zmin

    return {
        'base_center': (cx, cy, zmin),
        'radius': radius,
        'height': height,
        'axis_dir': (0.0, 0.0, 1.0),
    }


def _is_closed_surface(face: TopoDS_Face) -> bool:
    """
    判断曲面是否为闭合曲面（球面、椭球面、环面等）

    闭合曲面的边界边全部是参数接缝（seam）：每条边在边界中出现两次。
    有真正几何边界边（只出现一次）的曲面不是闭合曲面。
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
    from OCC.Core.BRep import BRep_Tool

    adaptor = BRepAdaptor_Surface(face)
    if not adaptor.IsUClosed() and not adaptor.IsVClosed():
        return False

    # 收集所有边界边的几何哈希，检查是否有只出现一次的边（真实边界）
    edge_hashes = []
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while wire_explorer.More():
        edge_explorer = TopExp_Explorer(wire_explorer.Current(), TopAbs_EDGE)
        while edge_explorer.More():
            edge = edge_explorer.Current()
            curve, first, last = BRep_Tool.Curve(edge)
            if curve is not None:
                p1 = curve.Value(first)
                p2 = curve.Value(last)
                # 用首尾坐标和中点坐标作为边的几何哈希
                mid = curve.Value((first + last) / 2)
                edge_key = (
                    round(p1.X(), 8), round(p1.Y(), 8), round(p1.Z(), 8),
                    round(p2.X(), 8), round(p2.Y(), 8), round(p2.Z(), 8),
                    round(mid.X(), 8), round(mid.Y(), 8), round(mid.Z(), 8),
                )
                edge_hashes.append(edge_key)
            edge_explorer.Next()
        wire_explorer.Next()

    # 统计每条边出现的次数
    from collections import Counter
    hash_counts = Counter(edge_hashes)

    # 如果有任何边只出现一次，则不是闭合曲面（有真实边界）
    for key, count in hash_counts.items():
        if count == 1:
            return False

    return True

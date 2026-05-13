"""
基础几何体曲面网格生成模块

支持从 OCC 基础几何体（长方体、圆柱体）直接生成曲面网格，
无需从文件导入。
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fileIO.occ_loader import ensure_occ_loaded
ensure_occ_loaded()

from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Ax2, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCC.Core.TopoDS import TopoDS_Face, topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepClass import BRepClass_FaceClassifier

from .surface_mesh import _export_combined_mesh
from .surface_front import SurfaceTriangle, NodeElement3D
from .surface_geometry import SurfaceGeometry

from utils.message import info


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


def _mesh_face_parametric(
    face: TopoDS_Face,
    spacing: float,
    node_id_offset: int,
) -> Tuple[List[SurfaceTriangle], List[NodeElement3D]]:
    """
    对单个面进行参数化网格剖分，节点投影到几何曲面上

    利用 OCC 曲面评估器在参数空间 (u, v) 生成均匀网格，
    将参数坐标映射到三维空间，确保节点严格落在几何曲面上。
    通过 BRepClass_FaceClassifier 确保节点在面的边界内。
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

    geometry = SurfaceGeometry()
    adaptor = BRepAdaptor_Surface(face)
    u_min = adaptor.FirstUParameter()
    u_max = adaptor.LastUParameter()
    v_min = adaptor.FirstVParameter()
    v_max = adaptor.LastVParameter()

    # 利用包围盒估算物理尺寸，确定网格数
    xmin, ymin, zmin, xmax, ymax, zmax = _get_face_bbox(face)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    phys_len = max(dx, dy, dz)

    nu = max(2, int(round(phys_len / spacing)) + 1)
    nv = max(2, int(round(phys_len / spacing)) + 1)

    # 第一遍：生成候选节点，检查是否在面内
    node_dict = {}  # (ju, jv) -> node
    nid = node_id_offset

    for ju in range(nu):
        for jv in range(nv):
            u = u_min + ju * (u_max - u_min) / (nu - 1) if nu > 1 else u_min
            v = v_min + jv * (v_max - v_min) / (nv - 1) if nv > 1 else v_min

            if not _is_point_in_face(u, v, face):
                continue

            pnt = adaptor.Value(u, v)
            coords = (pnt.X(), pnt.Y(), pnt.Z())

            uv = geometry.project_point_to_surface(coords, face)
            proj_coords = geometry.evaluate_point(uv[0], uv[1], face)
            normal = geometry.get_surface_normal(uv[0], uv[1], face)

            node = NodeElement3D(
                coords=proj_coords, idx=nid,
                surface=face, uv_params=uv, normal=normal,
            )
            node_dict[(ju, jv)] = node
            nid += 1

    # 第二遍：生成三角形，连接相邻的面内节点
    triangles = []
    tri_idx = 0

    for ju in range(nu - 1):
        for jv in range(nv - 1):
            n00 = node_dict.get((ju, jv))
            n10 = node_dict.get((ju + 1, jv))
            n01 = node_dict.get((ju, jv + 1))
            n11 = node_dict.get((ju + 1, jv + 1))

            if n00 and n10 and n01:
                triangles.append(SurfaceTriangle(n00, n10, n01, idx=tri_idx))
                tri_idx += 1
            if n10 and n11 and n01:
                triangles.append(SurfaceTriangle(n10, n11, n01, idx=tri_idx))
                tri_idx += 1

    nodes = list(node_dict.values())
    return triangles, nodes


def _mesh_faces(
    faces: List[TopoDS_Face],
    face_types: Dict[int, str],
    spacing: float = 1.0,
) -> PrimitiveMeshResult:
    """
    对一组面逐个生成面网格

    利用 OCC 几何模型的参数空间进行网格剖分，
    节点通过投影确保落在几何模型的真实曲面上。
    """
    result = PrimitiveMeshResult()
    result.num_faces = len(faces)
    result.face_types = face_types

    node_hash_set = set()
    node_id_offset = 0

    for i, face in enumerate(faces):
        ftype = face_types.get(i, "unknown")
        info(f"生成面 {i + 1}/{len(faces)} ({ftype}) 网格...")

        triangles, face_nodes = _mesh_face_parametric(
            face, spacing, node_id_offset,
        )

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


def generate_cube_mesh(
    corner1: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    corner2: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spacing: float = 0.5,
    output_vtk: str = None,
) -> PrimitiveMeshResult:
    """
    生成长方体的曲面网格

    Args:
        corner1: 长方体第一个角点坐标 (x, y, z)
        corner2: 长方体对角点坐标 (x, y, z)
        spacing: 全局网格尺寸
        output_vtk: 输出VTK文件路径（可选）

    Returns:
        PrimitiveMeshResult: 包含网格数据和面映射的结果

    Raises:
        ValueError: 如果角点坐标无效
    """
    for i in range(3):
        if abs(corner2[i] - corner1[i]) < 1e-12:
            raise ValueError(
                f"长方体第 {i} 方向长度为零: corner1[{i}]={corner1[i]}, corner2[{i}]={corner2[i]}"
            )

    p1 = gp_Pnt(*corner1)
    p2 = gp_Pnt(*corner2)
    shape = BRepPrimAPI_MakeBox(p1, p2).Shape()

    faces = _extract_faces(shape)
    face_types = _classify_cube_faces(faces, corner1, corner2)

    result = _mesh_faces(faces, face_types, spacing=spacing)

    if output_vtk:
        _export_combined_mesh(result.triangles, output_vtk)

    return result


def generate_cylinder_mesh(
    base_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    height: float = 1.0,
    spacing: float = 0.5,
    output_vtk: str = None,
) -> PrimitiveMeshResult:
    """
    生成圆柱体的曲面网格

    Args:
        base_center: 底面圆心坐标 (x, y, z)
        radius: 圆柱半径
        height: 圆柱高度（沿Z轴正方向）
        spacing: 全局网格尺寸
        output_vtk: 输出VTK文件路径（可选）

    Returns:
        PrimitiveMeshResult: 包含网格数据和面映射的结果

    Raises:
        ValueError: 如果半径或高度不是正数
    """
    if radius <= 0:
        raise ValueError(f"圆柱半径必须为正数: {radius}")
    if height <= 0:
        raise ValueError(f"圆柱高度必须为正数: {height}")

    axis = gp_Ax2(gp_Pnt(*base_center), gp_Dir(0, 0, 1))
    shape = BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()

    faces = _extract_faces(shape)
    base_z = base_center[2]
    top_z = base_center[2] + height
    face_types = _classify_cylinder_faces(faces, base_z, top_z)

    result = _mesh_faces(faces, face_types, spacing=spacing)

    if output_vtk:
        _export_combined_mesh(result.triangles, output_vtk)

    return result

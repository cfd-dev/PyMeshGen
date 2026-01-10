"""
几何文件导入模块，支持STEP、IGES、STL等格式
"""
import os
from typing import Union, List, Tuple
import numpy as np

try:
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Shell, TopoDS_Face, TopoDS_Wire, TopoDS_Edge, TopoDS_Vertex
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid
    from OCC.Core.Geom import Geom_Curve, Geom_Surface
    from OCC.Core.gp import gp_Pnt, gp_Vec
except ImportError:
    try:
        from OCP.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Shell, TopoDS_Face, TopoDS_Wire, TopoDS_Edge, TopoDS_Vertex
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
        from OCP.BRep import BRep_Tool
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid
        from OCP.Geom import Geom_Curve, Geom_Surface
        from OCP.gp import gp_Pnt, gp_Vec
    except ImportError:
        raise ImportError("无法导入OpenCASCADE库，请确保已安装pythonocc-core或OCP")


def import_geometry_file(filename: str) -> TopoDS_Shape:
    """
    导入几何文件，自动识别文件格式
    
    Args:
        filename: 几何文件路径
        
    Returns:
        TopoDS_Shape: 导入的几何形状
        
    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext in ['.step', '.stp']:
        return import_step_file(filename)
    elif file_ext in ['.iges', '.igs']:
        return import_iges_file(filename)
    elif file_ext in ['.stl']:
        return import_stl_file(filename)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


def import_step_file(filename: str) -> TopoDS_Shape:
    """
    导入STEP文件
    
    Args:
        filename: STEP文件路径
        
    Returns:
        TopoDS_Shape: 导入的几何形状
    """
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
    except ImportError:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
    
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    
    if status != IFSelect_RetDone:
        raise ValueError(f"无法读取STEP文件: {filename}")
    
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    return shape


def import_iges_file(filename: str) -> TopoDS_Shape:
    """
    导入IGES文件
    
    Args:
        filename: IGES文件路径
        
    Returns:
        TopoDS_Shape: 导入的几何形状
    """
    try:
        from OCC.Core.IGESControl import IGESControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
    except ImportError:
        from OCP.IGESControl import IGESControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
    
    iges_reader = IGESControl_Reader()
    status = iges_reader.ReadFile(filename)
    
    if status != IFSelect_RetDone:
        raise ValueError(f"无法读取IGES文件: {filename}")
    
    iges_reader.TransferRoots()
    shape = iges_reader.OneShape()
    
    return shape


def import_stl_file(filename: str) -> TopoDS_Shape:
    """
    导入STL文件
    
    Args:
        filename: STL文件路径
        
    Returns:
        TopoDS_Shape: 导入的几何形状
    """
    try:
        from OCC.Core.StlAPI import StlAPI_Reader
    except ImportError:
        from OCP.StlAPI import StlAPI_Reader
    
    stl_reader = StlAPI_Reader()
    shape = TopoDS_Shape()
    
    success = stl_reader.Read(shape, filename)
    
    if not success:
        raise ValueError(f"无法读取STL文件: {filename}")
    
    return shape


def extract_vertices_from_shape(shape: TopoDS_Shape) -> List[Tuple[float, float, float]]:
    """
    从TopoDS_Shape中提取所有顶点坐标
    
    Args:
        shape: OpenCASCADE形状
        
    Returns:
        顶点坐标列表 [(x, y, z), ...]
    """
    vertices = []
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    
    while explorer.More():
        vertex = explorer.Current()
        pnt = BRep_Tool.Pnt(vertex)
        vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
        explorer.Next()
    
    return vertices


def extract_edges_from_shape(shape: TopoDS_Shape) -> List[List[Tuple[float, float, float]]]:
    """
    从TopoDS_Shape中提取所有边
    
    Args:
        shape: OpenCASCADE形状
        
    Returns:
        边列表，每条边由多个点组成 [[(x1, y1, z1), (x2, y2, z2), ...], ...]
    """
    edges = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    
    while explorer.More():
        edge = explorer.Current()
        curve = BRep_Tool.Curve(edge)
        
        if curve:
            geom_curve, first, last = curve
            if geom_curve:
                # 在边上采样点
                num_points = max(10, int((last - first) * 10))
                points = []
                for i in range(num_points + 1):
                    param = first + (last - first) * i / num_points
                    pnt = geom_curve.Value(param)
                    points.append((pnt.X(), pnt.Y(), pnt.Z()))
                edges.append(points)
        
        explorer.Next()
    
    return edges


def extract_faces_from_shape(shape: TopoDS_Shape) -> List[List[List[Tuple[float, float, float]]]]:
    """
    从TopoDS_Shape中提取所有面
    
    Args:
        shape: OpenCASCADE形状
        
    Returns:
        面列表，每个面由多个三角形网格组成 [[[p1, p2, p3], [p1, p2, p3], ...], ...]
    """
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    
    while explorer.More():
        face = explorer.Current()
        surface = BRep_Tool.Surface(face)
        
        if surface:
            # 获取面的边界线
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            while wire_explorer.More():
                wire = wire_explorer.Current()
                edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                
                wire_points = []
                while edge_explorer.More():
                    edge = edge_explorer.Current()
                    curve = BRep_Tool.Curve(edge)
                    
                    if curve:
                        geom_curve, first, last = curve
                        if geom_curve:
                            num_points = max(10, int((last - first) * 10))
                            for i in range(num_points + 1):
                                param = first + (last - first) * i / num_points
                                pnt = geom_curve.Value(param)
                                wire_points.append((pnt.X(), pnt.Y(), pnt.Z()))
                    
                    edge_explorer.Next()
                
                if len(wire_points) >= 3:
                    # 将边界线三角化
                    triangulation = triangulate_wire(wire_points)
                    if triangulation:
                        faces.append(triangulation)
                
                wire_explorer.Next()
        
        explorer.Next()
    
    return faces


def triangulate_wire(points: List[Tuple[float, float, float]]) -> List[List[Tuple[float, float, float]]]:
    """
    将边界线点三角化
    
    Args:
        points: 边界点列表
        
    Returns:
        三角形列表 [[p1, p2, p3], ...]
    """
    if len(points) < 3:
        return []
    
    # 简单的扇形三角化（适用于凸多边形）
    triangles = []
    center = np.mean(points, axis=0)
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        triangles.append([tuple(center), p1, p2])
    
    return triangles


def get_shape_bounding_box(shape: TopoDS_Shape) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    获取形状的边界框
    
    Args:
        shape: OpenCASCADE形状
        
    Returns:
        (min_point, max_point): 最小点和最大点坐标
    """
    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
    except ImportError:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import brepbndlib_Add
    
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    min_point = (xmin, ymin, zmin)
    max_point = (xmax, ymax, zmax)
    
    return min_point, max_point


def get_shape_statistics(shape: TopoDS_Shape) -> dict:
    """
    获取形状的统计信息
    
    Args:
        shape: OpenCASCADE形状
        
    Returns:
        包含统计信息的字典
    """
    stats = {
        'num_vertices': 0,
        'num_edges': 0,
        'num_faces': 0,
        'num_solids': 0,
        'num_shells': 0,
        'bounding_box': None
    }
    
    # 统计各种几何元素
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        stats['num_vertices'] += 1
        explorer.Next()
    
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        stats['num_edges'] += 1
        explorer.Next()
    
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        stats['num_faces'] += 1
        explorer.Next()
    
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        stats['num_solids'] += 1
        explorer.Next()
    
    explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while explorer.More():
        stats['num_shells'] += 1
        explorer.Next()
    
    # 获取边界框
    stats['bounding_box'] = get_shape_bounding_box(shape)
    
    return stats

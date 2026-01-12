"""
几何文件导入模块，支持STEP、IGES、STL等格式
"""
import os
from typing import Union, List, Tuple
import numpy as np

# 导入 OpenCASCADE DLL 加载器
# 该模块负责预加载所有必要的 OpenCASCADE DLL，解决 Windows 平台上的 DLL 依赖问题
from fileIO.occ_loader import ensure_occ_loaded

# 确保 OpenCASCADE DLL 已加载
# 这必须在导入任何 OCC 模块之前调用
ensure_occ_loaded()

try:
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Shell, TopoDS_Face, TopoDS_Wire, TopoDS_Edge, TopoDS_Vertex
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid
    from OCC.Core.Geom import Geom_Curve, Geom_Surface
    from OCC.Core.gp import gp_Pnt, gp_Vec
except ImportError:
    raise ImportError("无法导入OpenCASCADE库，请确保已安装pythonocc-core")


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
        raise ImportError("无法导入STEP模块，请确保已安装pythonocc-core")
    
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
        raise ImportError("无法导入IGES模块，请确保已安装pythonocc-core")
    
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
        raise ImportError("无法导入STL模块，请确保已安装pythonocc-core")
    
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
        raise ImportError("无法导入边界框模块，请确保已安装pythonocc-core")
    
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


def extract_edges_with_info(shape: TopoDS_Shape) -> List[dict]:
    """
    从TopoDS_Shape中提取所有边及其详细信息
    
    Args:
        shape: OpenCASCADE形状
        
    Returns:
        边列表，每条边包含几何曲线、参数范围、长度等信息
        [
            {
                'edge': TopoDS_Edge,
                'curve': Geom_Curve,
                'first': float,
                'last': float,
                'length': float,
                'start_point': (x, y, z),
                'end_point': (x, y, z)
            },
            ...
        ]
    """
    edges_info = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    
    while explorer.More():
        edge = explorer.Current()
        curve = BRep_Tool.Curve(edge)
        
        if curve:
            geom_curve, first, last = curve
            if geom_curve:
                # 计算曲线长度
                try:
                    from OCC.Core.GCPnts import GCPnts_AbscissaPoint
                    length = GCPnts_AbscissaPoint.Length(geom_curve, first, last)
                except:
                    # 如果无法计算精确长度，使用近似值
                    length = abs(last - first)
                
                # 获取起点和终点
                start_pnt = geom_curve.Value(first)
                end_pnt = geom_curve.Value(last)
                
                edge_info = {
                    'edge': edge,
                    'curve': geom_curve,
                    'first': first,
                    'last': last,
                    'length': length,
                    'start_point': (start_pnt.X(), start_pnt.Y(), start_pnt.Z()),
                    'end_point': (end_pnt.X(), end_pnt.Y(), end_pnt.Z())
                }
                edges_info.append(edge_info)
        
        explorer.Next()
    
    return edges_info


def discretize_edge_by_size(edge_info: dict, max_size: float) -> List[Tuple[float, float, float]]:
    """
    根据最大网格尺寸离散化边
    
    Args:
        edge_info: 边信息字典（由extract_edges_with_info返回）
        max_size: 最大网格尺寸
        
    Returns:
        离散点列表 [(x1, y1, z1), (x2, y2, z2), ...]
    """
    geom_curve = edge_info['curve']
    first = edge_info['first']
    last = edge_info['last']
    length = edge_info['length']
    
    if length < 1e-10:
        return [edge_info['start_point'], edge_info['end_point']]
    
    # 计算需要的段数
    num_segments = max(1, int(length / max_size))
    
    # 生成离散点
    points = []
    for i in range(num_segments + 1):
        param = first + (last - first) * i / num_segments
        pnt = geom_curve.Value(param)
        points.append((pnt.X(), pnt.Y(), pnt.Z()))
    
    return points


def discretize_edge_by_count(edge_info: dict, num_points: int) -> List[Tuple[float, float, float]]:
    """
    根据指定点数离散化边
    
    Args:
        edge_info: 边信息字典（由extract_edges_with_info返回）
        num_points: 离散点数
        
    Returns:
        离散点列表 [(x1, y1, z1), (x2, y2, z2), ...]
    """
    geom_curve = edge_info['curve']
    first = edge_info['first']
    last = edge_info['last']
    
    if num_points < 2:
        num_points = 2
    
    # 生成离散点
    points = []
    for i in range(num_points):
        param = first + (last - first) * i / (num_points - 1)
        pnt = geom_curve.Value(param)
        points.append((pnt.X(), pnt.Y(), pnt.Z()))
    
    return points


def bind_edges_to_connectors(shape: TopoDS_Shape, parts: List) -> None:
    """
    将OCC读取的二维曲线edge与Connector绑定，并离散化后保存到Connector中
    
    Args:
        shape: OpenCASCADE形状
        parts: 部件列表，每个部件包含多个Connector对象
    """
    # 提取所有边及其信息
    edges_info = extract_edges_with_info(shape)
    
    if not edges_info:
        print("警告：未找到任何边")
        return
    
    # 为每条边分配一个索引
    for idx, edge_info in enumerate(edges_info):
        edge_info['idx'] = idx
    
    # 将边绑定到Connector
    # 这里使用简单的策略：按照边的顺序分配给各个部件的默认connector
    # 实际应用中可以根据几何特征、命名等进行更智能的匹配
    edge_idx = 0
    for part in parts:
        for connector in part.connectors:
            if connector.curve_name == "default":
                # 为默认connector分配边
                if edge_idx < len(edges_info):
                    edge_info = edges_info[edge_idx]
                    
                    # 根据connector的参数进行离散化
                    max_size = connector.param.max_size
                    
                    # 离散化边
                    discretized_points = discretize_edge_by_size(edge_info, max_size)
                    
                    # 将离散点保存到connector的front_list中
                    _create_fronts_from_points(connector, discretized_points)
                    
                    # 保存边信息到connector的cad_obj中
                    connector.cad_obj = edge_info
                    
                    edge_idx += 1
    
    print(f"成功绑定 {edge_idx} 条边到Connector")


def _create_fronts_from_points(connector, points: List[Tuple[float, float, float]]) -> None:
    """
    从离散点创建Front对象并保存到Connector中
    
    Args:
        connector: Connector对象
        points: 离散点列表 [(x1, y1, z1), (x2, y2, z2), ...]
    """
    if len(points) < 2:
        return
    
    # 延迟导入以避免循环依赖
    from data_structure.basic_elements import NodeElementALM
    from data_structure.front2d import Front
    
    # 清空现有的front_list
    connector.front_list = []
    
    # 为每对相邻点创建一个Front
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        
        # 创建节点元素
        node1 = NodeElementALM(
            coords=p1,
            idx=i,
            bc_type=None,
            match_bound=None
        )
        node2 = NodeElementALM(
            coords=p2,
            idx=i + 1,
            bc_type=None,
            match_bound=None
        )
        
        # 创建Front对象
        front = Front(
            node_elem1=node1,
            node_elem2=node2,
            idx=i,
            bc_type=None,
            part_name=connector.part_name
        )
        
        # 添加到connector的front_list中
        connector.front_list.append(front)


def bind_edges_by_curve_name(shape: TopoDS_Shape, parts: List, edge_curve_mapping: dict) -> None:
    """
    根据曲线名称将edge绑定到对应的Connector
    
    Args:
        shape: OpenCASCADE形状
        parts: 部件列表
        edge_curve_mapping: 边到曲线名称的映射 {edge_idx: curve_name}
    """
    # 提取所有边及其信息
    edges_info = extract_edges_with_info(shape)
    
    if not edges_info:
        print("警告：未找到任何边")
        return
    
    # 为每条边分配一个索引
    for idx, edge_info in enumerate(edges_info):
        edge_info['idx'] = idx
    
    # 根据映射关系绑定边到connector
    for edge_idx, curve_name in edge_curve_mapping.items():
        if edge_idx >= len(edges_info):
            continue
        
        edge_info = edges_info[edge_idx]
        
        # 查找对应的connector
        for part in parts:
            for connector in part.connectors:
                if connector.curve_name == curve_name:
                    # 根据connector的参数进行离散化
                    max_size = connector.param.max_size
                    
                    # 离散化边
                    discretized_points = discretize_edge_by_size(edge_info, max_size)
                    
                    # 将离散点保存到connector的front_list中
                    _create_fronts_from_points(connector, discretized_points)
                    
                    # 保存边信息到connector的cad_obj中
                    connector.cad_obj = edge_info
                    
                    print(f"边 {edge_idx} 已绑定到 {part.part_name}.{curve_name}")
                    break

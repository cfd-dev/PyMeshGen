"""
OCC Shape到VTK PolyData的转换模块
"""
import numpy as np
from typing import List, Tuple, Optional

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangulation

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    raise ImportError("无法导入VTK库，请确保已安装VTK")


def shape_to_vtk_polydata(shape: TopoDS_Shape, 
                         mesh_quality: float = 1.0,
                         relative: bool = False) -> vtk.vtkPolyData:
    """
    将TopoDS_Shape转换为VTK PolyData
    
    Args:
        shape: OpenCASCADE形状
        mesh_quality: 网格质量参数（越小越精细）
        relative: 是否使用相对网格质量
        
    Returns:
        vtk.vtkPolyData: VTK多边形数据对象
    """
    # 对形状进行网格化
    mesher = BRepMesh_IncrementalMesh(shape, mesh_quality, relative, True, True)
    mesher.Perform()
    
    # 创建VTK点集
    points = vtk.vtkPoints()
    
    # 创建VTK单元
    triangles = vtk.vtkCellArray()
    lines = vtk.vtkCellArray()
    
    # 存储所有顶点
    all_vertices = []
    vertex_map = {}
    
    # 遍历所有面
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        
        # 获取面的三角化
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        
        if triangulation:
            # 获取三角形数量
            nb_triangles = triangulation.NbTriangles()
            nb_nodes = triangulation.NbNodes()
            
            # 获取节点坐标
            for i in range(1, nb_nodes + 1):
                p = triangulation.Node(i)
                p.Transform(loc.Transformation())
                vertex = (p.X(), p.Y(), p.Z())
                
                if vertex not in vertex_map:
                    vertex_map[vertex] = len(all_vertices)
                    all_vertices.append(vertex)
            
            # 获取三角形索引
            for i in range(1, nb_triangles + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                
                # 转换为0-based索引
                idx1 = vertex_map[(triangulation.Node(n1).X(), 
                                  triangulation.Node(n1).Y(), 
                                  triangulation.Node(n1).Z())]
                idx2 = vertex_map[(triangulation.Node(n2).X(), 
                                  triangulation.Node(n2).Y(), 
                                  triangulation.Node(n2).Z())]
                idx3 = vertex_map[(triangulation.Node(n3).X(), 
                                  triangulation.Node(n3).Y(), 
                                  triangulation.Node(n3).Z())]
                
                # 创建VTK三角形
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, idx1)
                triangle.GetPointIds().SetId(1, idx2)
                triangle.GetPointIds().SetId(2, idx3)
                triangles.InsertNextCell(triangle)
        
        face_explorer.Next()
    
    # 添加顶点到VTK点集
    for vertex in all_vertices:
        points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
    
    # 创建PolyData对象
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    
    return polydata


def shape_edges_to_vtk_polydata(shape: TopoDS_Shape,
                               sample_rate: float = 0.1) -> vtk.vtkPolyData:
    """
    将TopoDS_Shape的边转换为VTK PolyData（仅边框）
    
    Args:
        shape: OpenCASCADE形状
        sample_rate: 采样率（控制边的采样密度）
        
    Returns:
        vtk.vtkPolyData: VTK多边形数据对象（仅包含边）
    """
    # 创建VTK点集
    points = vtk.vtkPoints()
    
    # 创建VTK线单元
    lines = vtk.vtkCellArray()
    
    # 存储所有顶点
    all_vertices = []
    vertex_map = {}
    
    # 遍历所有边
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        
        # 获取边的曲线
        curve, first, last = BRep_Tool.Curve(edge)
        
        if curve:
            # 计算采样点数量
            length = last - first
            num_points = max(2, int(length / sample_rate) + 1)
            
            # 采样边上的点
            edge_points = []
            for i in range(num_points):
                param = first + (last - first) * i / (num_points - 1)
                p = curve.Value(param)
                vertex = (p.X(), p.Y(), p.Z())
                
                if vertex not in vertex_map:
                    vertex_map[vertex] = len(all_vertices)
                    all_vertices.append(vertex)
                
                edge_points.append(vertex_map[vertex])
            
            # 创建VTK线段
            for i in range(len(edge_points) - 1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, edge_points[i])
                line.GetPointIds().SetId(1, edge_points[i + 1])
                lines.InsertNextCell(line)
        
        edge_explorer.Next()
    
    # 添加顶点到VTK点集
    for vertex in all_vertices:
        points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
    
    # 创建PolyData对象
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    
    return polydata


def shape_vertices_to_vtk_polydata(shape: TopoDS_Shape) -> vtk.vtkPolyData:
    """
    将TopoDS_Shape的顶点转换为VTK PolyData（仅顶点）
    
    Args:
        shape: OpenCASCADE形状
        
    Returns:
        vtk.vtkPolyData: VTK多边形数据对象（仅包含顶点）
    """
    # 创建VTK点集
    points = vtk.vtkPoints()
    
    # 遍历所有顶点
    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = vertex_explorer.Current()
        p = BRep_Tool.Pnt(vertex)
        points.InsertNextPoint(p.X(), p.Y(), p.Z())
        vertex_explorer.Next()
    
    # 创建PolyData对象
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    
    # 创建顶点单元
    vertices = vtk.vtkCellArray()
    for i in range(points.GetNumberOfPoints()):
        vertex_cell = vtk.vtkVertex()
        vertex_cell.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex_cell)
    
    polydata.SetVerts(vertices)
    
    return polydata


def create_shape_actor(shape: TopoDS_Shape,
                      mesh_quality: float = 1.0,
                      display_mode: str = 'surface',
                      color: Tuple[float, float, float] = (0.8, 0.8, 0.9),
                      opacity: float = 1.0,
                      edge_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                      edge_width: float = 1.0) -> vtk.vtkActor:
    """
    为TopoDS_Shape创建VTK Actor
    
    Args:
        shape: OpenCASCADE形状
        mesh_quality: 网格质量参数
        display_mode: 显示模式 ('surface', 'wireframe', 'points', 'surface_with_edges')
        color: 表面颜色 (R, G, B)
        opacity: 不透明度 (0.0-1.0)
        edge_color: 边框颜色 (R, G, B)
        edge_width: 边框宽度
        
    Returns:
        vtk.vtkActor: VTK Actor对象
    """
    # 根据显示模式创建PolyData
    if display_mode == 'surface':
        polydata = shape_to_vtk_polydata(shape, mesh_quality)
    elif display_mode == 'wireframe':
        polydata = shape_edges_to_vtk_polydata(shape)
    elif display_mode == 'points':
        polydata = shape_vertices_to_vtk_polydata(shape)
    elif display_mode == 'surface_with_edges':
        polydata = shape_to_vtk_polydata(shape, mesh_quality)
    else:
        polydata = shape_to_vtk_polydata(shape, mesh_quality)
    
    # 创建Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    # 创建Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # 设置颜色和属性
    actor.GetProperty().SetColor(color[0], color[1], color[2])
    actor.GetProperty().SetOpacity(opacity)
    
    # 根据显示模式设置属性
    if display_mode == 'wireframe':
        actor.GetProperty().SetRepresentationToWireframe()
    elif display_mode == 'points':
        actor.GetProperty().SetRepresentationToPoints()
        actor.GetProperty().SetPointSize(5.0)
    elif display_mode == 'surface_with_edges':
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(edge_color[0], edge_color[1], edge_color[2])
        actor.GetProperty().SetLineWidth(edge_width)
    
    return actor


def compute_shape_normals(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """
    计算PolyData的法向量
    
    Args:
        polydata: VTK多边形数据对象
        
    Returns:
        vtk.vtkPolyData: 包含法向量的PolyData对象
    """
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.Update()
    
    return normals.GetOutput()


def smooth_shape_mesh(polydata: vtk.vtkPolyData,
                     iterations: int = 20,
                     relaxation_factor: float = 0.1) -> vtk.vtkPolyData:
    """
    平滑PolyData网格
    
    Args:
        polydata: VTK多边形数据对象
        iterations: 迭代次数
        relaxation_factor: 松弛因子
        
    Returns:
        vtk.vtkPolyData: 平滑后的PolyData对象
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    
    return smoother.GetOutput()


def decimate_shape_mesh(polydata: vtk.vtkPolyData,
                       target_reduction: float = 0.5) -> vtk.vtkPolyData:
    """
    简化PolyData网格
    
    Args:
        polydata: VTK多边形数据对象
        target_reduction: 目标简化比例 (0.0-1.0)
        
    Returns:
        vtk.vtkPolyData: 简化后的PolyData对象
    """
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputData(polydata)
    decimator.SetTargetReduction(target_reduction)
    decimator.PreserveTopologyOn()
    decimator.Update()
    
    return decimator.GetOutput()


def create_vertex_actor(vertex: TopoDS_Vertex,
                       color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                       point_size: float = 8.0) -> vtk.vtkActor:
    """
    为顶点创建VTK Actor
    
    Args:
        vertex: OpenCASCADE顶点
        color: 颜色 (R, G, B)
        point_size: 点大小
        
    Returns:
        vtk.vtkActor: VTK Actor对象
    """
    from OCC.Core.BRep import BRep_Tool
    
    p = BRep_Tool.Pnt(vertex)
    
    points = vtk.vtkPoints()
    points.InsertNextPoint(p.X(), p.Y(), p.Z())
    
    vertices = vtk.vtkCellArray()
    vertex_id = vtk.vtkVertex()
    vertex_id.GetPointIds().SetId(0, 0)
    vertices.InsertNextCell(vertex_id)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0], color[1], color[2])
    actor.GetProperty().SetPointSize(point_size)
    
    return actor


def create_edge_actor(edge: TopoDS_Edge,
                     color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                     line_width: float = 2.0,
                     sample_rate: float = 0.1) -> vtk.vtkActor:
    """
    为边创建VTK Actor
    
    Args:
        edge: OpenCASCADE边
        color: 颜色 (R, G, B)
        line_width: 线宽
        sample_rate: 采样率（控制边的采样密度）
        
    Returns:
        vtk.vtkActor: VTK Actor对象
    """
    from OCC.Core.BRep import BRep_Tool
    
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    
    curve, first, last = BRep_Tool.Curve(edge)
    
    if curve:
        length = last - first
        num_points = max(2, int(length / sample_rate) + 1)
        
        edge_points = []
        for i in range(num_points):
            param = first + (last - first) * i / (num_points - 1)
            p = curve.Value(param)
            points.InsertNextPoint(p.X(), p.Y(), p.Z())
            edge_points.append(i)
        
        for i in range(len(edge_points) - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge_points[i])
            line.GetPointIds().SetId(1, edge_points[i + 1])
            lines.InsertNextCell(line)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0], color[1], color[2])
    actor.GetProperty().SetLineWidth(line_width)
    
    return actor


def create_face_actor(face: TopoDS_Face,
                     color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
                     opacity: float = 0.8,
                     mesh_quality: float = 1.0) -> vtk.vtkActor:
    """
    为面创建VTK Actor
    
    Args:
        face: OpenCASCADE面
        color: 颜色 (R, G, B)
        opacity: 不透明度 (0.0-1.0)
        mesh_quality: 网格质量参数
        
    Returns:
        vtk.vtkActor: VTK Actor对象
    """
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, loc)
    
    if triangulation:
        nb_triangles = triangulation.NbTriangles()
        nb_nodes = triangulation.NbNodes()
        
        vertex_map = {}
        all_vertices = []
        
        for i in range(1, nb_nodes + 1):
            p = triangulation.Node(i)
            p.Transform(loc.Transformation())
            vertex = (p.X(), p.Y(), p.Z())
            
            if vertex not in vertex_map:
                vertex_map[vertex] = len(all_vertices)
                all_vertices.append(vertex)
        
        for i in range(1, nb_triangles + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()
            
            idx1 = vertex_map[(triangulation.Node(n1).X(), 
                              triangulation.Node(n1).Y(), 
                              triangulation.Node(n1).Z())]
            idx2 = vertex_map[(triangulation.Node(n2).X(), 
                              triangulation.Node(n2).Y(), 
                              triangulation.Node(n2).Z())]
            idx3 = vertex_map[(triangulation.Node(n3).X(), 
                              triangulation.Node(n3).Y(), 
                              triangulation.Node(n3).Z())]
            
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, idx1)
            triangle.GetPointIds().SetId(1, idx2)
            triangle.GetPointIds().SetId(2, idx3)
            triangles.InsertNextCell(triangle)
        
        for vertex in all_vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0], color[1], color[2])
    actor.GetProperty().SetOpacity(opacity)
    
    return actor


def create_solid_actor(solid: TopoDS_Shape,
                      color: Tuple[float, float, float] = (0.8, 0.8, 0.9),
                      opacity: float = 0.8,
                      mesh_quality: float = 1.0) -> vtk.vtkActor:
    """
    为实体创建VTK Actor
    
    Args:
        solid: OpenCASCADE实体
        color: 颜色 (R, G, B)
        opacity: 不透明度 (0.0-1.0)
        mesh_quality: 网格质量参数
        
    Returns:
        vtk.vtkActor: VTK Actor对象
    """
    polydata = shape_to_vtk_polydata(solid, mesh_quality)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0], color[1], color[2])
    actor.GetProperty().SetOpacity(opacity)
    
    return actor

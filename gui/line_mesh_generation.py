# -*- coding: utf-8 -*-
"""
线网格生成核心模块
提供离散化算法、Connector生成和front2d结构生成功能
"""

import numpy as np
from math import tanh, sqrt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from data_structure.basic_elements import NodeElement, NodeElementALM, Connector, Part
from data_structure.front2d import Front
from utils.geom_toolkit import calculate_distance
from utils.message import error, info, warning, debug


@dataclass
class LineMeshParams:
    """线网格生成参数"""
    method: str = "uniform"  # uniform, geometric, tanh
    num_elements: int = 10
    start_size: float = 0.1
    end_size: float = 0.2
    growth_rate: float = 1.2
    tanh_factor: float = 2.0
    bc_type: str = "wall"
    part_name: str = "default_line"


def generate_discretization_params(
    start_point: Tuple[float, float, float],
    end_point: Tuple[float, float, float],
    params: LineMeshParams
) -> List[float]:
    """
    根据离散化方法生成参数坐标（0到1之间的值）
    
    Args:
        start_point: 起点坐标
        end_point: 终点坐标
        params: 线网格生成参数
        
    Returns:
        参数坐标列表 [0, t1, t2, ..., 1]
    """
    num_points = params.num_elements + 1
    
    if params.method == "uniform":
        return generate_uniform_params(num_points)
    elif params.method == "geometric":
        return generate_geometric_params(
            num_points, 
            params.start_size, 
            params.end_size, 
            params.growth_rate
        )
    elif params.method == "tanh":
        return generate_tanh_params(
            num_points,
            params.start_size,
            params.end_size,
            params.tanh_factor
        )
    else:
        return generate_uniform_params(num_points)


def generate_uniform_params(num_points: int) -> List[float]:
    """生成均匀分布的参数坐标"""
    return [i / (num_points - 1) for i in range(num_points)]


def generate_geometric_params(
    num_points: int,
    start_size: float,
    end_size: float,
    growth_rate: float
) -> List[float]:
    """
    生成几何级数分布的参数坐标
    
    几何级数分布的特点是相邻线段长度按固定比率增长
    """
    if num_points < 2:
        return [0.0, 1.0]
    
    total_length = 0.0
    segment_lengths = []
    
    # 计算各段长度
    for i in range(num_points - 1):
        if i == 0:
            length = start_size
        else:
            length = segment_lengths[-1] * growth_rate
        
        # 检查是否超出剩余长度
        if total_length + length > 1.0:
            break
            
        segment_lengths.append(length)
        total_length += length
    
    # 确保最后一段到达终点
    if total_length < 1.0 and len(segment_lengths) > 0:
        segment_lengths[-1] += (1.0 - total_length)
    
    # 生成参数坐标
    params = [0.0]
    cumulative = 0.0
    for length in segment_lengths:
        cumulative += length
        params.append(min(cumulative, 1.0))
    
    # 确保最后一个点是1.0
    if params[-1] < 1.0:
        params.append(1.0)
    
    return params


def generate_tanh_params(
    num_points: int,
    start_size: float,
    end_size: float,
    tanh_factor: float
) -> List[float]:
    """
    生成Tanh函数分布的参数坐标
    
    Tanh分布的特点是在起点和终点附近网格较密，中间较疏
    通过调整tanh_factor可以控制疏密程度
    """
    if num_points < 2:
        return [0.0, 1.0]
    
    params = []
    
    for i in range(num_points):
        t = i / (num_points - 1)
        
        # 使用tanh函数进行非线性变换
        # 将[0,1]映射到[-a,a]，然后通过tanh变换
        scaled_t = (2 * t - 1) * tanh_factor
        transformed_t = (tanh(scaled_t) + 1) / 2
        
        params.append(transformed_t)
    
    # 归一化到[0,1]
    min_val = min(params)
    max_val = max(params)
    if max_val - min_val > 1e-10:
        params = [(p - min_val) / (max_val - min_val) for p in params]
    
    # 确保端点正好是0和1
    params[0] = 0.0
    params[-1] = 1.0
    
    return params


def discretize_line(
    start_point: Tuple[float, float, float],
    end_point: Tuple[float, float, float],
    params: LineMeshParams,
    curve=None
) -> List[Tuple[float, float, float]]:
    """
    对线段进行离散化
    
    Args:
        start_point: 起点坐标
        end_point: 终点坐标
        params: 线网格生成参数
        curve: 几何曲线对象（可选，如果提供则将点投影到曲线上）
        
    Returns:
        离散点列表 [(x1, y1, z1), (x2, y2, z2), ...]
    """
    # 使用现成的函数生成 [0, 1] 范围内的参数坐标
    param_coords = generate_discretization_params(start_point, end_point, params)
    
    # 如果提供了几何曲线，使用参数坐标映射进行离散化
    if curve is not None:
        try:
            curve_first = curve.FirstParameter()
            curve_last = curve.LastParameter()
            curve_length = curve_last - curve_first
            
            if curve_length < 1e-10:
                raise ValueError("曲线参数范围过小")
            
            # 将 [0, 1] 范围的参数坐标映射到曲线参数范围
            curve_param_coords = [curve_first + t * curve_length for t in param_coords]
            
            # 使用参数坐标在曲线上采样点
            points = []
            for t in curve_param_coords:
                point = curve.Value(t)
                points.append((point.X(), point.Y(), point.Z()))
            
            return points
            
        except Exception as e:
            warning(f"曲线离散化失败，使用线性插值: {e}")
            # 如果曲线离散化失败，回退到线性插值
            start = np.array(start_point)
            end = np.array(end_point)
            
            points = []
            for t in param_coords:
                point = start + t * (end - start)
                points.append(tuple(point))
            
            return points
    else:
        error("无法进行离散化：未提供几何曲线对象")
        return []


def create_fronts_from_points(
    points: List[Tuple[float, float, float]],
    bc_type: str,
    part_name: str
) -> List[Front]:
    """
    从离散点创建Front对象列表
    
    Args:
        points: 离散点列表
        bc_type: 边界类型
        part_name: 部件名称
        
    Returns:
        Front对象列表
    """
    if len(points) < 2:
        return []
    
    fronts = []
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        
        node1 = NodeElementALM(
            coords=p1,
            idx=-1,
            bc_type=bc_type,
            part_name=part_name
        )
        node2 = NodeElementALM(
            coords=p2,
            idx=-1,
            bc_type=bc_type,
            part_name=part_name
        )
        
        front = Front(
            node_elem1=node1,
            node_elem2=node2,
            idx=i,
            bc_type=bc_type,
            part_name=part_name
        )
        
        fronts.append(front)
    
    return fronts


def create_connector_from_edge(
    edge_info: Dict,
    params: LineMeshParams
) -> Connector:
    """
    从几何边信息创建Connector
    
    Args:
        edge_info: 边信息字典（包含start_point, end_point, curve等）
        params: 线网格生成参数
        
    Returns:
        Connector对象
    """
    start_point = edge_info['start_point']
    end_point = edge_info['end_point']
    curve = edge_info.get('curve')
    
    debug(f"create_connector_from_edge: curve 类型 = {type(curve)}")
    
    valid_curve = False
    if curve is not None:
        try:
            if hasattr(curve, 'FirstParameter'):
                valid_curve = True
                debug("curve 是有效的几何曲线对象")
            else:
                warning("curve 没有 FirstParameter 方法，将使用线性插值")
        except Exception as e:
            error(f"检查 curve 类型时出错: {e}")
    
    # 离散化线段
    points = discretize_line(start_point, end_point, params, curve=curve if valid_curve else None)
    
    # 创建Front列表
    fronts = create_fronts_from_points(points, params.bc_type, params.part_name)
    
    # 创建Connector参数
    from data_structure.parameters import MeshParameters
    connector_params = MeshParameters(
        part_name=params.part_name,
        max_size=1e6,
        PRISM_SWITCH="off"
    )
    
    # 创建Connector
    connector = Connector(
        part_name=params.part_name,
        curve_name=edge_info.get('name', 'default'),
        param=connector_params,
        cad_obj=edge_info
    )
    connector.front_list = fronts
    
    return connector


def create_part_from_connectors(
    part_name: str,
    connectors: List[Connector]
) -> Part:
    """
    从Connector列表创建Part
    
    Args:
        part_name: 部件名称
        connectors: Connector列表
        
    Returns:
        Part对象
    """
    from data_structure.parameters import MeshParameters
    
    part_params = MeshParameters(
        part_name=part_name,
        max_size=0.1,
        PRISM_SWITCH="off"
    )
    
    part = Part(part_name, part_params, connectors)
    part.init_part_front_list()
    part.sync_connector_params()
    
    return part


def generate_line_mesh(
    edges_info: List[Dict],
    params: LineMeshParams
) -> Tuple[List[Connector], List[Part]]:
    """
    生成线网格
    
    Args:
        edges_info: 边信息列表
        params: 线网格生成参数
        
    Returns:
        (Connector列表, Part列表)
    """
    connectors = []
    
    for idx, edge_info in enumerate(edges_info):
        edge_params = LineMeshParams(
            method=params.method,
            num_elements=params.num_elements,
            start_size=params.start_size,
            end_size=params.end_size,
            growth_rate=params.growth_rate,
            tanh_factor=params.tanh_factor,
            bc_type=params.bc_type,
            part_name=f"{params.part_name}_{idx}"
        )
        
        connector = create_connector_from_edge(edge_info, edge_params)
        connectors.append(connector)
    
    # 按部件名称分组
    part_dict = {}
    for conn in connectors:
        if conn.part_name not in part_dict:
            part_dict[conn.part_name] = []
        part_dict[conn.part_name].append(conn)
    
    # 创建Part列表
    parts = []
    for part_name, conn_list in part_dict.items():
        part = create_part_from_connectors(part_name, conn_list)
        parts.append(part)
    
    return connectors, parts


def convert_connectors_to_unstructured_grid(
    connectors: List[Connector],
    grid_dimension: int = 2
) -> 'Unstructured_Grid':
    """
    将Connector列表转换为Unstructured_Grid对象
    
    Args:
        connectors: Connector列表
        grid_dimension: 网格维度（2或3）
        
    Returns:
        Unstructured_Grid对象
    """
    from data_structure.unstructured_grid import Unstructured_Grid, GenericCell
    from data_structure.basic_elements import NodeElement
    
    # 收集所有节点
    node_map = {}
    node_coords = []
    node_idx = 0
    
    for conn in connectors:
        if hasattr(conn, 'front_list') and conn.front_list:
            for front in conn.front_list:
                if hasattr(front, 'node_elems') and len(front.node_elems) >= 2:
                    for node_elem in front.node_elems:
                        if hasattr(node_elem, 'coords'):
                            # 确保 coords 是数值列表，而不是 numpy 类型对象
                            coords = tuple(float(x) for x in node_elem.coords)
                            if coords not in node_map:
                                node_map[coords] = node_idx
                                node_coords.append([float(x) for x in coords])
                                node_idx += 1
    
    # 创建边界节点
    boundary_nodes = []
    for idx, coords in enumerate(node_coords):
        boundary_nodes.append(NodeElement(coords, idx, bc_type="wall"))
    
    # 创建线段单元（使用GenericCell）
    cell_container = []
    cell_idx = 0
    
    for conn in connectors:
        if hasattr(conn, 'front_list') and conn.front_list:
            for front in conn.front_list:
                if hasattr(front, 'node_elems') and len(front.node_elems) >= 2:
                    node1 = front.node_elems[0]
                    node2 = front.node_elems[1]
                    
                    if hasattr(node1, 'coords') and hasattr(node2, 'coords'):
                        coords1 = tuple(float(x) for x in node1.coords)
                        coords2 = tuple(float(x) for x in node2.coords)
                        
                        if coords1 in node_map and coords2 in node_map:
                            node_id1 = node_map[coords1]
                            node_id2 = node_map[coords2]
                            
                            # 创建线段单元（使用GenericCell，只保存节点索引）
                            line_segment = GenericCell(
                                node_ids=[node_id1, node_id2],
                                part_name=conn.part_name,
                                idx=cell_idx
                            )
                            cell_container.append(line_segment)
                            cell_idx += 1
    
    # 创建Unstructured_Grid
    grid = Unstructured_Grid(
        cell_container=cell_container,
        node_coords=node_coords,
        boundary_nodes=boundary_nodes,
        grid_dimension=grid_dimension,
        mesh_type="line_mesh"
    )
    
    # 设置部件信息
    parts_info = {}
    for conn in connectors:
        if conn.part_name not in parts_info:
            parts_info[conn.part_name] = {
                'part_name': conn.part_name,
                'connectors': []
            }
        parts_info[conn.part_name]['connectors'].append(conn.curve_name)
    
    grid.parts_info = parts_info
    
    # 设置边界信息
    boundary_info = {}
    for conn in connectors:
        if conn.part_name not in boundary_info:
            boundary_info[conn.part_name] = {
                'bc_type': 'wall',
                'faces': []
            }
    grid.boundary_info = boundary_info
    
    return grid


def get_all_fronts_from_parts(parts: List[Part]) -> List[Front]:
    """
    从Part列表获取所有Front
    
    Args:
        parts: Part列表
        
    Returns:
        所有Front的列表
    """
    all_fronts = []
    for part in parts:
        all_fronts.extend(part.front_list)
    return all_fronts


def extract_edge_info_from_geometry(geometry_obj) -> List[Dict]:
    """
    从几何对象提取边信息
    
    Args:
        geometry_obj: 几何对象（OCC形状或其他）
        
    Returns:
        边信息列表
    """
    edges_info = []
    
    if geometry_obj is None:
        return edges_info
    
    try:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core.BRep import BRep_Tool
        
        explorer = TopExp_Explorer(geometry_obj, TopAbs_EDGE)
        
        idx = 0
        while explorer.More():
            edge = explorer.Current()
            
            # 获取边的几何信息
            curve = BRep_Tool.Curve(edge)
            if curve:
                from OCC.Core.GProp import GProp_GProps
                from OCC.Core.BRepGProp import brepgprop
                
                gprop = GProp_GProps()
                brepgprop.LinearProperties(edge, gprop)
                length = gprop.Mass()
                
                # 获取端点
                from OCC.Core.TopoDS import TopoDS_Vertex
                from OCC.Core.TopExp import TopExp_Explorer as TopExp_Explorer_V
                from OCC.Core.TopAbs import TopAbs_VERTEX
                
                v_explorer = TopExp_Explorer_V(edge, TopAbs_VERTEX)
                
                vertices = []
                while v_explorer.More():
                    v = TopoDS_Vertex(v_explorer.Current())
                    pnt = BRep_Tool.Pnt(v)
                    vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
                    v_explorer.Next()
                
                if len(vertices) >= 2:
                    edge_info = {
                        'idx': idx,
                        'curve': curve,
                        'length': length,
                        'start_point': vertices[0],
                        'end_point': vertices[-1],
                        'name': f"edge_{idx}"
                    }
                    edges_info.append(edge_info)
                    info(f"提取边 {idx}: 起点 {vertices[0]}, 终点 {vertices[-1]}, 长度 {length:.4f}")
                    idx += 1
            
            explorer.Next()
            
    except ImportError:
        pass
    
    return edges_info

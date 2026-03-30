"""
三维曲面阵面数据结构

定义用于曲面网格生成的阵面、节点和单元数据结构
"""
import heapq
import numpy as np
from typing import List, Tuple, Optional, Any
from data_structure.basic_elements import is_node_element


class NodeElement3D:
    """
    三维节点元素
    
    用于曲面网格生成的节点数据结构
    """
    __slots__ = [
        'coords',
        'idx',
        'bc_type',
        'part_name',
        'surface',
        'uv_params',
        'hash',
        'normal',
        'bbox',
    ]
    
    def __init__(
        self,
        coords: Tuple[float, float, float],
        idx: int = -1,
        bc_type: str = None,
        part_name: str = None,
        surface: Any = None,
        uv_params: Tuple[float, float] = None,
        normal: Tuple[float, float, float] = None
    ):
        """
        初始化三维节点
        
        Args:
            coords: 节点坐标
            idx: 节点索引
            bc_type: 边界类型
            part_name: 所属部件名称
            surface: 所属曲面 (OCC TopoDS_Face)
            uv_params: 曲面参数坐标
            normal: 节点法向量
        """
        self.coords = coords
        self.idx = idx
        self.bc_type = bc_type
        self.part_name = part_name
        self.surface = surface
        self.uv_params = uv_params
        self.normal = normal
        
        coords_hash = hash(tuple(f"{c:.8f}" for c in coords))
        self.hash = coords_hash
        
        self.bbox = (coords[0], coords[1], coords[2], coords[0], coords[1], coords[2])
    
    def __hash__(self):
        return self.hash
    
    def __eq__(self, other):
        if not isinstance(other, NodeElement3D):
            return False
        return self.hash == other.hash
    
    def __repr__(self):
        return f"NodeElement3D(idx={self.idx}, coords={self.coords})"


class SurfaceTriangle:
    """
    曲面三角形单元
    
    用于曲面网格的三角形单元
    """
    __slots__ = [
        'nodes',
        'node_ids',
        'idx',
        'surface',
        'normal',
        'area',
        'quality',
        'hash',
        'bbox',
    ]
    
    def __init__(
        self,
        node1: NodeElement3D,
        node2: NodeElement3D,
        node3: NodeElement3D,
        surface: Any = None,
        idx: int = -1
    ):
        """
        初始化曲面三角形
        
        Args:
            node1, node2, node3: 三个顶点节点
            surface: 所属曲面
            idx: 单元索引
        """
        self.nodes = [node1, node2, node3]
        self.node_ids = [node1.idx, node2.idx, node3.idx]
        self.surface = surface
        self.idx = idx
        
        self.normal = self._compute_normal()
        self.area = self._compute_area()
        self.quality = self._compute_quality()
        self.bbox = self._compute_bbox()
        
        self.hash = hash(tuple(sorted(self.node_ids)))
    
    def _compute_normal(self) -> np.ndarray:
        """计算三角形法向量"""
        p1 = np.array(self.nodes[0].coords)
        p2 = np.array(self.nodes[1].coords)
        p3 = np.array(self.nodes[2].coords)
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        
        return normal
    
    def _compute_area(self) -> float:
        """计算三角形面积"""
        p1 = np.array(self.nodes[0].coords)
        p2 = np.array(self.nodes[1].coords)
        p3 = np.array(self.nodes[2].coords)
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        return 0.5 * np.linalg.norm(np.cross(v1, v2))
    
    def _compute_quality(self) -> float:
        """
        计算三角形质量（形状因子）
        
        使用面积与边长平方和的比值
        质量范围: 0 ~ 1，1 表示等边三角形
        """
        p1 = np.array(self.nodes[0].coords)
        p2 = np.array(self.nodes[1].coords)
        p3 = np.array(self.nodes[2].coords)
        
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        
        s = (a + b + c) / 2.0
        
        if s < 1e-12:
            return 0.0
        
        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
        
        sum_sq = a * a + b * b + c * c
        if sum_sq < 1e-12:
            return 0.0
        
        quality = 4.0 * np.sqrt(3) * area / sum_sq
        
        return min(1.0, max(0.0, quality))
    
    def _compute_bbox(self) -> Tuple[float, float, float, float, float, float]:
        """计算三角形边界框"""
        p1 = self.nodes[0].coords
        p2 = self.nodes[1].coords
        p3 = self.nodes[2].coords
        
        min_x = min(p1[0], p2[0], p3[0])
        max_x = max(p1[0], p2[0], p3[0])
        min_y = min(p1[1], p2[1], p3[1])
        max_y = max(p1[1], p2[1], p3[1])
        min_z = min(p1[2], p2[2], p3[2])
        max_z = max(p1[2], p2[2], p3[2])
        
        return (min_x, min_y, min_z, max_x, max_y, max_z)
    
    def __hash__(self):
        return self.hash
    
    def __eq__(self, other):
        if not isinstance(other, SurfaceTriangle):
            return False
        return self.hash == other.hash
    
    def __repr__(self):
        return f"SurfaceTriangle(idx={self.idx}, nodes={self.node_ids}, quality={self.quality:.4f})"


class SurfaceFront:
    """
    三维曲面阵面
    
    用于曲面网格生成的阵面数据结构
    表示曲面上的一个边界边
    """
    __slots__ = [
        'node_elems',
        'idx',
        'bc_type',
        'part_name',
        'surface',
        'priority',
        'al',
        'center',
        'length',
        'direction',
        'normal',
        'tangent_normal',
        'bbox',
        'hash',
        'node_ids',
        'left_triangle',
        'right_triangle',
    ]
    
    def __init__(
        self,
        node_elem1: NodeElement3D,
        node_elem2: NodeElement3D,
        surface: Any = None,
        idx: int = None,
        bc_type: str = None,
        part_name: str = None,
        al: float = 3.0
    ):
        """
        初始化曲面阵面
        
        Args:
            node_elem1, node_elem2: 阵面两端节点
            surface: 所属曲面
            idx: 阵面ID
            bc_type: 边界类型
            part_name: 所属部件
            al: 搜索范围系数
        """
        self.node_elems = [node_elem1, node_elem2]
        self.surface = surface
        self.idx = idx
        self.bc_type = bc_type
        self.part_name = part_name
        
        self.priority = False
        self.al = al
        self.left_triangle = None
        self.right_triangle = None
        
        node1 = node_elem1.coords
        node2 = node_elem2.coords
        self.node_ids = [node_elem1.idx, node_elem2.idx]
        
        self.length = np.linalg.norm(np.array(node2) - np.array(node1))
        
        if self.length < 1e-12:
            raise ValueError("阵面两端节点不能重合")
        
        self.center = tuple((a + b) / 2 for a, b in zip(node1, node2))
        
        direction = np.array(node2) - np.array(node1)
        self.direction = tuple(direction / self.length)
        
        n1 = node_elem1.normal
        n2 = node_elem2.normal
        if n1 is not None and n2 is not None:
            avg_normal = np.array(n1) + np.array(n2)
            norm = np.linalg.norm(avg_normal)
            if norm > 1e-12:
                self.normal = tuple(avg_normal / norm)
            else:
                self.normal = (0.0, 0.0, 1.0)
        else:
            self.normal = (0.0, 0.0, 1.0)
        
        self._compute_tangent_normal()
        
        self._compute_bbox()
        
        length_hash = hash(f"{self.length:.6f}")
        center_hash = hash(tuple(f"{c:.6f}" for c in self.center))
        self.hash = hash((center_hash, length_hash))
    
    def _compute_tangent_normal(self):
        """计算切平面内的法向（用于理想点计算）"""
        direction = np.array(self.direction)
        normal = np.array(self.normal)
        
        tangent_normal = np.cross(direction, normal)
        norm = np.linalg.norm(tangent_normal)
        if norm > 1e-12:
            tangent_normal = tangent_normal / norm
        else:
            tangent_normal = np.array([0.0, 0.0, 1.0])
        
        self.tangent_normal = tuple(tangent_normal)
    
    def _compute_bbox(self):
        """计算边界框"""
        node1 = self.node_elems[0].coords
        node2 = self.node_elems[1].coords
        
        min_x = min(node1[0], node2[0])
        max_x = max(node1[0], node2[0])
        min_y = min(node1[1], node2[1])
        max_y = max(node1[1], node2[1])
        min_z = min(node1[2], node2[2])
        max_z = max(node1[2], node2[2])
        
        self.bbox = (min_x, min_y, min_z, max_x, max_y, max_z)
    
    def __lt__(self, other):
        """优先队列比较：优先级高的在前，其次长度短的在前"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.length < other.length
    
    def __eq__(self, other):
        return self.hash == other.hash
    
    def __hash__(self):
        return self.hash
    
    def __repr__(self):
        return f"SurfaceFront(idx={self.idx}, nodes={self.node_ids}, length={self.length:.4f})"


def create_initial_fronts_from_surface(
    surface,
    geometry_handler,
    sizing_field,
    bc_type: str = "wall",
    part_name: str = "default"
) -> List[SurfaceFront]:
    """
    从曲面边界创建初始阵面
    
    Args:
        surface: OCC TopoDS_Face
        geometry_handler: 曲面几何操作对象
        sizing_field: 尺寸场
        bc_type: 边界类型
        part_name: 部件名称
    
    Returns:
        初始阵面列表
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.Geom import Geom_Curve
    
    fronts = []
    node_dict = {}
    node_idx = 0
    
    explorer = TopExp_Explorer(surface, TopAbs_WIRE)
    wires = []
    while explorer.More():
        wires.append(explorer.Current())
        explorer.Next()
    
    for wire in wires:
        edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        edges = []
        while edge_explorer.More():
            edges.append(edge_explorer.Current())
            edge_explorer.Next()
        
        for edge in edges:
            curve_handle, first, last = BRep_Tool.Curve(edge)
            
            if curve_handle is None:
                continue
            
            # BRep_Tool.Curve() 返回的已经是 Geom_Curve 对象，不需要 GetObject()
            curve = curve_handle
            
            num_points = max(2, int((last - first) / sizing_field.global_spacing))
            
            prev_node = None
            for i in range(num_points + 1):
                t = first + i * (last - first) / num_points
                
                pnt = curve.Value(t)
                coords = (pnt.X(), pnt.Y(), pnt.Z())
                
                coords_key = tuple(f"{c:.6f}" for c in coords)
                if coords_key in node_dict:
                    current_node = node_dict[coords_key]
                else:
                    uv = geometry_handler.project_point_to_surface(coords, surface)
                    normal = geometry_handler.get_surface_normal(uv[0], uv[1], surface)
                    
                    current_node = NodeElement3D(
                        coords=coords,
                        idx=node_idx,
                        bc_type=bc_type,
                        part_name=part_name,
                        surface=surface,
                        uv_params=uv,
                        normal=normal
                    )
                    node_dict[coords_key] = current_node
                    node_idx += 1
                
                if prev_node is not None and prev_node.idx != current_node.idx:
                    front = SurfaceFront(
                        node_elem1=prev_node,
                        node_elem2=current_node,
                        surface=surface,
                        idx=len(fronts),
                        bc_type=bc_type,
                        part_name=part_name
                    )
                    fronts.append(front)
                
                prev_node = current_node
    
    return fronts

"""
三维曲面阵面数据结构

定义用于曲面网格生成的阵面、节点和单元数据结构
"""
import heapq
import math
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
        
        coords_hash = hash(tuple(f"{0.0 if round(c, 8) == 0 else round(c, 8):.8f}" for c in coords))
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
        """计算切平面内的推进方向（用于理想点计算）

        在阵面中点处计算曲面法向，再与边方向叉乘，得到切平面内
        垂直于边的单位向量。
        """
        direction = np.array(self.direction)

        # 在阵面中点处计算曲面法向
        mid_uv = self._compute_midpoint_surface_normal()

        tangent_normal = np.cross(direction, mid_uv)
        norm = np.linalg.norm(tangent_normal)
        if norm > 1e-12:
            tangent_normal = tangent_normal / norm
        else:
            tangent_normal = np.array([0.0, 0.0, 1.0])

        self.tangent_normal = tuple(tangent_normal)

    def _compute_midpoint_surface_normal(self):
        """计算阵面中点处的曲面法向量"""
        mid = self.center
        if self.surface is not None:
            try:
                from .surface_geometry import SurfaceGeometry
                geom = SurfaceGeometry()
                uv = geom.project_point_to_surface(mid, self.surface)
                normal = geom.get_surface_normal(uv[0], uv[1], self.surface)
                n = np.array(normal)
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    return n / norm
            except Exception:
                pass

        # 回退：使用两端节点法向平均
        n1 = self.node_elems[0].normal
        n2 = self.node_elems[1].normal
        if n1 is not None and n2 is not None:
            avg = np.array(n1) + np.array(n2)
            norm = np.linalg.norm(avg)
            if norm > 1e-12:
                return avg / norm
        return np.array([0.0, 0.0, 1.0])
    
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

            # 通过采样估算物理弧长，避免参数空间与物理空间的不一致
            n_samples = max(20, int(abs(last - first) * 2))
            arc_length = 0.0
            prev_pt = curve.Value(first)
            for s in range(1, n_samples + 1):
                t = first + s * (last - first) / n_samples
                pt = curve.Value(t)
                dx = pt.X() - prev_pt.X()
                dy = pt.Y() - prev_pt.Y()
                dz = pt.Z() - prev_pt.Z()
                arc_length += (dx * dx + dy * dy + dz * dz) ** 0.5
                prev_pt = pt

            num_points = max(2, int(arc_length / sizing_field.global_spacing))
            
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
    
    # 修复：确保所有初始阵面的方向一致（曲面在阵面左侧）
    fronts = _fix_front_orientation(fronts, surface, geometry_handler)
    
    return fronts


def _fix_front_orientation(
    fronts: List[SurfaceFront],
    surface,
    geometry_handler
) -> List[SurfaceFront]:
    """
    修复阵面方向，确保所有阵面的方向一致（曲面在阵面左侧）
    
    对于每个阵面，检查其方向是否正确。正确的方向应该使得：
    cross(阵面方向, 曲面法向) 指向曲面内部
    
    即：如果曲面法向朝上 [0,0,1]，阵面方向应该使得内部在左侧。
    
    Args:
        fronts: 初始阵面列表
        surface: 曲面
        geometry_handler: 几何处理器
        
    Returns:
        方向修正后的阵面列表
    """
    if not fronts:
        return fronts
    
    fixed_fronts = []
    
    for front in fronts:
        # 获取阵面中点处的曲面法向
        mid = np.array(front.center)
        try:
            uv = geometry_handler.project_point_to_surface(tuple(mid), surface)
            surface_normal = np.array(geometry_handler.get_surface_normal(uv[0], uv[1], surface))
        except Exception:
            surface_normal = np.array([0.0, 0.0, 1.0])
        
        # 归一化曲面法向
        norm = np.linalg.norm(surface_normal)
        if norm > 1e-12:
            surface_normal = surface_normal / norm
        
        # 计算当前阵面的切向推进方向
        current_tangent = np.array(front.tangent_normal)
        
        # 计算期望的推进方向：cross(阵面方向, 曲面法向)
        # 根据右手定则，这样可以保证推进方向指向曲面内部
        direction = np.array(front.direction)
        expected_tangent = np.cross(direction, surface_normal)
        norm = np.linalg.norm(expected_tangent)
        if norm > 1e-12:
            expected_tangent = expected_tangent / norm
        
        # 检查当前方向是否与期望方向一致
        dot = np.dot(current_tangent, expected_tangent)
        
        if dot < 0:
            # 方向相反，需要翻转阵面
            fixed_front = SurfaceFront(
                node_elem1=front.node_elems[1],
                node_elem2=front.node_elems[0],
                surface=surface,
                idx=front.idx,
                bc_type=front.bc_type,
                part_name=front.part_name
            )
            fixed_fronts.append(fixed_front)
        else:
            fixed_fronts.append(front)
    
    return fixed_fronts


def _compute_edge_key(edge, face=None) -> str:
    """
    计算 OCC 边的几何键，用于跨面识别同一条边

    使用曲线类型、起点、终点和中点构成键，方向无关（中点不随遍历方向改变）。
    当 3D 曲线为空时（如闭合曲面的退化缝合边），回退到 pcurve + 曲面参数化。

    Args:
        edge: OCC TopoDS_Edge
        face: OCC TopoDS_Face（可选，用于 NULL 3D 曲线回退）

    Returns:
        边几何键字符串，无法获取曲线时返回 None
    """
    from OCC.Core.BRep import BRep_Tool

    curve_handle, first, last = BRep_Tool.Curve(edge)
    if curve_handle is not None:
        curve = curve_handle
        type_name = curve.DynamicType().Name()
        p1 = curve.Value(first)
        p2 = curve.Value(last)
        mid = curve.Value((first + last) / 2.0)
        key = (
            type_name,
            round(p1.X(), 6), round(p1.Y(), 6), round(p1.Z(), 6),
            round(p2.X(), 6), round(p2.Y(), 6), round(p2.Z(), 6),
            round(mid.X(), 6), round(mid.Y(), 6), round(mid.Z(), 6),
        )
        return str(key)

    # 回退：通过 pcurve + 曲面参数化生成键
    if face is None:
        return None
    pcurve, srf, loc = BRep_Tool.CurveOnSurface(edge, face)
    if pcurve is None:
        return None
    p_first, p_last = pcurve.FirstParameter(), pcurve.LastParameter()
    if abs(p_last - p_first) < 1e-12:
        return None
    surface = BRep_Tool.Surface(face)
    if surface is None:
        return None
    # 处理无限参数范围：使用固定采样点
    if abs(p_last - p_first) > 1e10:
        uv1 = pcurve.Value(0.0)
        uv2 = pcurve.Value(math.pi)
        uv_mid = pcurve.Value(math.pi / 2.0)
    else:
        uv1 = pcurve.Value(p_first)
        uv2 = pcurve.Value(p_last)
        uv_mid = pcurve.Value((p_first + p_last) / 2.0)
    p1 = surface.Value(uv1.X(), uv1.Y())
    p2 = surface.Value(uv2.X(), uv2.Y())
    mid = surface.Value(uv_mid.X(), uv_mid.Y())
    srf_type = surface.DynamicType().Name()
    key = (
        srf_type + "_pcurve",
        round(p1.X(), 6), round(p1.Y(), 6), round(p1.Z(), 6),
        round(p2.X(), 6), round(p2.Y(), 6), round(p2.Z(), 6),
        round(mid.X(), 6), round(mid.Y(), 6), round(mid.Z(), 6),
    )
    return str(key)


def discretize_shape_edges(shape, sizing_field) -> dict:
    """
    离散化形状中所有唯一几何边，生成共享线网格

    遍历形状中的所有边，每条几何边只离散一次（通过边几何键去重）。
    共享顶点通过坐标字符串键去重，确保跨面一致性。

    Args:
        shape: OCC TopoDS_Shape
        sizing_field: 尺寸场（需提供 global_spacing 属性）

    Returns:
        Dict[str, List[dict]] — 边几何键 → 节点列表，每个 dict 含 'coords' 和 'idx'
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.BRep import BRep_Tool

    spacing = sizing_field.global_spacing
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)

    line_mesh = {}       # edge_key → [{'coords': ..., 'idx': ...}, ...]
    node_dict = {}       # coords_key → idx
    global_idx = 0

    # 提取面和曲面（用于 NULL 3D 曲线回退）
    face = None
    surface = None
    from OCC.Core.TopExp import TopExp_Explorer as _TE2
    from OCC.Core.TopAbs import TopAbs_FACE as _TA_FACE
    _fe = _TE2(shape, _TA_FACE)
    if _fe.More():
        from OCC.Core.BRep import BRep_Tool as _BT
        face = _fe.Current()
        surface = _BT.Surface(face)

    while edge_explorer.More():
        edge = edge_explorer.Current()
        edge_key = _compute_edge_key(edge, face)
        edge_explorer.Next()

        if edge_key is None or edge_key in line_mesh:
            continue

        curve_handle, first, last = BRep_Tool.Curve(edge)

        # 回退：当 3D 曲线为空时，通过 pcurve + 曲面参数化重建 3D 坐标
        if curve_handle is None:
            if surface is None:
                continue
            pcurve, srf, loc = BRep_Tool.CurveOnSurface(edge, face)
            if pcurve is None:
                continue
            p_first, p_last = pcurve.FirstParameter(), pcurve.LastParameter()
            dp = abs(p_last - p_first)
            if dp < 1e-12:
                continue
            # 处理无限参数范围（闭合曲面上的闭合曲线）
            if dp > 1e10:
                # 闭合曲线：自适应采样检测周期
                n_probe = 72
                p0 = pcurve.Value(0.0)
                pt0 = surface.Value(p0.X(), p0.Y())
                period = None
                for s in range(1, n_probe + 1):
                    t = s * math.pi / 18.0
                    uv = pcurve.Value(t)
                    pt = surface.Value(uv.X(), uv.Y())
                    d = ((pt.X()-pt0.X())**2 + (pt.Y()-pt0.Y())**2 + (pt.Z()-pt0.Z())**2)**0.5
                    if d < 1e-6 and t > 0.1:
                        period = t
                        break
                if period is None:
                    continue
                p_first_use, p_last_use = 0.0, period
            else:
                p_first_use, p_last_use = p_first, p_last
            n_samp = max(20, int(abs(p_last_use - p_first_use) * 10))
            pts_3d = []
            for s in range(n_samp + 1):
                t = p_first_use + s * (p_last_use - p_first_use) / n_samp
                uv = pcurve.Value(t)
                pnt = surface.Value(uv.X(), uv.Y())
                pts_3d.append((pnt.X(), pnt.Y(), pnt.Z()))
            arc_length = 0.0
            for s in range(1, len(pts_3d)):
                dx = pts_3d[s][0] - pts_3d[s - 1][0]
                dy = pts_3d[s][1] - pts_3d[s - 1][1]
                dz = pts_3d[s][2] - pts_3d[s - 1][2]
                arc_length += (dx * dx + dy * dy + dz * dz) ** 0.5
            num_points = max(2, int(arc_length / spacing))
            # 闭合曲线：去掉末尾重复节点
            if dp > 1e10:
                num_points = max(2, num_points)
            edge_nodes = []
            for i in range(num_points + 1):
                idx_f = i * (len(pts_3d) - 1) / num_points
                lo = int(idx_f)
                hi = min(lo + 1, len(pts_3d) - 1)
                frac = idx_f - lo
                coords = tuple(pts_3d[lo][k] + frac * (pts_3d[hi][k] - pts_3d[lo][k]) for k in range(3))
                coords_key = tuple(f"{c:.6f}" for c in coords)
                if coords_key in node_dict:
                    idx = node_dict[coords_key]
                else:
                    idx = global_idx
                    node_dict[coords_key] = idx
                    global_idx += 1
                edge_nodes.append({'coords': coords, 'idx': idx})
            # 跳过退化边（弧长接近零，例如球面极点处的边）
            if arc_length < 1e-8:
                continue
            line_mesh[edge_key] = edge_nodes
            continue

        curve = curve_handle

        # 弧长估算
        n_samples = max(20, int(abs(last - first) * 2))
        arc_length = 0.0
        prev_pt = curve.Value(first)
        for s in range(1, n_samples + 1):
            t = first + s * (last - first) / n_samples
            pt = curve.Value(t)
            dx = pt.X() - prev_pt.X()
            dy = pt.Y() - prev_pt.Y()
            dz = pt.Z() - prev_pt.Z()
            arc_length += (dx * dx + dy * dy + dz * dz) ** 0.5
            prev_pt = pt

        num_points = max(2, int(arc_length / spacing))

        edge_nodes = []
        for i in range(num_points + 1):
            t = first + i * (last - first) / num_points
            pnt = curve.Value(t)
            coords = (pnt.X(), pnt.Y(), pnt.Z())
            coords_key = tuple(f"{c:.6f}" for c in coords)

            if coords_key in node_dict:
                idx = node_dict[coords_key]
            else:
                idx = global_idx
                node_dict[coords_key] = idx
                global_idx += 1

            edge_nodes.append({'coords': coords, 'idx': idx})

        # 跳过退化边（弧长接近零，例如球面极点处的边）
        if arc_length < 1e-8:
            continue
        line_mesh[edge_key] = edge_nodes

    return line_mesh


def create_fronts_from_line_mesh(
    face,
    geometry_handler,
    line_mesh: dict,
    bc_type: str = "wall",
    part_name: str = "default",
) -> list:
    """
    从预离散线网格创建面的初始阵面

    使用线网格中的预离散节点，为每个面创建带有正确 UV 和法向量的
    NodeElement3D 和 SurfaceFront。

    Args:
        face: OCC TopoDS_Face
        geometry_handler: SurfaceGeometry 实例
        line_mesh: discretize_shape_edges 返回的线网格
        bc_type: 边界类型
        part_name: 部件名称

    Returns:
        List[SurfaceFront]
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
    from OCC.Core.BRep import BRep_Tool

    fronts = []
    created_nodes = {}  # idx → NodeElement3D（面内去重）

    def _get_or_create_node(node_info):
        """获取或创建面专属的 NodeElement3D"""
        idx = node_info['idx']
        if idx in created_nodes:
            return created_nodes[idx]

        coords = node_info['coords']
        uv = geometry_handler.project_point_to_surface(coords, face)
        normal = geometry_handler.get_surface_normal(uv[0], uv[1], face)

        node = NodeElement3D(
            coords=coords,
            idx=idx,
            bc_type=bc_type,
            part_name=part_name,
            surface=face,
            uv_params=uv,
            normal=normal,
        )
        created_nodes[idx] = node
        return node

    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while wire_explorer.More():
        wire = wire_explorer.Current()
        wire_explorer.Next()

        # 收集边并按端点连通性排序
        edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        raw_edges = []
        _surf = BRep_Tool.Surface(face)
        while edge_explorer.More():
            edge = edge_explorer.Current()
            edge_explorer.Next()
            edge_key = _compute_edge_key(edge, face)
            if edge_key is None or edge_key not in line_mesh:
                continue

            # 获取边的端点坐标（3D 曲线或 pcurve 回退）
            curve_handle, first, last = BRep_Tool.Curve(edge)
            if curve_handle is not None:
                p_first = curve_handle.Value(first)
                p_last = curve_handle.Value(last)
                start_pt = (p_first.X(), p_first.Y(), p_first.Z())
                end_pt = (p_last.X(), p_last.Y(), p_last.Z())
            elif _surf is not None:
                pcurve, srf, loc = BRep_Tool.CurveOnSurface(edge, face)
                if pcurve is None:
                    continue
                uv1 = pcurve.Value(pcurve.FirstParameter())
                uv2 = pcurve.Value(pcurve.LastParameter())
                p1 = _surf.Value(uv1.X(), uv1.Y())
                p2 = _surf.Value(uv2.X(), uv2.Y())
                start_pt = (p1.X(), p1.Y(), p1.Z())
                end_pt = (p2.X(), p2.Y(), p2.Z())
            else:
                continue

            raw_edges.append({
                'edge_key': edge_key,
                'start_pt': start_pt,
                'end_pt': end_pt,
            })

        if not raw_edges:
            continue

        # 按连通性排序：贪心链式连接
        ordered_edges = [raw_edges[0]]
        remaining = raw_edges[1:]
        while remaining:
            last_end = ordered_edges[-1]['end_pt']
            found = False
            for i, e in enumerate(remaining):
                # 直接连接
                d = sum((a - b) ** 2 for a, b in zip(last_end, e['start_pt']))
                if d < 1e-10:
                    ordered_edges.append(remaining.pop(i))
                    found = True
                    break
                # 反转连接
                d = sum((a - b) ** 2 for a, b in zip(last_end, e['end_pt']))
                if d < 1e-10:
                    e_flipped = {
                        'edge_key': e['edge_key'],
                        'start_pt': e['end_pt'],
                        'end_pt': e['start_pt'],
                    }
                    ordered_edges.append(e_flipped)
                    remaining.pop(i)
                    found = True
                    break
            if not found:
                # 无法继续连接，跳出（可能是开放 wire）
                break

        wire_nodes = []

        for oe in ordered_edges:
            edge_key = oe['edge_key']
            edge_nodes_info = list(line_mesh[edge_key])

            # 方向适配：根据排序后的 start_pt 决定节点顺序
            stored_first = edge_nodes_info[0]['coords']
            stored_last = edge_nodes_info[-1]['coords']
            d_first = sum((a - b) ** 2 for a, b in zip(oe['start_pt'], stored_first))
            d_last = sum((a - b) ** 2 for a, b in zip(oe['start_pt'], stored_last))
            if d_last < d_first:
                edge_nodes_info = list(reversed(edge_nodes_info))

            # 闭合边去重：仅当 wire 只有一条边时去掉末尾重复节点
            # 多边 wire 中，接缝处去重由下方的"跳过与上一条边重合的首节点"处理
            if len(ordered_edges) == 1 and len(edge_nodes_info) > 1:
                c0 = edge_nodes_info[0]['coords']
                c1 = edge_nodes_info[-1]['coords']
                dist_sq = (c0[0]-c1[0])**2 + (c0[1]-c1[1])**2 + (c0[2]-c1[2])**2
                if dist_sq < 1e-10:
                    edge_nodes_info = edge_nodes_info[:-1]

            # 跳过与上一条边重合的首节点（接缝处）
            if wire_nodes:
                last_node = wire_nodes[-1]
                first_info = edge_nodes_info[0]
                d_sq = sum((a - b) ** 2 for a, b in zip(last_node.coords, first_info['coords']))
                if d_sq < 1e-10:
                    edge_nodes_info = edge_nodes_info[1:]

            for info in edge_nodes_info:
                wire_nodes.append(_get_or_create_node(info))

        # 创建阵面
        for i in range(len(wire_nodes) - 1):
            if wire_nodes[i].idx != wire_nodes[i + 1].idx:
                front = SurfaceFront(
                    node_elem1=wire_nodes[i],
                    node_elem2=wire_nodes[i + 1],
                    surface=face,
                    idx=len(fronts),
                    bc_type=bc_type,
                    part_name=part_name,
                )
                fronts.append(front)

        # 闭合：首尾节点坐标接近时创建闭合阵面
        if len(wire_nodes) >= 3:
            c0 = wire_nodes[0].coords
            c1 = wire_nodes[-1].coords
            dist_sq = (c0[0]-c1[0])**2 + (c0[1]-c1[1])**2 + (c0[2]-c1[2])**2
            if dist_sq < 1e-10 and wire_nodes[0].idx != wire_nodes[-1].idx:
                front = SurfaceFront(
                    node_elem1=wire_nodes[-1],
                    node_elem2=wire_nodes[0],
                    surface=face,
                    idx=len(fronts),
                    bc_type=bc_type,
                    part_name=part_name,
                )
                fronts.append(front)

    return fronts

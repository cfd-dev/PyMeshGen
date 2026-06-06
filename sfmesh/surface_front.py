"""
三维曲面阵面数据结构

定义用于曲面网格生成的阵面、节点和单元数据结构
"""
import heapq
import math
import numpy as np
from typing import List, Tuple, Optional, Any
from data_structure.basic_elements import is_node_element, NodeElementALM
from data_structure.front2d import Front


class NodeElement3D(NodeElementALM):
    """
    三维节点元素，继承自 NodeElementALM

    用于曲面网格生成的节点数据结构，增加了曲面相关属性
    """

    def __init__(
        self,
        coords: Tuple[float, float, float],
        idx: int = -1,
        bc_type: str = None,
        part_name: str = None,
        surface: Any = None,
        uv_params: Tuple[float, float] = None,
        normal: Tuple[float, float, float] = None,
        **kwargs
    ):
        super().__init__(coords, idx, part_name=part_name, bc_type=bc_type, **kwargs)
        self.surface = surface
        self.uv_params = uv_params
        self.normal = normal
        # 覆盖 hash 为 8 位精度（用于跨面节点去重）
        self.hash = hash(tuple(f"{0.0 if round(c, 8) == 0 else round(c, 8):.8f}" for c in coords))
        # 覆盖 bbox 为 3D
        self.bbox = (coords[0], coords[1], coords[2], coords[0], coords[1], coords[2])

    def __eq__(self, other):
        if not isinstance(other, NodeElement3D):
            return False
        return self.hash == other.hash

    def __hash__(self):
        return self.hash

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


class SurfaceFront(Front):
    """
    三维曲面阵面，继承自 Front

    用于曲面网格生成的阵面数据结构
    表示曲面上的一个边界边
    """

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
        super().__init__(node_elem1, node_elem2, idx=idx, bc_type=bc_type,
                         part_name=part_name, al=al)
        self.surface = surface
        # 用节点法向覆盖 2D normal
        self._compute_surface_normal()
        # 计算切平面推进方向
        self.compute_tangent_normal(None)
        # 重新计算 3D bbox
        self._compute_bbox_3d()

    def _compute_surface_normal(self):
        """从节点法向计算阵面法向（覆盖父类的 2D normal）"""
        n1 = getattr(self.node_elems[0], 'normal', None)
        n2 = getattr(self.node_elems[1], 'normal', None)
        if n1 is not None and n2 is not None:
            avg = np.array(n1) + np.array(n2)
            norm = np.linalg.norm(avg)
            if norm > 1e-12:
                self.normal = tuple(avg / norm)
            else:
                self.normal = (0.0, 0.0, 1.0)
        else:
            self.normal = (0.0, 0.0, 1.0)

    def _compute_bbox_3d(self):
        """计算 3D 边界框（覆盖父类的 2D bbox）"""
        node1 = self.node_elems[0].coords
        node2 = self.node_elems[1].coords
        self.bbox = (
            min(node1[0], node2[0]), min(node1[1], node2[1]), min(node1[2], node2[2]),
            max(node1[0], node2[0]), max(node1[1], node2[1]), max(node1[2], node2[2]),
        )

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

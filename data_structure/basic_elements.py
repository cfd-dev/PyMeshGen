import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from vtk_io import write_vtk, parse_vtk_msh, VTK_ELEMENT_TYPE
from geom_toolkit import (
    calculate_distance,
    segments_intersect,
    triangle_area,
    triangle_intersect_triangle,
    quadrilateral_area,
    calculate_angle,
    quad_intersects_triangle,
    quad_intersects_quad,
)
from mesh_quality import (
    triangle_shape_quality,
    triangle_skewness,
    quadrilateral_shape_quality,
    quadrilateral_skewness,
    quadrilateral_aspect_ratio,
)
from message import info, debug, warning, verbose


class NodeElement:
    def __init__(self, coords, idx, part_name=None, bc_type=None):
        if isinstance(coords, (np.ndarray, list, tuple)):
            self.coords = list(coords)
        else:
            raise TypeError(f"坐标格式错误，期望可迭代类型，实际类型：{type(coords)}")

        self.idx = idx
        self.part_name = part_name
        self.bc_type = bc_type

        self.node2front = []  # 节点关联的阵面列表
        self.node2node = []  # 节点关联的节点列表

        # 使用处理后的坐标生成哈希
        # 此处注意！！！hash(-1.0)和hash(-2.0)的结果是一样的！！！因此必须使用字符串
        self.hash = hash(tuple(f"{coord:.6f}" for coord in coords))

        self.bbox = [
            min(self.coords[0], self.coords[0]),
            min(self.coords[1], self.coords[1]),
            max(self.coords[0], self.coords[0]),
            max(self.coords[1], self.coords[1]),
        ]

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if isinstance(other, NodeElement):
            # return self.coords == other.coords
            return self.hash == other.hash
        return False


# 继承自NodeElement，用于存储额外的信息
class NodeElementALM(NodeElement):  # 添加父类继承
    def __init__(
        self,
        coords,
        idx,
        bc_type=None,
        match_bound=None,
        convex_flag=False,
        concav_flag=False,
    ):
        super().__init__(coords, idx, bc_type)  # 调用父类构造函数

        self.marching_direction = []  # 节点推进方向
        self.marching_distance = 0.0  # 节点处的推进距离
        self.angle = 0.0  # 节点的角度
        self.convex_flag = False
        self.concav_flag = False
        self.num_multi_direction = 1  # 节点处的多方向数量
        self.local_step_factor = 1.0  # 节点处的局部步长因子
        self.corresponding_node = None  # 节点的对应节点
        self.matching_boundary = match_bound  # 节点所属的match边界

    @classmethod
    def from_existing_node(cls, node_elem):
        new_node = cls(
            node_elem.coords,
            node_elem.idx,
            bc_type=node_elem.bc_type,
        )
        return new_node


class LineSegment:
    def __init__(self, p1, p2):
        if isinstance(p1, NodeElement) and isinstance(p2, NodeElement):
            self.p1 = p1.coords
            self.p2 = p2.coords
        else:
            self.p1 = p1
            self.p2 = p2

        self.length = calculate_distance(self.p1, self.p2)
        self.bbox = [
            min(self.p1[0], self.p2[0]),
            min(self.p1[1], self.p2[1]),
            max(self.p1[0], self.p2[0]),
            max(self.p1[1], self.p2[1]),
        ]

    def is_intersect(self, line):
        """判断两条线段是否相交"""
        p3 = line.p1
        p4 = line.p2
        return segments_intersect(self.p1, self.p2, p3, p4)


class Triangle:
    def __init__(self, p1, p2, p3, part_name=None, idx=None, node_ids=None):
        if (
            isinstance(p1, NodeElement)
            and isinstance(p2, NodeElement)
            and isinstance(p3, NodeElement)
        ):
            self.p1 = p1.coords
            self.p2 = p2.coords
            self.p3 = p3.coords
            self.node_ids = [p1.idx, p2.idx, p3.idx]
        else:
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self.node_ids = node_ids

        self.part_name = part_name
        self.idx = idx
        # 生成几何级哈希
        coord_hash = hash(
            (
                tuple(f"{coord:.6f}" for coord in self.p1),
                tuple(f"{coord:.6f}" for coord in self.p2),
                tuple(f"{coord:.6f}" for coord in self.p3),
            )
        )
        # 生成逻辑级哈希
        id_hash = hash(tuple(sorted(self.node_ids))) if self.node_ids else 0
        # 组合哈希
        self.hash = hash((coord_hash, id_hash))

        self.area = None
        self.quality = None
        self.bbox = [
            min(self.p1[0], self.p2[0], self.p3[0]),  # (min_x, min_y, max_x, max_y)
            min(self.p1[1], self.p2[1], self.p3[1]),
            max(self.p1[0], self.p2[0], self.p3[0]),
            max(self.p1[1], self.p2[1], self.p3[1]),
        ]

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if isinstance(other, Triangle):
            return self.hash == other.hash
        return False

    def init_metrics(self):
        if self.area is None:
            self.area = triangle_area(self.p1, self.p2, self.p3)
            if self.area == 0.0:
                raise ValueError(
                    f"三角形面积异常：{self.area}，顶点：{self.p1}, {self.p2}, {self.p3}"
                )
        if self.quality is None:
            self.quality = triangle_shape_quality(self.p1, self.p2, self.p3)
            if self.quality == 0.0:
                raise ValueError(
                    f"三角形质量异常：{self.quality}，顶点：{self.p1}, {self.p2}, {self.p3}"
                )
    
    def get_quality(self):
        if self.quality is None:
            self.init_metrics()
        return self.quality
        
    def get_area(self):
        if self.area is None:
            self.area = triangle_area(self.p1, self.p2, self.p3)
        return self.area

    def get_skewness(self):
        return triangle_skewness(self.p1, self.p2, self.p3)

    def get_element_size(self):
        if self.area is None:
            self.get_area()
        return sqrt(self.area)

    def is_intersect(self, triangle):
        p4 = triangle.p1
        p5 = triangle.p2
        p6 = triangle.p3
        # 采用bbox快速判断
        if (
            self.bbox[0] > triangle.bbox[2]
            or self.bbox[2] < triangle.bbox[0]
            or self.bbox[1] > triangle.bbox[3]
            or self.bbox[3] < triangle.bbox[1]
        ):
            return False

        return triangle_intersect_triangle(self.p1, self.p2, self.p3, p4, p5, p6)


class Quadrilateral:
    def __init__(self, p1, p2, p3, p4, part_name=None, idx=None, node_ids=None):
        if (
            isinstance(p1, NodeElement)
            and isinstance(p2, NodeElement)
            and isinstance(p3, NodeElement)
            and isinstance(p4, NodeElement)
        ):
            self.p1 = p1.coords
            self.p2 = p2.coords
            self.p3 = p3.coords
            self.p4 = p4.coords
            self.node_ids = [p1.idx, p2.idx, p3.idx, p4.idx]
        else:
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self.p4 = p4
            self.node_ids = node_ids

        self.part_name = part_name
        self.idx = idx

        # 生成几何级哈希
        coord_hash = hash(
            (
                tuple(f"{coord:.6f}" for coord in self.p1),
                tuple(f"{coord:.6f}" for coord in self.p2),
                tuple(f"{coord:.6f}" for coord in self.p3),
                tuple(f"{coord:.6f}" for coord in self.p4),
            )
        )
        # 生成逻辑级哈希
        id_hash = hash(tuple(sorted(self.node_ids))) if self.node_ids else 0
        # 组合哈希
        self.hash = hash((coord_hash, id_hash))

        self.area = None
        self.quality = None
        self.bbox = [  # (min_x, min_y, max_x, max_y)
            min(self.p1[0], self.p2[0], self.p3[0], self.p4[0]),
            min(self.p1[1], self.p2[1], self.p3[1], self.p4[1]),
            max(self.p1[0], self.p2[0], self.p3[0], self.p4[0]),
            max(self.p1[1], self.p2[1], self.p3[1], self.p4[1]),
        ]

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if isinstance(other, Triangle):
            return self.hash == other.hash
        return False

    def init_metrics(self):
        if self.area is None:
            self.area = quadrilateral_area(self.p1, self.p2, self.p3, self.p4)
            if self.area == 0.0:
                warning(
                    f"四边形面积异常：{self.area}，顶点：{self.p1}, {self.p2}, {self.p3}, {self.p4}"
                )
        if self.quality is None:
            # self.quality = quadrilateral_shape_quality(self.p1, self.p2, self.p3, self.p4)
            self.quality = self.get_skewness()
            if self.quality == 0.0:
                warning(
                    f"四边形质量异常：{self.quality}，顶点：{self.p1}, {self.p2}, {self.p3}, {self.p4}"
                )
        
    def get_area(self):
        if self.area is None:
            self.area = quadrilateral_area(self.p1, self.p2, self.p3, self.p4)
        return self.area

    def get_quality(self):
        if self.quality is None:
            self.quality = quadrilateral_shape_quality(self.p1, self.p2, self.p3, self.p4)
        return self.quality

    def get_element_size(self):
        if self.area is None:
            self.get_area()
        return sqrt(self.area)

    def get_aspect_ratio(self):
        """基于最大/最小边长的长宽比"""
        return quadrilateral_aspect_ratio(self.p1, self.p2, self.p3, self.p4)

    def get_aspect_ratio2(self):
        """基于底边/侧边的长宽比"""
        # 计算所有边长
        edges = [
            calculate_distance(self.p1, self.p2),
            calculate_distance(self.p2, self.p3),
            calculate_distance(self.p3, self.p4),
            calculate_distance(self.p4, self.p1),
        ]

        return (edges[0] + edges[2]) / (edges[1] + edges[3])

    def get_skewness(self):
        return quadrilateral_skewness(self.p1, self.p2, self.p3, self.p4)

    def is_intersect_triangle(self, triangle):
        if not isinstance(triangle, Triangle):
            raise TypeError("只接受Triangle对象作为输入！")

        p5 = triangle.p1
        p6 = triangle.p2
        p7 = triangle.p3
        # 采用bbox快速判断
        if (
            self.bbox[0] > triangle.bbox[2]
            or self.bbox[2] < triangle.bbox[0]
            or self.bbox[1] > triangle.bbox[3]
            or self.bbox[3] < triangle.bbox[1]
        ):
            return False

        return quad_intersects_triangle(self.p1, self.p2, self.p3, self.p4, p5, p6, p7)

    def is_intersect_quad(self, quad):
        if not isinstance(quad, Quadrilateral):
            raise TypeError("只接受Quadrilateral对象作为输入！")

        p5 = quad.p1
        p6 = quad.p2
        p7 = quad.p3
        p8 = quad.p4
        # 采用bbox快速判断
        if (
            self.bbox[0] > quad.bbox[2]
            or self.bbox[2] < quad.bbox[0]
            or self.bbox[1] > quad.bbox[3]
            or self.bbox[3] < quad.bbox[1]
        ):
            return False

        quad1 = [self.p1, self.p2, self.p3, self.p4]
        quad2 = [p5, p6, p7, p8]

        return quad_intersects_quad(quad1, quad2)


class Unstructured_Grid:
    def __init__(self, cell_container, node_coords, boundary_nodes):
        self.cell_container = cell_container
        self.node_coords = node_coords
        self.boundary_nodes = boundary_nodes
        self.boundary_nodes_list = [node_elem.idx for node_elem in boundary_nodes]

        self.num_cells = len(cell_container)
        self.num_nodes = len(node_coords)
        self.num_boundary_nodes = len(boundary_nodes)
        self.num_edges = 0
        self.num_faces = 0
        self.edges = []
        self.node2node = None
        self.node2cell = None

        self.dim = len(node_coords[0])

    def calculate_edges(self):
        """计算网格的边"""
        edge_set = set()
        for cell in self.cell_container:
            for i in range(len(cell.node_ids)):
                edge = tuple(
                    sorted(
                        [
                            cell.node_ids[i],
                            cell.node_ids[(i + 1) % len(cell.node_ids)],
                        ]
                    )
                )
                if edge not in edge_set:
                    edge_set.add(edge)

        self.edges = list(edge_set)
        self.num_edges = len(self.edges)

    def merge(self, other_grid):
        """以各向异性网格为基础，合并两个Unstructured_Grid对象"""
        # 合并单元容器
        self.cell_container.extend(other_grid.cell_container)

        # 更新单元数量
        self.num_cells = len(self.cell_container)

        # 节点坐标已经合并过，但是laplacian优化后，节点坐标有更新，此处应采用优化后的节点坐标
        self.node_coords = other_grid.node_coords

        # 更新节点数量
        self.num_nodes = len(self.node_coords)

        # 更新边界点
        self.boundary_nodes.extend(other_grid.boundary_nodes)
        merged_boundary_nodes = set()
        node_hash_list = set()
        for node_elem in self.boundary_nodes:
            if node_elem.bc_type == "interior":
                continue

            if node_elem.hash not in node_hash_list:
                merged_boundary_nodes.add(node_elem)
                node_hash_list.add(node_elem.hash)

        self.boundary_nodes = list(merged_boundary_nodes)
        self.boundary_nodes_list = [node_elem.idx for node_elem in self.boundary_nodes]
        self.num_boundary_nodes = len(self.boundary_nodes)

    def save_debug_file(self, status):
        """保存调试文件"""
        self.num_cells = len(self.cell_container)
        file_path = f"./out/debug_mesh_{status}.vtk"
        self.save_to_vtkfile(file_path)

    def summary(self):
        """输出网格信息"""
        self.num_cells = len(self.cell_container)
        self.num_nodes = len(self.node_coords)
        self.num_boundary_nodes = len(self.boundary_nodes)
        self.calculate_edges()
        self.num_edges = len(self.edges)
        print(f"Mesh Summary:")
        print(f"  Dimension: {self.dim}")
        print(f"  Number of Cells: {self.num_cells}")
        print(f"  Number of Nodes: {self.num_nodes}")
        print(f"  Number of Boundary Nodes: {self.num_boundary_nodes}")
        print(f"  Number of Edges: {self.num_edges}")
        print(f"  Number of Faces: {self.num_faces}")

        # 计算所有单元的质量
        for c in self.cell_container:
            c.init_metrics()

        quality_values = [
            c.quality for c in self.cell_container if c.quality is not None
        ]

        area_values = [c.area for c in self.cell_container if c.area is not None]

        # 输出质量信息
        print(f"Quality Statistics:")
        print(f"  Min Quality: {min(quality_values):.4f}")
        print(f"  Max Quality: {max(quality_values):.4f}")
        print(f"  Mean Quality: {np.mean(quality_values):.4f}")
        print(f"  Min Area: {min(area_values):.4e}")

    def quality_histogram(self):
        """绘制质量直方图"""
        # 绘制直方图
        quality_values = [
            c.quality for c in self.cell_container if c.quality is not None
        ]
        plt.figure(figsize=(10, 6))
        plt.hist(quality_values, bins=10, alpha=0.7, color="blue")
        plt.xlim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlabel("Skewness Quality")
        plt.ylabel("Number of Cells")
        plt.title("Quality Histogram")
        plt.show(block=False)

    def visualize_unstr_grid_2d(self, visual_obj=None):
        """可视化二维网格"""
        if visual_obj.ax is None:
            return

        fig, ax = visual_obj.fig, visual_obj.ax

        # 绘制所有节点
        xs = [n[0] for n in self.node_coords]
        ys = [n[1] for n in self.node_coords]
        ax.scatter(xs, ys, c="white", s=10, alpha=0.3, label="Nodes")

        # 绘制节点编号
        # for i, (x, y) in enumerate(self.node_coords):
        # ax.text(x, y, str(i), fontsize=8, ha="center", va="center")

        # 绘制边
        if self.dim == 2:
            self.calculate_edges()
        for edge in self.edges:
            x = [self.node_coords[i][0] for i in edge]
            y = [self.node_coords[i][1] for i in edge]
            ax.plot(x, y, c="blue", alpha=0.5, lw=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Unstructured Grid Visualization")
        ax.axis("equal")

        plt.show(block=False)

    def load_from_vtkfile(self, file_path):
        """从VTK文件加载网格"""
        cls = parse_vtk_msh(file_path)
        return cls

    def save_to_vtkfile(self, file_path):
        """将网格保存到VTK文件"""
        cell_idx_container = []
        cell_type_container = []
        for cell in self.cell_container:
            cell_idx_container.append(cell.node_ids)
            if isinstance(cell, Quadrilateral):
                vtk_cell_type = VTK_ELEMENT_TYPE.QUAD.value
            elif isinstance(cell, Triangle):
                vtk_cell_type = VTK_ELEMENT_TYPE.TRI.value
            cell_type_container.append(vtk_cell_type)

        write_vtk(
            file_path,
            self.node_coords,
            cell_idx_container,
            self.boundary_nodes_list,
            cell_type_container,
        )

    def init_node2node(self):
        """初始化节点关联的节点列表"""
        self.calculate_edges()

        self.node2node = {}
        # 按照边连接关系构建
        for edge in self.edges:
            for node_idx in edge:
                if node_idx not in self.node2node:
                    self.node2node[node_idx] = []
                for other_node_idx in edge:
                    if other_node_idx != node_idx:
                        self.node2node[node_idx].append(other_node_idx)

        # 去除重复
        for node_idx in self.node2node:
            self.node2node[node_idx] = list(set(self.node2node[node_idx]))
    
    def init_node2node_by_cell(self):
        """初始化节点关联的节点列表"""
        self.calculate_edges()
        
        self.node2node = {}
        # 按照点相连的单元构建
        for cell in self.cell_container:
            for node_idx in cell.node_ids:
                if node_idx not in self.node2node:
                    self.node2node[node_idx] = []
                for other_node_idx in cell.node_ids:
                    if other_node_idx != node_idx:
                        self.node2node[node_idx].append(other_node_idx)
                        
        # 去除重复
        for node_idx in self.node2node:
            self.node2node[node_idx] = list(set(self.node2node[node_idx]))
    
    def cyclic_node2node(self):
        """根据edge的连接关系将node2node构建为首尾相连成环的方式"""
        if not self.node2node:
            # self.init_node2node()
            self.init_node2node_by_cell()

        # 创建节点坐标字典加速查询
        coord_dict = {
            idx: np.array(coords) for idx, coords in enumerate(self.node_coords)
        }

        for node_idx in self.node2node:
            neighbors = self.node2node[node_idx]
            if len(neighbors) < 2:
                continue

            # 获取当前节点坐标
            current_coord = coord_dict[node_idx]

            # 计算相邻节点相对当前节点的极角并排序
            sorted_neighbors = sorted(
                neighbors,
                key=lambda n: np.arctan2(
                    coord_dict[n][1] - current_coord[1],
                    coord_dict[n][0] - current_coord[0],
                ),
            )

            # 验证环形连接完整性
            prev_node = sorted_neighbors[-1]
            for curr_node in sorted_neighbors:
                edge = tuple(sorted([prev_node, curr_node]))
                if edge not in self.edges:
                    warning(
                        f"节点 {node_idx} 的邻接节点 {prev_node} 和 {curr_node} 未直接连接"
                    )
                prev_node = curr_node

            self.node2node[node_idx] = sorted_neighbors

    def init_node2cell(self):
        """初始化节点关联的单元列表"""
        self.node2cell = {}
        for cell in self.cell_container:
            for node_idx in cell.node_ids:
                if node_idx not in self.node2cell:
                    self.node2cell[node_idx] = []
                self.node2cell[node_idx].append(cell)

        # 去除重复
        for node_idx in self.node2cell:
            self.node2cell[node_idx] = list(set(self.node2cell[node_idx]))
    

class Connector:
    """曲线对象，包含曲线的几何对象、网格生成参数和所属部件名称，以及曲线网格本身"""

    def __init__(self, part_name, curve_name, param, cad_obj=None):
        self.part_name = part_name  # 部件名称
        self.curve_name = curve_name  # 曲线名称
        self.cad_obj = cad_obj  # 绑定的几何对象，预留给CAD引擎
        self.param = param  # 面网格生成参数
        self.front_list = []  # 曲线网格
        self.matching_boundary = None  # connector需要匹配的对象

    @property
    def start_point(self):
        """通过端点统计获取拓扑起点"""
        endpoints = []
        for front in self.front_list:
            endpoints.extend(front.node_elems)

        # 使用哈希值统计出现次数
        count = {}
        for p in endpoints:
            count[p.hash] = count.get(p.hash, 0) + 1

        # 找出只出现一次的端点（拓扑端点）
        endpoints = [p for p in endpoints if count[p.hash] % 2 != 0]
        return endpoints[0] if endpoints else None

    @property
    def end_point(self):
        """通过端点统计获取拓扑终点"""
        endpoints = []
        for front in self.front_list:
            endpoints.extend(front.node_elems)

        count = {}
        for p in endpoints:
            count[p.hash] = count.get(p.hash, 0) + 1

        endpoints = [p for p in endpoints if count[p.hash] % 2 != 0]
        return endpoints[-1] if len(endpoints) >= 2 else None

    def check_fronts_are_linear(self):
        """检查front_list中的所有front是否共线"""
        all_points = []
        for front in self.front_list:
            all_points.extend([front.node_elems[0].coords, front.node_elems[1].coords])

        # 找到直线起点终点（任意方向）
        points_array = np.array(all_points)
        centroid = np.mean(points_array, axis=0)
        _, _, v = np.linalg.svd(points_array - centroid)
        direction_vector = v[0]

        # 计算各点在主方向的投影
        projections = np.dot(points_array - centroid, direction_vector)
        start_idx = np.argmin(projections)
        end_idx = np.argmax(projections)
        start = points_array[start_idx]
        end = points_array[end_idx]

        # 验证共线性
        vec = end - start
        for p in points_array:
            if np.linalg.norm(np.cross(p - start, vec)) > 1e-6:
                raise ValueError("线段不共线，无法进行直线离散化")

    def rediscretize_conn_to_match_wall(self, wall_part):
        from adlayers2 import MatchingBoundary
        from front2d import Front

        """按照wall_part的边界层参数重新对当前connector进行离散化（暂时只考虑connector为直线）"""
        # 检查conn中的front_list是否共线
        self.check_fronts_are_linear()

        start = self.start_point
        end = self.end_point

        # 为了计算当前connector的推进方向，要找到与wall_part共点的点，作为起点
        found = False
        for front in wall_part.front_list:
            # 如果start_point出现在front的两个端点中，则该端点为真正的起点
            if (
                front.node_elems[0].hash == start.hash
                or front.node_elems[1].hash == start.hash
            ):
                found = True
                break
            # 如果end_point出现在front的两个端点中，则该端点为真正的起点
            elif (
                front.node_elems[0].hash == end.hash
                or front.node_elems[1].hash == end.hash
            ):
                start, end = end, start
                found = True
                break

        if not found:
            raise ValueError("connector与wall_part不共点，无法进行匹配重离散！")

        marching_vector = np.array(end.coords) - np.array(start.coords)
        total_length = np.linalg.norm(marching_vector)
        if total_length < 1e-6:
            raise ValueError("线段长度过小，无法进行离散化")
        marching_vector /= total_length

        match_bound = MatchingBoundary(start, end, marching_vector, self.curve_name)
        self.matching_boundary = match_bound

        # 计算新的离散化参数
        first_height = wall_part.part_params.first_height
        growth_rate = wall_part.part_params.growth_rate
        growth_method = wall_part.part_params.growth_method
        max_layers = wall_part.part_params.max_layers

        if growth_method != "geometric":
            raise ValueError("目前只支持几何增长方法")

        # 生成几何增长离散点
        discretized_points = [np.array(start.coords)]
        cumulative = 0.0
        ilayer = 0
        while True:
            step = first_height * (growth_rate**ilayer)
            ilayer += 1
            # 检查步长是否超出剩余长度
            if cumulative + step > total_length:
                break  # 提前终止循环

            cumulative += step
            new_point = np.array(start.coords) + marching_vector * cumulative
            discretized_points.append(new_point)

        # 确保最后一个点正好是终点（考虑浮点精度）
        if np.linalg.norm(discretized_points[-1] - np.array(end.coords)) > 1e-3:
            discretized_points.append(np.array(end.coords))

        # 创建connector的新front_list
        old_bc_type = self.front_list[0].bc_type
        old_direction = self.front_list[0].direction
        self.front_list = []
        for i in range(0, len(discretized_points) - 1):
            p1 = discretized_points[i].tolist()
            p2 = discretized_points[i + 1].tolist()

            # 注意此处要保持新阵面的方向与原始阵面一致，仅仅只改变了坐标位置
            if np.dot(old_direction, marching_vector) < 0:
                p1, p2 = p2, p1

            node1 = NodeElementALM(
                coords=p1, idx=-1, bc_type=old_bc_type, match_bound=match_bound
            )
            node2 = NodeElementALM(
                coords=p2, idx=-1, bc_type=old_bc_type, match_bound=match_bound
            )
            self.front_list.append(Front(node1, node2, -1, old_bc_type, self.part_name))

        debug(f"connector {self.curve_name} 匹配 {wall_part.part_name} 重离散化完成！")


class Part:
    """部件对象，包含网格生成参数和所有曲线对象"""

    def __init__(self, part_name, part_params, connectors):
        self.part_params = part_params  # 部件级网格生成参数
        self.connectors = connectors  # 部件所包含的曲线对象
        self.part_name = part_name  # 部件名称
        self.front_list = []  # 部件的阵面列表=所有曲线的阵面列表
        # 后续可扩展部件包含的曲面对象等等

    def match_fronts_with_connectors(self, front_list):
        """匹配初始阵面到曲线对象connector中"""
        # TODO 后续几何引擎做好之后要改为按照几何curve与connector对应的方式
        # 要将每个阵面归类到不同的connector中去，归类的依据是：
        # 1. 优先按照curve_name匹配
        # 2. 若没有匹配到curve_name，则放到对应part的最后一个默认connector中
        for conn in self.connectors:
            for front in front_list:
                # 如果curve_name不为"default"，则只匹配该名称的曲线
                if conn.curve_name == front.part_name:
                    conn.front_list.append(front)

                # 如果curve_name为"default"，则将阵面归类到与conn同一个part中去
                if conn.curve_name == "default" and front.part_name == conn.part_name:
                    conn.front_list.append(front)

    def init_part_front_list(self):
        """初始化part的front_list"""
        self.front_list = []
        for conn in self.connectors:
            self.front_list.extend(conn.front_list)

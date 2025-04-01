import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from vtk_io import write_vtk, VTK_ELEMENT_TYPE
from geometry_info import (
    calculate_distance,
    segments_intersect,
    triangle_area,
    triangle_quality,
    triangle_intersect_triangle,
    quadrilateral_area,
    quadrilateral_quality,
    calculate_angle,
)


class NodeElement:
    def __init__(self, coords, idx, bc_type=None):
        self.coords = coords
        self.idx = idx

        # 使用处理后的坐标生成哈希
        # 此处注意！！！hash(-1.0)和hash(-2.0)的结果是一样的！！！因此必须使用字符串
        self.hash = hash(tuple(f"{coord:.6f}" for coord in coords))

        self.bc_type = bc_type

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
    def __init__(self, coords, idx, bc_type=None, convex_flag=False, concav_flag=False):
        super().__init__(coords, idx, bc_type)  # 调用父类构造函数
        self.node2front = []  # 节点关联的阵面列表
        self.node2node = []  # 节点关联的节点列表
        self.marching_direction = []  # 节点推进方向
        self.marching_distance = 0.0  # 节点处的推进距离
        self.angle = 0.0  # 节点的角度
        self.convex_flag = False
        self.concav_flag = False
        self.num_multi_direction = 1  # 节点处的多方向数量
        self.local_step_factor = 1.0  # 节点处的局部步长因子
        self.corresponding_node = None  # 节点的对应节点

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
    def __init__(self, p1, p2, p3, idx=None, node_ids=None):
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
            self.quality = triangle_quality(self.p1, self.p2, self.p3)
            if self.quality == 0.0:
                raise ValueError(
                    f"三角形质量异常：{self.quality}，顶点：{self.p1}, {self.p2}, {self.p3}"
                )

    def is_intersect(self, triangle):
        p4 = triangle.p1
        p5 = triangle.p2
        p6 = triangle.p3
        return triangle_intersect_triangle(self.p1, self.p2, self.p3, p4, p5, p6)


class Quadrilateral:
    def __init__(self, p1, p2, p3, p4, idx=None, node_ids=None):
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

    def init_metrics(self):
        if self.area is None:
            self.area = quadrilateral_area(self.p1, self.p2, self.p3, self.p4)
            if self.area == 0.0:
                raise ValueError(
                    f"四边形面积异常：{self.area}，顶点：{self.p1}, {self.p2}, {self.p3}, {self.p4}"
                )
        if self.quality is None:
            # self.quality = quadrilateral_quality(self.p1, self.p2, self.p3, self.p4)
            self.quality = self.get_skewness()
            if self.quality == 0.0:
                raise ValueError(
                    f"四边形质量异常：{self.quality}，顶点：{self.p1}, {self.p2}, {self.p3}, {self.p4}"
                )

    def get_area(self):
        self.area = quadrilateral_area(self.p1, self.p2, self.p3, self.p4)
        return self.area

    def get_quality(self):
        self.quality = quadrilateral_quality(self.p1, self.p2, self.p3, self.p4)
        return self.quality

    def get_element_size(self):
        if self.area is None:
            self.get_area()
        return sqrt(self.area)

    def get_aspect_ratio(self):
        """基于最大/最小边长的长宽比"""
        # 计算所有边长
        edges = [
            calculate_distance(self.p1, self.p2),
            calculate_distance(self.p2, self.p3),
            calculate_distance(self.p3, self.p4),
            calculate_distance(self.p4, self.p1),
        ]

        max_edge = max(edges)
        min_edge = min(edges)
        return max_edge / min_edge if min_edge > 1e-12 else 0.0

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
        # 四边形的四个内角
        angles = [
            calculate_angle(self.p1, self.p2, self.p3),
            calculate_angle(self.p2, self.p3, self.p4),
            calculate_angle(self.p3, self.p4, self.p1),
            calculate_angle(self.p4, self.p1, self.p2),
        ]
        # 检查内角和是否为360度
        if abs(np.sum(angles) - 360) > 1e-3:
            skewness = 0.0
            return skewness

        # 计算最大和最小角度
        max_angle = max(angles)
        min_angle = min(angles)

        skew1 = (max_angle - 90) / 90
        skew2 = (90 - min_angle) / 90
        skewness = 1.0 - max([skew1, skew2])

        return skewness


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

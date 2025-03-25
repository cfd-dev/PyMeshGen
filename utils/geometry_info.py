from math import sqrt, isnan, isinf
import numpy as np
import matplotlib.pyplot as plt


def normal_vector2d(front):
    """计算二维平面阵面的单位法向量"""
    node1, node2 = [front.node_elems[i].coords for i in range(2)]
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]

    # 计算向量模长
    magnitude = sqrt(dx**2 + dy**2)

    # 处理零向量情况
    if magnitude < 1e-12:
        return (0.0, 0.0)

    # 单位化法向量
    return (-dy / magnitude, dx / magnitude)


def calculate_distance(p1, p2):
    """计算二维/三维点间距"""
    return sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calculate_distance2(p1, p2):
    """计算二维/三维点间距"""
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


def triangle_area(p1, p2, p3):
    """计算三角形面积（支持2D/3D点）"""
    # 向量叉积法计算面积
    v1 = (
        [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
        if len(p1) > 2
        else [p2[0] - p1[0], p2[1] - p1[1], 0]
    )
    v2 = (
        [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]]
        if len(p1) > 2
        else [p3[0] - p1[0], p3[1] - p1[1], 0]
    )
    cross = (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )
    return 0.5 * sqrt(sum(x**2 for x in cross))


def triangle_quality(p1, p2, p3):
    """计算三角形网格质量（两种方法）"""
    a = calculate_distance(p1, p2)
    b = calculate_distance(p2, p3)
    c = calculate_distance(p3, p1)
    area = triangle_area(p1, p2, p3)

    # 方法1：内外接圆半径比（注释保留）
    # denominator = a + b + c
    # if denominator == 0:
    #     return 0.0
    # r = 2.0 * area / denominator  # 内切圆半径
    # R = (a * b * c) / (4.0 * area) if area != 0 else 0  # 外接圆半径
    # quality_method = 3.0 * r / R if R != 0 else 0
    # upper_limit = 1.5

    # 方法2：面积与边长平方比（默认使用该方法）
    denominator = a**2 + b**2 + c**2
    quality_method = 4.0 * sqrt(3.0) * area / denominator if denominator != 0 else 0
    upper_limit = 1.0
    lower_limit = 0.0
    # 新增异常检测
    if (
        isnan(quality_method)
        or isinf(quality_method)
        or quality_method < lower_limit
        or quality_method > upper_limit
    ):
        raise ValueError(
            f"三角形质量计算异常：quality={quality_method}，节点坐标：{p1}, {p2}, {p3}"
        )

    return quality_method


def is_left2d(p1, p2, p3):
    """判断点p3是否在p1-p2向量的左侧"""
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p1[0], p3[1] - p1[1])
    # 向量叉积
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    return cross_product > 0


# 判断四边形是否为凸
def is_convex(a, b, c, d, node_coords):
    ab = np.array(node_coords[b]) - np.array(node_coords[a])
    ac = np.array(node_coords[c]) - np.array(node_coords[a])
    ad = np.array(node_coords[d]) - np.array(node_coords[a])
    cross_c = np.cross(ab, ac)
    cross_d = np.cross(ab, ad)
    return cross_c * cross_d < 0  # 符号不同则在两侧


# 计算三角形最小角
def calculate_min_angle(cell, node_coords):
    if isinstance(cell, Triangle):
        cell = cell.node_ids

    if len(cell) != 3:
        return 0.0
    p1, p2, p3 = cell
    angles = []
    for i in range(3):
        prev = cell[(i - 1) % 3]
        next_p = cell[(i + 1) % 3]
        v1 = np.array(node_coords[prev]) - np.array(node_coords[cell[i]])
        v2 = np.array(node_coords[next_p]) - np.array(node_coords[cell[i]])
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angles.append(np.rad2deg(angle))
    return min(angles) if angles else 0.0


# 检查三角形是否有效（非退化）
def is_valid_triangle(cell, node_coords):
    if isinstance(cell, Triangle):
        cell = cell.node_ids

    if len(cell) != 3:
        return False
    p1, p2, p3 = cell

    # 面积计算
    v1 = np.array(node_coords[p2]) - np.array(node_coords[p1])
    v2 = np.array(node_coords[p3]) - np.array(node_coords[p1])
    area = 0.5 * np.abs(np.cross(v1, v2))  # 使用向量叉积计算面积
    return area > 1e-10


# 检查两点是否重合
def points_equal(p1, p2, epsilon=1e-6):
    return abs(p1[0] - p2[0]) < epsilon and abs(p1[1] - p2[1]) < epsilon


def segments_intersect(a1, a2, b1, b2):
    """判断两个二维线段是否相交（包含共线部分重叠但不包含端点重合的情况）

    参数：
        a1, a2: 线段A的端点坐标 (x, y)
        b1, b2: 线段B的端点坐标 (x, y)

    返回值：
        bool: 是否相交（True表示相交）
    """
    # 阶段1：快速排除端点重合的情况
    if any(points_equal(a, b) for a in (a1, a2) for b in (b1, b2)):
        return False

    # 阶段2：轴对齐包围盒（AABB）快速排斥实验
    # 计算线段A的包围盒
    a_min_x = min(a1[0], a2[0])
    a_max_x = max(a1[0], a2[0])
    a_min_y = min(a1[1], a2[1])
    a_max_y = max(a1[1], a2[1])

    # 计算线段B的包围盒
    b_min_x = min(b1[0], b2[0])
    b_max_x = max(b1[0], b2[0])
    b_min_y = min(b1[1], b2[1])
    b_max_y = max(b1[1], b2[1])

    # 包围盒不相交则直接返回False
    if (
        (a_max_x < b_min_x)
        or (b_max_x < a_min_x)
        or (a_max_y < b_min_y)
        or (b_max_y < a_min_y)
    ):
        return False

    # 阶段3：跨立实验（使用向量叉积判断线段相对位置）
    def cross(o, a, b):
        """计算向量oa和ob的叉积（z分量）"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 计算四个关键叉积值
    c1 = cross(a1, a2, b1)  # b1相对a1a2的位置
    c2 = cross(a1, a2, b2)  # b2相对a1a2的位置
    c3 = cross(b1, b2, a1)  # a1相对b1b2的位置
    c4 = cross(b1, b2, a2)  # a2相对b1b2的位置

    # 标准相交情况：线段AB相互跨立
    if (c1 * c2 < 0) and (c3 * c4 < 0):
        return True

    # 阶段4：处理共线情况（所有叉积为0时）
    if c1 == 0 and c2 == 0 and c3 == 0 and c4 == 0:
        # 检查坐标轴投影是否有重叠（允许部分重叠但不完全重合）
        x_overlap = max(a_min_x, b_min_x) <= min(a_max_x, b_max_x)
        y_overlap = max(a_min_y, b_min_y) <= min(a_max_y, b_max_y)
        return x_overlap and y_overlap

    return False


def is_point_inside_or_on(p, a, b, c):
    # 检查点p是否在三角形内部或边上，但排除共享顶点的情况
    if any(points_equal(p, vtx) for vtx in [a, b, c]):
        return False  # 共享顶点，视为不相交

    def cross_sign(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = cross_sign(a, b, p)
    d2 = cross_sign(b, c, p)
    d3 = cross_sign(c, a, p)

    has_neg = (d1 < -1e-8) or (d2 < -1e-8) or (d3 < -1e-8)
    has_pos = (d1 > 1e-8) or (d2 > 1e-8) or (d3 > 1e-8)

    return not (has_neg and has_pos)


class NodeElement:
    def __init__(self, coords, idx, bc_type=None):
        self.coords = coords
        self.idx = idx
        # 使用处理后的坐标生成哈希
        self.hash = hash(tuple([round(coord, 6) for coord in self.coords]))
        self.bc_type = bc_type

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
        # 快速排斥实验
        if (
            self.bbox[0] > line.bbox[2]
            or self.bbox[2] < line.bbox[0]
            or self.bbox[1] > line.bbox[3]
            or self.bbox[3] < line.bbox[1]
        ):
            return False

        # 跨立实验
        if is_left2d(self.p1, self.p2, line.p1) != is_left2d(
            self.p1, self.p2, line.p2
        ) and is_left2d(line.p1, line.p2, self.p1) != is_left2d(
            line.p1, line.p2, self.p2
        ):
            # 新增端点重合检查
            if (
                self.p1 == line.p1
                or self.p1 == line.p2
                or self.p2 == line.p1
                or self.p2 == line.p2
            ):
                return False
            return True

        return False


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
            self.node_ids = (p1.idx, p2.idx, p3.idx)
        else:
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self.node_ids = node_ids

        self.idx = idx
        # 生成几何级哈希
        coord_hash = hash(
            (
                tuple([round(c, 6) for c in self.p1]),
                tuple([round(c, 6) for c in self.p2]),
                tuple([round(c, 6) for c in self.p3]),
            )
        )
        # 生成逻辑级哈希
        id_hash = hash(tuple(sorted(self.node_ids)))
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

    def init_metrics(self):
        self.area = triangle_area(self.p1, self.p2, self.p3)
        self.quality = triangle_quality(self.p1, self.p2, self.p3)

    def is_intersect(self, triangle):
        self_edges = [(self.p1, self.p2), (self.p2, self.p3), (self.p3, self.p1)]
        tri_edges = [
            (triangle.p1, triangle.p2),
            (triangle.p2, triangle.p3),
            (triangle.p3, triangle.p1),
        ]

        # 检查边是否相交（排除共边）
        for e1 in self_edges:
            a1, a2 = e1
            for e2 in tri_edges:
                b1, b2 = e2
                if segments_intersect(a1, a2, b1, b2):
                    # 检查是否为共边（顶点完全相同）
                    if (points_equal(a1, b1) and points_equal(a2, b2)) or (
                        points_equal(a1, b2) and points_equal(a2, b1)
                    ):
                        continue  # 共边，视为不相交
                    else:
                        return True  # 非共边的相交

        # 检查顶点是否在另一个三角形内部或边上（严格内部或边上，但排除共享顶点）
        for p in [self.p1, self.p2, self.p3]:
            if any(
                points_equal(p, tri_p)
                for tri_p in [triangle.p1, triangle.p2, triangle.p3]
            ):
                continue
            if is_point_inside_or_on(p, triangle.p1, triangle.p2, triangle.p3):
                return True

        for p in [triangle.p1, triangle.p2, triangle.p3]:
            if any(points_equal(p, self_p) for self_p in [self.p1, self.p2, self.p3]):
                continue
            if is_point_inside_or_on(p, self.p1, self.p2, self.p3):
                return True

        # 检查当前三角形是否完全在另一个三角形内部（非共享顶点）
        all_in_self = True
        for p in [self.p1, self.p2, self.p3]:
            if not is_point_inside_or_on(p, triangle.p1, triangle.p2, triangle.p3):
                all_in_self = False
                break
        if all_in_self:
            return True

        # 检查另一个三角形是否完全在当前三角形内部
        all_in_other = True
        for p in [triangle.p1, triangle.p2, triangle.p3]:
            if not is_point_inside_or_on(p, self.p1, self.p2, self.p3):
                all_in_other = False
                break
        if all_in_other:
            return True

        return False


def quadrilateral_area(p1, p2, p3, p4):
    """使用鞋带公式计算任意简单四边形面积（顶点需按顺序排列）"""
    # 将顶点坐标转换为numpy数组并按顺序排列
    points = np.array([p1, p2, p3, p4])

    # 鞋带公式计算面积
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def is_valid_quadrilateral(p1, p2, p3, p4):
    """检查四边形是否非退化（面积>阈值）"""
    area = quadrilateral_area(p1, p2, p3, p4)
    return area > 1e-10


def quadrilateral_quality(p1, p2, p3, p4):
    """计算四边形质量（基于子三角形质量的最小组合）"""
    # 检查四边形有效性
    if not is_valid_quadrilateral(p1, p2, p3, p4):
        return 0.0

    # 凸性检查（使用现有is_convex函数适配）
    if not is_convex([0, 1, 2, 3], [p1, p2, p3, p4], lambda i: [p1, p2, p3, p4][i]):
        return 0.0

    # 计算对角线交点
    seg1 = LineSegment(p1, p3)
    seg2 = LineSegment(p2, p4)
    if not seg1.is_intersect(seg2):
        return 0.0

    # 获取交点坐标（新增交点计算逻辑）
    def line_intersection(line1, line2):
        # 实现线段交点计算
        x1, y1 = line1.p1
        x2, y2 = line1.p2
        x3, y3 = line2.p1
        x4, y4 = line2.p2

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return None
        u = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        x = x1 + u * (x2 - x1)
        y = y1 + u * (y2 - y1)
        return (x, y)

    po = line_intersection(seg1, seg2)

    try:
        # 计算四个子三角形质量
        q1 = triangle_quality(p1, p2, po)
        q2 = triangle_quality(p2, p3, po)
        q3 = triangle_quality(p3, p4, po)
        q4 = triangle_quality(p4, p1, po)
    except ValueError:
        return 0.0

    # 组合质量指标
    tmp = sorted([q1, q2, q3, q4])
    if tmp[-1] == 0:
        return 0.0
    quality = (tmp[0] * tmp[1]) / (tmp[2] * tmp[3])

    # 异常检测（保持与triangle_quality一致）
    if isnan(quality) or isinf(quality):
        raise ValueError(f"四边形质量异常：{quality}，顶点：{p1}, {p2}, {p3}, {p4}")

    return quality


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
            self.node_ids = (p1.idx, p2.idx, p3.idx, p4.idx)
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
                tuple(round(c, 6) for c in self.p1),
                tuple(round(c, 6) for c in self.p2),
                tuple(round(c, 6) for c in self.p3),
                tuple(round(c, 6) for c in self.p4),
            )
        )
        # 生成逻辑级哈希
        id_hash = hash(tuple(sorted(self.node_ids)))
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

    def init_metrics(self):
        self.area = quadrilateral_area(self.p1, self.p2, self.p3, self.p4)
        self.quality = quadrilateral_quality(self.p1, self.p2, self.p3, self.p4)


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

    def visualize_unstr_grid_2d(self):
        """可视化二维网格"""
        fig, ax = plt.subplots(figsize=(10, 8))

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
        # 分类二维单元
        tri_cells = [c.node_ids for c in self.cell_container if len(c.node_ids) == 3]
        quad_cells = [c.node_ids for c in self.cell_container if len(c.node_ids) == 4]

        # 准备单元数据
        cell_data = []
        for cells, n_nodes in [(tri_cells, 3), (quad_cells, 4)]:
            cell_data.extend([f"{n_nodes} " + " ".join(map(str, c)) for c in cells])

        # 计算单元类型
        cell_types = []
        cell_types.extend([5] * len(tri_cells))  # VTK_TRIANGLE = 5
        cell_types.extend([9] * len(quad_cells))  # VTK_QUAD = 9

        with open(file_path, "w") as file:
            file.write("# vtk DataFile Version 2.0\n")
            file.write("Unstructured Grid\n")
            file.write("ASCII\n")
            file.write("DATASET UNSTRUCTURED_GRID\n")
            file.write(f"POINTS {self.num_nodes} float\n")
            for coord in self.node_coords:
                coord_tmp = coord.copy()
                if len(coord) == 2:
                    coord_tmp.append(0.0)  # 添加Z坐标
                file.write(" ".join(map(str, coord_tmp)) + "\n")

            # 写入单元信息
            total_cells = len(tri_cells) + len(quad_cells)
            total_data_size = len(tri_cells) * 4 + len(quad_cells) * 5  # 3+1 和 4+1
            file.write(f"CELLS {total_cells} {total_data_size}\n")
            file.write("\n".join(cell_data) + "\n")

            file.write(f"CELL_TYPES {total_cells}\n")
            file.write("\n".join(map(str, cell_types)) + "\n")

            # 写入边界节点标记
            file.write(f"POINT_DATA {self.num_nodes}\n")
            file.write("SCALARS node_id int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for i in range(self.num_nodes):
                file.write(
                    f"{1 if i in self.boundary_nodes_list else 0}\n"
                )  # 标记边界节点

            file.write(f"CELL_DATA {total_cells}\n")
            file.write("SCALARS cell_id int 1\n")
            file.write("LOOKUP_TABLE default\n")
            file.write("\n".join(map(str, range(total_cells))) + "\n")

        print(f"网格已保存到 {file_path}")

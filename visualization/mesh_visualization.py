import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


class Visualization:
    def __init__(self, SWITCH=False):
        self.ax = None
        self.fig = None

        if SWITCH:
            self.create_figure()

    def create_figure(self, figsize=(10, 8)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.axis("equal")

    def plot_mesh(self, mesh, boundary_only=False):
        if self.ax is None:
            return

        visualize_mesh_2d(mesh, self.ax, boundary_only)


def visualize_mesh_2d(grid, ax=None, BoundaryOnly=False):
    """可视化完整的2D网格结构"""
    if ax is None:
        return

    if not BoundaryOnly:
        # 绘制所有网格节点（半透明显示）
        xs = [n[0] for n in grid["nodes"]]
        ys = [n[1] for n in grid["nodes"]]
        ax.scatter(xs, ys, c="gray", s=8, alpha=0.3, label="All Nodes")

    # 定义边界类型颜色映射
    bc_colors = {
        "interior": "black",  # 内部面
        "wall": "red",  # 壁面
        "pressure-far-field": "blue",  # 压力入口
        "symmetry": "green",  # 压力出口
        "pressure-outlet": "purple",  # 对称面
        "pressure-inlet": "cyan",  # 速度入口
        "unspecified": "orange",  # 其他类型
    }

    # 遍历所有面区域
    existing_types = set()
    for zone in grid["zones"].values():
        if zone["type"] != "faces":
            continue

        if BoundaryOnly and zone["bc_type"] == "interior":
            continue

        # 获取边界类型和颜色
        bc_type = zone.get("bc_type", "INTERIOR")
        color = bc_colors.get(bc_type, bc_colors["unspecified"])

        existing_types.add(bc_type.lower())

        # 绘制面结构
        for face in zone["data"]:
            # 转换为0-based索引并获取节点坐标
            node_indices = [n - 1 for n in face["nodes"]]  # Fluent使用1-based索引
            coords = [grid["nodes"][i] for i in node_indices]

            # 绘制面线段
            x = [c[0] for c in coords]
            y = [c[1] for c in coords]

            # 闭合面（首尾相连）
            if len(coords) > 1:
                x.append(x[0])
                y.append(y[0])

            ax.plot(
                x,
                y,
                color=color,
                linewidth=1 if bc_type == "interior" else 2,
                alpha=0.7 if bc_type == "interior" else 1.0,
                marker=".",
            )

    # 创建图例代理
    legend_elements = []
    for bc_type, color in bc_colors.items():
        label = {
            "interior": "Internal Faces",
            "wall": "Wall",
            "pressure-far-field": "Pressure Farfield",
            "symmetry": "Symmetry",
            "pressure-outlet": "Pressure Outlet",
            "pressure-inlet": "Pressure Inlet",
            "unspecified": "Other Boundaries",
        }.get(bc_type, bc_type.title())

        if bc_type in existing_types or (
            bc_type == "unspecified" and len(existing_types - bc_colors.keys()) > 0
        ):
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=label))

    # 图形设置
    ax.set_title("2D Mesh Structure Visualization")
    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")
    ax.legend(handles=legend_elements, loc="upper right")
    ax.axis("equal")
    plt.tight_layout()
    plt.show(block=False)


def visualize_wall_structure_2d(grid, wall_nodes, ax=None, vector_scale=0.3):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # 预处理：计算每个wall面的边长（排序节点避免重复）
    face_length = {}
    for zone in grid["zones"].values():
        if zone.get("bc_type") == "wall" and zone["type"] == "faces":
            for face in zone["data"]:
                sorted_nodes = sorted(face["nodes"])
                nodes = [grid["nodes"][n - 1] for n in sorted_nodes]
                dx = nodes[1][0] - nodes[0][0]
                dy = nodes[1][1] - nodes[0][1]
                length = np.hypot(dx, dy)
                face_length[tuple(sorted_nodes)] = length

    # 绘制所有节点
    xs = [n[0] for n in grid["nodes"]]
    ys = [n[1] for n in grid["nodes"]]
    ax.scatter(xs, ys, c="gray", s=10, alpha=0.3, label="All Nodes")

    # 绘制Wall节点
    wall_xs = [n["coords"][0] for n in wall_nodes]
    wall_ys = [n["coords"][1] for n in wall_nodes]
    ax.scatter(wall_xs, wall_ys, c="red", s=20, label="Wall Nodes")

    # 绘制Wall面结构
    for zone in grid["zones"].values():
        if zone.get("bc_type") == "wall" and zone["type"] == "faces":
            for face in zone["data"]:
                coords = [grid["nodes"][n - 1] for n in face["nodes"]]
                x = [c[0] for c in coords]
                y = [c[1] for c in coords]
                ax.plot(x, y, c="orange", alpha=0.5, lw=1.5)

    # 绘制推进向量（使用quiver优化性能）
    x, y, u, v = [], [], [], []
    for node_info in wall_nodes:
        vec = node_info.get("march_vector")
        if not vec:
            continue

        vec_norm = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
        assert np.allclose(vec_norm, 1), "march_vector length is not 1"

        faces = node_info.get("node_wall_faces", [])
        if not faces:
            continue

        # 计算平均面长
        # total_length = 0
        # for face in faces:
        #     sorted_nodes = sorted(face["nodes"])
        #     total_length += face_length.get(tuple(sorted_nodes), 0.0)
        # avg_length = total_length / len(faces)
        xmin, xmax = min(wall_xs), max(wall_xs)
        ymin, ymax = min(wall_ys), max(wall_ys)
        avg_length = min(xmax - xmin, ymax - ymin)

        scale = vector_scale * avg_length

        x.append(node_info["coords"][0])
        y.append(node_info["coords"][1])
        u.append(vec[0] * scale)
        v.append(vec[1] * scale)

    ax.quiver(
        x,
        y,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1,
        headwidth=3,
        headlength=4,
        color="blue",
        alpha=0.7,
        width=0.003,
    )

    # 图形设置
    ax.set_title("2D Mesh Visualization with March Vectors")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.axis("equal")
    plt.show()


def visualize_unstr_grid_2d(unstr_grid, ax=None):
    """可视化非结构化网格"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制节点
    xs = [n[0] for n in unstr_grid.node_coords]
    ys = [n[1] for n in unstr_grid.node_coords]

    ax.scatter(xs, ys, c="red", s=1, alpha=0.7, label="Nodes")
    # 绘制边
    if unstr_grid.dim == 2:
        unstr_grid.calculate_edges()
    for edge in unstr_grid.edges:
        x = [unstr_grid.node_coords[i][0] for i in edge]
        y = [unstr_grid.node_coords[i][1] for i in edge]
        ax.plot(x, y, c="red", alpha=0.5, lw=1.5)

    # 图形设置
    if ax.get_title() == "":
        ax.set_title("2D Unstructured Mesh Visualization")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.axis("equal")
    plt.show(block=False)

def plot_polygon(polygon_coords, ax, color="blue", alpha=0.5):
    # 绘制多边形
    polygon = Polygon(polygon_coords, closed=True, fill=True, color=color, alpha=alpha)
    ax.add_patch(polygon)
    
    # polygon = polygon_coords
    # polygon = np.vstack([polygon, polygon[0]])  # 使用numpy正确闭合多边形
    # ax.clear()
    # if len(polygon) >= 3:  # 至少3个点才能绘制多边形
    #     x, y = zip(*polygon)
    #     ax.plot(x, y, "g-", alpha=0.5)

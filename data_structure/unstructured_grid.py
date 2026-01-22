#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
非结构化网格类。

数据模型约定：
- cell_container 是唯一的单元真源（单元对象或 GenericCell）。
- cells 属性仅提供只读的节点索引视图；写入请使用 set_cells。
- bbox 为惰性计算：当 node_coords 变化时标记失效，读取时再重算。
"""

import numpy as np
import matplotlib.pyplot as plt

from fileIO.vtk_io import write_vtk
from data_structure.vtk_types import VTKCellType
from utils.message import warning, info


class GenericCell:
    """通用单元容器，仅保存节点索引。

    用于无法映射到具体单元类型时的占位，保持拓扑信息可用。
    """

    def __init__(self, node_ids, part_name=None, idx=None):
        self.node_ids = list(node_ids)
        self.part_name = part_name
        self.idx = idx
        self.hash = hash(tuple(self.node_ids))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if isinstance(other, GenericCell):
            return self.hash == other.hash
        return False


class Unstructured_Grid:
    """非结构化网格数据结构。

    主要字段说明：
    - node_coords: 节点坐标列表（二维/三维）。
    - cell_container: 单元对象容器（唯一真源）。
    - boundary_nodes: 边界节点对象列表（NodeElement）。
    - bbox: 网格包围盒，惰性计算。
    """
    def __init__(
        self,
        cell_container=None,
        node_coords=None,
        boundary_nodes=None,
        grid_dimension=2,
        file_path=None,
        mesh_type=None,
    ):
        self.dimension = grid_dimension
        # 单元容器：唯一真源，避免与 cells 视图重复存储
        self.cell_container = cell_container if cell_container is not None else []
        self._bbox = None
        self._bbox_dirty = True
        self._node_coords = []
        self.node_coords = node_coords if node_coords is not None else []
        self.boundary_nodes = boundary_nodes if boundary_nodes is not None else []

        self.num_edges = 0
        self.num_faces = 0
        self.edges = []
        self.node2node = None
        self.node2cell = None

        self.file_path = file_path
        self.mesh_type = mesh_type
        self.vtk_poly_data = None
        self.parts_info = {}
        self.boundary_info = {}
        self.quality_data = {}

        self.point_data = {}
        self.cell_data_dict = {}
        self.metadata = {}
        self.cell_groups = []
        self.volume_cells = []
        self.line_cells = []

        self._bbox = None
        self._bbox_dirty = True

    @property
    def node_coords(self):
        """节点坐标列表；设置时会使 bbox 失效。"""
        return self._node_coords

    @node_coords.setter
    def node_coords(self, value):
        """设置节点坐标，并标记 bbox 失效。"""
        self._node_coords = value if value is not None else []
        if hasattr(self, "_bbox_dirty"):
            self._bbox_dirty = True

    @property
    def num_nodes(self):
        return len(self.node_coords) if self.node_coords is not None else 0

    @property
    def num_cells(self):
        return len(self.cell_container) if self.cell_container is not None else 0

    @property
    def num_boundary_nodes(self):
        return len(self.boundary_nodes) if self.boundary_nodes is not None else 0

    @property
    def boundary_nodes_list(self):
        """边界节点索引列表（只读视图）。"""
        if not self.boundary_nodes:
            return []
        return [node_elem.idx for node_elem in self.boundary_nodes]

    @property
    def bbox(self):
        """包围盒，读取时按需计算。"""
        if self._bbox_dirty:
            self._update_bbox()
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        self._bbox = value
        self._bbox_dirty = False

    @property
    def num_points(self):
        return self.num_nodes

    @property
    def cells(self):
        """单元节点索引只读视图（由 cell_container 派生）。"""
        if not self.cell_container:
            return []
        cells = []
        for cell in self.cell_container:
            if cell is None:
                continue
            if hasattr(cell, "node_ids"):
                cells.append(list(cell.node_ids))
            elif isinstance(cell, (list, tuple)):
                cells.append(list(cell))
        return cells

    def set_cells(self, value, grid_dimension=None):
        """设置单元数据。

        支持两类输入：
        - 单元对象序列（包含 node_ids）
        - 节点索引序列（list/tuple）
        """
        if not value:
            self.cell_container = []
            return
        first = value[0]
        if hasattr(first, "node_ids"):
            self.cell_container = list(value)
        else:
            build_dimension = self.dimension if grid_dimension is None else grid_dimension
            self.cell_container = self._build_cell_objects(self.node_coords, value, build_dimension)

    def _update_bbox(self):
        """按当前 node_coords 计算包围盒。"""
        if self.node_coords is None or len(self.node_coords) == 0:
            if self.dimension >= 3:
                self._bbox = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                self._bbox = [0.0, 0.0, 0.0, 0.0]
            self._bbox_dirty = False
            return

        self._bbox = [
            min(coord[0] for coord in self.node_coords),
            min(coord[1] for coord in self.node_coords),
            max(coord[0] for coord in self.node_coords),
            max(coord[1] for coord in self.node_coords),
        ]
        if self.dimension >= 3:
            self._bbox.extend([
                min(coord[2] for coord in self.node_coords),
                max(coord[2] for coord in self.node_coords),
            ])
        self._bbox_dirty = False

    def update_counts(self):
        """标记包围盒需要更新（保持历史接口兼容）。"""
        self._bbox_dirty = True

    @staticmethod
    def _build_cell_objects(node_coords, cells, grid_dimension=2):
        """将节点索引列表转换为具体单元对象集合。"""
        if not cells:
            return []
        from data_structure.basic_elements import (
            Triangle,
            Quadrilateral,
            Tetrahedron,
            Pyramid,
            Prism,
            Hexahedron,
        )

        cell_container = []
        for idx, cell_nodes in enumerate(cells):
            if cell_nodes is None:
                continue
            node_ids = list(cell_nodes)
            if not node_ids:
                continue
            if any((not isinstance(n, int)) or n < 0 or n >= len(node_coords) for n in node_ids):
                continue

            coords = [node_coords[nid] for nid in node_ids]
            cell = None
            if len(node_ids) == 3:
                cell = Triangle(coords[0], coords[1], coords[2], "interior-triangle", idx, node_ids=node_ids)
            elif len(node_ids) == 4:
                if grid_dimension == 3:
                    cell = Tetrahedron(coords[0], coords[1], coords[2], coords[3], "interior-tetrahedron", idx, node_ids=node_ids)
                else:
                    cell = Quadrilateral(coords[0], coords[1], coords[2], coords[3], "interior-quadrilateral", idx, node_ids=node_ids)
            elif len(node_ids) == 2:
                cell = GenericCell(node_ids, idx=idx)
            elif len(node_ids) == 5 and grid_dimension == 3:
                cell = Pyramid(coords[0], coords[1], coords[2], coords[3], coords[4], "interior-pyramid", idx, node_ids=node_ids)
            elif len(node_ids) == 6 and grid_dimension == 3:
                cell = Prism(coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], "interior-prism", idx, node_ids=node_ids)
            elif len(node_ids) == 8 and grid_dimension == 3:
                cell = Hexahedron(coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[6], coords[7], "interior-hexahedron", idx, node_ids=node_ids)
            else:
                cell = GenericCell(node_ids, idx=idx)
            cell_container.append(cell)
        return cell_container

    @classmethod
    def from_cells(
        cls,
        node_coords,
        cells,
        boundary_nodes_idx=None,
        grid_dimension=2,
        cell_dimension=None,
    ):
        """从节点与单元索引创建 Unstructured_Grid。"""
        from data_structure.basic_elements import NodeElement

        node_coords = [list(coord) for coord in node_coords] if node_coords else []
        boundary_nodes_idx = boundary_nodes_idx or []
        boundary_nodes = []
        for idx in boundary_nodes_idx:
            if 0 <= idx < len(node_coords):
                boundary_nodes.append(NodeElement(node_coords[idx], idx, bc_type="boundary"))

        build_dimension = grid_dimension if cell_dimension is None else cell_dimension
        cell_container = cls._build_cell_objects(node_coords, cells, build_dimension)
        grid = cls(cell_container, node_coords, boundary_nodes, grid_dimension)
        return grid

    def calculate_edges(self):
        """计算网格的边（无向、去重）。"""
        if self.edges:
            return

        edge_set = set()
        for cell in self.cell_container:
            if cell is None or not hasattr(cell, "node_ids"):
                continue
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
        """以各向异性网格为基础，合并两个Unstructured_Grid对象。

        合并后保留 self 的部件信息；若 other_grid 有新增部件则补充。
        """
        # 合并单元容器
        self.cell_container.extend(other_grid.cell_container)

        # 节点坐标已经合并过，但是laplacian优化后，节点坐标有更新，此处应采用优化后的节点坐标
        self.node_coords = other_grid.node_coords

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

        # 保留原始的parts_info信息（如果存在）
        if hasattr(other_grid, 'parts_info') and other_grid.parts_info:
            if not hasattr(self, 'parts_info') or not self.parts_info:
                self.parts_info = other_grid.parts_info
            else:
                # 如果两个网格都有parts_info，合并它们
                for part_name, part_data in other_grid.parts_info.items():
                    if part_name not in self.parts_info:
                        self.parts_info[part_name] = part_data
        self.update_counts()

    def save_debug_file(self, status):
        """保存调试文件"""
        file_path = f"./out/debug_mesh_{status}.vtk"
        self.save_to_vtkfile(file_path)

    def refresh_cell_geometry(self):
        """同步单元顶点坐标到最新的 node_coords，并重置缓存的几何指标。

        仅更新包含 node_ids 的单元对象，避免破坏占位单元。
        """
        if not hasattr(self, "cell_container") or not hasattr(self, "node_coords"):
            return
        num_nodes = len(self.node_coords)
        for cell in self.cell_container:
            if cell is None or not hasattr(cell, "node_ids"):
                continue
            node_ids = cell.node_ids
            if not node_ids:
                continue
            if max(node_ids) >= num_nodes:
                continue

            coords = [self.node_coords[idx] for idx in node_ids]
            if hasattr(cell, "p1"):
                cell.p1 = list(coords[0])
            if hasattr(cell, "p2") and len(coords) > 1:
                cell.p2 = list(coords[1])
            if hasattr(cell, "p3") and len(coords) > 2:
                cell.p3 = list(coords[2])
            if hasattr(cell, "p4") and len(coords) > 3:
                cell.p4 = list(coords[3])
            if hasattr(cell, "p5") and len(coords) > 4:
                cell.p5 = list(coords[4])
            if hasattr(cell, "p6") and len(coords) > 5:
                cell.p6 = list(coords[5])
            if hasattr(cell, "p7") and len(coords) > 6:
                cell.p7 = list(coords[6])
            if hasattr(cell, "p8") and len(coords) > 7:
                cell.p8 = list(coords[7])

            x_vals = [coord[0] for coord in coords if len(coord) > 0]
            y_vals = [coord[1] for coord in coords if len(coord) > 1]
            z_vals = [coord[2] for coord in coords if len(coord) > 2]
            if x_vals and y_vals:
                if z_vals:
                    cell.bbox = [
                        min(x_vals),
                        min(y_vals),
                        min(z_vals),
                        max(x_vals),
                        max(y_vals),
                        max(z_vals),
                    ]
                else:
                    cell.bbox = [min(x_vals), min(y_vals), max(x_vals), max(y_vals)]

            if hasattr(cell, "area"):
                cell.area = None
            if hasattr(cell, "volume"):
                cell.volume = None
            if hasattr(cell, "quality"):
                cell.quality = None
            if hasattr(cell, "skewness"):
                cell.skewness = None

    def refresh_cell_metrics(self):
        """刷新所有单元的几何与质量指标。"""
        self.refresh_cell_geometry()
        for cell in self.cell_container:
            if hasattr(cell, "init_metrics"):
                cell.init_metrics()

    def summary(self, gui_instance=None):
        """输出网格信息，并计算质量统计信息。"""
        self.calculate_edges()
        self.num_edges = len(self.edges)

        mesh_summary = f"Mesh Summary:\n"
        mesh_summary += f"  Dimension: {self.dimension}\n"
        mesh_summary += f"  Number of Cells: {self.num_cells}\n"
        mesh_summary += f"  Number of Nodes: {self.num_nodes}\n"
        mesh_summary += f"  Number of Boundary Nodes: {self.num_boundary_nodes}\n"
        mesh_summary += f"  Number of Edges: {self.num_edges}\n"
        mesh_summary += f"  Number of Faces: {self.num_faces}\n"

        print(mesh_summary.rstrip())  # Print to console
        if gui_instance and hasattr(gui_instance, 'log_info'):
            gui_instance.log_info(mesh_summary.rstrip())  # Output to GUI

        # 计算所有单元的质量
        self.refresh_cell_metrics()

        quality_values = [
            c.quality for c in self.cell_container if c.quality is not None
        ]

        area_values = [c.area for c in self.cell_container if hasattr(c, 'area') and c.area is not None]
        volume_values = [c.volume for c in self.cell_container if hasattr(c, 'volume') and c.volume is not None]

        skewness_values = [
            c.skewness for c in self.cell_container if c.skewness is not None
        ]

        # 输出质量信息
        if quality_values and skewness_values:
            quality_stats = f"Quality Statistics:\n"
            quality_stats += f"  (Quality=1.0 is the best)\n"
            quality_stats += f"  Min Quality: {min(quality_values):.4f}\n"
            quality_stats += f"  Max Quality: {max(quality_values):.4f}\n"
            quality_stats += f"  Mean Quality: {np.mean(quality_values):.4f}\n\n"

            quality_stats += f"  Min Skewness: {min(skewness_values):.4f}\n"
            quality_stats += f"  Max Skewness: {max(skewness_values):.4f}\n"
            quality_stats += f"  Mean Skewness: {np.mean(skewness_values):.4f}\n"
            
            if area_values:
                quality_stats += f"  Min Area: {min(area_values):.4e}\n"
            if volume_values:
                quality_stats += f"  Min Volume: {min(volume_values):.4e}\n"

            print(quality_stats.rstrip())  # Print to console
            if gui_instance and hasattr(gui_instance, 'log_info'):
                gui_instance.log_info(quality_stats.rstrip())  # Output to GUI
        else:
            no_data_msg = "Quality Statistics:\n  No quality or skewness data available\n"
            print(no_data_msg.rstrip())  # Print to console
            if gui_instance and hasattr(gui_instance, 'log_info'):
                gui_instance.log_info(no_data_msg.rstrip())  # Output to GUI

    def quality_histogram(self, ax=None):
        """绘制质量直方图。"""
        self.refresh_cell_geometry()
        # 计算所有单元的质量值（如果尚未计算）
        quality_values = []
        for c in self.cell_container:
            if c.quality is None:
                # 如果质量未计算，则计算它
                c.quality = c.get_quality()
            if c.quality is not None:
                quality_values.append(c.quality)

        if not quality_values:
            # 如果没有有效的质量值，显示提示信息
            if ax is None:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No quality data available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Quality")
                ax.set_ylabel("Number of Cells")
                ax.set_title("Quality Histogram")
                plt.show(block=False)
            else:
                ax.text(0.5, 0.5, 'No quality data available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Quality")
                ax.set_ylabel("Number of Cells")
                ax.set_title("Quality Histogram")
            return

        if ax is None:
            # 如果没有提供ax，则创建新的figure（用于命令行模式）
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(quality_values, bins=10, alpha=0.7, color="blue")
            plt.xlim(0, 1)
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.xlabel("Quality")
            plt.ylabel("Number of Cells")
            plt.title("Quality Histogram")
            plt.show(block=False)
        else:
            # 如果提供了ax（用于GUI模式），则使用现有的axes
            ax.hist(quality_values, bins=10, alpha=0.7, color="blue")
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.set_xlabel("Quality")
            ax.set_ylabel("Number of Cells")
            ax.set_title("Quality Histogram")

    def skewness_histogram(self, ax=None):
        """绘制偏斜度直方图。"""
        self.refresh_cell_geometry()
        # 计算所有单元的偏斜度值（如果尚未计算）
        skewness_values = []
        for c in self.cell_container:
            if c.skewness is None:
                # 如果偏斜度未计算，则计算它
                c.skewness = c.get_skewness()
            if c.skewness is not None:
                skewness_values.append(c.skewness)

        if not skewness_values:
            # 如果没有有效的偏斜度值，显示提示信息
            if ax is None:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No skewness data available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Skewness")
                ax.set_ylabel("Number of Cells")
                ax.set_title("Skewness Histogram")
                plt.show(block=False)
            else:
                ax.text(0.5, 0.5, 'No skewness data available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Skewness")
                ax.set_ylabel("Number of Cells")
                ax.set_title("Skewness Histogram")
            return

        if ax is None:
            # 如果没有提供ax，则创建新的figure（用于命令行模式）
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(skewness_values, bins=10, alpha=0.7, color="red")
            plt.xlim(0, 1)
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.xlabel("Skewness")
            plt.ylabel("Number of Cells")
            plt.title("Skewness Histogram")
            plt.show(block=False)
        else:
            # 如果提供了ax（用于GUI模式），则使用现有的axes
            ax.hist(skewness_values, bins=10, alpha=0.7, color="red")
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.set_xlabel("Skewness")
            ax.set_ylabel("Number of Cells")
            ax.set_title("Skewness Histogram")

    def visualize_unstr_grid_2d(self, visual_obj=None):
        """可视化二维网格（主要用于 GUI 或调试）。"""
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
        if self.dimension == 2:
            self.calculate_edges()

        for edge in self.edges:
            x = [self.node_coords[i][0] for i in edge]
            y = [self.node_coords[i][1] for i in edge]
            ax.plot(x, y, c="blue", alpha=0.5, lw=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Unstructured Grid Visualization")
        ax.axis("equal")

        # 只有在非GUI模式下才调用plt.show()
        # 在GUI模式下，由GUI程序负责更新画布
        # 当visual_obj是GUI传入的对象时，不需要调用plt.show()
        if not hasattr(visual_obj, 'fig') or visual_obj.fig != fig:
            plt.show(block=False)

    @classmethod
    def load_from_vtkfile(cls, file_path):
        """从 VTK 文件加载网格。"""
        from fileIO.vtk_io import parse_vtk_msh  # This will work with main script adding paths
        return parse_vtk_msh(file_path)

    def save_to_vtkfile(self, file_path):
        """将网格保存到 VTK 文件（仅支持三角形与四边形）。"""
        cell_idx_container = []
        cell_type_container = []
        cell_part_names = []
        for cell in self.cell_container:
            # 跳过None单元格
            if cell is None:
                continue
            cell_idx_container.append(cell.node_ids)

            # 初始化vtk_cell_type为None，避免未定义错误
            vtk_cell_type = None
            # if isinstance(cell, Quadrilateral):
            #     vtk_cell_type = VTKCellType.QUAD
            # elif isinstance(cell, Triangle):
            #     vtk_cell_type = VTKCellType.TRIANGLE
            # 使用类名字符串进行类型判断，避免导入问题
            cell_class_name = cell.__class__.__name__

            if cell_class_name == 'Quadrilateral':
                vtk_cell_type = VTKCellType.QUAD
            elif cell_class_name == 'Triangle':
                vtk_cell_type = VTKCellType.TRIANGLE
            else:
                # 如果遇到未知类型的单元，可以选择跳过或抛出错误
                warning(f"未知单元类型: {type(cell)}, 跳过保存")
                continue

            cell_type_container.append(vtk_cell_type)

            # 获取单元的部件名称，如果没有则默认为'Fluid'
            part_name = getattr(cell, 'part_name', 'Fluid')
            # Ensure part_name is always a string to avoid comparison issues
            part_name = str(part_name) if part_name is not None else 'Fluid'
            cell_part_names.append(part_name)

        # 只有在有有效单元时才写入文件
        if cell_idx_container and cell_type_container:
            write_vtk(
            file_path,
            self.node_coords,
            cell_idx_container,
            self.boundary_nodes_list,
            cell_type_container,
            cell_part_names,
            )
        else:
            warning("没有有效的单元可以保存到VTK文件")

    def init_node2node(self):
        """初始化节点关联的节点列表（基于边）。"""
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
        """初始化节点关联的节点列表（基于单元拓扑）。"""
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
        """根据边关系将 node2node 构建为首尾相连的环形顺序。"""
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
                # if edge not in self.edges:
                # warning(
                # f"节点 {node_idx} 的邻接节点 {prev_node} 和 {curr_node} 未直接连接"
                # )
                prev_node = curr_node

            self.node2node[node_idx] = sorted_neighbors

    def build_topological_ring(self, node_id):
        """基于单元拓扑构建节点的一环有序邻居。

        若拓扑不一致则返回 None。
        """
        next_map = {}
        prev_map = {}
        neighbors = set()

        for cell in self.cell_container:
            if cell is None or not hasattr(cell, "node_ids"):
                continue
            node_ids = cell.node_ids
            if not node_ids:
                continue
            try:
                idx = node_ids.index(node_id)
            except ValueError:
                continue

            prev_id = node_ids[idx - 1]
            next_id = node_ids[(idx + 1) % len(node_ids)]

            neighbors.add(prev_id)
            neighbors.add(next_id)

            if prev_id in next_map and next_map[prev_id] != next_id:
                return None
            if next_id in prev_map and prev_map[next_id] != prev_id:
                return None
            next_map[prev_id] = next_id
            prev_map[next_id] = prev_id

        if not next_map:
            return None

        start = next(iter(next_map))
        ordered = [start]
        current = start
        while True:
            nxt = next_map.get(current)
            if nxt is None:
                return None
            if nxt == start:
                break
            if nxt in ordered:
                return None
            ordered.append(nxt)
            current = nxt

        if len(ordered) != len(neighbors):
            return None

        return ordered

    def init_node2cell(self):
        """初始化节点关联的单元列表。"""
        self.node2cell = {}
        for cell in self.cell_container:
            for node_idx in cell.node_ids:
                if node_idx not in self.node2cell:
                    self.node2cell[node_idx] = []
                self.node2cell[node_idx].append(cell)

        # 去除重复
        for node_idx in self.node2cell:
            self.node2cell[node_idx] = list(set(self.node2cell[node_idx]))

    def merge_elements(self):
        """
        合并相邻的三角形单元，形成四边形单元。
        
        此方法调用 utils.mesh_utils.merge_triangles_to_quads 工具函数。
        
        合并条件：
        1. 两个三角形必须共享一条边
        2. 合并后的四边形是凸多边形
        3. 四边形质量高于合并前三角形质量中位数
        
        Returns:
            Unstructured_Grid: 合并后的新网格对象（不修改原始网格）
        """
        from utils.mesh_utils import merge_triangles_to_quads
        return merge_triangles_to_quads(self)

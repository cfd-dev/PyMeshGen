#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
非结构化网格类
用于管理非结构化网格的数据和操作
"""

import numpy as np
import matplotlib.pyplot as plt

from fileIO.vtk_io import write_vtk, parse_vtk_msh, VTK_ELEMENT_TYPE
from utils.message import warning


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
        self.bbox = [
            min(coord[0] for coord in node_coords),
            min(coord[1] for coord in node_coords),
            max(coord[0] for coord in node_coords),
            max(coord[1] for coord in node_coords),
        ]
        if self.dim >= 3:
            self.bbox.extend([
                min(coord[2] for coord in node_coords),
                max(coord[2] for coord in node_coords),
            ])

    def calculate_edges(self):
        """计算网格的边"""
        if self.edges:
            return

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

        # 保留原始的parts_info信息（如果存在）
        if hasattr(other_grid, 'parts_info') and other_grid.parts_info:
            if not hasattr(self, 'parts_info') or not self.parts_info:
                self.parts_info = other_grid.parts_info
            else:
                # 如果两个网格都有parts_info，合并它们
                for part_name, part_data in other_grid.parts_info.items():
                    if part_name not in self.parts_info:
                        self.parts_info[part_name] = part_data

    def save_debug_file(self, status):
        """保存调试文件"""
        self.num_cells = len(self.cell_container)
        file_path = f"./out/debug_mesh_{status}.vtk"
        self.save_to_vtkfile(file_path)

    def summary(self, gui_instance=None):
        """输出网格信息"""
        self.num_cells = len(self.cell_container)
        self.num_nodes = len(self.node_coords)
        self.num_boundary_nodes = len(self.boundary_nodes)
        self.calculate_edges()
        self.num_edges = len(self.edges)

        mesh_summary = f"Mesh Summary:\n"
        mesh_summary += f"  Dimension: {self.dim}\n"
        mesh_summary += f"  Number of Cells: {self.num_cells}\n"
        mesh_summary += f"  Number of Nodes: {self.num_nodes}\n"
        mesh_summary += f"  Number of Boundary Nodes: {self.num_boundary_nodes}\n"
        mesh_summary += f"  Number of Edges: {self.num_edges}\n"
        mesh_summary += f"  Number of Faces: {self.num_faces}\n"

        print(mesh_summary.rstrip())  # Print to console
        if gui_instance and hasattr(gui_instance, 'log_info'):
            gui_instance.log_info(mesh_summary.rstrip())  # Output to GUI

        # 计算所有单元的质量
        for c in self.cell_container:
            c.init_metrics()

        quality_values = [
            c.quality for c in self.cell_container if c.quality is not None
        ]

        area_values = [c.area for c in self.cell_container if c.area is not None]

        skewness_values = [
            c.skewness for c in self.cell_container if c.skewness is not None
        ]

        # 输出质量信息
        if quality_values and skewness_values:
            quality_stats = f"Quality Statistics:\n"
            quality_stats += f"  Min Quality: {min(quality_values):.4f}\n"
            quality_stats += f"  Max Quality: {max(quality_values):.4f}\n"
            quality_stats += f"  Mean Quality: {np.mean(quality_values):.4f}\n\n"

            quality_stats += f"  Min Skewness: {min(skewness_values):.4f}\n"
            quality_stats += f"  Max Skewness: {max(skewness_values):.4f}\n"
            quality_stats += f"  Mean Skewness: {np.mean(skewness_values):.4f}\n"
            quality_stats += f"  Min Area: {min(area_values):.4e}\n"

            print(quality_stats.rstrip())  # Print to console
            if gui_instance and hasattr(gui_instance, 'log_info'):
                gui_instance.log_info(quality_stats.rstrip())  # Output to GUI
        else:
            no_data_msg = "Quality Statistics:\n  No quality or skewness data available\n"
            print(no_data_msg.rstrip())  # Print to console
            if gui_instance and hasattr(gui_instance, 'log_info'):
                gui_instance.log_info(no_data_msg.rstrip())  # Output to GUI

    def quality_histogram(self, ax=None):
        """绘制质量直方图"""
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
        """绘制偏斜度直方图"""
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

        # 只有在非GUI模式下才调用plt.show()
        # 在GUI模式下，由GUI程序负责更新画布
        # 当visual_obj是GUI传入的对象时，不需要调用plt.show()
        if not hasattr(visual_obj, 'fig') or visual_obj.fig != fig:
            plt.show(block=False)

    @classmethod
    def load_from_vtkfile(cls, file_path):
        """从VTK文件加载网格"""
        from fileIO.vtk_io import parse_vtk_msh  # This will work with main script adding paths
        return parse_vtk_msh(file_path)

    def save_to_vtkfile(self, file_path):
        """将网格保存到VTK文件"""
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
            #     vtk_cell_type = VTK_ELEMENT_TYPE.QUAD.value
            # elif isinstance(cell, Triangle):
            #     vtk_cell_type = VTK_ELEMENT_TYPE.TRI.value
            # 使用类名字符串进行类型判断，避免导入问题
            cell_class_name = cell.__class__.__name__

            if cell_class_name == 'Quadrilateral':
                vtk_cell_type = VTK_ELEMENT_TYPE.QUAD.value
            elif cell_class_name == 'Triangle':
                vtk_cell_type = VTK_ELEMENT_TYPE.TRI.value
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
                # if edge not in self.edges:
                # warning(
                # f"节点 {node_idx} 的邻接节点 {prev_node} 和 {curr_node} 未直接连接"
                # )
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

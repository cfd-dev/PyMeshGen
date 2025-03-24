import numpy as np
from math import pi
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
from geometry_info import NodeElementALM


class PartMeshParameters:
    """网格生成部件参数"""

    def __init__(
        self,
        part_name,
        max_size=1.0,
        PRISM_SWITCH=False,
        first_height=0.1,
        growth_rate=1.2,
        growth_method="geometric",
    ):
        self.part_name = part_name  # 部件名称
        self.max_size = max_size  # 最大网格尺寸
        self.PRISM_SWITCH = PRISM_SWITCH  # 是否生成边界层网格
        self.first_height = first_height  # 第一层网格高度
        self.growth_rate = growth_rate  # 网格高度增长比例
        self.growth_method = growth_method  # 网格高度增长方法
        self.front_heap = None  # 阵面堆


class Adlayers2:
    def __init__(self, part_params, initial_front):

        # 层推进全局参数
        self.max_layers = 100  # 最大推进层数
        self.full_layers = 0  # 完整推进层数
        self.multi_direction = False  # 是否多方向推进

        self.initial_front = initial_front  # 初始阵面堆
        self.part_params = part_params  # 层推进部件参数
        self.normal_points = []  # 节点推进方向
        self.normal_fronts = []  # 阵面法向
        self.front_node_list = []  # 当前阵面节点列表
        self.relax_factor = 0.5  # 节点坐标松弛因子

    def generate_elements(self):
        """生成边界层网格"""
        for i in range(self.max_layers):
            print("第{}层推进...".format(i + 1))

            self.prepare_geometry_info()

            self.compute_point_normals()

            self.advancing_fronts()

            # self.update_cells()

        return unstr_grid

    def advancing_fronts(self):
        # for part in self.part_params:
        pass

    def compute_point_normals(self):
        """计算节点推进方向"""
        for node in self.front_node_list:
            if len(node.node2front) < 2:
                continue

            front1, front2 = node.node2front[:2]
            normal1 = np.array(front1.normal)
            normal2 = np.array(front2.normal)

            # 计算推进方向（法向量平均）
            avg_direction = (normal1 + normal2) / 2.0
            norm = np.linalg.norm(avg_direction)
            if norm > 1e-6:
                node_elem.marching_direction = tuple(avg_direction / norm)
            else:
                node_elem.marching_direction = tuple(normal1)

            # 加权光滑
            w1 = self.relax_factor
            wf1 = (1 - self.relax_factor) / 2
            wf2 = 1 - w1 - wf1

            iterations = 0
            smooth = True
            while smooth:
                iterations += 1
                old_direction = node.marching_direction

                node.marching_direction = tuple(
                    w1 * old_direction + wf1 * normal1 + +wf2 * normal2
                ) / np.linalg.norm(w1 * old_direction + wf1 * normal1 + +wf2 * normal2)

                # 计算法向与相邻面的夹角及其与平均夹角的偏差df
                angle1 = np.arccos(np.dot(node.marching_direction, front1.normal))
                angle2 = np.arccos(np.dot(node.marching_direction, front2.normal))
                avg_angle = (angle1 + angle2) / 2

                # 计算夹角偏差
                df1 = abs(angle1 - avg_angle)
                df2 = abs(angle2 - avg_angle)

                wf1_bar = wf1 * (1 - df1 / avg_angle)
                wf2_bar = wf2 * (1 - df2 / avg_angle)
                wf1 = wf1_bar + wf1 * (1 - (wf1_bar + wf2_bar))
                wf2 = wf2_bar + wf2 * (1 - (wf1_bar + wf2_bar))

                if (
                    np.linalg.norm(np.array(node.marching_direction) - old_direction)
                    < 1e-3
                ):
                    smooth = False

    def prepare_geometry_info(self):
        """准备几何信息"""
        self.compute_front_geometry()

        self.compute_point_normals()

        self.match_parts_with_fronts()

    def compute_front_geometry(self):
        """计算阵面几何信息"""
        # 计算node2front
        self.front_node_list = []
        processed_nodes = set()
        hash_idx_map = {}  # 节点hash值到节点索引的映射
        for front in self.initial_front:
            for i, node_elem in enumerate(front.node_elems):
                if (
                    not isinstance(node_elem, NodeElementALM)
                    and node_elem.hash not in processed_nodes
                ):
                    front.node_elems[i] = NodeElementALM.from_existing_node(node_elem)
                    processed_nodes.add(node_elem.hash)
                    hash_idx_map[node_elem.hash] = front.node_elems[i]
                    self.front_node_list.append(front.node_elems[i])
                else:
                    front.node_elems[i] = hash_idx_map[node_elem.hash]

                front.node_elems[i].node2front.append(front)

        # 计算阵面间夹角、凹凸标记、局部步长因子、多方向推进数量
        for node in self.front_node_list:
            if len(node.node2front) < 2:
                continue

            front1, front2 = node.node2front[:2]
            normal1 = np.array(front1.normal)
            normal2 = np.array(front2.normal)
            # 计算夹角（0-180度）
            cos_theta = np.dot(normal1, normal2) / (
                np.linalg.norm(normal1) * np.linalg.norm(normal2)
            )
            node_elem.angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            # 判断凹凸性（通过叉积符号）
            # 判断连接顺序：front2在前，front1在后，则交换normal顺序，便于叉乘
            if front2.node_ids[1] == front1.node_ids[0]:
                temp = normal2
                normal2 = normal1
                normal1 = temp

            thetam = 0  # 局部步长因子计算中的夹角
            cross = np.cross(normal1, normal2)
            if cross < -1e-6:  # 凸角
                node_elem.convex_flag = True
                node_elem.concav_flag = False
                thetam = np.radians(node_elem.angle)
            elif cross > 1e-6:  # 凹角
                node_elem.convex_flag = False
                node_elem.concav_flag = True
                thetam = -np.radians(node_elem.angle)
            else:  # 共线
                node_elem.convex_flag = False
                node_elem.concav_flag = False

            # 计算多方向推进数量和局部步长因子
            if self.multi_direction and node_elem.convex_flag:
                node_elem.num_multi_direction = (
                    int(np.radians(node_elem.angle) / (1.1 * pi / 3)) + 1
                )
                delta = np.radians(node_elem.angle) / (num_multi_direction - 1)
                initial_vectors = normal1

                for i in range(node_elem.num_multi_direction):
                    angle = i * delta
                    rotation_matrix = np.array(
                        [
                            [np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)],
                        ]
                    )
                    rotated_vector = np.dot(rotation_matrix, initial_vectors)
                    node_elem.multi_direction.append(tuple(rotated_vector))

                node_elem.local_step_factor = 1.0
            else:
                node_elem.num_multi_direction = 1
                node_elem.local_step_factor = 1 - np.sign(thetam) * abs(thetam) / pi

        kkk = 1

        # if self.multi_direction:
        #     if node_elem.convex_flag:
        #         node_elem.num_multi_direction = (
        #             int(np.radians(node_elem.angle) / (1.1 * pi / 3)) + 1
        #         )
        #         node_elem.local_step_factor = 1.0
        #     elif node_elem.concav_flag:
        #         node_elem.local_step_factor = 1 - sign(thetam) * abs(thetam) / pi
        # else:
        #     node_elem.num_multi_direction = 1
        #     node_elem.local_step_factor = 1 - sign(thetam) * abs(thetam) / pi

    def match_parts_with_fronts(self):
        """匹配部件和阵面堆"""
        for front in self.initial_front_heap:
            for part in self.part_params:
                if part.part_name == front.bc_name:
                    part.front_heap.append(front)
                    break

    # def compute_point_normals(self):
    #     """计算节点推进方向"""

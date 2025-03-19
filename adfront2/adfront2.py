import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
import geometry_info as geo_info


class Adfront2:
    def __init__(self, initial_front, sizing_system):
        # 阵面推进参数
        self.al = 3.0  # 在几倍范围内搜索
        self.coeff = 0.85  # Pbest质量系数，coeff越小，选择Pbest的概率越小
        self.mesh_type = 1  # 1-三角形，2-直角三角形，3-三角形/四边形混合
        self.quality_criteria = 0.5  # 单元质量标准，值越大，要求越高
        self.sort_front = True  # 是否对阵面排序
        self.plot_front = True  # 是否实时绘图

        self.front_list = initial_front
        self.sizing_system = sizing_system

    @staticmethod
    def generate_elements(self):
        while self.front_list:
            smallest = heapq.heappop(self.front_list)

            spacing = self.sizing_system(smallest.front_center)

            pbest = self.add_new_point(smallest, spacing)

            node_candidates, front_candidates = self.search_candidates(
                pbest, self.al * spacing
            )

            pselect = select_point(node_candidates, front_candidates)

    def select_point(self, node_candidates, front_candidates):
        pass

    def search_candidates(self, point, radius):
        node_candidates = []
        face_candidates = []
        radius2 = radius * radius

        possible_fronts = []
        for front in self.front_list:
            if (
                point[0] > front.bbox[0] - radius
                and point[0] < front.bbox[1] + radius
                and point[1] > front.bbox[2] - radius
                and point[1] < front.bbox[3] + radius
            ):
                possible_fronts.append(front)

        for front in possible_fronts:
            for node in front.nodes_coords:
                if geo_info.calculate_distance2(point, node) <= radius2:
                    node_candidates.append(node)
                    face_candidates.append(front)
        return node_candidates, face_candidates

    def add_new_point(self, front, spacing):
        normal_vec = normal_vector(smallest)
        if mesh_type == 1:
            pbest = smallest.front_center + normal_vec * spacing
        elif mesh_type == 2:
            pbest = smallest.nodes_coords[0] + normal_vec * spacing
        elif mesh_type == 3:
            pass
        else:
            pbest = smallest.front_center + normal_vec * spacing
        return pbest

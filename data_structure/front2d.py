import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
import geometry_info as geo_info
import heapq
import matplotlib.pyplot as plt


class Front:
    def __init__(self, nodes_idx_old, nodes_coords, bc_type, bc_name):
        self.nodes_idx_old = (
            nodes_idx_old  # 节点对应的原始索引，从原始cas网格中读入的值，编号从1开始
        )
        self.node_coords = [round(coord, 6) for coord in node_coords]  # 节点坐标属性
        self.bc_type = bc_type  # 边界类型
        self.bc_name = bc_name  # 新增边界名称属性

        self.front_center = None  # 阵面中心坐标
        self.length = None  # 阵面长度
        self.bbox = None  # 边界框

        # 计算长度
        node1, node2 = self.nodes_coords
        self.length = geo_info.calculate_distance(node1, node2)
        self.front_center = [(a + b) / 2 for a, b in zip(node1, node2)]

        # 计算边界框
        min_x = min(node1[0], node2[0])  # 最小x坐标
        max_x = max(node1[0], node2[0])  # 最大x坐标
        min_y = min(node1[1], node2[1])  # 最小y坐标
        max_y = max(node1[1], node2[1])  # 最大y坐标

        self.bbox = (min_x, max_x, min_y, max_y)

    def __lt__(self, other):
        return self.length < other.length

    def __eq__(self, other):
        # 通过节点坐标判断是否相同
        return self.nodes_coords == other.nodes_coords

    def draw_front(self, marker="b-", ax=None):
        """绘制阵面"""
        node1, node2 = (
            self.nodes_coords[0],
            self.nodes_coords[1],
        )
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], marker)


def construct_initial_front(grid):
    """
    从网格数据中构造初始的阵面，并按长度排序
    返回格式：优先队列 [(length, Front)]
    """
    heap = []
    processed_edges = set()  # 新增已处理边记录

    # 遍历所有面，筛选边界面
    for face in grid["faces"]:
        if face["left_cell"] == 0:
            raise ValueError(
                f"发现无效左单元 (face ID: {grid['faces'].index(face)})，左单元为0，请检查网格读入！"
            )

        # 仅处理有两个节点的线性面（边界面）
        if len(face["nodes"]) == 2 and (face["right_cell"] == 0):
            # 获取原始节点顺序
            u, v = face["nodes"]

            # 使用冻结集合确保边的唯一性
            edge_key = frozenset({u, v})
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)

            # 获取节点坐标（保持原始顺序）
            node1 = grid["nodes"][u - 1]
            node2 = grid["nodes"][v - 1]

            # 创建Front对象并压入堆
            heapq.heappush(
                heap,
                Front(
                    nodes_idx_old=(u, v),  # 保持原始顺序，编号从1开始
                    nodes_coords=(node1, node2),
                    bc_type=face["bc_type"],
                    bc_name=face["bc_name"],
                ),
            )

    return heap


# 使用示例
# front_heap = construct_initial_front(grid_data)
# smallest_front = heapq.heappop(front_heap)

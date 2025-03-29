import heapq
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
from geometry_info import NodeElement, calculate_distance
from timer import TimeSpan


class Front:
    def __init__(self, node_elem1, node_elem2, idx=None, bc_type=None, part_name=None):
        if not isinstance(node_elem1, NodeElement) or not isinstance(
            node_elem1, NodeElement
        ):
            raise TypeError("node1 和 node2 必须是 NodeElement 类型")

        self.node_elems = [node_elem1, node_elem2]
        self.idx = idx  # 阵面ID
        self.bc_type = bc_type  # 边界类型
        self.part_name = part_name  # 边界所属部件

        self.priority = False  # 优先推进标记
        self.early_stop_flag = False  # 提前停止标志
        self.al = 3.0  # 候选点搜索范围系数
        self.center = None  # 阵面中心坐标
        self.length = None  # 阵面长度
        self.direction = None  # 单位方向向量
        self.normal = None  # 单位法向量
        self.bbox = None  # 边界框
        self.hash = None  # 阵面hash值
        self.node_ids = (node_elem1.idx, node_elem2.idx)  # 节点元素列表
        # self.node_pair = [round(self.node_elems[i].coords, 6) for i in range(2)]

        node1 = node_elem1.coords
        node2 = node_elem2.coords
        self.length = calculate_distance(node1, node2)  # 长度
        if self.length < 1e-12:
            raise ValueError("node1 和 node2 不能重合")

        self.center = [(a + b) / 2 for a, b in zip(node1, node2)]  # 中心坐标

        self.direction = [
            (b - a) / self.length for a, b in zip(node1, node2)
        ]  # 单位方向向量

        self.normal = [
            -self.direction[1],
            self.direction[0],
        ]  # 单位法向量

        # 计算边界框
        min_x = min(node1[0], node2[0])  # 最小x坐标
        max_x = max(node1[0], node2[0])  # 最大x坐标
        min_y = min(node1[1], node2[1])  # 最小y坐标
        max_y = max(node1[1], node2[1])  # 最大y坐标

        self.bbox = (min_x, min_y, max_x, max_y)

        length_hash = hash(round(self.length, 3))

        center_hash = hash(tuple(f"{round(coord, 6):.6f}" for coord in self.center))
        self.hash = hash((center_hash, length_hash))

    def __lt__(self, other):
        # 优先比较priority属性，其次比较长度
        if self.priority != other.priority:
            return self.priority > other.priority  # True值优先
        return self.length < other.length

    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return self.hash

    def draw_front(self, marker="b-", ax=None, linewidth=1):
        """绘制阵面"""
        if ax is None:
            ax = plt.gca()

        node1, node2 = [self.node_elems[i].coords for i in range(2)]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], marker, linewidth=linewidth)

        for node_elem in self.node_elems:
            ax.text(
                node_elem.coords[0],
                node_elem.coords[1],
                str(node_elem.idx),
                fontsize=8,
                ha="center",
                va="top",
            )

        plt.show(block=False)


def process_initial_front(grid):
    """从网格数据中构造初始阵面，并按长度排序"""
    heap = []
    processed_edges = set()  # 新增已处理边记录

    # 遍历所有面，筛选边界面
    front_count = 0
    for face in grid["faces"]:
        # 仅处理有两个节点的线性面（边界面）
        if len(face["nodes"]) == 2 and (face["right_cell"] == 0):
            # 获取原始节点顺序
            u, v = face["nodes"]

            # 使用冻结集合确保边的唯一性
            edge_key = frozenset({u, v})
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)

            # 获取节点坐标，fluent网格从1开始计数
            node1 = grid["nodes"][u - 1]
            node2 = grid["nodes"][v - 1]

            # 创建NodeElement对象
            node_elem1 = NodeElement(
                coords=node1,
                idx=-1,
                bc_type=face["bc_type"],
            )

            node_elem2 = NodeElement(
                coords=node2,
                idx=-1,
                bc_type=face["bc_type"],
            )

            # 创建Front对象并压入堆
            heapq.heappush(
                heap,
                Front(
                    node_elem1=node_elem1,
                    node_elem2=node_elem2,
                    idx=front_count,
                    bc_type=face["bc_type"],
                    part_name=face["part_name"],
                ),
            )
            front_count += 1

    return heap


def construct_initial_front(grid):
    """从网格数据中构造初始阵面，并按长度排序"""
    timer = TimeSpan("构造初始阵面...")
    front_heap = process_initial_front(grid)

    # 重新计算节点索引,对初始阵面的节点重新编号
    node_count = 0
    node_hash_list = set()
    hash_idx_map = {}  # 节点hash值到节点索引的映射
    for front in front_heap:
        front.node_ids = []
        for node_elem in front.node_elems:
            if node_elem.hash not in node_hash_list:
                node_elem.idx = node_count
                hash_idx_map[node_elem.hash] = node_elem.idx
                node_hash_list.add(node_elem.hash)
                node_count += 1
            else:
                node_elem.idx = hash_idx_map[node_elem.hash]

            front.node_ids.append(node_elem.idx)

    timer.show_to_console("构造初始阵面..., Done.")
    return front_heap


# 使用示例
# front_heap = construct_initial_front(grid_data)
# smallest_front = heapq.heappop(front_heap)

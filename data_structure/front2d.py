import heapq
import matplotlib.pyplot as plt
from geom_toolkit import calculate_distance
from basic_elements import NodeElement
from utils.timer import TimeSpan


# 使用__slots__减少内存占用
class Front:
    __slots__ = [
        "node_elems",
        "idx",
        "bc_type",
        "part_name",
        "priority",
        "al",
        "center",
        "length",
        "direction",
        "normal",
        "bbox",
        "hash",
        "node_ids",
        "early_stop_flag",
    ]

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
        self.node_ids = [node_elem1.idx, node_elem2.idx]  # 节点元素列表
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

        length_hash = hash(f"{self.length:.3f}")
        center_hash = hash(tuple(f"{coord:.6f}" for coord in self.center))
        # 这里之所以用center和length生成hash，是因为在阵面推进时，阵面会出现2次，
        # 只是方向不同，对于这种方向不同的阵面，我们认为其是相同的阵面，应该具有相同的hash值
        self.hash = hash((center_hash, length_hash))

        # 备用精确坐标生成唯一hash
        # self.hash = hash(
        #     (
        #         tuple(f"{x:.8f}" for x in node_elem1.coords),
        #         tuple(f"{x:.8f}" for x in node_elem2.coords),
        #     )
        # )

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


def reorder_node_front_index_and(front_list):
    # 重新计算节点索引,对初始阵面的节点重新编号
    node_count = 0
    front_count = 0
    processed_nodes = set()
    node_dict = {}  # 节点hash值到节点的映射
    node_coords = []
    boundary_nodes = []
    for front in front_list:
        front.idx = front_count
        front_count += 1
        front.node_ids = []
        for i, node_elem in enumerate(front.node_elems):
            if node_elem.hash not in processed_nodes:
                front.node_elems[i].idx = node_count
                node_dict[node_elem.hash] = front.node_elems[i]
                node_coords.append(node_elem.coords)
                boundary_nodes.append(front.node_elems[i])

                processed_nodes.add(node_elem.hash)
                node_count += 1
            else:
                front.node_elems[i] = node_dict[node_elem.hash]

            front.node_ids.append(front.node_elems[i].idx)

    return front_list, node_coords, boundary_nodes


def construct_initial_front(grid):
    """从网格数据中构造初始阵面，并按长度排序"""
    timer = TimeSpan("构造初始阵面...")
    front_heap = process_initial_front(grid)

    # reorder_node_index_and_front(front_heap)

    timer.show_to_console("构造初始阵面..., Done.")
    return front_heap


# 使用示例
# front_heap = construct_initial_front(grid_data)
# smallest_front = heapq.heappop(front_heap)

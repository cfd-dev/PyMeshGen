import heapq
from math import sqrt

class Front:
    def __init__(self, nodes, length, bc_type, bc_name, front_center, nodes_coords):
        self.nodes = nodes    # 节点对（排序后）
        self.length = length  # 阵面长度
        self.bc_type = bc_type # 边界类型
        self.bc_name = bc_name  # 新增边界名称属性
        self.front_center = front_center  # 阵面中心坐标
        self.nodes_coords = nodes_coords  # 节点坐标属性
        
    def __lt__(self, other):
        return self.length < other.length

def calculate_distance(p1, p2):
    """计算二维/三维点间距"""
    return sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

def construct_initial_front(grid):
    """
    从网格数据中构造初始的阵面，并按长度排序
    返回格式：优先队列 [(length, Front)]
    """
    heap = []
    processed_edges = set()  # 新增已处理边记录
    
    # 遍历所有面，筛选边界面
    for face in grid['faces']:
        if face['left_cell'] == 0:
            raise ValueError(f"发现无效左单元 (face ID: {grid['faces'].index(face)})，左单元为0，请检查网格读入！")
  
        # 仅处理有两个节点的线性面（边界面）
        if len(face['nodes']) == 2 and (face['right_cell'] == 0):
            # 获取原始节点顺序
            u, v = face['nodes']
            
            # 使用冻结集合确保边的唯一性
            edge_key = frozenset({u, v})
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)
            
            # 获取节点坐标（保持原始顺序）
            node1 = grid['nodes'][u-1]
            node2 = grid['nodes'][v-1]
            
            # 计算长度
            length = calculate_distance(node1, node2)
            center = [(a + b) / 2 for a, b in zip(node1, node2)]

            # 创建Front对象并压入堆
            heapq.heappush(heap, Front(
                nodes=(u, v),  # 保持原始顺序，编号从1开始
                length=length,
                bc_type=face['bc_type'],
                bc_name=face['bc_name'],
                front_center=center,
                nodes_coords=(node1, node2)
            ))
    
    return heap

# 使用示例
# front_heap = construct_initial_front(grid_data)
# smallest_front = heapq.heappop(front_heap)

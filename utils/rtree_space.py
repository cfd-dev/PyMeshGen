from rtree import index
from collections import defaultdict

# global space_index, front_dict


def build_space_index_with_cartesian_grid(fronts, grid_size=1.0):
    """构建空间索引加速相交检测"""
    space_index = defaultdict(list)

    for front in fronts:
        # 计算包围盒
        x_min = front.bbox[0]
        x_max = front.bbox[2]
        y_min = front.bbox[1]
        y_max = front.bbox[3]

        # 计算网格索引
        i_min = int(x_min // grid_size)
        i_max = int(x_max // grid_size)
        j_min = int(y_min // grid_size)
        j_max = int(y_max // grid_size)

        # 将阵面注册到覆盖的网格
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                space_index[(i, j)].append(front)

    return space_index


def add_fronts_to_space_index_with_cartesian_grid(fronts, grid_size=1.0):
    for front in fronts:
        # 计算包围盒
        x_min = front.bbox[0]
        x_max = front.bbox[2]
        y_min = front.bbox[1]
        y_max = front.bbox[3]

        # 计算网格索引
        i_min = int(x_min // grid_size)
        i_max = int(x_max // grid_size)
        j_min = int(y_min // grid_size)
        j_max = int(y_max // grid_size)

        # 将阵面注册到覆盖的网格
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                space_index[(i, j)].append(front)

    return space_index

    def get_candidate_fronts(front, space_index, grid_size, search_radius):
        # 计算网格索引范围
        x_min = front.bbox[0] - search_radius
        x_max = front.bbox[2] + search_radius
        y_min = front.bbox[1] - search_radius
        y_max = front.bbox[3] + search_radius

        # 计算网格索引边界
        i_min = int(x_min // grid_size)
        i_max = int(x_max // grid_size)
        j_min = int(y_min // grid_size)
        j_max = int(y_max // grid_size)

        # 生成网格坐标范围
        i_range = range(i_min, i_max + 1)
        j_range = range(j_min, j_max + 1)

        # 使用集合推导式快速获取候选网格
        grid_coords = {(i, j) for i in i_range for j in j_range}

        # 批量获取候选阵面并去重
        candidates = set()
        for coord in grid_coords:
            candidates.update(space_index.get(coord, []))

        return list(candidates)


def build_space_index_with_RTree(fronts):
    """构建空间索引加速相交检测"""
    # 由于front的hash值不是一一对应的，因此暂时先不用hash值创建字典，后续再考虑
    # front_dict = {hash(f): f for f in fronts}  # 存储front对象的字典
    front_dict = {id(f): f for f in fronts}

    space_index = index.Index()
    for f_id, front in front_dict.items():
        space_index.insert(f_id, front.bbox)

    return front_dict, space_index


def add_fronts_to_space_index_with_RTree(fronts, space_index, front_dict):
    """构建空间索引加速相交检测"""
    for fro in fronts:
        try:
            space_index.insert(id(fro), fro.bbox)
            # 注意此处必须同步更新front_dict
            front_dict[id(fro)] = fro
        except index.RTreeError as e:
            print(f"R树插入失败: {e}")

    return space_index, front_dict


def get_candidate_fronts_id(front, space_index, search_radius):
    """获取可能相交的阵面"""
    # 计算搜索区域
    x_min = front.bbox[0] - search_radius
    x_max = front.bbox[2] + search_radius
    y_min = front.bbox[1] - search_radius
    y_max = front.bbox[3] + search_radius

    # 使用R树进行范围查询
    query_bbox = (x_min, y_min, x_max, y_max)
    candidates = list(space_index.intersection(query_bbox))

    return candidates

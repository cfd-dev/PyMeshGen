from rtree import index
from collections import defaultdict


def build_space_index_with_cartesian_grid(elems, grid_size=1.0):
    """构建空间索引加速相交检测"""
    space_index = defaultdict(list)

    for ele in elems:
        # 计算包围盒
        x_min = ele.bbox[0]
        x_max = ele.bbox[2]
        y_min = ele.bbox[1]
        y_max = ele.bbox[3]

        # 计算网格索引
        i_min = int(x_min // grid_size)
        i_max = int(x_max // grid_size)
        j_min = int(y_min // grid_size)
        j_max = int(y_max // grid_size)

        # 将阵面注册到覆盖的网格
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                space_index[(i, j)].append(ele)

    return space_index


def add_elems_to_space_index_with_cartesian_grid(elems, grid_size=1.0):
    for ele in elems:
        # 计算包围盒
        x_min = ele.bbox[0]
        x_max = ele.bbox[2]
        y_min = ele.bbox[1]
        y_max = ele.bbox[3]

        # 计算网格索引
        i_min = int(x_min // grid_size)
        i_max = int(x_max // grid_size)
        j_min = int(y_min // grid_size)
        j_max = int(y_max // grid_size)

        # 将阵面注册到覆盖的网格
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                space_index[(i, j)].append(ele)

    return space_index


def get_candidate_elements(base_elem, space_index, grid_size, search_radius):
    # 计算网格索引范围
    x_min = base_elem.bbox[0] - search_radius
    x_max = base_elem.bbox[2] + search_radius
    y_min = base_elem.bbox[1] - search_radius
    y_max = base_elem.bbox[3] + search_radius

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


def build_space_index_with_RTree(elems):
    """构建空间索引加速相交检测"""
    # 由于front的hash值不是一一对应的，因此暂时先不用hash值创建字典，后续再考虑
    # elem_dict = {hash(f): f for f in elems}  # 存储front对象的字典
    elem_dict = {id(e): e for e in elems}

    space_index = index.Index()
    for e_id, elem in elem_dict.items():
        space_index.insert(e_id, elem.bbox)

    return elem_dict, space_index


def add_elems_to_space_index_with_RTree(elems, space_index, elem_dict):
    """构建空间索引加速相交检测"""
    for ele in elems:
        try:
            space_index.insert(id(ele), ele.bbox)
            # 注意此处必须同步更新elem_dict
            elem_dict[id(ele)] = ele
        except index.RTreeError as e:
            print(f"R树插入失败: {e}")

    return space_index, elem_dict


def get_candidate_elements_id(base_elem, space_index, search_radius):
    """获取可能相交的阵面"""
    # 计算搜索区域
    x_min = base_elem.bbox[0] - search_radius
    x_max = base_elem.bbox[2] + search_radius
    y_min = base_elem.bbox[1] - search_radius
    y_max = base_elem.bbox[3] + search_radius

    # 使用R树进行范围查询
    query_bbox = (x_min, y_min, x_max, y_max)
    candidates = list(space_index.intersection(query_bbox))

    return candidates

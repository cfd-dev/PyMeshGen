
def construct_initial_front(grid):
    """
    从网格数据中构造初始的阵面。
    """
    initial_front = []
    for zone in grid['zones'].values():
        if zone['type'] == 'faces' and zone.get('bc_type') != 'interior':
            for face in zone['data']:
                initial_front.append(face)
                initial_front['size'] = 0 # face的大小  
    return initial_front

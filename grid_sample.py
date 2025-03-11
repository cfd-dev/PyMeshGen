import mesh_reconstruction as mesh_recons
    
def get_march_vector(grid, node_1based, current_face):
      
    adjacent_faces = mesh_recons.get_adjacent_node(grid, node_1based, current_face)
    
    for adj_face in adjacent_faces:        
        face = adj_face['face']       
        nodes = face['nodes']
        
        try:
            idx = nodes.index(node_1based)
            adj_node = nodes[(idx + 1) % len(nodes)]  # 取下一个节点
            break
        except ValueError:
            # 处理节点不在当前面的情况
            adj_node = None
    else:
        raise ValueError("No adjacent node found")
                
        # if nodes[0] == node_1based:
        #     adj_node = nodes[1]
        # else:
        #     adj_node = nodes[0]
    
    node_0based = node_1based - 1
    adj_node = adj_node - 1
    node1_coord = grid['nodes'][node_0based]
    node2_coord = grid['nodes'][adj_node]

    return unit_direction_vector(node1_coord, node2_coord)

# def unit_direction_vector(node1, node2):
#     if len(node1)==3 and len(node2)==3 :
#         dim = 3
#     elif len(node1)==2 and len(node2)==2:
#         dim = 2
#     else:
#         print("Error: incorrect input coordinates! Nodes must be 2D or 3D and of the same dimension.")
#         return None  # Exit early on invalid input
        
#     dx = node2[0] - node1[0]
#     dy = node2[1] - node1[1]
#     dz = node2[2] - node1[2] if dim==3 else 0.0

#     length = (dx**2 + dy**2 + (dz**2 if dim == 3 else 0.0)) ** 0.5
#     if length == 0:
#         return (0.0, 0.0, 0.0) if dim==3 else (0.0, 0.0)
#     return (dx/length, dy/length, dz/length) if dim==3 else (dx/length, dy/length)
    
def unit_direction_vector(node1, node2):
    """计算单位方向向量（增强版）"""
    dim = len(node1)
    if dim not in (2,3) or len(node2) != dim:
        raise ValueError("Nodes must be 2D or 3D with same dimension")
    
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]
    dz = (node2[2] - node1[2]) if dim == 3 else 0.0

    length = (dx**2 + dy**2 + dz**2)**0.5
    if length == 0:
        return (0.0,)*dim
    return (dx/length, dy/length, dz/length)[:dim]
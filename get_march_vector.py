from collections import defaultdict

def get_adjacent_node(grid, node_1based, current_face):
    
    # 创建节点到所有面的映射（使用0基索引）
    node_to_faces = defaultdict(list)
    
    # 收集所有面并建立映射（处理所有区域）
    for zone in grid['zones'].values():
        if zone['type'] == 'faces':
            for face in zone['data']:
                # 转换节点索引为0基
                converted_nodes = [n - 1 for n in face['nodes']]
                # 保存原始面和转换后的节点
                face_info = {
                    'original': face,
                    'nodes': converted_nodes,
                    'zone': zone  # 保留区域信息
                }
                # 建立节点到面的映射
                for node in converted_nodes:
                    node_to_faces[node].append(face_info)
    
    # 收集所有wall面
    wall_faces = []
    for zone in grid['zones'].values():
        if zone['type'] == 'faces' and zone.get('bc_type') == 'wall':
            for face in zone['data']:
                # 转换节点索引为0基
                converted_nodes = [n - 1 for n in face['nodes']]
                wall_faces.append({
                    'original': face,
                    'nodes': converted_nodes,
                    'zone': zone
                })
    
    # 查找相邻面
    adjacent_faces_list = []
    adjacent = []
    # 收集所有可能的相邻面
    node_0based = node_1based - 1
    candidate_faces = []    
    candidate_faces.extend(node_to_faces.get(node_0based, []))
         
    # 去重并排除自身
    seen = set()
    for face_info in candidate_faces:
        face = face_info['original']
        zone = face_info['zone']
        if zone['bc_type'] == 'wall':
            continue
        if face is current_face:
            continue
        # 使用原始面对象的id保证唯一性
        if id(face) not in seen:
            seen.add(id(face))
            adjacent.append({
                'face': face,
                'zone_type': face_info['zone'].get('bc_type', 'internal')
            })
        
        adjacent_faces_list.append(adjacent)
   
    return adjacent_faces_list
    

def get_march_vector(grid, node_1based, current_face):
      
    adjacent_faces = get_adjacent_node(grid, node_1based, current_face)
    
    for adj_face in adjacent_faces[0]:        
        face = adj_face['face']
        nodes = face['nodes']
        if nodes[0] == node_1based:
            adj_node = nodes[1]
        else:
            adj_node = nodes[0]
    
    node_0based = node_1based - 1
    adj_node = adj_node - 1
    node1_coord = grid['nodes'][node_0based]
    node2_coord = grid['nodes'][adj_node]

    return unit_direction_vector(node1_coord, node2_coord)

def unit_direction_vector(node1, node2):
    if len(node1)==3 and len(node2)==3 :
        dim = 3
    elif len(node1)==2 and len(node2)==2:
        dim = 2
    else:
        print("Error: incorrect input coordinates! Nodes must be 2D or 3D and of the same dimension.")
        return None  # Exit early on invalid input
        
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]
    dz = node2[2] - node1[2] if dim==3 else 0.0

    length = (dx**2 + dy**2 + (dz**2 if dim == 3 else 0.0)) ** 0.5
    if length == 0:
        return (0.0, 0.0, 0.0) if dim==3 else (0.0, 0.0)
    return (dx/length, dy/length, dz/length) if dim==3 else (dx/length, dy/length)
    

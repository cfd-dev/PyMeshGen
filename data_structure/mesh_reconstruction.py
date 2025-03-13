from collections import defaultdict

def get_adjacent_node(grid, node_1based, current_face):
    
    node_to_faces = grid['preprocessed']['node_to_faces']
    
    # 收集所有可能的相邻面
    node_0based = node_1based - 1
    candidate_faces = node_to_faces.get(node_0based, [])  
    adjacent = []
      
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
   
    return adjacent

def preprocess_grid(grid):
    # 创建节点到所有面的映射（使用0基索引）
    node_to_faces = defaultdict(list)
    wall_faces = []   
    # 收集所有面并建立映射（处理所有区域）
    for zone in grid['zones'].values():
        if zone['type'] != 'faces':
            continue
        
        for face in zone['data']:
            # 转换节点索引为0基
            converted_nodes = [n - 1 for n in face['nodes']]
            # 保存原始面和转换后的节点
            face_info = {
                'original': face,
                'nodes': converted_nodes,
                'zone': zone  # 保留区域信息
                }
            
            if zone.get('bc_type') == 'wall':
                wall_faces.append(face_info)
            else:
                # 建立节点到面的映射
                for node in converted_nodes:
                    node_to_faces[node].append(face_info)
                    
    grid['preprocessed'] = {
        'node_to_faces': node_to_faces,
        'wall_faces': wall_faces
    }   
                   
def find_2d_wall_adjacent_faces(grid):
    """查找二维网格中每个wall面的相邻面（通过共享节点）"""
    
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
    for wall_face in wall_faces:
        adjacent = []
        current_face = wall_face['original']
        current_nodes = wall_face['nodes']
        
        # 收集所有可能的相邻面
        candidate_faces = []
        for node in current_nodes:
            candidate_faces.extend(node_to_faces.get(node, []))
        
        # 去重并排除自身
        seen = set()
        for face_info in candidate_faces:
            face = face_info['original']
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

def find_adjacent_faces(grid):
    """查找每个wall面的相邻面"""
    
    # 创建边到面的映射字典
    edge_to_faces = defaultdict(list)
    
    # 遍历所有面建立边面映射
    for zone in grid['zones'].values():
        if zone['type'] == 'faces':
            for face in zone['data']:
                nodes = face['nodes']
                num_nodes = len(nodes)
                for i in range(num_nodes):
                    a = nodes[i]
                    b = nodes[(i+1) % num_nodes]
                    edge = tuple(sorted((a, b)))  # 创建有序边元组
                    edge_to_faces[edge].append(face)
    
    # 收集所有wall面
    wall_faces = []
    for zone in grid['zones'].values():
        if zone['type'] == 'faces' and zone.get('bc_type') == 'wall':
            wall_faces.extend(zone['data'])
    
    # 查找相邻面
    adjacent_faces_list = []
    for wall_face in wall_faces:
        adjacent_faces = set()
        nodes = wall_face['nodes']
        num_nodes = len(nodes)
        
        for i in range(num_nodes):
            a = nodes[i]
            b = nodes[(i+1) % num_nodes]
            edge = tuple(sorted((a, b)))
            
            # 获取共享边的所有面
            connected_faces = edge_to_faces.get(edge, [])
            for face in connected_faces:
                if face != wall_face:  # 排除自身
                    adjacent_faces.add(face)
        
        adjacent_faces_list.append(adjacent_faces)
    
    return adjacent_faces_list

# # 使用示例
# adjacent_faces = find_adjacent_faces(grid)

# # 打印第一个wall面的相邻面数量
# if adjacent_faces:
#     print(f"第一个wall面有 {len(adjacent_faces[0])} 个相邻面")
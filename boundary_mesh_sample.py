from read_cas import parse_fluent_msh

file_path = './sample/convex.cas'
grid = parse_fluent_msh(file_path)
node_coords = grid['nodes']

wall_faces = []
wall_nodes_coords = []
# 遍历所有区域
for zone in grid['zones'].values():
    # 检查是否为面区域且边界类型为wall
    if zone['type'] == 'faces' and zone.get('bc_type') == 'wall':
        # 遍历该区域的所有面
        for face in zone['data']:
            wall_faces.append(face)
            # 获取面的节点索引（注意Fluent节点索引从1开始）
            node_indices = [n - 1 for n in face['nodes']]  # 转换为Python的0基索引
            # 收集节点坐标
            nodes = [node_coords[i] for i in node_indices]
            wall_nodes_coords.append(nodes)

# 输出第一个wall面的节点坐标
if wall_nodes_coords:
    print("第一个wall面的节点坐标：")
    for coord in wall_nodes_coords[0]:
        print(f"({coord[0]:.3f}, {coord[1]:.3f})")

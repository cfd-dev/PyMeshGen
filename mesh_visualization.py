import matplotlib.pyplot as plt

def visualize_mesh_2d(grid, wall_nodes, vector_scale=1.0):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制所有节点
    all_nodes = grid['nodes']
    xs = [n[0] for n in all_nodes]
    ys = [n[1] for n in all_nodes]
    ax.scatter(xs, ys, c='gray', s=10, alpha=0.3, label='All Nodes')
    
    # 绘制Wall节点
    wall_xs = [n['coords'][0] for n in wall_nodes]
    wall_ys = [n['coords'][1] for n in wall_nodes]
    ax.scatter(wall_xs, wall_ys, c='red', s=20, label='Wall Nodes')
    
    # 绘制Wall面结构
    for zone in grid['zones'].values():
        if zone.get('bc_type') == 'wall' and zone['type'] == 'faces':
            for face in zone['data']:
                # 转换为0-based索引
                coords = [all_nodes[n-1] for n in face['nodes']]
                x = [c[0] for c in coords]
                y = [c[1] for c in coords]
                ax.plot(x, y, c='orange', alpha=0.5, lw=1.5)
    
    # 绘制推进向量
    for node_info in wall_nodes:
        vec = node_info['march_vector']
        if not vec:
            continue
            
        x, y = node_info['coords'][0], node_info['coords'][1]
        dx, dy = vec[0], vec[1]
        
        # 绘制箭头
        ax.arrow(x, y, 
                 dx * vector_scale, 
                 dy * vector_scale,
                 head_width=0.05,
                 head_length=0.1,
                 fc='blue',
                 ec='blue',
                 alpha=0.7,
                 length_includes_head=True)
    
    # 设置图形属性
    ax.set_title("2D Mesh Visualization with March Vectors")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.axis('equal')
    plt.show()
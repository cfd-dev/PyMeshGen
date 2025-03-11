import matplotlib.pyplot as plt
import numpy as np
def visualize_mesh_2d(grid, wall_nodes, vector_scale=0.3):
    fig, ax = plt.subplots(figsize=(10, 8))
      
    # 预处理：计算每个wall面的边长（排序节点避免重复）
    face_length = {}
    for zone in grid['zones'].values():
        if zone.get('bc_type') == 'wall' and zone['type'] == 'faces':
            for face in zone['data']:
                sorted_nodes = sorted(face['nodes'])
                nodes = [grid['nodes'][n-1] for n in sorted_nodes]
                dx = nodes[1][0] - nodes[0][0]
                dy = nodes[1][1] - nodes[0][1]
                length = np.hypot(dx, dy)
                face_length[tuple(sorted_nodes)] = length

    # 绘制所有节点
    xs = [n[0] for n in grid['nodes']]
    ys = [n[1] for n in grid['nodes']]
    ax.scatter(xs, ys, c='gray', s=10, alpha=0.3, label='All Nodes')
    
    # 绘制Wall节点
    wall_xs = [n['coords'][0] for n in wall_nodes]
    wall_ys = [n['coords'][1] for n in wall_nodes]
    ax.scatter(wall_xs, wall_ys, c='red', s=20, label='Wall Nodes')

    # 绘制Wall面结构
    for zone in grid['zones'].values():
        if zone.get('bc_type') == 'wall' and zone['type'] == 'faces':
            for face in zone['data']:
                coords = [grid['nodes'][n-1] for n in face['nodes']]
                x = [c[0] for c in coords]
                y = [c[1] for c in coords]
                ax.plot(x, y, c='orange', alpha=0.5, lw=1.5)

    # 绘制推进向量（使用quiver优化性能）
    x, y, u, v = [], [], [], []
    for node_info in wall_nodes:
        vec = node_info.get('march_vector')
        if not vec:
            continue
            
        faces = node_info.get('node_wall_faces', [])
        if not faces:
            continue
            
        # 计算平均面长
        total_length = 0
        for face in faces:
            sorted_nodes = sorted(face['nodes'])
            total_length += face_length.get(tuple(sorted_nodes), 0.0)
        avg_length = total_length / len(faces)
            
        scale = vector_scale * avg_length
        
        x.append(node_info['coords'][0])
        y.append(node_info['coords'][1])
        u.append(vec[0] * scale)
        v.append(vec[1] * scale)
    
    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, 
              headwidth=3, headlength=4, color='blue', alpha=0.7, width=0.003)

    # 图形设置
    ax.set_title("2D Mesh Visualization with March Vectors")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.axis('equal')
    plt.show()
    
# def visualize_mesh_2d(grid, wall_nodes, vector_scale=0.3):
#     fig, ax = plt.subplots(figsize=(10, 8))

#     # 预处理：计算每个wall面的边长（排序节点避免重复）
#     face_length = {}
#     for zone in grid['zones'].values():
#         if zone.get('bc_type') == 'wall' and zone['type'] == 'faces':
#             for face in zone['data']:
#                 sorted_nodes = sorted(face['nodes'])
#                 nodes = [grid['nodes'][n-1] for n in sorted_nodes]
#                 dx = nodes[1][0] - nodes[0][0]
#                 dy = nodes[1][1] - nodes[0][1]
#                 length = np.hypot(dx, dy)
#                 face_length[tuple(sorted_nodes)] = length

#     # 绘制所有节点
#     xs = [n[0] for n in grid['nodes']]
#     ys = [n[1] for n in grid['nodes']]
#     ax.scatter(xs, ys, c='gray', s=10, alpha=0.3, label='All Nodes')
    
#     # 绘制Wall节点
#     wall_xs = [n['coords'][0] for n in wall_nodes]
#     wall_ys = [n['coords'][1] for n in wall_nodes]
#     ax.scatter(wall_xs, wall_ys, c='red', s=20, label='Wall Nodes')

#     # 绘制Wall面结构
#     for zone in grid['zones'].values():
#         if zone.get('bc_type') == 'wall' and zone['type'] == 'faces':
#             for face in zone['data']:
#                 coords = [grid['nodes'][n-1] for n in face['nodes']]
#                 x = [c[0] for c in coords]
#                 y = [c[1] for c in coords]
#                 ax.plot(x, y, c='orange', alpha=0.5, lw=1.5)

#     # 预处理推进向量数据
#     directions = []  # 归一化方向向量
#     initial_scales = []  # 初始缩放因子 (vector_scale * avg_length)
#     x_data, y_data = [], []
    
#     for node_info in wall_nodes:
#         vec = node_info.get('march_vector')
#         if not vec:
#             continue
            
#         faces = node_info.get('node_wall_faces', [])
#         if not faces:
#             continue
            
#         # 计算平均面长
#         total_length = 0
#         for face in faces:
#             sorted_nodes = sorted(face['nodes'])
#             total_length += face_length.get(tuple(sorted_nodes), 0.0)
#         avg_length = total_length / len(faces)
        
#         # 归一化向量
#         vec_np = np.array(vec)
#         vec_norm = np.linalg.norm(vec_np)
#         if vec_norm < 1e-6:
#             continue
#         vec_np_normalized = vec_np / vec_norm
        
#         # 保存数据
#         directions.append(vec_np_normalized)
#         initial_scales.append(vector_scale * avg_length)
#         x_data.append(node_info['coords'][0])
#         y_data.append(node_info['coords'][1])

#     # 绘制初始箭头
#     u_init = [d[0] * s for d, s in zip(directions, initial_scales)]
#     v_init = [d[1] * s for d, s in zip(directions, initial_scales)]
#     q = ax.quiver(
#         x_data, y_data, u_init, v_init,
#         angles='xy', 
#         scale_units='xy', 
#         scale=1,
#         headwidth=3, 
#         headlength=4, 
#         color='blue', 
#         alpha=0.7,
#         width=0.003
#     )

#     # 设置坐标轴等比例并获取初始范围
#     ax.axis('equal')
#     plt.draw()
#     initial_xlim = ax.get_xlim()
#     initial_ylim = ax.get_ylim()

#     # 缩放事件回调函数
#     def on_zoom(event):
#         ax = event.axes
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
        
#         # 计算当前视图范围与初始范围的比例
#         current_width = xlim[1] - xlim[0]
#         current_height = ylim[1] - ylim[0]
#         if current_width == 0 or current_height == 0:
#             return
        
#         view_scale_x = (initial_xlim[1] - initial_xlim[0]) / current_width
#         view_scale_y = (initial_ylim[1] - initial_ylim[0]) / current_height
#         view_scale = (view_scale_x + view_scale_y) / 2  # 取平均值
        
#         # 计算动态缩放后的箭头分量
#         new_u = [d[0] * s * view_scale for d, s in zip(directions, initial_scales)]
#         new_v = [d[1] * s * view_scale for d, s in zip(directions, initial_scales)]
        
#         # 更新箭头
#         q.set_UVC(new_u, new_v)
#         plt.draw()

#     # 绑定事件
#     ax.callbacks.connect('xlim_changed', on_zoom)
#     ax.callbacks.connect('ylim_changed', on_zoom)

#     # 图形设置
#     ax.set_title("2D Mesh Visualization with Dynamic March Vectors")
#     ax.set_xlabel("X Coordinate")
#     ax.set_ylabel("Y Coordinate")
#     ax.legend()
#     plt.show()
        
# def visualize_mesh_2d(grid, wall_nodes, vector_scale=1.0):
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     # 绘制所有节点
#     all_nodes = grid['nodes']
#     xs = [n[0] for n in all_nodes]
#     ys = [n[1] for n in all_nodes]
#     ax.scatter(xs, ys, c='gray', s=10, alpha=0.3, label='All Nodes')
    
#     # 绘制Wall节点
#     wall_xs = [n['coords'][0] for n in wall_nodes]
#     wall_ys = [n['coords'][1] for n in wall_nodes]
#     ax.scatter(wall_xs, wall_ys, c='red', s=20, label='Wall Nodes')
    
#     # 绘制Wall面结构
#     for zone in grid['zones'].values():
#         if zone.get('bc_type') == 'wall' and zone['type'] == 'faces':
#             for face in zone['data']:
#                 # 转换为0-based索引
#                 coords = [all_nodes[n-1] for n in face['nodes']]
#                 x = [c[0] for c in coords]
#                 y = [c[1] for c in coords]
#                 ax.plot(x, y, c='orange', alpha=0.5, lw=1.5)
    
#     # 绘制推进向量
#     for node_info in wall_nodes:
#         vec = node_info['march_vector']
#         if not vec:
#             continue
            
#         x, y = node_info['coords'][0], node_info['coords'][1]
#         dx, dy = vec[0], vec[1]
        
#         # 绘制箭头
#         ax.arrow(x, y, 
#                  dx * vector_scale, 
#                  dy * vector_scale,
#                  head_width=0.05,
#                  head_length=0.1,
#                  fc='blue',
#                  ec='blue',
#                  alpha=0.7,
#                  length_includes_head=True)
    
#     # 设置图形属性
#     ax.set_title("2D Mesh Visualization with March Vectors")
#     ax.set_xlabel("X Coordinate")
#     ax.set_ylabel("Y Coordinate")
#     ax.legend()
#     ax.axis('equal')
#     plt.show()
    
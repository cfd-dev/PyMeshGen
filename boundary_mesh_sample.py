import read_cas as rc
import mesh_reconstruction as mesh_recons
import grid_sample as gsam 
import mesh_visualization as mesh_vis

file_path = './sample/convex.cas'
grid = rc.parse_fluent_msh(file_path)
mesh_recons.preprocess_grid(grid)

wall_faces = []
wall_nodes = []
march_vector = []
processed_nodes = set()
# 遍历所有区域
for zone in grid['zones'].values():
    # 检查是否为面区域且边界类型为wall
    if zone['type'] == 'faces' and zone.get('bc_type') == 'wall':
        # 遍历该区域的所有面
        for face in zone['data']:
            wall_faces.append(face)
            # 获取面的节点索引（注意Fluent节点索引从1开始）
            node_indices_0based = [n - 1 for n in face['nodes']]  # 转换为Python的0基索引
            
            for node_idx in node_indices_0based:
                if node_idx in processed_nodes:
                    continue
                processed_nodes.add(node_idx)
                
                node_1based = node_idx + 1
                
                # 计算推进向量
                try:
                    march_vector = gsam.get_march_vector(grid, node_1based, face)
                except Exception as e:
                    print(f"Error calculating vector for node {node_1based}: {e}")
                    print(f"Face nodes: {face['nodes']}")
                    vector = None 
                                   
                wall_nodes.append({'original_indices': node_idx,
                                   'coords': grid['nodes'][node_idx],
                                   'march_vector': march_vector
                                   })

# 过滤掉无效向量
valid_wall_nodes = [n for n in wall_nodes if n['march_vector']]

print(f"Total wall nodes: {len(wall_nodes)}")
print(f"Valid vectors: {len(valid_wall_nodes)}")
                      
mesh_vis.visualize_mesh_2d(grid, valid_wall_nodes, vector_scale=0.3)




import os
import read_cas as rc
import mesh_reconstruction as mesh_recons
import grid_sample as gsam 

def process_single_file(file_path):
    """
    处理单个网格文件，提取wall节点和推进向量
    
    参数:
    file_path (str): 文件路径
    
    返回:
    dict: 包含wall_faces, wall_nodes, valid_wall_nodes等信息
    """
    print(f"Processing file: {file_path}")
    
    # 解析网格文件
    try:
        grid = rc.parse_fluent_msh(file_path)
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None
    
    # 预处理网格
    mesh_recons.preprocess_grid(grid)

    # 初始化存储结构
    wall_faces = []
    node_dict = {}  # 使用字典暂存节点信息，避免重复
    
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
                    # 初始化节点信息
                    if node_idx not in node_dict:
                        node_dict[node_idx] = {
                            'original_indices': node_idx,
                            'coords': grid['nodes'][node_idx],
                            'node_wall_faces': [],          # 存储该节点所属的所有wall面
                            'march_vector': None  # 后续计算
                        }                       
                        
                    # 添加当前面到节点的faces列表
                    node_dict[node_idx]['node_wall_faces'].append(face)
                      
# 转换为列表并计算推进向量
    wall_nodes = list(node_dict.values())
    for node_info in wall_nodes:
        node_1based = node_info['original_indices'] + 1
        # 选择第一个关联的面进行计算（可根据需求调整策略）           
        if node_info['node_wall_faces']:
            face = node_info['node_wall_faces'][0]
            try:
                node_info['march_vector'] = gsam.get_march_vector(grid, node_1based, face)
            except Exception as e:
                print(f"Error calculating vector for node {node_1based}: {e}")
    
    # 过滤无效向量
    valid_wall_nodes = [n for n in wall_nodes if n['march_vector']]
    
    # 打印统计信息
    print(f"File: {file_path}")
    print(f"Total wall nodes: {len(wall_nodes)}")
    print(f"Valid vectors: {len(valid_wall_nodes)}")
    
    # mesh_vis.visualize_mesh_2d(grid, valid_wall_nodes, vector_scale=0.3)
    
    return {
        'file_path': file_path,
        'grid': grid,
        'wall_faces': wall_faces,
        'wall_nodes': wall_nodes,
        'valid_wall_nodes': valid_wall_nodes
    }                                    

    
def batch_process_files(folder_path):
    """
    批量处理指定文件夹中的所有网格文件
    
    参数:
    folder_path (str): 文件夹路径
    
    返回:
    list: 每个文件的处理结果列表
    """
    results = []
    
    # 遍历文件夹中的所有.cas文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.cas'):
            file_path = os.path.join(folder_path, file_name)
            result = process_single_file(file_path)
            if result:
                results.append(result)
    
    return results
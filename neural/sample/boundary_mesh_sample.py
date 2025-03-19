import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent /"fileIO"))
sys.path.append(str(Path(__file__).parent.parent /"sample"))
sys.path.append(str(Path(__file__).parent.parent /"data_structure"))
import grid_sample as gsam
import read_cas as rc
import mesh_reconstruction as mesh_recons

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
    
    # === 新增归一化预处理 ===
    # 获取所有节点坐标
    all_nodes = grid['nodes']
    if not all_nodes:
        print("No nodes found in grid.")
        return None
    
    # 计算各维度范围
    xs = [n[0] for n in all_nodes]
    ys = [n[1] for n in all_nodes]
    zs = [n[2] for n in all_nodes] if len(all_nodes[0]) > 2 else []
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = (min(zs), max(zs)) if zs else (0.0, 0.0)
    
    # 生成归一化后的坐标列表
    normalized_nodes = []
    for node in all_nodes:
        # 处理每个坐标轴
        x_range = x_max - x_min
        norm_x = (node[0] - x_min) / x_range if x_range != 0 else 0.5
        
        y_range = y_max - y_min
        norm_y = (node[1] - y_min) / y_range if y_range != 0 else 0.5
        
        if len(node) > 2 and zs:
            z_range = z_max - z_min
            norm_z = (node[2] - z_min) / z_range if z_range != 0 else 0.5
            normalized_nodes.append((norm_x, norm_y, norm_z))
        else:
            normalized_nodes.append((norm_x, norm_y))

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
                            'coords': normalized_nodes[node_idx], 
                            'node_wall_faces': [], # 存储与该节点相关的所有wall面         
                            'march_vector': None  
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
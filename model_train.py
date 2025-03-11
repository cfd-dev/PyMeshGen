import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt  
import boundary_mesh_sample as bl_samp 
import mesh_visualization as mesh_vis

def build_graph_data(wall_nodes, wall_faces):
    # 创建原始索引到wall_nodes索引的映射
    index_map = {node['original_indices']: i for i, node in enumerate(wall_nodes)}
    
    edge_set = set()  # 使用集合去重
    
    for face in wall_faces:
        # 获取面的节点索引（转换为0-based）
        nodes_0based = [n-1 for n in face['nodes']]
        
        # 根据面类型决定连接方式
        if len(nodes_0based) == 2:  # 线性面
            i, j = nodes_0based
            if i in index_map and j in index_map:
                a, b = index_map[i], index_map[j]
                edge_set.add((a, b))
                edge_set.add((b, a))  # 无向图
        else:  # 多边形面（三角形/四边形等）
            # 循环连接相邻节点
            for i in range(len(nodes_0based)):
                current = nodes_0based[i]
                next_node = nodes_0based[(i+1) % len(nodes_0based)]
                if current in index_map and next_node in index_map:
                    a, b = index_map[current], index_map[next_node]
                    edge_set.add((a, b))
                    edge_set.add((b, a))
    
    # 转换为tensor格式
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    
    # 处理节点特征和标签
    x = torch.tensor([node['coords'][:2] for node in wall_nodes], dtype=torch.float)
    y = torch.tensor([node['march_vector'][:2] for node in wall_nodes], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)


def visualize_graph_structure(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制节点
    ax.scatter(data.x[:,0], data.x[:,1], c='blue', s=20)
    
    # 绘制边
    for i in range(data.edge_index.shape[1]):
        a = data.edge_index[0, i].item()
        b = data.edge_index[1, i].item()
        ax.plot([data.x[a,0], data.x[b,0]],
                [data.x[a,1], data.x[b,1]],
                c='gray', alpha=0.3)
    
    ax.set_title("Graph Structure Visualization")
    plt.show()
    
class EnhancedGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(2, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, 32)
        self.fc2 = torch.nn.Linear(32, 2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
     
def visualize_graph_structure(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制节点
    ax.scatter(data.x[:,0], data.x[:,1], c='blue', s=20)
    
    # 绘制边
    for i in range(data.edge_index.shape[1]):
        a = data.edge_index[0, i].item()
        b = data.edge_index[1, i].item()
        ax.plot([data.x[a,0], data.x[b,0]],
                [data.x[a,1], data.x[b,1]],
                c='gray', alpha=0.3)
    
    ax.set_title("Graph Structure Visualization")
    plt.show()


def visualize_predictions(data, model, vector_scale=0.3):
    """
    可视化真实向量与预测向量对比
    
    参数:
    data (Data): 图数据对象
    model (torch.nn.Module): 训练好的GNN模型
    vector_scale (float): 向量缩放因子（防止箭头过长）
    """
    model.eval()
    with torch.no_grad():
        pred = model(data).cpu().numpy()
    true = data.y.cpu().numpy()
    coords = data.x.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制节点位置
    ax.scatter(coords[:, 0], coords[:, 1], c='black', s=20, label='Nodes')
    
    # 绘制真实向量（蓝色）
    for i in range(len(true)):
        if not np.any(np.isnan(true[i])):
            dx, dy = true[i] * vector_scale
            ax.arrow(coords[i, 0], coords[i, 1], dx, dy,
                     head_width=0.05, head_length=0.1,
                     fc='blue', ec='blue', alpha=0.6,
                     length_includes_head=True)
    
    # 绘制预测向量（红色）
    for i in range(len(pred)):
        if not np.any(np.isnan(pred[i])):
            dx, dy = pred[i] * vector_scale
            ax.arrow(coords[i, 0], coords[i, 1], dx, dy,
                     head_width=0.05, head_length=0.1,
                     fc='red', ec='red', alpha=0.6,
                     length_includes_head=True)
    
    # 创建图例
    blue_arrow = plt.Line2D([0], [0], color='blue', lw=2, label='True Vector')
    red_arrow = plt.Line2D([0], [0], color='red', lw=2, label='Predicted Vector')
    ax.legend(handles=[blue_arrow, red_arrow])
    
    ax.set_title("March Vector Prediction vs Ground Truth")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.axis('equal')
    plt.tight_layout()
    plt.show()    

# 主程序入口
if __name__ == "__main__":
    folder_path = './sample'  
    all_results = bl_samp.batch_process_files(folder_path)
    
    # 示例：可视化第一个文件的结果
    if all_results:
        first_result = all_results[3]
        mesh_vis.visualize_mesh_2d(
            first_result['grid'], 
            first_result['valid_wall_nodes'], 
            vector_scale=0.3
        )
        
    # 1. 数据预处理
    data = build_graph_data(valid_wall_nodes, wall_faces)
    visualize_graph_structure(data)
    # 查看图结构
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges}")
    print(f"特征维度: {data.num_node_features}")

    # 2. 模型初始化
    model = EnhancedGNN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # 3. 训练循环
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # 4. 可视化验证
    visualize_predictions(data, model)

    input("按回车键退出...")
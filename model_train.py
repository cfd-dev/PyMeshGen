import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv  
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
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

def add_edge_features(data):
    row, col = data.edge_index
    edge_attr = data.x[row, :dim_coords] - data.x[col, :dim_coords]  # 相对坐标差
    data.edge_attr = edge_attr
    return data

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

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        # 图注意力卷积层
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=0.2
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,  # 多头输出拼接后的维度
            out_channels=hidden_channels,
            heads=heads,
            dropout=0.2
        )
        self.conv3 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,  # 最后一层用单头，简化输出维度
            concat=False  # 不拼接，输出维度保持hidden_channels
        )
        
        # 全连接层
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 图注意力卷积部分
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        
        # 全连接输出
        x = self.fc(x)
        return x

class EnhancedGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        
        self.coord_encoder = nn.Linear(2, hidden_dim)

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)  # 新增BN层

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)  # 新增BN层

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)  # 新增BN层

        self.fc1 = torch.nn.Linear(hidden_channels, 32)
        self.bn_fc1 = torch.nn.BatchNorm1d(32)  # 全连接层后BN
        
        self.fc2 = torch.nn.Linear(32, 2)
        self.tanh = torch.nn.Tanh()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 编码坐标
        x = F.relu(self.coord_encoder(x))

        # 第一层残差
        identity1 = x  # 保存当前层输入
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        identity1 = self._adjust_identity(identity1, x)  # 新增维度适配
        x = x + identity1  # 与当前层输入相加

        # 第二层残差
        identity2 = x  # 使用上一层的输出作为下一层残差输入
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = x + identity2

        # 第三层残差
        identity3 = x
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = x + identity3

        # 全连接部分
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.tanh(self.fc2(x))
        return x

    def _adjust_identity(self, identity, x):
        """维度适配器，当残差连接维度不匹配时进行线性变换"""
        if identity.size(1) != x.size(1):
            return torch.nn.Linear(identity.size(1), x.size(1)).to(x.device)(identity)
        return identity

def visualize_predictions(data, model, vector_scale=None, head_scale=None):
    """
    可视化真实向量与预测向量对比

    参数:
    data (Data): 图数据对象
    model (torch.nn.Module): 训练好的GNN模型
    vector_scale (float): [可选] 手动指定向量缩放因子，None表示自动计算
    head_scale (float): [可选] 手动指定箭头尺寸缩放因子，None表示自动计算
    """
    model.eval()
    with torch.no_grad():
        pred = model(data).cpu().numpy()
    true = data.y.cpu().numpy()
    coords = data.x.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制节点位置
    ax.scatter(coords[:, 0], coords[:, 1], c='black', s=20, label='Nodes')

    # 计算动态箭头参数
    # 计算坐标范围
    x_min, x_max = coords[:,0].min(), coords[:,0].max()
    y_min, y_max = coords[:,1].min(), coords[:,1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    avg_range = (x_range + y_range) / 2

    # 自动计算参数（新增部分）
    if vector_scale is None:
        vector_scale = 0.15 / avg_range if avg_range > 0 else 0.3
    if head_scale is None:
        head_scale = 0.02 / avg_range if avg_range > 0 else 0.01

    # 统一箭头尺寸参数（保持原有逻辑）
    base_size = avg_range * head_scale
    head_width = base_size * 2.5
    head_length = base_size * 4

    # 修改箭头绘制部分（原117-136行）
    # 绘制真实向量（蓝色）
    for i in range(len(true)):
        if not np.any(np.isnan(true[i])):
            dx, dy = true[i] * vector_scale
            ax.arrow(coords[i, 0], coords[i, 1], dx, dy,
                     head_width=head_width, 
                     head_length=head_length,  # 替换固定数值
                     fc='blue', ec='blue', alpha=0.6,
                     length_includes_head=True)

    # 绘制预测向量（红色）
    for i in range(len(pred)):
        if not np.any(np.isnan(pred[i])):
            dx, dy = pred[i] * vector_scale
            ax.arrow(coords[i, 0], coords[i, 1], dx, dy,
                     head_width=head_width,
                     head_length=head_length,  # 替换固定数值
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
    plt.show(block=True)

if __name__ == "__main__":
    # -------------------------- 初始化配置 --------------------------
    # 硬件设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前运行设备: {device}")
    
    # 路径配置
    folder_path = './sample'  # 原始数据目录
    model_save_path = './model/saved_model.pth'  # 模型保存路径
    
    # -------------------------- 超参数配置 --------------------------
    config = {
        'hidden_channels': 64,  # GNN隐藏层维度
        'learning_rate': 0.001,  # 降低学习率
        'total_epochs': 50,      # 总训练轮次（新增）
        'epochs_per_dataset': 1000, # 每个数据集每次迭代训练次数（新增）
        'log_interval': 20
    }

    # -------------------------- 数据准备 --------------------------
    # 批量处理边界采样数据
    try:
        all_results = bl_samp.batch_process_files(folder_path)
        print(f"成功加载 {len(all_results)} 个数据集")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit(1)

    # -------------------------- 模型初始化 --------------------------
    # 创建模型实例并转移到指定设备
    model = EnhancedGNN(hidden_channels=config['hidden_channels']).to(device)
    print("\n网络层详细信息：")
    print(model)
    # 新增参数数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总可训练参数数量：{total_params:,}")

    # 优化器和损失函数配置
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()

    # -------------------------- 训练监控 --------------------------
    # 初始化实时损失曲线
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    train_losses = []  # 全局损失记录
    line, = ax.plot([], [], 'r-')
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Accumulated Epochs")
    ax.set_ylabel("Loss")

    # -------------------------- 训练流程 --------------------------
    try:
        # 外层循环：总训练轮次
        for total_epoch in range(config['total_epochs']):
           # 内层循环：遍历所有数据集
            for dataset_idx, result in enumerate(all_results):
                # 1. 数据预处理
                wall_nodes = result['valid_wall_nodes']
                wall_faces = result['wall_faces']
            
                # 构建图数据结构
                data = build_graph_data(wall_nodes, wall_faces).to(device)
                add_edge_features(data)
                print(f"\n数据集 {dataset_idx+1}/{len(all_results)} 包含 {data.num_nodes} 个节点")
            
                # 可视化初始图结构
                # visualize_graph_structure(data.cpu())

                # 单个数据集训练阶段
                model.train()
                for epoch in range(config['epochs_per_dataset']):
                    optimizer.zero_grad()
                    out = model(data)
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()

                    # 记录损失值
                    train_losses.append(loss.item())

                    # 定期更新训练信息（调整日志格式）
                    if epoch % config['log_interval'] == 0:
                        print(f"总轮次[{total_epoch+1}/{config['total_epochs']}] " 
                              f"数据集[{dataset_idx+1}/{len(all_results)}] "
                              f"局部轮次[{epoch}/{config['epochs_per_dataset']}] "
                              f"损失: {loss.item():.4f}")                    
                        # 更新损失曲线
                        line.set_data(range(len(train_losses)), train_losses)
                        ax.relim()
                        ax.autoscale_view()
                        plt.draw()
                        plt.pause(0.01)  # 维持图像响应

                # 3. 单数据集验证
                # plt.ioff()
                # visualize_predictions(data.cpu(), model.cpu())
                # plt.ion()
            
        model.to(device)  # 确保模型回到正确设备
        # 每个总epoch结束后保存模型
        torch.save(model.state_dict(), model_save_path)
        print(f"\n模型已保存至 {model_save_path}（第{total_epoch+1}轮）")

    except KeyboardInterrupt:
        print("\n训练被用户中断！")
    finally:
        # -------------------------- 收尾工作 --------------------------
        # 最终保存模型
        torch.save(model.state_dict(), model_save_path)
        print(f"\n模型已保存至 {model_save_path}")
        
    # 关闭交互式绘图
    plt.ioff()
    plt.close()

    input("训练完成，按回车键退出...")
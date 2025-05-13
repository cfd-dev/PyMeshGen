import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from grid_sample import batch_process_files
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
from utils.message import info, warning, error, debug


def add_edge_features(data):
    row, col = data.edge_index
    edge_attr = data.x[col, :2] - data.x[row, :2]  # 相对坐标差
    data.edge_attr = edge_attr
    return data


def build_graph_data(wall_nodes, wall_faces):
    # 创建原始索引到wall_nodes索引的映射
    index_map = {node["original_indices"]: i for i, node in enumerate(wall_nodes)}

    edge_set = set()  # 使用集合去重

    for face in wall_faces:
        # 获取面的节点索引（转换为0-based）
        nodes_0based = [n - 1 for n in face["nodes"]]

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
                next_node = nodes_0based[(i + 1) % len(nodes_0based)]
                if current in index_map and next_node in index_map:
                    a, b = index_map[current], index_map[next_node]
                    edge_set.add((a, b))
                    edge_set.add((b, a))

    # 转换为tensor格式
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()

    # 处理节点特征和标签
    x = torch.tensor([node["coords"][:2] for node in wall_nodes], dtype=torch.float)
    y = torch.tensor(
        [node["march_vector"][:2] for node in wall_nodes], dtype=torch.float
    )

    data = Data(x=x, edge_index=edge_index, y=y)

    data = add_edge_features(data)

    return data


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(
            in_channels,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=2,
            dropout=0.2,
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=2,
            dropout=0.2,
        )
        self.conv3 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            edge_dim=2,
            dropout=0.2,
        )
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # 第一层：多头注意力 + 边特征
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(x)
        # 第二层
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(x)
        # 第三层（单头）
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        # 全连接输出
        x = self.fc(x)
        return x


class EnhancedGNN(torch.nn.Module):

    def __init__(
        self,
        hidden_channels=64,
        num_gcn_layers=4,
        residual_switch=True,
        dropout=0.3,
        normalization="None",
    ):
        super().__init__()
        self.residual_switch = residual_switch  # 新增残差开关
        self.normalization = normalization  # 新增标准化类型
        self.hidden_channels = hidden_channels  # 新增隐藏层维度
        self.num_gcn_layers = num_gcn_layers  # 新增GCN层数

        self.norm_layers = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(hidden_channels),  # 节点级标准化
                torch.nn.BatchNorm1d(hidden_channels),  # 批量标准化
                torch.nn.InstanceNorm1d(hidden_channels),  # 图级标准化
            ]
        )

        self.coord_encoder = torch.nn.Linear(2, hidden_channels)

        self.convs = torch.nn.ModuleList(
            [
                GCNConv(hidden_channels, hidden_channels)
                for _ in range(num_gcn_layers)  # 增加层数
            ]
        )

        self.residual_adapter = torch.nn.Linear(hidden_channels, hidden_channels)

        # 添加Dropout
        self.dropout = torch.nn.Dropout(dropout)

        self.fc1 = torch.nn.Linear(hidden_channels, 32)
        self.fc2 = torch.nn.Linear(32, 2)
        self.tanh = torch.nn.Tanh()

    def complexity_check(self, data):
        with torch.no_grad():
            out = self(data)
            mem_usage = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
        return mem_usage  # 返回显存占用情况

    def Normalization(self, x):
        if self.normalization == "None":
            return x
        elif self.normalization == "Layer":
            x = self.norm_layers[0](x)  # 初始编码后使用LayerNorm
        elif self.normalization == "Batch":
            x = self.norm_layers[1](x)  # 初始编码后使用BatchNorm
        elif self.normalization == "Instance":
            x = self.norm_layers[2](x)  # 初始编码后使用InstanceNorm

        return x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        self.toggle_residual(self.residual_switch)

        # 编码坐标
        x = F.relu(self.coord_encoder(x))
        x = self.Normalization(x)  # 初始编码后使用LayerNorm

        # 多层GCN
        for conv in self.convs:
            res = x
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.residual_switch:
                res = self._adjust_identity(res, x)
                x = x + res
            else:
                x = x
            x = self.Normalization(x)
            x = self.dropout(x)

        # 全连接部分
        x = F.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x

    def toggle_residual(self, mode: bool):
        """残差连接开关"""
        self.residual_switch = mode

    def _adjust_identity(self, identity, x):
        """维度适配器，当残差连接维度不匹配时进行线性变换"""
        # if identity.size(1) != x.size(1):
        #     return torch.nn.Linear(identity.size(1), x.size(1)).to(x.device)(identity)
        # return identity
        if identity.size(1) != x.size(1):
            # return self.residual_adapter(identity)
            raise ValueError(
                "残差连接的输入输出维度不一致，请确保hidden_channels一致！"
            )
        return identity


if __name__ == "__main__":
    # -------------------------- 初始化配置 --------------------------
    # 硬件设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    info(f"当前运行设备: {device}")

    # 路径配置
    current_dir = Path(__file__).parent
    folder_path = current_dir / "sample_grids/training"  # 原始数据目录
    model_save_path = current_dir / "model/saved_model.pth"  # 模型保存路径

    # -------------------------- 超参数配置 --------------------------
    config = {
        "train_ratio": 1.0,  # 训练集比例
        "batch_size": 3,  # 批量大小
        "hidden_channels": 64,  # GNN隐藏层维度
        "learning_rate": 0.001,  # 学习率
        "total_epochs": 20000,  # 总训练轮次
        "log_interval": 50,
        "validation_interval": 200,  # 验证间隔
    }

    # -------------------------- 数据准备 --------------------------
    # 批量处理边界采样数据
    try:
        all_results = batch_process_files(folder_path)
        info(f"成功加载 {len(all_results)} 个数据集")
    except Exception as e:
        error(f"数据加载失败: {str(e)}")
        exit(1)

    dataset = [
        build_graph_data(result["valid_wall_nodes"], result["wall_faces"])
        for result in all_results
    ]
    train_size = int(config["train_ratio"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    # -------------------------- 临时措施 --------------------------
    val_loader = train_loader
    val_dataset = train_dataset

    # -------------------------- 模型初始化 --------------------------
    # 创建模型实例并转移到指定设备
    model = EnhancedGNN(
        hidden_channels=config["hidden_channels"],
        num_gcn_layers=4,
        residual_switch=False,
        dropout=0.0,
        normalization="Batch",
    ).to(device)
    info("\n网络层详细信息：")
    info(model)
    total_params = sum(p.numel() for p in model.parameters())
    info(f"\n总可训练参数数量：{total_params:,}")

    # 优化器和损失函数配置
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.MSELoss()

    # -------------------------- 训练监控 --------------------------
    # 初始化实时损失曲线
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    train_losses, val_losses = [], []
    (line_train,) = ax.plot([], [], "r-", label="Train Loss")
    (line_val,) = ax.plot([], [], "b-", label="Val Loss")
    ax.set_title("Training & Validation Loss Curve")
    ax.set_xlabel("Accumulated Steps")
    ax.set_ylabel("Loss")
    ax.legend()

    # -------------------------- 训练流程 --------------------------
    try:
        global_step = 0  # 新增全局步数计数器
        for epoch in range(config["total_epochs"]):
            model.train()
            for batch_data in train_loader:
                global_step += 1
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                out = model(batch_data)
                loss = criterion(out, batch_data.y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                # 定期更新训练信息
                if global_step % config["log_interval"] == 0:
                    info(
                        f"当前步数[{global_step}] 轮次[{epoch+1}/{config['total_epochs']}]"
                        f" 损失: {loss.item():.4f}"
                    )

                if global_step % config["validation_interval"] == 0:
                    model.eval()
                    total_val_loss = 0.0
                    with torch.no_grad():
                        for val_data in val_loader:  # 遍历整个验证集
                            val_data = val_data.to(device)
                            val_out = model(val_data)
                            val_loss = criterion(val_out, val_data.y)
                            total_val_loss += val_loss.item() * val_data.num_graphs
                    avg_val_loss = total_val_loss / len(val_dataset)  # 计算平均验证损失
                    val_losses.append(avg_val_loss)
                    model.train()
                    info(f"训练损失: {loss.item():.4f} 验证损失: {avg_val_loss:.4f}")

                # 更新损失曲线
                if global_step % config["log_interval"] == 0:
                    line_train.set_data(range(len(train_losses)), train_losses)
                    line_val.set_data(
                        [
                            i * config["validation_interval"]
                            for i in range(len(val_losses))
                        ],
                        val_losses,
                    )
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.01)  # 维持图像响应

                model.to(device)  # 确保模型回到正确设备

    except KeyboardInterrupt:
        error("\n训练被用户中断！")
    finally:
        torch.save(model.state_dict(), model_save_path)
        info(f"\n模型已保存至 {model_save_path}")
        plt.ioff()
        input("训练完成，按回车键退出...")
        plt.close()

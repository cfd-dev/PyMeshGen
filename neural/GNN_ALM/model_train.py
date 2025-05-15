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
from config import MODEL_CONFIG, TRAINING_CONFIG

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
                # edge_set.add((b, a))  # 无向图
        else:  # 多边形面（三角形/四边形等）
            # 循环连接相邻节点
            for i in range(len(nodes_0based)):
                current = nodes_0based[i]
                next_node = nodes_0based[(i + 1) % len(nodes_0based)]
                if current in index_map and next_node in index_map:
                    a, b = index_map[current], index_map[next_node]
                    edge_set.add((a, b))
                    # edge_set.add((b, a))

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

    def __init__(
        self,
        hidden_channels=64,
        num_gat_layers=3,
        residual_switch=True,
        dropout=0.3,
        normalization="None",
        heads=4,
    ):
        super().__init__()
        self.residual_switch = residual_switch
        self.normalization = normalization
        self.hidden_channels = hidden_channels
        self.num_gat_layers = num_gat_layers
        self.heads = heads

        self.norm_layers = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.InstanceNorm1d(hidden_channels),
            ]
        )

        self.coord_encoder = torch.nn.Linear(2, hidden_channels)

        self.gat_layers = torch.nn.ModuleList(
            [
                GATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=heads if i < num_gat_layers - 1 else 1,
                    # concat=(i < num_gat_layers - 1),
                    concat=False,
                    edge_dim=2,
                    dropout=dropout,
                )
                for i in range(num_gat_layers)
            ]
        )

        self.residual_adapter = torch.nn.Linear(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_channels, 32)
        self.fc2 = torch.nn.Linear(32, 2)
        self.tanh = torch.nn.Tanh()

    def Normalization(self, x):
        if self.normalization == "None":
            return x
        elif self.normalization == "LayerNorm":
            x = self.norm_layers[0](x)
        elif self.normalization == "BatchNorm":
            x = self.norm_layers[1](x)
        elif self.normalization == "InstanceNorm":
            x = self.norm_layers[2](x)
        return x

    def _adjust_identity(self, identity, x):
        if identity.size(1) != x.size(1):
            # return self.residual_adapter(identity)
            raise ValueError(
                "残差连接的输入输出维度不一致，请确保hidden_channels一致！"
            )
        return identity

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.coord_encoder(x))
        x = self.Normalization(x)
        for gat in self.gat_layers:
            res = x
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            if self.residual_switch:
                res = self._adjust_identity(res, x)
                x = x + res
            x = self.Normalization(x)
            x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
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
        elif self.normalization == "LayerNorm":
            x = self.norm_layers[0](x)  # 初始编码后使用LayerNorm
        elif self.normalization == "BatchNorm":
            x = self.norm_layers[1](x)  # 初始编码后使用BatchNorm
        elif self.normalization == "InstanceNorm":
            x = self.norm_layers[2](x)  # 初始编码后使用InstanceNorm

        return x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

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

    def _adjust_identity(self, identity, x):
        """维度适配器，当残差连接维度不匹配时进行线性变换"""
        if identity.size(1) != x.size(1):
            # return self.residual_adapter(identity)
            raise ValueError(
                "残差连接的输入输出维度不一致，请确保hidden_channels一致！"
            )
        return identity


class LossPlotter:

    def __init__(self, model_name, visualize=True):
        self.visualize = visualize
        self.train_losses = []
        self.val_losses = []
        if self.visualize:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            (self.line_train,) = self.ax.plot([], [], "r-", label="Train Loss")
            (self.line_val,) = self.ax.plot([], [], "b-", label="Val Loss")
            self.ax.set_title(f"Training & Validation Loss Curve({model_name})")
            self.ax.set_xlabel("Accumulated Steps")
            self.ax.set_ylabel("Loss")
            self.ax.legend()

    def append_train(self, loss):
        self.train_losses.append(loss)

    def append_val(self, val_loss):
        self.val_losses.append(val_loss)

    def update(self, global_step, validation_interval, log_interval):
        if self.visualize and global_step % log_interval == 0:
            self.line_train.set_data(range(len(self.train_losses)), self.train_losses)
            self.line_val.set_data(
                [i * validation_interval for i in range(len(self.val_losses))],
                self.val_losses,
            )
            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

    def save(self, save_path):
        """保存loss曲线图片"""
        if self.visualize:
            self.fig.savefig(save_path, dpi=200, bbox_inches="tight")

    def close(self):
        if self.visualize:
            plt.ioff()
            plt.close()


def train_model():
    # -------------------------- 初始化配置 --------------------------
    # 硬件设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    info(f"当前运行设备: {device}")

    # 路径配置
    current_dir = Path(__file__).parent
    folder_path = current_dir / "sample_grids/training/naca4digits"  # 原始数据目录
    model_save_path = (
        current_dir / f"model/saved_model_{MODEL_CONFIG['model_name']}.pth"
    )

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
    train_size = int(TRAINING_CONFIG["train_ratio"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG["batch_size"])
    # -------------------------- 临时措施 --------------------------
    val_loader = train_loader
    val_dataset = train_dataset

    # -------------------------- 模型初始化 --------------------------
    # 创建模型实例并转移到指定设备
    if MODEL_CONFIG["model_type"] == "GCN":
        model = EnhancedGNN(
            hidden_channels=MODEL_CONFIG["hidden_channels"],
            num_gcn_layers=MODEL_CONFIG["num_gcn_layers"],
            residual_switch=MODEL_CONFIG["residual_switch"],
            dropout=MODEL_CONFIG["dropout"],
            normalization=MODEL_CONFIG["normalization"],
        ).to(device)
    elif MODEL_CONFIG["model_type"] == "GAT":
        model = GATModel(
            hidden_channels=MODEL_CONFIG["hidden_channels"],
            num_gat_layers=MODEL_CONFIG["num_gcn_layers"],
            residual_switch=MODEL_CONFIG["residual_switch"],
            dropout=MODEL_CONFIG["dropout"],
            normalization=MODEL_CONFIG["normalization"],
        ).to(device)

    info("网络层详细信息：")
    info(model)
    total_params = sum(p.numel() for p in model.parameters())
    info(f"总可训练参数数量：{total_params:,}")

    # 优化器和损失函数配置
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TRAINING_CONFIG["learning_rate"]
    )
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAINING_CONFIG["lr_stepsize"],
        gamma=TRAINING_CONFIG["lr_gamma"],
    )

    # -------------------------- 训练监控 --------------------------
    loss_plotter = LossPlotter(
        MODEL_CONFIG["model_name"], visualize=TRAINING_CONFIG["visualize_training"]
    )

    # ----------- 加载已有模型参数或断点（如有） -----------
    checkpoint_path = model_save_path.with_suffix(".ckpt")
    global_step = 0
    start_epoch = 0
    if checkpoint_path.exists():
        info(f"检测到训练断点，正在恢复：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        # 恢复loss曲线
        loss_plotter.train_losses = checkpoint.get("train_losses", [])
        loss_plotter.val_losses = checkpoint.get("val_losses", [])
    elif model_save_path.exists():
        info(f"检测到已存在模型文件，正在加载参数并继续训练：{model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        info("未检测到已保存模型，将从头开始训练。")

    # -------------------------- 训练流程 --------------------------
    epoch = start_epoch - 1  # try前初始化epoch，避免异常时finally块报错
    try:
        for epoch in range(start_epoch, TRAINING_CONFIG["total_epochs"]):
            model.train()
            for batch_data in train_loader:
                global_step += 1
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                out = model(batch_data)
                loss = criterion(out, batch_data.y)
                loss.backward()
                optimizer.step()
                loss_plotter.append_train(loss.item())

                # 输出训练信息
                if global_step % TRAINING_CONFIG["log_interval"] == 0:
                    info(
                        f"当前步数[{global_step}] 轮次[{epoch+1}/{TRAINING_CONFIG['total_epochs']}]"
                        f" 损失: {loss.item():.4f}"
                    )

                # 验证集loss计算
                if global_step % TRAINING_CONFIG["validation_interval"] == 0:
                    model.eval()
                    total_val_loss = 0.0
                    with torch.no_grad():
                        for val_data in val_loader:  # 遍历整个验证集
                            val_data = val_data.to(device)
                            val_out = model(val_data)
                            val_loss = criterion(val_out, val_data.y)
                            total_val_loss += val_loss.item() * val_data.num_graphs
                    avg_val_loss = total_val_loss / len(val_dataset)  # 计算平均验证损失
                    loss_plotter.append_val(avg_val_loss)
                    model.train()
                    info(
                        f"训练损失: {loss.item():.4f},  验证损失: {avg_val_loss:.4f}, 学习率：{scheduler.get_last_lr()[0]:.2e}"
                    )

                # 更新损失曲线
                loss_plotter.update(
                    global_step,
                    TRAINING_CONFIG["validation_interval"],
                    TRAINING_CONFIG["log_interval"],
                )

                model.to(device)  # 确保模型回到正确设备
            # 学习率调度
            scheduler.step()

    except KeyboardInterrupt:
        error("\n训练被用户中断！")
    finally:
        # 保存模型参数
        torch.save(model.state_dict(), model_save_path)
        # 保存训练状态
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "train_losses": loss_plotter.train_losses,
            "val_losses": loss_plotter.val_losses,
        }
        checkpoint_path = model_save_path.with_suffix(".ckpt")
        torch.save(checkpoint, checkpoint_path)
        info(f"\n模型已保存至 {model_save_path}")
        info(f"训练断点已保存至 {checkpoint_path}")
        # 保存loss曲线
        loss_curve_path = model_save_path.with_suffix(".png")
        loss_plotter.save(loss_curve_path)
        info(f"Loss曲线已保存至 {loss_curve_path}")
        # loss_plotter.close()


if __name__ == "__main__":
    train_model()
    input("训练完成，按回车键退出...")

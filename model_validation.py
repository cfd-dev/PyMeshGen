import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import boundary_mesh_sample as bl_samp
from model_train import build_graph_data, visualize_predictions
from torch_geometric.nn import GCNConv

class EnhancedGNN(torch.nn.Module):
    """保持与训练模型一致的网络结构"""
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(2, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)  # 新增BN层
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)  # 新增BN层
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)  # 新增BN层
        self.fc1 = torch.nn.Linear(hidden_channels, 32)
        self.bn_fc1 = torch.nn.BatchNorm1d(32)  # 新增全连接层BN
        self.fc2 = torch.nn.Linear(32, 2)
        self.tanh = torch.nn.Tanh()  # 新增tanh激活

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))  # 调整前向传播顺序
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.relu(self.bn_fc1(self.fc1(x)))  # 全连接层后加BN
        return self.tanh(self.fc2(x))  # 添加tanh激活

def validate_model():
    # -------------------------- 初始化配置 --------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'hidden_channels': 128,
        'model_path': 'marching_model.pth',
        'validation_data_path': './validation_sample'
    }

    # -------------------------- 加载模型 --------------------------
    try:
        model = EnhancedGNN(config['hidden_channels']).to(device)
        model.load_state_dict(torch.load(config['model_path']))
        model.eval()
        print("成功加载训练模型")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # -------------------------- 加载验证数据 --------------------------
    try:
        val_results = bl_samp.batch_process_files(config['validation_data_path'])
        print(f"加载到 {len(val_results)} 个验证数据集")
    except Exception as e:
        print(f"验证数据加载失败: {str(e)}")
        return

    # -------------------------- 执行验证 --------------------------
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for idx, result in enumerate(val_results):
            # 数据准备
            data = build_graph_data(result['valid_wall_nodes'], 
                                  result['wall_faces']).to(device)
            
            # 模型预测
            pred = model(data)
            loss = criterion(pred, data.y)
            total_loss += loss.item()
            
            # 可视化最后一个样本的预测结果
            if idx == len(val_results) - 1:
                visualize_predictions(data.cpu(), model.cpu())
                plt.suptitle(f"验证样本 {idx+1} (Loss: {loss.item():.4f})")
                
            print(f"样本 {idx+1}/{len(val_results)} 验证损失: {loss.item():.4f}")

    # -------------------------- 输出统计结果 --------------------------
    avg_loss = total_loss / len(val_results)
    print(f"\n验证完成 | 平均损失: {avg_loss:.4f}")
    plt.figure()
    plt.bar(['Validation Loss'], [avg_loss], color='steelblue')
    plt.title("平均验证损失")
    plt.ylabel("MSE Loss")
    plt.show()

if __name__ == "__main__":
    validate_model()
    input("按回车键退出...")
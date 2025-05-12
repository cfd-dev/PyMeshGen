import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
from grid_sample import batch_process_files
from model_train import build_graph_data, EnhancedGNN
from visualization.visualization import visualize_predictions


def validate_model():
    current_dir = Path(__file__).parent
    validation_data_path = current_dir / "sample_grids/validation"  # 原始数据目录
    model_save_path = current_dir / "model/saved_model.pth"  # 模型保存路径
    # -------------------------- 初始化配置 --------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        "hidden_channels": 64,
        "model_path": model_save_path,
        "validation_data_path": validation_data_path,
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
        val_results = batch_process_files(config["validation_data_path"])
        print(f"加载到 {len(val_results)} 个验证数据集")
    except Exception as e:
        print(f"验证数据加载失败: {str(e)}")
        return

    # -------------------------- 执行验证 --------------------------
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    all_losses = []  # 用于保存每个样本的loss

    with torch.no_grad():
        for idx, result in enumerate(val_results):
            # 数据准备
            data = build_graph_data(result['valid_wall_nodes'], 
                                  result['wall_faces']).to(device)

            # 模型预测
            pred = model(data)
            loss = criterion(pred, data.y)
            total_loss += loss.item()
            all_losses.append(loss.item())

            # 可视化最后一个样本的预测结果
            if idx == len(val_results) - 1:
                fig, ax = visualize_predictions(data.cpu(), model.cpu())
                fig.suptitle(f"Case {idx+1} (Loss: {loss.item():.4f})")
                ax.set_title("March Vector Prediction vs Ground Truth")
                plt.show()
            print(f"样本 {idx+1}/{len(val_results)} 验证损失: {loss.item():.4f}")

    # -------------------------- 输出统计结果 --------------------------
    avg_loss = total_loss / len(val_results)
    print(f"\n验证完成 | 平均损失: {avg_loss:.4f}")
    plt.figure()
    plt.bar(range(1, len(all_losses) + 1), all_losses, color="steelblue")
    plt.title("Validation Loss for Each Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    validate_model()
    input("按回车键退出...")

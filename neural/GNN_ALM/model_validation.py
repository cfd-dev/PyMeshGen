import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
from grid_sample import batch_process_files
from model_train import build_graph_data, EnhancedGNN, GATModel
from visualization.visualization import visualize_predictions
from config import MODEL_CONFIG

def validate_model():
    current_dir = Path(__file__).parent
    validation_data_path = current_dir / "sample_grids/validation"  # 原始数据目录
    saved_model = current_dir / f"model/saved_model_{MODEL_CONFIG['model_name']}.pth"
    # -------------------------- 加载模型 --------------------------
    try:
        if MODEL_CONFIG["model_name"] == "GCN":
            model = EnhancedGNN(
                hidden_channels=MODEL_CONFIG["hidden_channels"],
                num_gcn_layers=MODEL_CONFIG["num_gcn_layers"],
                residual_switch=MODEL_CONFIG["residual_switch"],
                dropout=MODEL_CONFIG["dropout"],
                normalization=MODEL_CONFIG["normalization"],
            )
        elif MODEL_CONFIG["model_name"] == "GAT":
            model = GATModel(
                hidden_channels=MODEL_CONFIG["hidden_channels"],
                num_gat_layers=MODEL_CONFIG["num_gcn_layers"],
                residual_switch=MODEL_CONFIG["residual_switch"],
                dropout=MODEL_CONFIG["dropout"],
                normalization=MODEL_CONFIG["normalization"],
            )
        model.load_state_dict(torch.load(saved_model))
        model.eval()
        print("成功加载训练模型")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # -------------------------- 加载验证数据 --------------------------
    try:
        val_results = batch_process_files(validation_data_path)
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
            data = build_graph_data(result["valid_wall_nodes"], result["wall_faces"])

            # 计算wall_faces的平均长度
            avg_length = compute_average_face_length(
                result["wall_faces"], result["valid_wall_nodes"]
            )

            # 模型预测
            pred = model(data)
            loss = criterion(pred, data.y)
            total_loss += loss.item()
            all_losses.append(loss.item())

            print(f"样本 {idx+1}/{len(val_results)} 验证损失: {loss.item():.4f}")

            # 可视化最后一个样本的预测结果
            if idx == 0:
                fig, ax = visualize_predictions(
                    data, model, avg_length, result["normalize_coeff"]
                )
                ax.set_title(
                    f"March Vector Prediction vs Ground Truth | Case {idx+1} (Loss: {loss.item():.4f})"
                )
                plt.tight_layout(rect=[0, 0, 1, 1])
                plt.show(block=False)

    # -------------------------- 输出统计结果 --------------------------
    avg_loss = total_loss / len(val_results)
    print(f"\n验证完成 | 平均损失: {avg_loss:.4f}")
    # plt.figure()
    # plt.bar(range(1, len(all_losses) + 1), all_losses, color="steelblue")
    # plt.title("Validation Loss for Each Sample")
    # plt.xlabel("Sample Index")
    # plt.ylabel("MSE Loss")
    # plt.tight_layout()
    # plt.show(block=False)


def compute_average_face_length(wall_faces, wall_nodes):
    """计算wall_faces的平均长度"""
    total_length = 0.0
    for face in wall_faces:
        nodes = [n - 1 for n in face["nodes"]]
        # 从wall_nodes中取出节点的坐标 当nodes中的编号等于original_indices时，取出其coords
        for i in range(len(nodes)):
            for j in range(len(wall_nodes)):
                if nodes[i] == wall_nodes[j]["original_indices"]:
                    nodes[i] = j
                    break
            else:
                print(f"Warning: Node {nodes[i]} not found in wall_nodes")

        # 从wall_nodes取出坐标计算长度
        nodes = [wall_nodes[i]["coords"] for i in nodes]
        dx = nodes[1][0] - nodes[0][0]
        dy = nodes[1][1] - nodes[0][1]
        length = np.hypot(dx, dy)
        total_length += length
    return total_length / len(wall_faces) if wall_faces else 0.0


if __name__ == "__main__":
    validate_model()
    input("按回车键退出...")

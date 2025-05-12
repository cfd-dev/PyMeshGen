import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_graph_structure(graph):
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制节点
    ax.scatter(graph.x[:, 0], graph.x[:, 1], c="blue", s=20)

    # 绘制边
    for i in range(graph.edge_index.shape[1]):
        a = graph.edge_index[0, i].item()
        b = graph.edge_index[1, i].item()
        ax.plot(
            [graph.x[a, 0], graph.x[b, 0]],
            [graph.x[a, 1], graph.x[b, 1]],
            c="gray",
            alpha=0.3,
        )

    ax.set_title("Graph Structure Visualization")
    plt.show()


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

    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制节点位置
    ax.scatter(coords[:, 0], coords[:, 1], c="black", s=20, label="Nodes")

    # 计算动态箭头参数
    # 计算坐标范围
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    avg_range = np.sqrt(x_range**2 + y_range**2)

    # 自动计算参数（新增部分）
    if vector_scale is None:
        # vector_scale = 0.15 / avg_range if avg_range > 0 else 0.3
        vector_scale = (
            0.08 * avg_range / np.linalg.norm(true, axis=1).max()
            if np.linalg.norm(true, axis=1).max() > 0
            else 0.1
        )
    if head_scale is None:
        # head_scale = 0.02 / avg_range if avg_range > 0 else 0.01
        head_scale = 0.015 * avg_range

    # 统一箭头尺寸参数（保持原有逻辑）
    # base_size = avg_range * head_scale
    # head_width = base_size * 2.5
    # head_length = base_size * 4
    head_width = head_scale
    head_length = head_scale * 1.8

    # 修改箭头绘制部分（原117-136行）
    # 绘制真实向量（蓝色）
    for i in range(len(true)):
        if not np.any(np.isnan(true[i])):
            dx, dy = true[i] * vector_scale
            ax.arrow(
                coords[i, 0],
                coords[i, 1],
                dx,
                dy,
                head_width=head_width,
                head_length=head_length,  # 替换固定数值
                fc="blue",
                ec="blue",
                alpha=0.6,
                length_includes_head=True,
            )

    # 绘制预测向量（红色）
    for i in range(len(pred)):
        if not np.any(np.isnan(pred[i])):
            dx, dy = pred[i] * vector_scale
            ax.arrow(
                coords[i, 0],
                coords[i, 1],
                dx,
                dy,
                head_width=head_width,
                head_length=head_length,  # 替换固定数值
                fc="red",
                ec="red",
                alpha=0.6,
                length_includes_head=True,
            )

    # 创建图例
    blue_arrow = plt.Line2D([0], [0], color="blue", lw=2, label="True Vector")
    red_arrow = plt.Line2D([0], [0], color="red", lw=2, label="Predicted Vector")
    ax.legend(handles=[blue_arrow, red_arrow])

    # ax.set_title("March Vector Prediction vs Ground Truth")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.axis("equal")
    plt.tight_layout()
    # plt.show(block=True)
    return fig, ax

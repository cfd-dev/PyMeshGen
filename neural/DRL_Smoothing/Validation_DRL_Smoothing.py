import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "fileIO"))
sys.path.append(str(root_dir / "data_structure"))
sys.path.append(str(root_dir / "meshsize"))
sys.path.append(str(root_dir / "visualization"))
sys.path.append(str(root_dir / "adfront2"))
sys.path.append(str(root_dir / "optimize"))
sys.path.append(str(root_dir / "utils"))

import torch
import matplotlib.pyplot as plt
import numpy as np
from DRL_Smoothing import Actor, DRLSmoothingEnv
from stl_io import parse_stl_msh
from mesh_visualization import Visualization
from basic_elements import Unstructured_Grid
from mesh_visualization import visualize_unstr_grid_2d
from message import info


def plot_mesh_comparison(original_mesh, optimized_mesh):
    """
    并列显示原始网格和优化后的网格。

    :param original_mesh: 原始网格对象
    :param optimized_mesh: 优化后的网格对象
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 创建并列的两个子图

    # 绘制原始网格
    axes[0].set_title("Original Mesh")
    visualize_unstr_grid_2d(original_mesh, axes[0])

    # 绘制优化后的网格
    axes[1].set_title("Optimized Mesh")
    visualize_unstr_grid_2d(optimized_mesh, axes[1])

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


def load_actor(model_path, state_dim, action_dim):
    """
    加载训练好的 DDPG agent 的 actor 模型。

    :param model_path: 模型文件路径
    :param state_dim: 状态维度
    :param action_dim: 动作维度
    :return: 加载好的 actor 模型
    """
    actor = Actor(state_dim, action_dim)
    actor.load_state_dict(torch.load(model_path / "actor_final.pth"))
    actor.eval()  # 设置为评估模式
    return actor


def optimize_mesh_with_env(actor, env, iterations=1):
    """
    使用 DRLSmoothingEnv 和训练好的 actor 模型优化网格。

    :param actor: 训练好的 actor 模型
    :param env: DRLSmoothingEnv 环境
    :return: 优化后的网格和累计奖励
    """

    for i in range(iterations):
        state = env.reset(True)
        total_reward = 0
        while not env.done:
            # 使用 actor 模型预测动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 添加 batch 维度
            action = actor(state_tensor).detach().numpy().squeeze(0)

            # 执行动作并获取奖励
            state, reward, done = env.step(action)
            total_reward += reward
        info(f"Iteration {i + 1}/{iterations}: Total reward: {total_reward:.2f}")

    return env.initial_grid, total_reward


def main():
    # 模型路径
    model_path = Path(__file__).parent / "agent/"

    # 定义状态和动作维度（与训练时一致）
    state_dim = 16
    action_dim = 2

    # 加载训练好的 actor 模型
    actor = load_actor(model_path, state_dim, action_dim)

    # 加载 STL 网格
    input_mesh_path = Path(__file__).parent / "training_mesh/validation1.stl"
    original_mesh = parse_stl_msh(input_mesh_path)

    # 拷贝一份原始网格，以免后续操作影响原始网格
    original_mesh_bak = Unstructured_Grid(
        original_mesh.cell_container,
        original_mesh.node_coords.copy(),
        original_mesh.boundary_nodes,
    )
    original_mesh.summary()

    # 创建 DRLSmoothingEnv 环境
    param_obj = {
        "viz_enabled": False,
        "max_ring_nodes": 8,
        "shape_coeff": 0.0,
        "min_coeff": 1.0,
    }
    visual_obj = Visualization(param_obj["viz_enabled"])

    # 使用 actor 模型优化网格
    env = DRLSmoothingEnv(
        initial_grid=original_mesh, visual_obj=visual_obj, param_obj=param_obj
    )
    optimized_mesh, total_reward = optimize_mesh_with_env(actor, env, iterations=3)
    optimized_mesh.summary()
    plot_mesh_comparison(original_mesh_bak, optimized_mesh)

    # 保存优化后的网格
    output_mesh_path = Path(__file__).parent / "training_mesh/optimized_mesh.vtk"
    optimized_mesh.save_to_vtkfile(output_mesh_path)

    info(f"Final total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()

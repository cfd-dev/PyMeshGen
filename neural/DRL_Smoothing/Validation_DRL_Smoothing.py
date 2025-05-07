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
from DRL_Smoothing import Actor, DRLSmoothingEnv
from stl_io import parse_stl_msh
import numpy as np
from mesh_visualization import Visualization
from basic_elements import UnstructuredGrid


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
    total_reward = 0.0
    state = env.reset()
    for i in range(iterations):
        while not env.done:
            # 使用 actor 模型预测动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 添加 batch 维度
            action = actor(state_tensor).detach().numpy().squeeze(0)

            # 执行动作并获取奖励
            state, reward, done = env.step(action)
            total_reward += reward
        print(f"Iteration {i + 1}/{iterations}: Total reward: {total_reward}")

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
    input_mesh_path = Path(__file__).parent / "validation1.stl"
    mesh = parse_stl_msh(input_mesh_path)

    visual_obj = Visualization()
    mesh.visualize_unstr_grid_2d(visual_obj)
    # 创建 DRLSmoothingEnv 环境
    param_obj = {
        "viz_enabled": True,
        "max_ring_nodes": 8,
        "shape_coeff": 0.0,
        "min_coeff": 1.0,
    }
    env = DRLSmoothingEnv(initial_grid=mesh, visual_obj=visual_obj, param_obj=param_obj)

    # 使用 actor 模型优化网格
    optimized_mesh, total_reward = optimize_mesh_with_env(actor, env, iterations=3)

    # 保存优化后的网格
    output_mesh_path = Path(__file__).parent / "optimized_mesh.stl"
    optimized_mesh.export(output_mesh_path)
    print(f"Optimized mesh saved to {output_mesh_path}")
    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    main()

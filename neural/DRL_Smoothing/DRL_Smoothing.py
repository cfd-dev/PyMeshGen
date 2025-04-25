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
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import gym
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.buffers import ReplayBuffer

from optimize import node_perturbation
from geom_toolkit import (
    point_in_polygon,
    calculate_distance2,
    centroid,
    calculate_distance,
)
from stl_io import parse_stl_msh
from mesh_visualization import Visualization, plot_polygon
from geom_normalization import normalize_ploygon, denormalize_point, normalize_point


class DRLSmoothingEnv(gym.Env):
    def __init__(
        self,
        initial_grid=None,
        visual_obj=None,
        param_obj=None,
    ):
        super(DRLSmoothingEnv, self).__init__()
        self.action_dim = 2  # 动作维度
        self.viz_enabled = param_obj["viz_enabled"]
        self.max_ring_nodes = param_obj["max_ring_nodes"]

        # shape quality所占权重，skewness权重为1-shape_coeff
        self.shape_coeff = param_obj["shape_coeff"]
        # minQuality在reward中的权重
        self.min_coeff = param_obj["min_coeff"]

        self.initial_grid = initial_grid  # 初始网格
        self.original_node_coords = np.copy(initial_grid.node_coords)  # 备份初始坐标

        self.ax = visual_obj.ax  # 绘图对象
        self.visual_obj = visual_obj  # 绘图对象

        self.current_node_id = 0  # 当前节点ID
        self.ring_coords = []  # 当前环节点坐标
        self.normalized_ring_coords = []  # 归一化后的环节点坐标
        self.state = []  # 当前状态
        self.done = False  # 是否完成
        self.local_range = []  # 当前状态的局部范围，用于归一化和反归一化

        self.init_env()

    def init_env(self):
        # 初始化节点邻居环节点
        self.initial_grid.cyclic_node2node()
        self.initial_grid.init_node2cell()

        # 初始化观测和动作空间
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.max_ring_nodes, 2), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1, self.action_dim), dtype=np.float32
        )

    def init_state(self):
        # 遇到边界节点时，跳过并寻找下一个非边界节点
        while self.current_node_id in self.initial_grid.boundary_nodes_list:
            self.current_node_id += 1

        if self.current_node_id >= self.initial_grid.num_nodes:
            self.done = True
            self.state = np.zeros((self.max_ring_nodes, 2))
            return

        # 获取当前节点的环状邻居坐标
        node_ids = self.initial_grid.node2node[self.current_node_id]
        self.ring_coords = np.array(
            [self.initial_grid.node_coords[i] for i in node_ids]
        )

        normalized_coords, self.local_range = normalize_ploygon(self.ring_coords)
        self.normalized_ring_coords = normalized_coords

        # 填充至固定长度
        if len(normalized_coords) < self.max_ring_nodes:
            pad = np.zeros((self.max_ring_nodes - len(normalized_coords), 2))
            normalized_coords = np.vstack([normalized_coords, pad])

        # 截断超长部分
        normalized_coords = normalized_coords[: self.max_ring_nodes]

        self.state = normalized_coords.copy()

    def reset(self):
        self.done = False
        self.current_node_id = 0

        # 恢复网格坐标到初始状态
        self.initial_grid.node_coords = np.copy(self.original_node_coords)

        # 初始化状态
        self.init_state()

        return self.get_obs()

    def get_obs(self):
        return self.state.copy()

    def plot_normalized_ring(self, new_point):
        # 绘制归一化后的环节点
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_polygon(self.normalized_ring_coords, ax, color="blue", alpha=0.5)

        new_point = np.squeeze(new_point)
        current_point = self.initial_grid.node_coords[self.current_node_id]
        current_point = normalize_point(current_point, self.local_range)
        ax.scatter(current_point[0], current_point[1], color="green")
        ax.scatter(new_point[0], new_point[1], color="red")
        ax.set_aspect('equal')

    def plot_ring_and_action(self, new_point):
        # self.plot_normalized_ring(new_point)

        # 绘制当前环节点和动作
        new_point = denormalize_point(new_point, self.local_range)
        plot_polygon(self.ring_coords, self.ax, color="blue", alpha=0.5)  # 绘制环
        self.ax.scatter(self.ring_coords[:, 0], self.ring_coords[:, 1], color="green")
        self.ax.scatter(new_point[0], new_point[1], color="red")
        plt.pause(0.001)

    def step(self, action):
        observation = self.get_obs()
        reward = 0
        self.done = False

        # 对于超出最大环节点数的动作，使用形心代替
        if len(self.ring_coords) > self.max_ring_nodes:
            action = centroid(self.normalized_ring_coords)
        else:
            action = action + centroid(self.normalized_ring_coords)

        if self.viz_enabled:
            self.ax.clear()
            self.initial_grid.visualize_unstr_grid_2d(self.visual_obj)
            self.plot_ring_and_action(action)

        reward = self.compute_reward(action)

        new_point_denormalized = denormalize_point(action, self.local_range)  # 反归一化
        self.initial_grid.node_coords[self.current_node_id] = new_point_denormalized

        # 计算下一个状态，如果当前步给了惩罚，则退出重来
        if not (reward < 0 or self.done):
            self.current_node_id += 1
            self.init_state()

        return self.get_obs(), reward, self.done

    def compute_reward(self, new_point):
        # 判断new_point是否在当前环内，如果在环外，则惩罚并终止
        reward = 0.0
        new_point = np.squeeze(new_point)
        if not point_in_polygon(new_point, self.normalized_ring_coords):
            self.done = True
            center_point = centroid(self.normalized_ring_coords)
            dis = calculate_distance(new_point, center_point)
            reward = -1.0 * dis  # 距离惩罚
            return reward

        # 当前节点的邻居单元及其质量
        neigbor_cells = self.initial_grid.node2cell[self.current_node_id]
        min_shape = np.min([cell.get_quality() for cell in neigbor_cells])
        avg_shape = np.mean([cell.get_quality() for cell in neigbor_cells])
        min_skew = np.min([cell.get_skewness() for cell in neigbor_cells])
        avg_skew = np.mean([cell.get_skewness() for cell in neigbor_cells])

        shape = self.min_coeff * min_shape + (1 - self.min_coeff) * avg_shape
        skew = self.min_coeff * min_skew + (1 - self.min_coeff) * avg_skew

        reward = self.shape_coeff * shape + (1 - self.shape_coeff) * skew

        return reward


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=16):
        super(Critic, self).__init__()
        # State processing path
        self.state_path = nn.Sequential(
            nn.Linear(state_dim, 4 * hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
        )

        # Action processing path
        self.action_path = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            # nn.Linear(action_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim // 2),
        )
        # Common path
        self.common_path = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            # nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            module.bias.data.zero_()

    def forward(self, state, action):
        state_out = self.state_path(state)
        action_out = self.action_path(action)
        combined = torch.cat([state_out, action_out], dim=1)
        return self.common_path(combined)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=16):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 2 * hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # nn.Tanh(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            module.bias.data.zero_()

    def forward(self, state):
        return self.net(state)


class DDPGAgent:
    def __init__(self, env):
        state_dim = np.prod(env.observation_space.shape)
        action_dim = np.prod(env.action_space.shape)

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.batch_size = 64
        self.gamma = 0.99  # 折扣因子，用于计算未来奖励的衰减系数
        self.tau = 1e-3  # 目标网络软更新系数（滑动平均系数）

        # 目标网络初始化
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # OU噪声
        self.noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(env.action_space.shape),
            sigma=0.3 * np.ones(env.action_space.shape),
            theta=0.15,
            dt=1e-2,
        )

        # GaussianNoise噪声：
        # self.noise = GaussianNoise(
        #     mu=np.zeros(env.action_space.shape),
        #     sigma=0.3,  # 初始标准差，可逐步衰减
        #     action_dim=env.action_space.shape[0]
        # )

        self.replay_buffer = MeshReplayBuffer(
            buffer_size=1e6  # 只需要传递buffer_size参数
        )

        # self.replay_buffer = MeshReplayBufferSB3(
        #     buffer_size=1e6,  # 经验回放缓冲区大小
        #     observation_space=env.observation_space,  # 环境的 observation space
        #     action_space=env.action_space,  # 环境的 action space
        # )

    # 新增软更新方法
    def soft_update(self):
        with torch.no_grad():
            for target_param, param in zip(
                self.target_actor.parameters(), self.actor.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

    # 新增训练步骤
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区采样
        # batch = self.replay_buffer.sample(self.batch_size)
        # states = torch.FloatTensor(batch.observations)
        # actions = torch.FloatTensor(batch.actions)
        # rewards = torch.FloatTensor(batch.rewards)
        # next_states = torch.FloatTensor(batch.next_observations)
        # dones = torch.FloatTensor(batch.dones)

        obs, next_obs, actions, rewards, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = torch.FloatTensor(
            obs.reshape(self.batch_size, -1)
        )  # 展平为(batch_size, 16)
        actions = torch.FloatTensor(
            actions.reshape(self.batch_size, -1)
        )  # (batch_size, 2)
        # rewards = torch.FloatTensor(rewards)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_obs.reshape(self.batch_size, -1))
        # dones = torch.FloatTensor(dones)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Critic更新
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, target_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor更新
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.flatten())
            action = self.actor(state)
            noise = torch.tensor(self.noise(), dtype=torch.float32)
            action = action + noise
            action = np.clip(action.detach().numpy().reshape(-1, 2), -1, 1)
            return action


# 新增经验回放缓冲区（兼容SB3）
class MeshReplayBufferSB3(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space):
        # 将二维观测空间展平以适应SB3的ReplayBuffer
        self.original_obs_shape = observation_space.shape
        modified_obs_space = gym.spaces.Box(
            low=observation_space.low.flatten(),
            high=observation_space.high.flatten(),
            shape=(np.prod(observation_space.shape),),  # 展平为1D
            dtype=observation_space.dtype,
        )

        # 调整动作空间维度
        modified_act_space = gym.spaces.Box(
            low=action_space.low.flatten(),  # Flatten the low bounds
            high=action_space.high.flatten(),  # Flatten the high bounds
            shape=(np.prod(action_space.shape),),  # 展平为1D
            dtype=action_space.dtype,
        )

        super().__init__(
            int(buffer_size), modified_obs_space, modified_act_space, device="cpu"
        )

    def add(self, obs, next_obs, action, reward, done):
        super().add(
            obs=obs.reshape(-1),  # 二维转一维
            next_obs=next_obs.reshape(-1),
            action=action.reshape(-1),
            reward=reward,
            done=done,
        )


class MeshReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = int(buffer_size)
        self.buffer = deque(maxlen=self.buffer_size)  # 使用双端队列替代列表

    def add(self, obs, next_obs, action, reward, done):
        experience = (obs.copy(), next_obs.copy(), action.copy(), reward, done)
        self.buffer.append(experience)  # 自动处理缓冲区满的情况

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        # # 改为优先经验回放
        # priorities = np.abs(np.array([item[3] for item in self.buffer])) + 1e-5  # 使用奖励绝对值作为优先级
        # probabilities = priorities / np.sum(priorities)
        # indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        batch = [self.buffer[i] for i in indices]

        # 保持原有数据处理逻辑不变
        observations = np.array([item[0] for item in batch])
        next_observations = np.array([item[1] for item in batch])
        actions = np.array([item[2] for item in batch])
        rewards = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])

        return (observations, next_observations, actions, rewards, dones)

    def __len__(self):
        return len(self.buffer)

class GaussianNoise:
    def __init__(self, mu, sigma, action_dim):
        self.mu = mu
        self.sigma = sigma
        self.action_dim = action_dim

    def __call__(self):
        return np.random.normal(self.mu, self.sigma, self.action_dim)


def train_drl(train_grid, visual_obj, param_obj):
    env = DRLSmoothingEnv(
        initial_grid=train_grid,
        visual_obj=visual_obj,
        param_obj=param_obj,
    )
    agent = DDPGAgent(env)

    max_episodes = param_obj["max_episodes"]
    max_steps = env.initial_grid.num_nodes
    save_interval = param_obj["save_interval"]
    viz_history = param_obj["viz_history"]

    if viz_history:
        # 初始化奖励记录
        episodes_list = []
        rewards_list = []

        # 创建绘图对象
        plt.ion()  # 启用交互模式
        fig, ax = plt.subplots()
        ax.set_title("Training Progress")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")

    for ep in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.replay_buffer.add(state, next_state, action, reward, done)
            agent.train_step()

            state = next_state
            episode_reward += reward

            if done:
                break

        if viz_history:
            # 记录奖励数据
            episodes_list.append(ep + 1)
            rewards_list.append(episode_reward)

            # 每100个episode更新一次曲线
            if (ep + 1) % 100 == 0:
                ax.clear()
                ax.plot(episodes_list, rewards_list, "b-", label="Episode Reward")
                ax.legend()
                plt.pause(0.0001)  # 短暂暂停更新图表

        if (ep + 1) % save_interval == 0:
            torch.save(
                agent.actor.state_dict(),
                f"./neural/DRL_Smoothing/agent/actor_{ep+1}.pth",
            )
            torch.save(
                agent.critic.state_dict(),
                f"./neural/DRL_Smoothing/agent/critic_{ep+1}.pth",
            )

        print(
            f"Episode {ep+1}/{max_episodes} completed | Total Steps: {step+1} | Total Reward: {episode_reward:.2f}"
        )

    if viz_history:
        # 训练结束后保存图表
        plt.ioff()
        plt.savefig("./neural/DRL_Smoothing/training_progress.png")
        plt.close()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 参数
    param_obj = {
        "max_ring_nodes": 8,
        "node_perturb": False,
        "viz_enabled": False,
        "shape_coeff": 0.0,
        "min_coeff": 1.0,
        "max_episodes": 100000,
        "save_interval": 1000,
        "viz_history": False,
    }

    visual_obj = Visualization(True)

    train_grid = parse_stl_msh("./neural/DRL_Smoothing/training_mesh/training_mesh.stl")
    # train_grid.visualize_unstr_grid_2d(visual_obj)

    # 初始化网格节点扰动
    if param_obj["node_perturb"]:
        train_grid = node_perturbation(train_grid)

    train_drl(train_grid, visual_obj, param_obj)

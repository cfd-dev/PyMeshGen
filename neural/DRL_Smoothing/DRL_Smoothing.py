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
from geom_toolkit import point_in_polygon
from stl_io import parse_stl_msh
from mesh_visualization import Visualization


class DRLSmoothingEnv(gym.Env):
    def __init__(
        self,
        max_ring_nodes=8,
        initial_grid=None,
        visual_obj=None,
    ):
        super(DRLSmoothingEnv, self).__init__()
        self.min_coeff = 1.0  # minQuality在reward中的权重
        self.shape_coeff = 0.0  # shape quality所占权重，skewness权重为1-shape_coeff
        self.node_perturb = False  # 是否进行节点扰动
        self.max_ring_nodes = max_ring_nodes  # 最大允许的环节点数量
        self.action_dim = 2  # 动作维度

        self.initial_grid = initial_grid  # 初始网格
        self.original_node_coords = np.copy(initial_grid.node_coords)  # 备份坐标

        self.ax = visual_obj.ax  # 绘图对象
        self.visual_obj = visual_obj  # 绘图对象

        self.current_node_id = 0  # 当前节点ID
        self.ring_coords = []  # 当前环节点坐标
        self.state = []  # 当前状态
        self.done = False  # 是否完成
        self.local_range = []  # 当前状态的局部范围，用于归一化和反归一化
        self.action_storage = []  # 动作存储

        self.init_env()

    def init_env(self):
        # 初始化节点邻居环节点
        self.initial_grid.init_node2node_by_cell()
        self.initial_grid.init_node2cell()

        # 初始化网格节点扰动
        if self.node_perturb:
            self.initial_grid = node_perturbation(self.initial_grid)

        self.initial_grid.visualize_unstr_grid_2d(self.visual_obj)

        # 初始化观测和动作空间
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.max_ring_nodes, 2), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

    def init_state(self):
        # 获取当前节点的环状邻居坐标
        node_ids = self.initial_grid.node2node[self.current_node_id]
        ring_coords = np.array([self.initial_grid.node_coords[i] for i in node_ids])
        self.ring_coords = ring_coords

        # 计算坐标范围
        x_min, x_max = np.min(ring_coords[:, 0]), np.max(ring_coords[:, 0])
        y_min, y_max = np.min(ring_coords[:, 1]), np.max(ring_coords[:, 1])
        ref_d = max(x_max - x_min, y_max - y_min) + 1e-8  # 防止除零

        # 归一化处理
        normalized_coords = ring_coords - np.array([[x_min, y_min]])
        normalized_coords /= ref_d

        # 断言：归一化后的坐标必须在 [0, 1] 范围内
        assert np.all(
            (normalized_coords >= 0) & (normalized_coords <= 1)
        ), f"Normalized coordinates out of [0,1]: min={np.min(normalized_coords)}, max={np.max(normalized_coords)}"

        # 填充至固定长度
        if len(normalized_coords) < self.max_ring_nodes:
            pad = np.zeros((self.max_ring_nodes - len(normalized_coords), 2))
            normalized_coords = np.vstack([normalized_coords, pad])

        # 截断超长部分
        normalized_coords = normalized_coords[: self.max_ring_nodes]
        # 再次断言：填充和截断后的结果必须在 [0,1] 范围内
        assert np.all(
            (normalized_coords >= 0) & (normalized_coords <= 1)
        ), "Normalized coordinates after padding/truncation are out of [0, 1]!"

        self.state = normalized_coords.copy()
        self.local_range = (x_min, x_max, y_min, y_max)  # 保存范围用于逆变换

    def reset(self):
        self.done = False
        self.current_node_id = 0
        self.action_storage = np.zeros(
            (self.initial_grid.num_nodes, 2)
        )  # 初始化动作存储矩阵

        # 恢复网格坐标到初始状态
        self.initial_grid.node_coords = np.copy(self.original_node_coords)
        # 初始化状态
        self.init_state()

        return self.get_obs()

    def get_obs(self):
        # 返回当前观测值
        return self.state.copy()

    def anti_normalize(self, action):
        # 反归一化
        x_min, x_max, y_min, y_max = self.local_range
        ref_d = max(x_max - x_min, y_max - y_min) + 1e-8  # 防止除零
        new_point = action * ref_d + np.array([x_min, y_min])

        return new_point

    def centroid(self, points):
        # 计算多边形的形心
        x = np.mean(points[:, 0])
        y = np.mean(points[:, 1])
        return np.array([x, y])

    def plot_ring_and_action(self):
        # 绘制当前环节点和动作
        self.ax.scatter(
            self.ring_coords[:, 0], self.ring_coords[:, 1], label="Ring Nodes"
        )
        self.ax.scatter(
            self.action_storage[self.current_node_id, 0],
            self.action_storage[self.current_node_id, 1],
            color="red",
            label="Action",
        )
        self.ax.legend()
        self.ax.set_xlim(
            self.initial_grid.bbox[0], self.initial_grid.bbox[2]
        )  # 限制x轴范围
        self.ax.set_ylim(
            self.initial_grid.bbox[1], self.initial_grid.bbox[3]
        )  # 限制y轴范围
        self.ax.set_aspect("equal", "box")  # 保持x和y轴比例相等
        plt.pause(0.001)  # 暂停以更新图形
        self.ax.clear()  # 清除上一帧的内容

    def step(self, action):
        observation = self.get_obs()
        reward = 0
        self.done = False

        # 对于超出最大环节点数的动作，使用形心代替
        if len(self.ring_coords) > self.max_ring_nodes:
            action = self.centroid(self.ring_coords)  # 计算多边形的形心

        new_point = self.anti_normalize(action)  # 反归一化

        # 绘图
        self.plot_ring_and_action()

        reward = self.compute_reward(new_point)  # 计算奖励

        self.initial_grid.node_coords[self.current_node_id] = new_point

        # 计算下一个状态
        self.current_node_id += 1
        if self.current_node_id >= self.initial_grid.num_nodes:
            self.done = True
        else:
            self.init_state()  # 初始化下一个状态

        return self.get_obs(), reward, self.done, {}

    def compute_reward(self, new_point):
        # 判断new_point是否在当前环内，如果在环外，则惩罚并终止
        if not point_in_polygon(new_point, self.ring_coords):
            self.done = True
            return -100  # 惩罚

        # 当前节点的邻居单元及其质量
        neigbor_cells = self.initial_grid.node2cell[self.current_node_id]

        shape_quality, skewness = self.neighbor_cells_quality(neigbor_cells)
        shape_min, shape_max, shape_avg = shape_quality
        skewness_min, skewness_max, skewness_avg = skewness

        shape = self.min_coeff * shape_min + (1 - self.min_coeff) * shape_avg
        skew = self.min_coeff * skewness_min + (1 - self.min_coeff) * skewness_avg

        self.reward = self.shape_coeff * shape + (1 - self.shape_coeff) * skew

        return self.reward

    def neighbor_cells_quality(self, cells):
        sum_quality = 0
        min_quality = np.inf
        max_quality = -np.inf
        for cell in cells:
            cell_shape_q = cell.get_quality()
            sum_quality += cell_shape_q
            min_quality = min(min_quality, cell_shape_q)
            max_quality = max(max_quality, cell_shape_q)

        avg_quality = sum_quality / len(cells) if len(cells) > 0 else 0
        shape_quality = (min_quality, max_quality, avg_quality)

        sum_quality = 0
        min_quality = np.inf
        max_quality = -np.inf
        for cell in cells:
            cell_skew_q = cell.get_skewness()
            sum_quality += cell_skew_q
            min_quality = min(min_quality, cell_skew_q)
            max_quality = max(max_quality, cell_skew_q)

        avg_quality = sum_quality / len(cells) if len(cells) > 0 else 0
        skewness_quality = (min_quality, max_quality, avg_quality)

        return shape_quality, skewness_quality


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.state_path = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.action_path = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.common_path = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
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
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
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
        action_dim = env.action_space.shape[0]

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)

        # self.buffer = deque(maxlen=1000000)
        self.replay_buffer = MeshReplayBuffer(
            buffer_size=1e6,  # 经验回放缓冲区大小
            observation_space=env.observation_space,  # 环境的 observation space
            action_space=env.action_space,  # 环境的 action space
        )
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 1e-3

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

        # 训练参数
        self.replay_buffer = MeshReplayBuffer(
            buffer_size=1e6,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

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
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch.observations)
        actions = torch.FloatTensor(batch.actions)
        rewards = torch.FloatTensor(batch.rewards)
        next_states = torch.FloatTensor(batch.next_observations)
        dones = torch.FloatTensor(batch.dones)

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

    # def select_action(self, state, noise_scale=0.3):
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state.flatten())
    #         action = self.actor(state)
    #         action += noise_scale * torch.randn_like(action)
    #         return action.numpy().reshape(-1, 2)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.flatten())
            action = self.actor(state)
            # 使用 OU 噪声
            noise = torch.tensor(self.noise(), dtype=torch.float32)
            action = action + noise
            return action.numpy().reshape(-1, 2)


# 新增经验回放缓冲区（兼容SB3）
class MeshReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space):
        # ... existing code ...
        
        # 验证修改后的空间定义
        modified_obs_space = gym.spaces.Box(
            low=0.0, 
            high=1.0,
            shape=(16,),  # 1D形状符合SB3要求
            dtype=np.float32
        )
        
        modified_act_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),   # 1D动作空间
            dtype=np.float32
        )

        super().__init__(
            buffer_size=int(buffer_size),
            observation_space=modified_obs_space,
            action_space=modified_act_space,
            device="cpu"
        )

    # def __init__(self, buffer_size, observation_space, action_space):
    #     # 将二维观测空间展平以适应SB3的ReplayBuffer
    #     self.original_obs_shape = observation_space.shape
    #     modified_obs_space = gym.spaces.Box(
    #         low=observation_space.low.flatten(),
    #         high=observation_space.high.flatten(),
    #         shape=(np.prod(observation_space.shape),),  # 展平为1D
    #         dtype=observation_space.dtype,
    #     )

    #     # 调整动作空间维度
    #     modified_act_space = gym.spaces.Box(
    #         low=action_space.low.flatten(),  # Flatten the low bounds
    #         high=action_space.high.flatten(),  # Flatten the high bounds
    #         shape=(np.prod(action_space.shape),),  # 展平为1D
    #         dtype=action_space.dtype,
    #     )

    #     super().__init__(int(buffer_size), modified_obs_space, modified_act_space, device="cpu")

    def add(self, obs, next_obs, action, reward, done):
        # 保持原有reshape逻辑，但使用展平后的形状
        super().add(
            obs=obs.reshape(-1),  # 二维转一维
            next_obs=next_obs.reshape(-1),
            action=action.reshape(-1),
            reward=reward,
            done=done,
        )


def train_drl(train_grid, visual_obj):
    env = DRLSmoothingEnv(
        max_ring_nodes=8, initial_grid=train_grid, visual_obj=visual_obj
    )
    agent = DDPGAgent(env)

    max_episodes = 50000
    max_steps = env.initial_grid.num_nodes or 1000
    save_interval = 10000

    for ep in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.replay_buffer.add(state, next_state, action, reward, done)
            agent.train_step()

            state = next_state
            episode_reward += reward

            if done:
                break

        if (ep + 1) % save_interval == 0:
            torch.save(agent.actor.state_dict(), f"./agent/actor_{ep+1}.pth")
            torch.save(agent.critic.state_dict(), f"./agent/critic_{ep+1}.pth")

        print(f"Episode {ep+1}, Reward: {episode_reward:.2f}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    visual_obj = Visualization(True)

    train_grid = parse_stl_msh("./neural/DRL_Smoothing/training_mesh/training_mesh.stl")

    # train_grid.visualize_unstr_grid_2d(visual_obj)

    train_drl(train_grid, visual_obj)

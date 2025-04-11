import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from optimize import node_perturbation

class DRLSmoothingEnv(gym.Env):
    def __init__(
        self, max_ring_nodes=8,initial_grid = None
    ):
        super(DRLSmoothingEnv, self).__init__()
        self.reward_coeff = 1.0 # minQuality在reward中的权重
        self.lamda = 0.0 # shape quality所占权重，skewness权重为1-lamda
        self.max_ring_nodes = max_ring_nodes # 最大允许的环节点数量
        self.initial_grid = initial_grid # 初始网格
        
        self.current_node_id = 0 # 当前节点ID
        self.state = [] # 当前状态
        self.done = False # 是否完成
        self.local_range =[] # 当前状态的局部范围，用于归一化和反归一化
        self.init_env()

    def init_env(self):
        # 初始化节点邻居环节点
        self.initial_grid.init_node2node_by_cell()
        
        # 初始化网格节点扰动
        self.initial_grid = node_perturbation(self.initial_grid)
        
        # 初始化观测和动作空间
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_ring_nodes, 2), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32
        )

    def init_state(self):
        # 获取当前节点的环状邻居坐标
        node_ids = self.initial_grid.node2node[self.current_node_id]
        ring_coords = np.array([self.initial_grid.node_coords[i] for i in node_ids])
        
        # 计算坐标范围
        x_min, x_max = np.min(ring_coords[:,0]), np.max(ring_coords[:,0])
        y_min, y_max = np.min(ring_coords[:,1]), np.max(ring_coords[:,1])
        ref_d = max(x_max - x_min, y_max - y_min) + 1e-8  # 防止除零
        
        # 归一化处理
        normalized_coords = ring_coords - np.array([[x_min, y_min]])
        normalized_coords /= ref_d
        
        # 填充至固定长度
        if len(normalized_coords) < self.max_ring_nodes:
            pad = np.zeros((self.max_ring_nodes - len(normalized_coords), 2))
            normalized_coords = np.vstack([normalized_coords, pad])
        
        self.state = normalized_coords[:self.max_ring_nodes]  # 截断超长部分
        self.local_range = (x_min, x_max, y_min, y_max)  # 保存范围用于逆变换

    def reset(self):
        self.done = False
        self.current_node_id = 0
        self.init_state()

        return self._get_obs()

    def step(self, action):
        # 实现环境交互逻辑
        self.actionStorage = action
        self.IsDone = True  # 简化示例
        reward = self._calculate_reward()
        return self._get_obs(), reward, self.IsDone, {}



    def _get_obs(self):
        # 返回当前观测值（示例数据）
        return np.random.randn(self.max_ringNodes, 2).astype(np.float32)

    def _calculate_reward(self):
        # 计算奖励逻辑（示例）
        return 1.0


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.state_path = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.action_path = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2), nn.ReLU()
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
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        action_dim = env.action_space.shape[0] * env.action_space.shape[1]

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)

        self.buffer = deque(maxlen=1000000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 1e-3

# 新增目标网络初始化
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 新增OU噪声
        self.noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(env.action_space.shape),
            sigma=0.3 * np.ones(env.action_space.shape),
            theta=0.15,
            dt=1e-2)
            
        # 新增训练参数
        self.replay_buffer = MeshReplayBuffer(
            buffer_size=1e6,
            observation_space=env.observation_space,
            action_space=env.action_space)

    # 新增软更新方法
    def soft_update(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data)

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

    def select_action(self, state, noise_scale=0.3):
        with torch.no_grad():
            state = torch.FloatTensor(state.flatten())
            action = self.actor(state)
            action += noise_scale * torch.randn_like(action)
            return action.numpy().reshape(-1, 2)


# 新增经验回放缓冲区（兼容SB3）
class MeshReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space):
        super().__init__(buffer_size, observation_space, action_space)

    def add(self, obs, next_obs, action, reward, done):
        super().add(
            obs=obs.flatten(),
            next_obs=next_obs.flatten(),
            action=action.flatten(),
            reward=reward,
            done=done,
        )

def train_drl():
    env = DRLEnv(max_ring_nodes=8)
    agent = DDPGAgent(env)
    
    max_episodes = 50000
    max_steps = env.nNodes or 1000
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
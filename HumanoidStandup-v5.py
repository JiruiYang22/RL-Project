import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import time
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# 检查是否有 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# 定义策略网络（Policy Network）
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        return mean

# 定义价值网络（Value Network）
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# PPO 算法类
class PPO:
    def __init__(self, env, policy_net, value_net, lr=3e-4, gamma=0.99, clip_epsilon=0.2, batch_size=64, epochs=10, entropy_coef=0.01):
        self.env = env
        self.policy_net = policy_net.to(device)  # 将策略网络移到 CUDA 上
        self.value_net = value_net.to(device)    # 将价值网络移到 CUDA 上
        self.optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.memory = deque(maxlen=10000)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # 移动到 GPU
        mean = self.policy_net(state_tensor)
        dist = Normal(mean, 1.0)  # 假设标准差为1（可以调整）
        action = dist.sample()
        return action, dist.log_prob(action)

    def compute_returns(self, rewards, dones, next_state, values):
        returns = []
        next_value = self.value_net(torch.FloatTensor(next_state).unsqueeze(0).to(device)).item()  # 移动到 GPU
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            returns.insert(0, rewards[t] + self.gamma * next_value)
            next_value = returns[0]
        return torch.tensor(returns).to(device)  # 移动到 GPU

    def update(self):
        # 使用 torch.stack 堆叠张量列表
        states = torch.stack([torch.tensor(m[0], dtype=torch.float32).to(device) for m in self.memory])  # 状态张量
        actions = torch.stack([m[1].clone().detach().float().to(device) for m in self.memory])  # 修正：堆叠动作张量
        old_log_probs = torch.stack([m[2].clone().detach().float().to(device) for m in self.memory])  # 修正：堆叠旧的 log_prob
        returns = torch.stack([m[3].clone().detach().float().to(device) for m in self.memory])  # 修正：堆叠 returns

        # 计算优势函数
        advantages = returns - self.value_net(states).squeeze()

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                # Get batch
                batch_states = states[i:i+self.batch_size]
                batch_actions = actions[i:i+self.batch_size]
                batch_old_log_probs = old_log_probs[i:i+self.batch_size]
                batch_returns = returns[i:i+self.batch_size]
                batch_advantages = advantages[i:i+self.batch_size]

                # Compute new log probabilities
                mean = self.policy_net(batch_states)
                dist = Normal(mean, 1.0)  # 假设标准差为1
                log_probs = dist.log_prob(batch_actions)

                # Compute the ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clip objective
                obj = ratio * batch_advantages.unsqueeze(1)
                obj_clipped = torch.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages.unsqueeze(1)
                loss_policy = -torch.min(obj, obj_clipped).mean()

                # Value loss (mean squared error)
                loss_value = (self.value_net(batch_states).squeeze() - batch_returns).pow(2).mean()

                # Entropy loss (to encourage exploration)
                entropy_loss = dist.entropy().mean()

                # Total loss
                loss = loss_policy + 0.5 * loss_value - self.entropy_coef * entropy_loss

                # Update networks
                loss.backward(retain_graph=True)
                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                self.optimizer_policy.step()
                self.optimizer_value.step()

    def train(self, num_episodes=1000):
        cum_rewards = []
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            episode_rewards = 0
            episode_states, episode_actions, episode_log_probs, episode_rewards_list, episode_dones = [], [], [], [], []

            done = False
            gamma_k = 1
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action.squeeze(0).cpu().numpy())

                episode_rewards += reward * gamma_k
                gamma_k *= self.gamma

                episode_states.append(state)
                episode_actions.append(action.squeeze(0))
                episode_log_probs.append(log_prob.squeeze(0))
                episode_rewards_list.append(reward)
                episode_dones.append(done)

                state = next_state

                if done or truncated:
                    returns = self.compute_returns(episode_rewards_list, episode_dones, next_state, self.value_net)
                    for i in range(len(episode_states)):
                        self.memory.append((episode_states[i], episode_actions[i], episode_log_probs[i], returns[i]))

                    self.update()
                    self.memory.clear()
                    print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_rewards}")
                    cum_rewards.append(episode_rewards)
                    break
        return cum_rewards

num_episodes = 1000
batch_size = 200


# 创建环境
env = gym.make('HumanoidStandup-v5', render_mode=None) # render_mode="human"

# 获取状态和动作空间的维度: 348,17
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]  # Humanoid 是连续动作空间

# 创建策略网络和值网络
policy_net = PolicyNetwork(obs_dim, act_dim)
value_net = ValueNetwork(obs_dim)

# 创建 PPO 实例
ppo = PPO(env, policy_net, value_net, lr=3e-3, gamma=0.9, clip_epsilon=0.2, batch_size=batch_size, epochs=10)


# 训练 PPO
cum_rewards = ppo.train(num_episodes=num_episodes)

# 绘制结果
plt.plot(range(num_episodes), cum_rewards, label='PPO')
plt.title(f'Cumulative Reward vs. Episodes')
plt.xlabel('Episodes')
plt.ylabel('Average Cumulative Reward')
plt.legend(fontsize=12, loc="lower right")
plt.grid()
plt.savefig('PPO.png')
plt.show()

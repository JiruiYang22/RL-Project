'''
@Author  ：Yan JP
@Created on Date：2023/4/19 17:31 
'''
# https://blog.csdn.net/dgvv4/article/details/129496576?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-129496576-blog-117329002.235%5Ev29%5Epc_relevant_default_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-129496576-blog-117329002.235%5Ev29%5Epc_relevant_default_base3&utm_relevant_index=6
# 代码用于离散环境的模型
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ----------------------------------- #
# 构建策略网络--actor
# ----------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens, dtype = torch.float32)
        self.fc2 = nn.Linear(n_hiddens, n_actions, dtype = torch.float32)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b, n_actions]
        x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
        return x


# ----------------------------------- #
# 构建价值网络--critic
# ----------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x


# ----------------------------------- #
# 构建模型
# ----------------------------------- #

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    # 动作选择
    def take_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # 移动到 GPU
        mean = self.actor(state_tensor)
        dist = torch.distributions.Normal(mean, 1.0)  # 假设标准差为1（可以调整）
        action = dist.sample()
        return action.squeeze().tolist(), dist.log_prob(action).squeeze().tolist()

    # 训练
    def learn(self, transition_dict):
        # 提取数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)
        log_probs = torch.tensor(transition_dict['log_probs'], dtype=torch.float).to(self.device).view(-1, 1)

        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value

        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式 :计算优势函数估计，使用时间差分误差和上一个时间步的优势函数估计
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = log_probs

        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            mean = self.actor(states)
            dist = torch.distributions.Normal(mean, 1.0)  # 假设标准差为1
            log_probs = dist.log_prob(actions)
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 梯度清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()

import matplotlib.pyplot as plt
import gymnasium as gym
import torch
if __name__ == '__main__':


    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    # ----------------------------------------- #
    # 参数设置
    # ----------------------------------------- #

    num_episodes = 300  # 总迭代次数
    gamma = 0.9  # 折扣因子
    actor_lr = 1e-3  # 策略网络的学习率
    critic_lr = 1e-2  # 价值网络的学习率
    n_hiddens = 16  # 隐含层神经元个数
    env_name = 'HumanoidStandup-v5'
    return_list = []  # 保存每个回合的return

    # ----------------------------------------- #
    # 环境加载
    # ----------------------------------------- #

    env = gym.make(env_name, render_mode = None)
    n_states = env.observation_space.shape[0]  # 状态数 4
    n_actions = env.action_space.shape[0]  # 动作数 2

    # ----------------------------------------- #
    # 模型构建
    # ----------------------------------------- #

    agent = PPO(n_states=n_states,  # 状态数
                n_hiddens=n_hiddens,  # 隐含层数
                n_actions=n_actions,  # 动作数
                actor_lr=actor_lr,  # 策略网络学习率
                critic_lr=critic_lr,  # 价值网络学习率
                lmbda=0.95,  # 优势函数的缩放因子
                epochs=10,  # 一组序列训练的轮次
                eps=0.2,  # PPO中截断范围的参数
                gamma=gamma,  # 折扣因子
                device=device
                )

    # ----------------------------------------- #
    # 训练--回合更新 on_policy
    # ----------------------------------------- #

    for i in range(num_episodes):

        state = env.reset()  # 环境重置
        state = state[0]
        done = False  # 任务完成的标记
        truncated = False  # 任务是否被截断的标记
        episode_return = 0  # 累计每回合的reward

        # 构造数据集，保存每个回合的状态数据
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
            'log_probs': []
        }

        while not done and not truncated:
            action, log_prob = agent.take_action(state)  # 动作选择
            next_state, reward, done, truncated, _ = env.step(action)  # 环境更新
            # 保存每个时刻的状态\动作\...
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['log_probs'].append(log_prob)
            # 更新状态
            state = next_state
            # 累计回合奖励
            episode_return += reward

        # 保存每个回合的return
        return_list.append(episode_return)
        # 模型训练
        agent.learn(transition_dict)

        # 打印回合信息
        print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

    # -------------------------------------- #
    # 绘图
    # -------------------------------------- #

    plt.plot(return_list)
    plt.title('return')
    plt.show()

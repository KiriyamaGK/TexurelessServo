import numpy as np
from typing import Callable, Tuple, List
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DAGGER:
    def __init__(
            self,
            expert_policy: Callable[[np.ndarray], np.ndarray],
            learner_policy: nn.Module,
            env,
            epochs_per_iter: int = 5,
            batch_size: int = 32,
            learning_rate: float = 1e-3,
            num_iterations: int = 10,
            rollout_steps: int = 1000,
            device: str = "cpu"
    ):
        """
        DAGGER 训练框架初始化

        参数:
            expert_policy: 专家策略函数，输入状态，返回动作
            learner_policy: 待训练的学员策略(PyTorch模型)
            env: 训练环境(OpenAI Gym风格)
            epochs_per_iter: 每次迭代的训练epoch数
            batch_size: 训练batch大小
            learning_rate: 学习率
            num_iterations: DAGGER迭代次数
            rollout_steps: 每次迭代的rollout步数
            device: 训练设备(cpu/cuda)
        """
        self.expert_policy = expert_policy
        self.learner_policy = learner_policy.to(device)
        self.env = env
        self.epochs_per_iter = epochs_per_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.rollout_steps = rollout_steps
        self.device = device

        self.criterion = nn.MSELoss()  # 假设连续动作空间
        self.optimizer = optim.Adam(self.learner_policy.parameters(), lr=learning_rate)

        # 存储所有收集的数据
        self.dataset = defaultdict(list)

    def collect_expert_data(self, num_episodes: int = 10) -> None:
        """收集初始专家数据"""
        states, actions = [], []
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.expert_policy(state)
                states.append(state)
                actions.append(action)
                state, _, done, _ = self.env.step(action)

        self.dataset["states"].extend(states)
        self.dataset["actions"].extend(actions)

    def rollout(self, policy: Callable, steps: int) -> List[Tuple]:
        """
        使用给定策略在环境中rollout收集数据

        返回:
            轨迹列表[(state, action, ...)]
        """
        trajectories = []
        state = self.env.reset()
        for _ in range(steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = policy(state_tensor).cpu().numpy()[0]

            next_state, _, done, _ = self.env.step(action)
            trajectories.append((state, action))

            state = next_state if not done else self.env.reset()
        return trajectories

    def train_iteration(self) -> float:
        """执行一次训练迭代，返回平均训练loss"""
        states = torch.FloatTensor(np.array(self.dataset["states"])).to(self.device)
        actions = torch.FloatTensor(np.array(self.dataset["actions"])).to(self.device)

        dataset = TensorDataset(states, actions)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        losses = []
        for _ in range(self.epochs_per_iter):
            for batch_states, batch_actions in loader:
                self.optimizer.zero_grad()
                pred_actions = self.learner_policy(batch_states)
                loss = self.criterion(pred_actions, batch_actions)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

        return np.mean(losses)

    def run(self) -> nn.Module:
        """执行完整的DAGGER训练流程"""
        # 初始专家数据收集
        print("Collecting initial expert data...")
        self.collect_expert_data()

        for it in range(self.num_iterations):
            print(f"\nDAGGER Iteration {it + 1}/{self.num_iterations}")

            # 1. 训练学员策略
            train_loss = self.train_iteration()
            print(f"Training loss: {train_loss:.4f}")

            # 2. 使用学员策略rollout收集新数据
            print("Rolling out current policy...")
            trajectories = self.rollout(self.learner_policy, self.rollout_steps)

            # 3. 对新收集的状态查询专家动作
            print("Querying expert for new data...")
            new_states = [t[0] for t in trajectories]
            new_actions = [self.expert_policy(s) for s in new_states]

            # 4. 将新数据添加到数据集
            self.dataset["states"].extend(new_states)
            self.dataset["actions"].extend(new_actions)

            # 可选: 评估当前策略性能
            self.evaluate()

        return self.learner_policy

    def evaluate(self, num_episodes: int = 5) -> float:
        """评估当前学员策略的性能"""
        total_rewards = []
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action = self.learner_policy(state_tensor).cpu().numpy()[0]

                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Evaluation - Average reward: {avg_reward:.2f}")
        return avg_reward


# 使用示例
if __name__ == "__main__":
    import gym

    # 1. 定义环境和专家策略
    env = gym.make("Pendulum-v1")  # 示例环境


    def expert_policy(state):
        """简化的专家策略示例"""
        # 在实际应用中替换为真实的专家策略
        return np.array([0.1])  # 固定动作


    # 2. 定义学员策略网络
    class LearnerPolicy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

        def forward(self, x):
            return self.net(x)


    # 3. 初始化并运行DAGGER
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dagger = DAGGER(
        expert_policy=expert_policy,
        learner_policy=LearnerPolicy(state_dim, action_dim),
        env=env,
        num_iterations=10,
        rollout_steps=2000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trained_policy = dagger.run()
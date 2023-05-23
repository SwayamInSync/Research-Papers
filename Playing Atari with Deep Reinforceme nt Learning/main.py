import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch import optim

from utils import make_env


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
                        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--capture-video", type=bool, default=True, help="Capture the environment recording")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
                        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=100,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=800,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
                        help="the frequency of training")

    args = parser.parse_args()
    return args


class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.num_actions = env.single_action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity, observation_space, action_space, n_envs):
        self.n_envs = n_envs
        self.buffer_size = max(capacity // n_envs, 1)
        self.obs_shape = observation_space.sample().shape
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape),
                                          dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, 1), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done, infos):
        obs = obs.reshape((self.n_envs, *self.obs_shape))
        next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, 1))

        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=(batch_size,))
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        next_obs = self.next_observations[batch_inds, env_indices, :]
        obs = self.observations[batch_inds, env_indices, :]
        actions = self.actions[batch_inds, env_indices, :]
        rewards = self.rewards[batch_inds, env_indices].reshape(-1, 1)
        dones = self.dones[batch_inds, env_indices].reshape(-1, 1)

        data = (obs, actions, next_obs, dones, rewards)

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array):
        return torch.tensor(array)


def get_epsilon(initial_epsilon, final_epsilon, epsilon_duration, timestep):
    slope = (final_epsilon - initial_epsilon) / epsilon_duration
    eps = slope * timestep + initial_epsilon
    return max(eps, final_epsilon)  # note: final_epsilon < initial_epsilon


if __name__ == "__main__":
    args = get_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed) for _ in range(1)])  # since replay buffer support single environment

    q_network = QNetwork(envs)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    target_network = QNetwork(envs)
    target_network.load_state_dict(q_network.state_dict())  # since target_network will be in sync with q_network

    buffer = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, 1)

    start_time = time.time()
    obs = envs.reset()[0]
    for global_step in range(args.total_timesteps):
        epsilon = get_epsilon(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs))
            actions = torch.argmax(F.softmax(q_values, dim=-1), dim=-1).cpu().numpy()

        next_obs, rewards, episode_terminations, dones, infos = envs.step(actions)

        real_next_obs = next_obs.copy()

        buffer.add(obs, real_next_obs, actions, rewards, dones, [infos])

        # training
        obs = next_obs
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = buffer.sample(args.batch_size)

                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max

                q_values = q_network(data.observations)

                # gathering Q-values for the action taken originally while playing
                q_values = torch.gather(q_values, dim=1, index=data.actions).squeeze()
                loss = loss_fn(q_values, td_target)

                print(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_param.data
                    )

    envs.close()

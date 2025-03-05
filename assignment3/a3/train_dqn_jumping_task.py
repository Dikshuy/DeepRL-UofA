import argparse
import gymnasium as gym
import agent_environment
import numpy as np
import matplotlib.pyplot as plt
import jumping_task
from jumping_task.envs import JumpTaskEnv
import epsilon_greedy_explorers
import torch
import torch.nn as nn
import replay_buffer
import dqn
import double_dqn
import pandas as pd
import os


CCID="ddikshan"

class LinearDecayEpsilonGreedyExploration:
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, start_epsilon, end_epsilon, decay_steps, num_actions):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        assert start_epsilon >= end_epsilon >= 0
        self.epsilon = start_epsilon
        self.decay_steps = decay_steps
        self.num_actions = num_actions
        self.steps = 0

    def select_action(self, action_values) -> int:
        epsilon_decay_step_size = (self.start_epsilon - self.end_epsilon) / self.decay_steps
        epsilon = max(self.start_epsilon - self.steps * epsilon_decay_step_size, self.end_epsilon)
        action_probs = epsilon_greedy_explorers.compute_epsilon_greedy_action_probs(action_values, epsilon)
        self.steps += 1
        return np.random.choice(len(action_probs), p=action_probs)


def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return"):

    # Compute running average
    running_avg = np.mean(np.array(episode_returns_list), axis=0)
    new_running_avg = running_avg.copy()
    for i in range(len(running_avg)):
        new_running_avg[i] = np.mean(running_avg[max(0, i-100):min(len(running_avg), i + 100)])
    running_avg = new_running_avg

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the running average
    plt.plot(
        range(0, len(running_avg)),
        running_avg,
        color='r',
        label=label
    )

    # Adding labels and title
    plt.title(f"({CCID})Episodic Returns")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)

def extract_config(filename_without_ext):
    configs = ['config_1', 'config_2', 'config_3']
    for configuration in configs:
        if configuration in filename_without_ext:
            return configuration
    return None

def produce_plots_for_all_configs():
    configs = ['config_1', 'config_2', 'config_3']
    data_dict = {}
    for configuration in configs:
        data_dict[configuration] = []
    files = os.listdir("data")
    for file in files:
        full_path = os.path.join("data", file)
        if os.path.isfile(full_path):
            if os.path.splitext(file)[-1] == '.csv':
                config = extract_config(os.path.splitext(file)[0])
                assert config is not None, f"{file} is not in the required format."
                df = pd.read_csv(full_path)
                data_dict[config].append(np.squeeze(df.values))

    for configuration in configs:
        if data_dict[configuration]:
            plot_alg_results(data_dict[configuration], f"dqn_jumping_{configuration}.png")

def get_env(config_num, render=False):
    if config_num == 1:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 25, 30],
                        agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                        obstacle_position=0, obstacle_size=(9,10),
                        rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                        max_number_of_steps=300, two_obstacles=False, finish_jump=False)
    elif config_num == 2:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[30, 40],
                        agent_w=7, agent_h=7, agent_init_pos=0, agent_speed=1,
                        obstacle_position=0, obstacle_size=(11,17),
                        rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                        max_number_of_steps=300, two_obstacles=False, finish_jump=False,
                        jump_height=24)
    else:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 30, 40],
                    agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                    obstacle_position=0, obstacle_size=(9,10),
                    rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                    max_number_of_steps=300, two_obstacles=True, finish_jump=False)
    return env

class JumpingTaskQNetwork(nn.Module):
    def __init__(self, _, num_actions):
        super(JumpingTaskQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, input):
        if len(input.shape) == 3:
            input = input.unsqueeze(0)
        return self.network(input).squeeze()

def input_preprocessor(x):
    # your code here (optional)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    # end your code 
    return x

def reward_phi(r):
    # your code here (optional)
    reward = (r - 1)/100 # remove per step +1 reward and normalize it
    # end your code
    return reward
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Which environment", type=int, choices=[1,2,3], default=1)
    parser.add_argument("--num-training-episodes", help="How many episodes you want to train your agent", default=5000, type=int)
    parser.add_argument("--run-label", help="Akin to a random seed", default=1, type=int)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--track-q", action='store_true')
    args = parser.parse_args()

    env = get_env(args.config, args.render)
    num_actions = env.action_space.n

    # hyperparameters
    n_seeds = 1
    lr = 0.0003
    init_eps = 1.0
    final_eps = 0.001
    decay_steps = 10000
    buffer_size = 50000
    discount_factor = 0.99
    target_update_interval = 100
    minibatch_size = 128
    min_replay_size_before_updates = 1000
    n_step = 7
    gradient_update_frequency = 1

    for seed in range(n_seeds):
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
        explorer = LinearDecayEpsilonGreedyExploration(init_eps, final_eps, decay_steps, num_actions)
        q_network = JumpingTaskQNetwork(env.observation_space.low.size, num_actions)
        optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
        buffer = replay_buffer.ReplayBuffer(buffer_size, discount_factor, n_step)

        agent = double_dqn.DoubleDQN(q_network, 
                                    optimizer=optimizer,
                                    replay_buffer=buffer, 
                                    explorer=explorer, 
                                    discount=discount_factor, 
                                    gradient_updates_per_target_refresh=target_update_interval, 
                                    gradient_update_frequency=gradient_update_frequency,
                                    input_preprocessor=input_preprocessor,
                                    minibatch_size=minibatch_size, 
                                    min_replay_size_before_updates=min_replay_size_before_updates, 
                                    reward_phi=reward_phi)
        episode_returns, _ = agent_environment.agent_environment_episode_loop(agent, env, args.num_training_episodes, args.debug, args.track_q)
        df = pd.DataFrame(episode_returns)
        df.to_csv(f'data/config_{args.config}_run_{seed}.csv', index=False)
        produce_plots_for_all_configs()

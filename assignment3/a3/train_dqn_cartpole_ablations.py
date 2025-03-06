import gymnasium as gym
import agent_environment
import numpy as np
import epsilon_greedy_explorers
import dqn
import double_dqn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import replay_buffer
import argparse

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



class CartpoleQNetwork(nn.Module):

    def __init__(self, input_size, num_actions):
        super().__init__()
        self.network = torch.nn.Sequential(nn.Linear(input_size, 64), 
                                           nn.ReLU(),
                                           nn.Linear(64, 64),
                                           nn.ReLU(),
                                           nn.Linear(64, num_actions))

    def forward(self, input):
        return self.network(input)


# Adapted from ChatGPT
def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return", title="Episodic Returns"):

    # Compute running average
    running_avg = np.mean(np.array(episode_returns_list), axis=0)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the original data
    # plt.plot(episode_rewards, marker='o', linestyle='-', color='b', label='Original Data')

    # Plot the running average
    plt.plot(
        range(0, len(running_avg)),
        running_avg,
        color='r',
        label=label
    )

    # Adding labels and title
    plt.title(f"({CCID}){title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


def plot_many_algs(lists, labels, colors, file, ylabel="Return", title="Episodic Returns"):
    # Define a function to calculate the running average

    running_avgs = []
    for i in range(len(lists)):
        running_avgs.append(np.mean(np.array(lists[i]), axis=0))

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the original data
    # plt.plot(episode_rewards, marker='o', linestyle='-', color='b', label='Original Data')

    for i in range(len(lists)):
        # Plot the running average
        plt.plot(
            range(0, len(running_avgs[i])),
            running_avgs[i],
            color=colors[i],
            label=labels[i],
        )

    # Adding labels and title
    plt.title(f"({CCID}){title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--target-network-ablation", action="store_true", default=False)
    parser.add_argument("--replay-buffer-ablation", action="store_true", default=False)
    args = parser.parse_args()

    num_seeds = args.num_runs
    lr = 0.0001
    optimizer_eps = 1e-2
    initial_epsilon = 1.0
    final_epsilon = 0.001
    epsilon_decay_steps = 12500
    discount = 0.99
    min_replay_size_before_updates = 500
    num_training_episodes = 500

    agent_class_to_text = {dqn.DQN: 'DQN'} 

    n_step = 1
    colors = ['r', 'b', 'g', 'o']
    agent_classes = [dqn.DQN]

    if args.target_network_ablation:
        buffer_size = 25000
        minibatch_size = 128
        target_update_intervals = [1, 10, 100]

        perf_dict = {}
        q_val_dict = {}
        for target_update_interval in target_update_intervals:
            perf_dict[target_update_interval] = {}
            q_val_dict[target_update_interval] = {}
            for agent_class in agent_classes:
                agent_text = agent_class_to_text[agent_class]
                alg_returns = []
                alg_q_values = []
                for seed in range(num_seeds):
                    env = gym.make("CartPole-v1")
                    num_actions = env.action_space.n
                    q_network = CartpoleQNetwork(env.observation_space.low.size, num_actions)
                    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
                    explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
                    buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)
                    agent = agent_class(q_network, optimizer, buffer, explorer, discount, target_update_interval,
                                    min_replay_size_before_updates=min_replay_size_before_updates, minibatch_size=minibatch_size)
                    episode_returns, q_values = agent_environment.agent_environment_episode_loop(agent, env, num_training_episodes, args.debug, args.track_q)
                    alg_returns.append(episode_returns)
                    alg_q_values.append(q_values)

                perf_dict[target_update_interval][agent_text] = alg_returns
                q_val_dict[target_update_interval][agent_text] = alg_q_values

        for agent_class in agent_classes:
            agent_text = agent_class_to_text[agent_class]
            plot_many_algs([perf_dict[target_update_interval][agent_text] for target_update_interval in target_update_intervals],
                        [f"{target_update_interval}-update-interval {agent_text}"for target_update_interval in target_update_intervals], colors, f"{agent_text}_cartpole_target_update_intervals.png")
            
    if args.replay_buffer_ablation:
        buffer_sizes = [100, 500, 5000, 25000]
        minibatch_size = 64
        target_update_intervals = 100

        perf_dict = {}
        q_val_dict = {}
        for buffer_size in buffer_sizes:
            perf_dict[buffer_size] = {}
            q_val_dict[buffer_size] = {}
            for agent_class in agent_classes:
                agent_text = agent_class_to_text[agent_class]
                alg_returns = []
                alg_q_values = []
                for seed in range(num_seeds):
                    env = gym.make("CartPole-v1")
                    num_actions = env.action_space.n
                    q_network = CartpoleQNetwork(env.observation_space.low.size, num_actions)
                    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
                    explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
                    buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)
                    agent = agent_class(q_network, optimizer, buffer, explorer, discount, target_update_interval,
                                    min_replay_size_before_updates=min_replay_size_before_updates, minibatch_size=minibatch_size)
                    episode_returns, q_values = agent_environment.agent_environment_episode_loop(agent, env, num_training_episodes, args.debug, args.track_q)
                    alg_returns.append(episode_returns)
                    alg_q_values.append(q_values)

                perf_dict[buffer_size][agent_text] = alg_returns
                q_val_dict[buffer_size][agent_text] = alg_q_values

        for agent_class in agent_classes:
            agent_text = agent_class_to_text[agent_class]
            plot_many_algs([perf_dict[buffer_size][agent_text] for buffer_size in buffer_sizes],
                        [f"{buffer_size}-step {agent_text}"for buffer_size in buffer_sizes], colors, f"{agent_text}_cartpole_buffers.png")

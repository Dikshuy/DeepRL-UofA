import gymnasium as gym
import agent_environment
import numpy as np
import epsilon_greedy_explorers
import td3
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import replay_buffer
import argparse

CCID="ddikshan"

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


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

def plot_timestep_returns(eval_returns_list, eval_timesteps_list, file, env_name, title="Learning Curve", window_size=100):
    plt.figure(figsize=(10, 6))
    
    smoothed_eval_returns_list = []
    for eval_returns in eval_returns_list:
        smoothed_returns = []
        for i in range(len(eval_returns)):
            start = max(0, i - window_size // 2)
            end = min(len(eval_returns), i + window_size // 2 + 1)
            window_mean = np.mean(eval_returns[start:end])
            smoothed_returns.append(window_mean)
        smoothed_eval_returns_list.append(smoothed_returns)
    
    for i, (eval_returns, eval_timesteps) in enumerate(zip(smoothed_eval_returns_list, eval_timesteps_list)):
        plt.plot(eval_timesteps, eval_returns, alpha=0.3, color='r', linestyle='-')
    
    mean_smoothed_eval_returns = np.mean(smoothed_eval_returns_list, axis=0)
    plt.plot(eval_timesteps_list[0], mean_smoothed_eval_returns, color='r', linewidth=2, label='Mean Evaluation Return')
    
    plt.title(f"({CCID}) {title}")
    plt.xlabel("Time Steps")
    plt.ylabel("Evaluation Return")
    plt.legend()
    plt.grid(True)
    plt.savefig(file)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=1000000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_seeds = args.num_runs
    lr = 0.0003
    optimizer_eps = 1e-8
    buffer_size = 1000000
    discount = 0.99
    min_replay_size_before_updates = 25000
    minibatch_size = 256
    total_steps = args.total_steps
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    tau = 0.005
    exploration_noise = 0.1

    environments = [
        "Ant-v4",
        "Walker2d-v4",
        # "MountainCarContinuous-v0",
        # "Reacher-v4",
        # "InvertedPendulum-v4"
    ]

    for env_name in environments:
        print(f"\n==== Training TD3 on {env_name} ====")
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        all_eval_returns = []
        all_eval_timesteps = []

        for seed in range(num_seeds):
            print(f"running seed: {seed}")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            actor = Actor(state_dim, action_dim, max_action)
            critic = Critic(state_dim, action_dim)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, eps=optimizer_eps)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, eps=optimizer_eps)
            explorer = epsilon_greedy_explorers.GaussianNoiseExplorer(std_dev=exploration_noise*max_action, max_action=max_action)
            buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount)
            agent = td3.TD3(actor, actor_optimizer, critic, critic_optimizer, buffer, explorer, discount, policy_noise=policy_noise*max_action, 
                            noise_clip=noise_clip*max_action, policy_update_frequency=policy_freq, minibatch_size=minibatch_size,
                            min_replay_size_before_updates=min_replay_size_before_updates, tau=tau, max_action=max_action, device=device)
            eval_returns, eval_timesteps, _ = agent_environment.agent_environment_step_loop(agent, env, total_steps, min_replay_size_before_updates, debug=args.debug)

            all_eval_returns.append(eval_returns)
            all_eval_timesteps.append(eval_timesteps)

        plot_timestep_returns(all_eval_returns, all_eval_timesteps, f"td3_{env_name.lower().replace('-', '_')}.png", env_name, title=f"TD3 Performance on {env_name}")

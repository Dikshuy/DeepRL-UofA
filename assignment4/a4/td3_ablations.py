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
    def __init__(self, state_dim, action_dim, use_two_q=True):
        super(Critic, self).__init__()
        self.use_two_q = use_two_q

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        if use_two_q:
            # Q2 architecture
            self.l4 = nn.Linear(state_dim + action_dim, 256)
            self.l5 = nn.Linear(256, 256)
            self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        if self.use_two_q:
            q2 = F.relu(self.l4(sa))
            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)
            return q1, q2
        else:
            return q1, q1
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class ModifiedTD3(td3.TD3):
    def __init__(self, actor, actor_optimizer, critic, critic_optimizer, replay_buffer, explorer, discount, policy_noise=0.2, noise_clip=0.5, policy_update_frequency=2, input_preprocessor=lambda x:x, minibatch_size=32, min_replay_size_before_updates=32, tau=0.005, reward_phi=lambda reward: reward, max_action=1, use_two_q=True, use_target_smoothing=True):
        super().__init__(actor, actor_optimizer, critic, critic_optimizer, replay_buffer, explorer, discount, policy_noise, noise_clip, policy_update_frequency, input_preprocessor, minibatch_size, min_replay_size_before_updates, tau, reward_phi, max_action)

        self.use_two_q = use_two_q
        self.use_target_smoothing = use_target_smoothing

    def compute_targets(self, batched_rewards, batched_actions, batched_next_states, batched_discounts, batch_terminated):
        with torch.no_grad():
            if self.use_target_smoothing:
                noise = (torch.randn_like(batched_actions)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_actions = (self.actor_target(batched_next_states) + noise).clamp(-self.max_action, self.max_action)
            else:
                next_actions = self.actor_target(batched_next_states).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(batched_next_states, next_actions)
            
            if self.use_two_q:
                target_q = torch.min(target_q1, target_q2).squeeze(-1)
            else:
                target_q = target_q1.squeeze(-1)

            target_q = batched_rewards + (1 - batch_terminated) * batched_discounts * target_q
        return target_q

def plot_comparison(env_name, results):
    plt.figure(figsize=(12, 8))
    
    colors = ['r', 'b', 'g', 'c', 'm']
    
    max_timestep = 0
    for config_name, data in results.items():
        for timesteps in data["timesteps"]:
            if timesteps[-1] > max_timestep:
                max_timestep = timesteps[-1]
    
    common_x = np.linspace(0, max_timestep, 100)
    
    for i, (config_name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        for returns, timesteps in zip(data["returns"], data["timesteps"]):
            plt.plot(timesteps, returns, alpha=0.2, color=color)
        
        interpolated_returns = []
        for returns, timesteps in zip(data["returns"], data["timesteps"]):
            interpolated_y = np.interp(common_x, timesteps, returns)
            interpolated_returns.append(interpolated_y)
        
        mean_returns = np.mean(interpolated_returns, axis=0)
        plt.plot(common_x, mean_returns, color=color, linewidth=2, label=config_name)
    
    plt.title(f"({CCID}) TD3 Ablation Study on {env_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"td3_ablation_returns_{env_name.lower().replace('-', '_')}.png")
    
    plt.figure(figsize=(12, 8))
    for i, (config_name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        all_q = []
        for q_values in data["q_values"]:
            if q_values:
                all_q.extend(q_values)
        
        if all_q:
            window_size = 1000
            q_windows = [np.mean(all_q[i:i+window_size]) for i in range(0, len(all_q), window_size)]
            plt.plot(range(0, len(q_windows)*window_size, window_size), q_windows, color=color, linewidth=2, label=config_name)
    
    plt.title(f"({CCID}) TD3 Q-Value Estimation on {env_name}")
    plt.xlabel("Training Steps")
    plt.ylabel("Average Q-Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"td3_q_values_{env_name.lower().replace('-', '_')}.png")

# Adapted from Claude
def plot_timestep_returns(returns_list, timesteps_list, file, env_name, title="Learning Curve"):
    plt.figure(figsize=(10, 6))
    
    # Determine the maximum number of timesteps
    max_timesteps = max([ts[-1] for ts in timesteps_list])

    # Create a common x-axis for interpolation
    common_x = np.linspace(0, max_timesteps, 100)
    interpolated_returns = []
    for i, (returns, timesteps) in enumerate(zip(returns_list, timesteps_list)):
        # Use numpy interpolation to get values at common timesteps
        interpolated_y = np.interp(common_x, timesteps, returns)
        interpolated_returns.append(interpolated_y)
        
        plt.plot(timesteps, returns, alpha=0.3, color='r', linestyle='-')

    # Calculate and plot the mean performance
    mean_returns = np.mean(interpolated_returns, axis=0)
    plt.plot(common_x, mean_returns, color='r', linewidth=2, label='Mean Return')
    
    plt.title(f"({CCID}) {title}")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True)
    plt.savefig(file)
    plt.close()

def run_experiments(env_name, num_seeds=5, total_steps=1000000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # hyperparameters
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

    results = {}

    configurations = [
        {"name": "TD3 (Twin Critics)", "use_twin_critics": True, "use_target_smoothing": True},
        {"name": "TD3 (Single Critic)", "use_twin_critics": False, "use_target_smoothing": True}
    ]

    for config in configurations:
        print(f"\n==== Training {config['name']} on {env_name} ====")

        all_returns = []
        all_timesteps = []
        all_q_values = []

        for seed in range(num_seeds):
            print(f"running seed: {seed}")
            actor = Actor(state_dim, action_dim, max_action)
            critic = Critic(state_dim, action_dim, use_two_q=config['use_twin_critics'])
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, eps=optimizer_eps)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, eps=optimizer_eps)
            explorer = epsilon_greedy_explorers.GaussianNoiseExplorer(std_dev=exploration_noise*max_action, max_action=max_action)
            buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount)
            agent = ModifiedTD3(actor, actor_optimizer, critic, critic_optimizer, buffer, explorer, discount, policy_noise=policy_noise*max_action, 
                                noise_clip=noise_clip*max_action, policy_update_frequency=policy_freq, minibatch_size=minibatch_size,
                                min_replay_size_before_updates=min_replay_size_before_updates, tau=tau, max_action=max_action, use_two_q=config['use_twin_critics'], use_target_smoothing=config['use_target_smoothing'])
            
            episode_returns, episode_timesteps, q_values = agent_environment.agent_environment_step_loop(agent, env, total_steps, debug=True, track_q=True)
            all_returns.append(episode_returns)
            all_timesteps.append(episode_timesteps)
            all_q_values.append(q_values)

        results[config["name"]] = {
            "returns": all_returns,
            "timesteps": all_timesteps,
            "q_values": all_q_values
        }

    plot_comparison(env_name, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--total-steps", type=int, default=1000000)
    args = parser.parse_args()

    environments = [
        "Ant-v4",
        "Walker2d-v4"
    ]

    for env_name in environments:
        run_experiments(env_name, args.num_runs, args.total_steps)
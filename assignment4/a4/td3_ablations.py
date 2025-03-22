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
    def __init__(self, actor, actor_optimizer, critic, critic_optimizer, replay_buffer, explorer, discount, policy_noise=0.2, noise_clip=0.5, policy_update_frequency=2, input_preprocessor=..., minibatch_size=32, min_replay_size_before_updates=32, tau=0.005, reward_phi=..., max_action=1, use_two_q=True, use_target_smoothing=True):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=1000000)
    args = parser.parse_args()

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
        "Walker2d-v4"
    ]

    for env_name in environments:
        print(f"\n==== Training TD3 on {env_name} ====")
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        all_returns = []
        all_timesteps = []
        all_q_values = []

        for seed in range(num_seeds):
            print(f"running seed: {seed}")
            actor = Actor(state_dim, action_dim, max_action)
            critic = Critic(state_dim, action_dim)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, eps=optimizer_eps)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, eps=optimizer_eps)
            explorer = epsilon_greedy_explorers.GaussianNoiseExplorer(std_dev=exploration_noise*max_action, max_action=max_action)
            buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount)
            agent = td3.TD3(actor, actor_optimizer, critic, critic_optimizer, buffer, explorer, discount, policy_noise=policy_noise*max_action, 
                            noise_clip=noise_clip*max_action, policy_update_frequency=policy_freq, minibatch_size=minibatch_size,
                            min_replay_size_before_updates=min_replay_size_before_updates, tau=tau, max_action=max_action)
            episode_returns, episode_timesteps, q_values = agent_environment.agent_environment_step_loop(agent, env, total_steps, args.debug, args.track_q)
            all_returns.append(episode_returns)
            all_timesteps.append(episode_timesteps)
            all_q_values.append(q_values)

        timesteps_list = []
        for timesteps in all_timesteps:
            timesteps_list.append(np.array(timesteps))

        plot_timestep_returns(all_returns, timesteps_list,f"td3_{env_name.lower().replace('-', '_')}.png", env_name, title=f"TD3 Performance on {env_name}")

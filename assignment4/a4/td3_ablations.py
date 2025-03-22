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
import os
import json

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
    def __init__(self, actor, actor_optimizer, critic, critic_optimizer, replay_buffer, explorer, discount, policy_noise=0.2, noise_clip=0.5, policy_update_frequency=2, input_preprocessor=lambda x:x, minibatch_size=32, min_replay_size_before_updates=32, tau=0.005, reward_phi=lambda reward: reward, max_action=1, use_two_q=True):
        super().__init__(actor, actor_optimizer, critic, critic_optimizer, replay_buffer, explorer, discount, policy_noise, noise_clip, policy_update_frequency, input_preprocessor, minibatch_size, min_replay_size_before_updates, tau, reward_phi, max_action)

        self.use_two_q = use_two_q

    def compute_targets(self, batched_rewards, batched_actions, batched_next_states, batched_discounts, batch_terminated):
        with torch.no_grad():
            noise = (torch.randn_like(batched_actions)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(batched_next_states) + noise).clamp(-self.max_action, self.max_action)
            
            target_q1, target_q2 = self.critic_target(batched_next_states, next_actions)
            
            if self.use_two_q:
                target_q = torch.min(target_q1, target_q2).squeeze(-1)
            else:
                target_q = target_q1.squeeze(-1)

            target_q = batched_rewards + (1 - batch_terminated) * batched_discounts * target_q
        return target_q

def plot_grouped_configs(env_name, results_dir, group_name, configs_to_plot, filename_suffix):
    plt.figure(figsize=(12, 8))
    
    colors = ['r', 'b', 'g', 'c', 'm', 'y']
    
    results = {}
    max_timestep = 0
    
    for config_name in configs_to_plot:
        config_folder = os.path.join(results_dir, config_name.replace(" ", "_").replace("(", "").replace(")", ""))
        results[config_name] = {"returns": [], "timesteps": []}
        
        for seed_file in os.listdir(config_folder):
            if seed_file.endswith(".json"):
                with open(os.path.join(config_folder, seed_file), 'r') as f:
                    seed_data = json.load(f)
                    results[config_name]["returns"].append(seed_data["returns"])
                    results[config_name]["timesteps"].append(seed_data["timesteps"])
                    
                    if seed_data["timesteps"][-1] > max_timestep:
                        max_timestep = seed_data["timesteps"][-1]
    
    if not results:
        print(f"No results found for the specified configurations: {configs_to_plot}")
        return
    
    common_x = np.linspace(0, max_timestep, 100)
    for i, config_name in enumerate(configs_to_plot):
        if config_name not in results:
            continue
            
        data = results[config_name]
        color = colors[i % len(colors)]
        for returns, timesteps in zip(data["returns"], data["timesteps"]):
            plt.plot(timesteps, returns, alpha=0.3, color=color)
        
        interpolated_returns = []
        for returns, timesteps in zip(data["returns"], data["timesteps"]):
            interpolated_y = np.interp(common_x, timesteps, returns)
            interpolated_returns.append(interpolated_y)
        
        mean_returns = np.mean(interpolated_returns, axis=0)
        plt.plot(common_x, mean_returns, color=color, linewidth=2, label=config_name)
    
    plt.title(f"({CCID}) TD3 Ablation Study - {group_name} on {env_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"td3_ablation_{env_name.lower().replace('-', '_')}_{filename_suffix}.png"))
    plt.close()

def run_single_config(env_name, config, seed, total_steps, output_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
  
    env = gym.make(env_name)
    env.reset(seed=seed)
    
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
    noise_clip = 0.5
    tau = 0.005
    exploration_noise = 0.1
    
    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim, use_two_q=config['use_twin_critics'])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, eps=optimizer_eps)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, eps=optimizer_eps)
    explorer = epsilon_greedy_explorers.GaussianNoiseExplorer(std_dev=exploration_noise*max_action, max_action=max_action)
    buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount)
    agent = ModifiedTD3(actor, actor_optimizer, critic, critic_optimizer, buffer, explorer, discount, 
                        policy_noise=config["policy_noise"] * max_action, 
                        noise_clip=noise_clip, 
                        policy_update_frequency=config["policy_freq"], 
                        minibatch_size=minibatch_size,
                        min_replay_size_before_updates=min_replay_size_before_updates, 
                        tau=tau, 
                        max_action=max_action, 
                        use_two_q=config['use_twin_critics'])
    
    print(f"Training {config['name']} with seed {seed} on {env_name}")
    episode_returns, episode_timesteps, q_values = agent_environment.agent_environment_step_loop(agent, env, total_steps, debug=False, track_q=True)
    
    config_folder = os.path.join(output_dir, config['name'].replace(" ", "_").replace("(", "").replace(")", ""))
    os.makedirs(config_folder, exist_ok=True)
    
    result = {
        "returns": episode_returns,
        "timesteps": episode_timesteps,
        "q_values": q_values,
        "config": config,
        "seed": seed
    }
    
    with open(os.path.join(config_folder, f"seed_{seed}.json"), 'w') as f:
        json.dump(result, f)
        
    return config['name'], seed

def create_plots(env_name, results_dir):    
    # 1. Twin Critics vs Single Critic
    plot_grouped_configs(
        env_name, 
        results_dir, 
        "Twin Critics vs Single Critic", 
        ["TD3 (Default)", "TD3 (Single Critic)"], 
        "critics_comparison"
    )
    
    # 2. Delayed Updates Comparison
    plot_grouped_configs(
        env_name, 
        results_dir, 
        "Policy Update Frequency", 
        ["TD3 (Default)", "TD3 (No Delayed Updates)", "TD3 (More Delayed Updates)"], 
        "policy_update_frequency"
    )
    
    # 3. Policy Noise Comparison
    plot_grouped_configs(
        env_name, 
        results_dir, 
        "Policy Noise", 
        ["TD3 (Default)", "TD3 (No policy noise)", "TD3 (More policy noise)"], 
        "policy_noise"
    )
    
    # 4. All configs combined
    all_configs = [
        "TD3 (Default)",
        "TD3 (Single Critic)",
        "TD3 (No Delayed Updates)",
        "TD3 (More Delayed Updates)",
        "TD3 (No policy noise)",
        "TD3 (More policy noise)"
    ]
    
    plot_grouped_configs(
        env_name, 
        results_dir, 
        "All Configurations", 
        all_configs, 
        "all_configs"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=1000000)
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--config-name", type=str, required=True, help="Configuration name")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--total-steps", type=int, default=1000000, help="Total steps to train")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--use-twin-critics", type=int, default=1, help="1 for twin critics, 0 for single critic")
    parser.add_argument("--debug", action="store_true", default=False, help="Track returns")
    parser.add_argument("--track-q", action="store_true", default=False, help="Track Q-values")
    parser.add_argument("--policy-freq", type=int, default=2, help="Policy update frequency")
    parser.add_argument("--policy-noise", type=float, default=0.2, help="Policy noise")
    parser.add_argument("--plot-only", action="store_true", help="Only create plots from existing results")
    args = parser.parse_args()
    args = parser.parse_args()

    if args.plot_only:
        print(f"Creating plots from results in {args.output_dir}")
        create_plots(args.env, args.output_dir)
    else:
        config = {
            "name": args.config_name,
            "use_twin_critics": args.use_twin_critics == 1,
            "policy_freq": args.policy_freq,
            "policy_noise": args.policy_noise
        }
        
        os.makedirs(args.output_dir, exist_ok=True)
        run_single_config(args.env, config, args.seed, args.total_steps, args.output_dir)
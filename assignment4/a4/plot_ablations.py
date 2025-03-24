import os
import json
import numpy as np
import matplotlib.pyplot as plt

CCID = "ddikshan"

def load_returns_from_jsons(base_directory, env_name, config_names):
    results = {}
    
    for config in config_names:
        sanitized_config = config.replace(" ", "_").replace("(", "").replace(")", "")
        
        config_dir = os.path.join(base_directory, env_name, sanitized_config)
        json_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        
        config_returns = []
        config_timesteps = []
        
        for filename in json_files:
            filepath = os.path.join(config_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                config_returns.append(data['returns'])
                config_timesteps.append(data['timesteps'])

        results[config] = {
                'returns': config_returns,
                'timesteps': config_timesteps
            }
        
    return results

def plot_timestep_returns(results, configs, file, env_name, title="Learning Curve"):
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    max_timesteps = 0
    
    for config in configs:
        config_results = results[config]
        timesteps_list = config_results['timesteps']
        max_timesteps = max(max_timesteps, max([ts[-1] for ts in timesteps_list]))
    
    common_x = np.linspace(0, max_timesteps, 100)
    
    for i, config in enumerate(configs):
        config_results = results[config]
        returns_list = config_results['returns']
        timesteps_list = config_results['timesteps']
        
        config_interpolated_returns = []
        for returns, timesteps in zip(returns_list, timesteps_list):
            interpolated_y = np.interp(common_x, timesteps, returns)
            config_interpolated_returns.append(interpolated_y)
        
        mean_returns = np.mean(config_interpolated_returns, axis=0)
        
        for returns, timesteps in zip(returns_list, timesteps_list):
            plt.plot(timesteps, returns, alpha=0.2, color=colors[i], linestyle='-')
        
        plt.plot(common_x, mean_returns, color=colors[i], linewidth=2, label=config)
    
    plt.title(f"({CCID}) {title} - {env_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file)
    plt.close()

def generate_ablation_plots(base_results_dir, output_dir):
    envs = ['Ant-v4', 'Walker2d-v4']
    
    plot_configs = [
        {
            'name': 'Critic Architecture Comparison',
            'configs': ['TD3_(Default)', 'TD3_(Single_Critic)']
        },
        {
            'name': 'Policy Update Frequency Comparison',
            'configs': ['TD3_(No_Delayed_Updates)', 'TD3_(Default)', 'TD3_(More_Delayed_Updates)']
        },
        {
            'name': 'Policy Noise Comparison',
            'configs': ['TD3_(No_policy_noise)', 'TD3_(Default)', 'TD3_(More_policy_noise)']
        },
        {
            'name': 'Target Network Update Rate Comparison',
            'configs': ['TD3_(Slow_Target_Update)', 'TD3_(Default)', 'TD3_(Fast_Target_Update)']
        },
        {
            'name': 'Exploration Strategy Comparison',
            'configs': ['TD3_(Default)', 'TD3_(Ornstein_Uhlenbeck)']
        }
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for env in envs:
        results = load_returns_from_jsons(base_results_dir, env, 
            ['TD3_(Default)', 'TD3_(Single_Critic)', 
             'TD3_(No_Delayed_Updates)', 'TD3_(More_Delayed_Updates)',
             'TD3_(No_policy_noise)', 'TD3_(More_policy_noise)',
             'TD3_(Slow_Target_Update)', 'TD3_(Fast_Target_Update)',
             'TD3_(Ornstein_Uhlenbeck)'])
        
        for plot_config in plot_configs:
            output_file = os.path.join(output_dir, f"{env}_{plot_config['name'].replace(' ', '_')}.png")
            plot_timestep_returns(results, plot_config['configs'], output_file, env, plot_config['name'])
            print(f"Generated plot for config {plot_config['name']} for environment {env}")

if __name__ == '__main__':
    BASE_RESULTS_DIR = 'results'
    OUTPUT_PLOT_DIR = 'results/plots'
    generate_ablation_plots(BASE_RESULTS_DIR, OUTPUT_PLOT_DIR)
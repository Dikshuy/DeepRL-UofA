import os
import json
import numpy as np
import matplotlib.pyplot as plt

CCID = "ddikshan"

def load_returns_from_jsons(directory, config_names):
    """
    Load returns from JSON files for specified configurations.
    
    Parameters:
    - directory: Path to the directory containing JSON result files
    - config_names: List of configuration names to plot
    
    Returns:
    - Dictionary with config names as keys and lists of returns/timesteps
    """
    results = {}
    for config in config_names:
        # Sanitize config name for file matching
        sanitized_config = config.replace(" ", "_").replace("(", "").replace(")", "")

        config_returns = []
        config_timesteps = []
        
        # Find all matching JSON files
        matching_files = [f for f in os.listdir(directory) if sanitized_config in f and f.endswith('.json')]
        print(f"Found {len(matching_files)} files for config: {config}")
        
        for filename in matching_files:
            filepath = os.path.join(directory, filename)
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
    """
    Plot learning curves for multiple configurations.
    
    Parameters:
    - results: Dictionary of results from load_returns_from_jsons
    - configs: List of configuration names to plot
    - file: Output file path for the plot
    - env_name: Name of the environment
    - title: Plot title
    """
    plt.figure(figsize=(12, 7))
    
    # Set color map for consistent colors
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # Prepare for interpolation
    max_timesteps = 0
    interpolated_returns = {}
    
    # First pass: find max timesteps
    for config in configs:
        config_results = results[config]
        timesteps_list = config_results['timesteps']
        max_timesteps = max(max_timesteps, max([ts[-1] for ts in timesteps_list]))
    
    # Common x-axis for interpolation
    common_x = np.linspace(0, max_timesteps, 100)
    
    # Plot each configuration
    for i, config in enumerate(configs):
        config_results = results[config]
        returns_list = config_results['returns']
        timesteps_list = config_results['timesteps']
        
        # Interpolate returns
        config_interpolated_returns = []
        for returns, timesteps in zip(returns_list, timesteps_list):
            interpolated_y = np.interp(common_x, timesteps, returns)
            config_interpolated_returns.append(interpolated_y)
        
        # Calculate mean and standard deviation
        mean_returns = np.mean(config_interpolated_returns, axis=0)
        std_returns = np.std(config_interpolated_returns, axis=0)
        
        # Plot individual runs
        for returns, timesteps in zip(returns_list, timesteps_list):
            plt.plot(timesteps, returns, alpha=0.2, color=colors[i], linestyle='-')
        
        # Plot mean with standard deviation
        plt.plot(common_x, mean_returns, color=colors[i], linewidth=2, label=config)
        plt.fill_between(common_x, 
                         mean_returns - std_returns, 
                         mean_returns + std_returns, 
                         color=colors[i], 
                         alpha=0.1)
    
    plt.title(f"({CCID}) {title} - {env_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file)
    plt.close()

def generate_ablation_plots(base_results_dir, output_dir):
    """
    Generate all specified ablation plots for each environment.
    
    Parameters:
    - base_results_dir: Base directory containing result JSON files
    - output_dir: Directory to save output plots
    """
    # Environments to process
    envs = ['Ant-v4', 'Walker2d-v4']
    
    # Plotting configurations
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
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each environment
    for env in envs:
        env_results_dir = os.path.join(base_results_dir, env)
        
        # Load results for this environment
        results = load_returns_from_jsons(env_results_dir, 
            ['TD3_(Default)', 'TD3_(Single_Critic)', 
             'TD3_(No_Delayed_Updates)', 'TD3_(More_Delayed_Updates)',
             'TD3_(No_policy_noise)', 'TD3_(More_policy_noise)',
             'TD3_(Slow_Target_Update)', 'TD3_(Fast_Target_Update)',
             'TD3_(Ornstein_Uhlenbeck)'])
        
        # Generate plots for this environment
        for plot_config in plot_configs:
            output_file = os.path.join(output_dir, f"{env}_{plot_config['name'].replace(' ', '_')}.png")
            plot_timestep_returns(results, 
                                  plot_config['configs'], 
                                  output_file, 
                                  env, 
                                  plot_config['name'])
            print(f"Generated plot: {output_file}")

if __name__ == '__main__':
    # Update these paths as needed
    BASE_RESULTS_DIR = 'results'
    OUTPUT_PLOT_DIR = 'ablation_plots'
    generate_ablation_plots(BASE_RESULTS_DIR, OUTPUT_PLOT_DIR)
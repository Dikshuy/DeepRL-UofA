import argparse
import gymnasium as gym
import agent_environment
import numpy as np
import semi_gradient_sarsa
import matplotlib.pyplot as plt
import jumping_task
from jumping_task.envs import JumpTaskEnv
import feature_extraction

import feature_extraction_dump

import epsilon_greedy_explorers

CCID="ddikshan"


def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return"):

    # Compute running average
    mean_curve = np.mean(np.array(episode_returns_list), axis=0)
    new_mean_curve = mean_curve.copy()
    for i in range(len(mean_curve)):
        new_mean_curve[i] = np.mean(mean_curve[max(0, i-10):min(len(mean_curve), i + 10)])
    mean_curve = new_mean_curve

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the running average
    plt.plot(range(0, len(mean_curve)), mean_curve, color='r',label=label)

    for returns in episode_returns_list:
        curve = np.array(returns)
        plt.plot(range(0, len(curve)), curve, color='r', alpha=0.25)  # Adjust alpha for transparency

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


class GreedyExploration:
    """Pure Greedy Exploration

    Args:
      num_actions: integer indicating the number of actions
    """

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def select_action(self, action_values) -> int:
        max_value = np.max(action_values)
        indices = np.where(action_values == max_value)[0]
        num_greedy_actions = len(indices)
        action_probs = np.zeros(action_values.shape)
        action_probs[indices] = 1. / num_greedy_actions
        assert np.sum(action_probs) == 1
        return np.random.choice(len(action_probs), p=action_probs)


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

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Which environment", type=int, choices=[1,2,3], default=1)
    parser.add_argument("--num-training-episodes", help="How many episodes you want to train your agent", default=5000, type=int)
    parser.add_argument("--num-seeds", help="How many episodes you want to train your agent", default=5, type=int)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    env = get_env(args.config, args.render)
    num_actions = env.action_space.n

    # explorer = epsilon_greedy_explorers.ConstantEpsilonGreedyExploration(1.0, num_actions)
    # explorer = epsilon_greedy_explorers.AdaptiveEpsilonGreedyExploration(1.0, 0.01, 0.0001, num_actions)

    feature_extractor = feature_extraction.good_feature_extractor
    # feature_extractor = feature_extraction_dump.test_features_extractor
    num_features = 7 * num_actions
    sarsa_episode_returns_list = []
    sarsa_episode_success_list = []
    np.random.seed(args.seed)
    for seed in range(args.num_seeds):
        explorer = epsilon_greedy_explorers.AdaptiveEpsilonGreedyExploration(1.0, 0.0, 0.001, num_actions)
        agent = semi_gradient_sarsa.SemiGradientSARSA(num_features, num_actions, feature_extractor, 0.01, explorer, 0.99, 10.)
        # agent = semi_gradient_sarsa.NStepSemiGradientSARSA(num_features, num_actions, feature_extractor, 0.1, 0.001, explorer, 0.99, 14, 5.0)
        episode_returns_sarsa = agent_environment.agent_environment_episode_loop(agent, env, args.num_training_episodes)
        episode_successes = [1 if episode_return > 140 else 0 for episode_return in episode_returns_sarsa]
        print("******************************************")
        sarsa_episode_returns_list.append(episode_returns_sarsa)
        sarsa_episode_success_list.append(episode_successes)
    plot_alg_results(sarsa_episode_returns_list, f"jumping_task_config_{args.config}.png", label="Semi-Gradient SARSA")
    plot_alg_results(sarsa_episode_success_list, f"jumping_task_successes_config_{args.config}.png", label="Semi-Gradient SARSA", ylabel="Success rate")


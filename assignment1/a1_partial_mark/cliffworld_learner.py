import gymnasium as gym
import agent_environment
import numpy as np
import epsilon_greedy_explorers
import q_learning
import sarsa
import matplotlib.pyplot as plt

CCID="ddikshan"

# Adapted from ChatGPT
def plot_alg_results(episode_rewards_list, file, label="Algorithm"):

    # Compute running average
    running_avg = np.clip(np.mean(np.array(episode_rewards_list), axis=0), -300, 0)

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
    plt.title(f"({CCID})Episodic Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


def plot_both_algs(episode_rewards_list, episode_rewards_2_list, label1, label2, file):
    # Define a function to calculate the running average

    # Compute running average
    running_avg_1 = np.clip(np.mean(np.array(episode_rewards_list), axis=0), -300, 0)
    running_avg_2 = np.clip(np.mean(np.array(episode_rewards_2_list), axis=0), -300, 0)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the original data
    # plt.plot(episode_rewards, marker='o', linestyle='-', color='b', label='Original Data')

    # Plot the running average
    plt.plot(
        range(0, len(running_avg_1)),
        running_avg_1,
        color='r',
        label=label1,
    )

    plt.plot(
        range(0, len(running_avg_2)),
        running_avg_2,
        color='b',
        label=label2,
    )

    # Adding labels and title
    plt.title(f"({CCID})Episodic Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    explorer = epsilon_greedy_explorers.ConstantEpsilonGreedyExploration(0.1, num_actions)

    q_learning_episode_rewards_list = []
    for seed in range(30):
        agent = q_learning.QLearning(num_states, num_actions, 0.1, explorer, 1.0)
        episode_rewards_q_learning = agent_environment.agent_environment_episode_loop(agent, env, 1000)
        q_learning_episode_rewards_list.append(episode_rewards_q_learning)
    plot_alg_results(q_learning_episode_rewards_list, "q_learning_cliff.png", label="Q-learning")

    sarsa_episode_rewards_list = []
    for seed in range(30):
        agent = sarsa.SARSA(num_states, num_actions, 0.1, explorer, 1.0)
        episode_rewards_sarsa = agent_environment.agent_environment_episode_loop(agent, env, 1000)
        sarsa_episode_rewards_list.append(episode_rewards_sarsa)
    plot_alg_results(sarsa_episode_rewards_list, "sarsa_cliff.png", label="SARSA")

    plot_both_algs(q_learning_episode_rewards_list, sarsa_episode_rewards_list, 'Q-learning', 'SARSA', "sarsa_q_learning_cliff.png")


import gymnasium as gym
import agent_environment
import numpy as np
import semi_gradient_sarsa
import matplotlib.pyplot as plt

from tile_coding import IHT, tiles

CCID="ddikshan"


def plot_alg_results(episode_rewards_list, file, label="Algorithm"):

    # Compute running average
    running_avg = np.clip(np.mean(np.array(episode_rewards_list), axis=0), -300, 0)

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
    plt.ylabel("Return")

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


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    num_actions = env.action_space.n
    num_state_action_features = 4096
    iht = IHT(num_state_action_features)

    def extract_state_action_features(obs, action):
        position = obs[0]
        velocity = obs[1]
        active_tiles = tiles(iht, 8, [8 * position / (0.5+1.2), 8 * velocity / (0.07 + 0.07)], [action]) # see footnote 1 on page 246 of http://incompleteideas.net/book/RLbook2020.pdf
        feature_vector = np.zeros(num_state_action_features)
        feature_vector[active_tiles] = 1.0
        return feature_vector

    explorer = GreedyExploration(num_actions)

    sarsa_episode_rewards_list = []
    for seed in range(30):
        agent = semi_gradient_sarsa.SemiGradientSARSA(num_state_action_features, num_actions, extract_state_action_features, 0.0625, explorer, 1.0)
        episode_rewards_sarsa = agent_environment.agent_environment_episode_loop(agent, env, 1000)
        sarsa_episode_rewards_list.append(episode_rewards_sarsa)
    plot_alg_results(sarsa_episode_rewards_list, "sarsa_mountain_car.png", label="Semi-Gradient SARSA")


import numpy as np
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import jumping_task
from jumping_task.envs import JumpTaskEnv


def compute_q_values(state_action_features, weights):
    """Takes in Q-values and produces epsilon-greedy action probabilities

    where ties are broken evenly.

    Args:
        state_action_features: a numpy array of state-action features
        weights: a numpy array of weights
         
    Returns:
        scalar numpy Q-value
    """
    # Your code here
    q_value = np.dot(state_action_features, weights)
    return q_value
    # end your code

def get_action_values(obs, feature_extractor, weights, num_actions):
    """Applies feature_extractor to observation and produces action values

    Args:
        obs: observation
        feature_extractor: extracts features for a state-action pair
        weights: a numpy array of weights
        num_actions: an integer number of actions
         
    Returns:
        a numpy array of Q-values
    """
    action_values = np.zeros(num_actions)
    for action in range(num_actions):
        action_values[action] = compute_q_values(feature_extractor(obs, action), weights)
    return action_values

class SemiGradientSARSA:
    """Class that implements Linear Semi-gradient SARSA."""

    def __init__(self,
                 num_state_action_features,
                 num_actions,
                 feature_extractor,
                 step_size,
                 explorer,
                 discount,
                 initial_weight_value=0.0):
        self.num_state_action_features = num_state_action_features
        self.num_actions = num_actions
        self.explorer = explorer
        self.step_size = step_size
        self.feature_extractor = feature_extractor
        self.w = np.full(num_state_action_features, initial_weight_value)
        self.discount = discount
        # Your code here: introduce any variables you may need
        self.prev_state = None
        self.prev_action = None
        # End your code here

    def update_q(self, obs, action, reward, next_obs, next_action, terminated):
        # Your code here
        features_ = self.feature_extractor(obs, action)
        features_next = self.feature_extractor(next_obs, next_action)
        td_error = reward + self.discount * compute_q_values(features_next, self.w) * (1 - terminated) - compute_q_values(features_, self.w)
        self.w += self.step_size * td_error * features_
        # End your code here
    

    def act(self, obs) -> int:
        """Returns an integer 
        """
        # Your code here
        self.prev_state = obs
        action_values = get_action_values(obs, self.feature_extractor, self.w, self.num_actions)
        action = self.explorer.select_action(action_values)
        self.prev_action = action
        # End your code here
        return action
        

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        state = self.prev_state
        action = self.prev_action
        next_state = obs
        next_action = self.act(obs)
        self.update_q(state, action, reward, next_state, next_action, terminated) # keep this line
        # End your code here

def compute_epsilon_greedy_action_probs(q_vals, epsilon):
	"""Takes in Q-values and produces epsilon-greedy action probabilities

	where ties are broken evenly.

	Args:
	    q_vals: a numpy array of action values
	    epsilon: epsilon-greedy epsilon in ([0,1])
	     
	Returns:
	    numpy array of action probabilities
	"""
	assert len(q_vals.shape) == 1
	# start your code
	action_probabilities = np.ones_like(q_vals) * epsilon / len(q_vals)
	best_actions = np.where(q_vals == np.max(q_vals))[0]
	action_probabilities[best_actions] += (1 - epsilon) / len(best_actions)
	# end your code
	assert action_probabilities.shape == q_vals.shape
	return action_probabilities	


class AdaptiveEpsilonGreedyExploration:
    """Epsilon-greedy with adaptive epsilon
    
    Args:
      initial_epsilon: indicating the initial value of epsilon
	  min_epsilon: indicating the minimum value of epsilon
	  decay_rate: indicating the decay rate of epsilon
	  num_actions: indicating the number of actions
    """
    def __init__(self, initial_epsilon, min_epsilon, decay_rate, num_actions):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.num_actions = num_actions
        self.episode_count = 0
        self.successful_episodes = 0
        
    def select_action(self, action_values) -> int:
        self.epsilon = max(self.min_epsilon, self.epsilon * (1 - self.decay_rate))
        
        action_probs = compute_epsilon_greedy_action_probs(action_values, self.epsilon)
        return np.random.choice(len(action_probs), p=action_probs)

class ConstantEpsilonGreedyExploration:
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, epsilon, num_actions):
        self.epsilon = epsilon
        self.num_actions = num_actions

    def select_action(self, action_values) -> int:
        action_probs = compute_epsilon_greedy_action_probs(action_values, self.epsilon)
        return np.random.choice(len(action_probs), p=action_probs)
    
def normalize_height(feature, max_height):
    return feature / max_height

def normalize_width(feature, max_width):
    return feature / max_width

def good_features_extractor(obs, action):
    num_actions = 2
    num_state_features = 9
    good_features = np.zeros(num_state_features * num_actions)

    # # visualize the observation
    # RESET = "\033[0m"       # Reset to default
    # GRAY = "\033[90m"       # Gray for 0.0
    # GREEN = "\033[92m"      # Green for 0.5
    # RED = "\033[91m"        # Red for 1.0

    # for row in obs[::-1]:
    #     colored_row = [
    #         f"{GRAY}0.0{RESET}" if num == 0.0 else
    #         f"{GREEN}0.5{RESET}" if num == 0.5 else
    #         f"{RED}1.0{RESET}"
    #         for num in row
    #     ]
    #     print(" ".join(colored_row))

    env = np.array(obs[::-1])
    height, width = env.shape
    
    floor_height = None

    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i

    agent_positions = np.where(env[:, :-1] == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1])
                      if y not in [0, height-1] and x not in [0, width-1] and y < floor_height]
    
    if agent_positions:
        agent_min_x = min(y for y, _ in agent_positions)
        agent_max_x = max(y for y, _ in agent_positions)
        agent_min_y = min(x for _, x in agent_positions)
        agent_max_y = max(x for _, x in agent_positions)

    # height and width of the agent
    # is this correct?
    agent_height = agent_max_x - agent_min_x + 1 + 1    # +1 to include the floor
    agent_width = agent_max_y - agent_min_y + 1

    # normalized agent height and width
    agent_height = normalize_height(agent_height, height)
    agent_width = normalize_width(agent_width, width)
    
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1]) if y < floor_height]

    if obstacle_positions:
        obstacle_min_y = min(y for y, _ in obstacle_positions)
        obstacle_max_y = max(y for y, _ in obstacle_positions)
        obstacle_min_x = min(x for _, x in obstacle_positions)
        obstacle_max_x = max(x for _, x in obstacle_positions)

    # height and width of the obstacle
    obstacle_height = obstacle_max_y - obstacle_min_y + 1 + 1   # +1 to include the floor
    obstacle_width = obstacle_max_x - obstacle_min_x + 1

    # normalized obstacle height and width
    obstacle_height = normalize_height(obstacle_height, height)
    obstacle_width = normalize_width(obstacle_width, width)

    
    
    return good_features

def agent_environment_episode_loop(agent, env, num_episodes):
    episode_returns = []
    for episode in range(num_episodes):
        observation, info = env.reset()
        # start your code
        episode_return = 0
        done = False
        while not done:
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.process_transition(next_observation, reward, terminated, truncated)
            observation = next_observation
            episode_return += reward
            done = terminated or truncated
        print(f"Episode {episode} return: {episode_return}")    # remove this line
        episode_returns.append(episode_return)
        # end your code
    return episode_returns


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
    plt.title(f"(Episodic Returns")
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

    explorer = AdaptiveEpsilonGreedyExploration(0.5, 0.05, 0.001, num_actions)

    feature_extractor = good_features_extractor
    num_features = 9 * num_actions
    sarsa_episode_returns_list = []
    sarsa_episode_success_list = []
    np.random.seed(args.seed)
    for seed in range(args.num_seeds):
        agent = SemiGradientSARSA(num_features, num_actions, feature_extractor, 0.01, explorer, 0.99, 10.)
        episode_returns_sarsa = agent_environment_episode_loop(agent, env, args.num_training_episodes)
        episode_successes = [1 if episode_return > 140 else 0 for episode_return in episode_returns_sarsa]
        sarsa_episode_returns_list.append(episode_returns_sarsa)
        sarsa_episode_success_list.append(episode_successes)
    plot_alg_results(sarsa_episode_returns_list, f"jumping_task_config_{args.config}.png", label="Semi-Gradient SARSA")
    plot_alg_results(sarsa_episode_success_list, f"jumping_task_successes_config_{args.config}.png", label="Semi-Gradient SARSA", ylabel="Success rate")


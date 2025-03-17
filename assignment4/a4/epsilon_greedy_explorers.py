import numpy as np

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
    

class LinearDecayEpsilonGreedyExploration:
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, start_epsilon, end_epsilon, decay_steps, num_actions):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        assert start_epsilon >= end_epsilon >= 0
        self.epsilon = start_epsilon
        self.decay_steps = decay_steps
        self.num_actions = num_actions
        self.steps = 0

    def select_action(self, action_values) -> int:
        epsilon_decay_step_size = (self.start_epsilon - self.end_epsilon) / self.decay_steps
        epsilon = max(self.start_epsilon - self.steps * epsilon_decay_step_size, self.end_epsilon)
        action_probs = compute_epsilon_greedy_action_probs(action_values, epsilon)
        self.steps += 1
        return np.random.choice(len(action_probs), p=action_probs)
    
class GaussianNoiseExplorer:
    """Adds Gaussian noise to continuous actions for exploration.
    
    Args:
        std_dev: Standard deviation of the Gaussian noise
        max_action: Maximum absolute value of action
    """
    def __init__(self, std_dev, max_action):
        self.std_dev = std_dev
        self.max_action = max_action
    
    def select_action(self, action_values):
        """Adds Gaussian noise to the action values"""
        noise = np.random.normal(0, self.std_dev, size=action_values.shape)
        noisy_action = action_values + noise
        return np.clip(noisy_action, -self.max_action, self.max_action)
    
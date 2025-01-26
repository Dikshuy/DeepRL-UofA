import numpy as np

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
    return 5 # replace this line
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
        # End your code here

    def update_q(self, obs, action, reward, next_obs, next_action, terminated):
        # Your code here
        pass # replace this line
        # End your code here
    

    def act(self, obs) -> int:
        """Returns an integer 
        """
        # Your code here
        action = None # replace this line
        # End your code here
        return action
        

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        # Your code here
        state = None # replace this line
        action = None # replace this line
        next_state = None # replace this line
        next_action = None # replace this line
        self.update_q(state, action, reward, next_state, next_action, terminated) # keep this line
        # End your code here

import numpy as np
from collections import deque

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

class NStepSemiGradientSARSA:
    """Class that implements N-step Linear Semi-gradient SARSA."""
    
    def __init__(self,
                 num_state_action_features,
                 num_actions,
                 feature_extractor,
                 step_size,
                 explorer,
                 discount,
                 n_steps,
                 initial_weight_value=0.0):
        self.num_state_action_features = num_state_action_features
        self.num_actions = num_actions
        self.explorer = explorer
        self.step_size = step_size
        self.feature_extractor = feature_extractor
        self.w = np.full(num_state_action_features, initial_weight_value)
        self.discount = discount
        self.n_steps = n_steps
        
        self.buffer = deque(maxlen=n_steps)
        self.total_steps = 0
        
        self.prev_state = None
        self.prev_action = None
    
    def compute_n_step_return(self, terminated=False):
        """Computes the n-step return from the buffer
        """
        if len(self.buffer) < self.n_steps and not terminated:
            return None
            
        n = len(self.buffer) if terminated else self.n_steps
        G = 0
        for i in range(n):
            G += (self.discount ** i) * self.buffer[i][2]
            
        if not terminated and len(self.buffer) >= self.n_steps:
            last_state = self.buffer[-1][3]
            last_action = self.buffer[-1][4]
            last_features = self.feature_extractor(last_state, last_action)
            q_value_n = compute_q_values(last_features, self.w)
            G += (self.discount ** n) * q_value_n
            
        return G
    
    def update_weights(self, n_step_return, initial_features):
        """Updates the weights of the agent
        """
        current_q = compute_q_values(initial_features, self.w)
        td_error = n_step_return - current_q
        step_size = self.step_size
        self.w += step_size * td_error * initial_features
        self.total_steps += 1
    
    def act(self, obs) -> int:
        """Returns an integer 
        """
        self.prev_state = obs
        action_values = get_action_values(obs, self.feature_extractor, self.w, self.num_actions)
        action = self.explorer.select_action(action_values)
        self.prev_action = action
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
        
        self.buffer.append((state, action, reward, next_state, next_action))
        
        n_step_return = self.compute_n_step_return(terminated)

        if n_step_return is not None:
            initial_features = self.feature_extractor(self.buffer[0][0], self.buffer[0][1])
            self.update_weights(n_step_return, initial_features)
            
            if not terminated:
                self.buffer.popleft()
        
        if terminated:
            self.buffer.clear()
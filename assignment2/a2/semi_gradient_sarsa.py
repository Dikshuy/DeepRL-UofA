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
    """Class that implements n-step Semi-gradient SARSA."""

    def __init__(self,
                 num_state_action_features,
                 num_actions,
                 feature_extractor,
                 step_size,
                 explorer,
                 discount,
                 n_step,
                 initial_weight_value=0.0):
        self.num_state_action_features = num_state_action_features
        self.num_actions = num_actions
        self.explorer = explorer
        self.step_size = step_size
        self.feature_extractor = feature_extractor
        self.w = np.full(num_state_action_features, initial_weight_value)
        self.discount = discount
        self.n_step = n_step

        # Buffer to store the last n steps of experience
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def update_q(self, state, action, reward, next_state, next_action, terminated):
        """Update the Q-values using n-step semi-gradient SARSA."""
        # Append the current experience to the buffer
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

        # If the buffer has enough experience, perform an update
        if len(self.state_buffer) >= self.n_step:
            # Compute the n-step return
            n_step_return = 0
            for i in range(self.n_step):
                n_step_return += (self.discount ** i) * self.reward_buffer[i]

            # Add the discounted value of the n-th next state
            if not terminated:
                features_next = self.feature_extractor(next_state, next_action)
                n_step_return += (self.discount ** self.n_step) * np.dot(features_next, self.w)

            # Get the features for the state-action pair at the start of the n-step window
            features = self.feature_extractor(self.state_buffer[0], self.action_buffer[0])

            # Compute the TD error
            td_error = n_step_return - np.dot(features, self.w)

            # Update the weights
            self.w += self.step_size * td_error * features

            # Remove the oldest experience from the buffer
            self.state_buffer.pop(0)
            self.action_buffer.pop(0)
            self.reward_buffer.pop(0)

    def act(self, obs) -> int:
        """Returns an integer action."""
        action_values = get_action_values(obs, self.feature_extractor, self.w, self.num_actions)
        action = self.explorer.select_action(action_values)
        return action

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly."""
        state = self.state_buffer[-1] if self.state_buffer else None
        action = self.action_buffer[-1] if self.action_buffer else None
        next_state = obs
        next_action = self.act(obs)

        if state is not None and action is not None:
            self.update_q(state, action, reward, next_state, next_action, terminated)

        # Append the current experience to the buffer
        self.state_buffer.append(obs)
        self.action_buffer.append(next_action)
        self.reward_buffer.append(reward)

        # If the episode terminates, clear the buffer
        if terminated or truncated:
            self.state_buffer.clear()
            self.action_buffer.clear()
            self.reward_buffer.clear()

import numpy as np
from typing import List, Tuple

class MonteCarloSemiGradientSARSA:
    """Class that implements Monte Carlo Semi-gradient SARSA."""
    
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
        
        # Initialize episode memory
        self.current_episode: List[Tuple[int, int, float]] = []  # [(state, action, reward)]
        self.prev_state = None
        self.prev_action = None

    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns for each step in the episode."""
        G = np.zeros(len(rewards))
        G[-1] = rewards[-1]
        for t in range(len(rewards)-2, -1, -1):
            G[t] = rewards[t] + self.discount * G[t+1]
        return G

    def update_episode(self, episode_states, episode_actions, episode_returns):
        """Update weights using the complete episode."""
        for t in range(len(episode_states)):
            state = episode_states[t]
            action = episode_actions[t]
            G_t = episode_returns[t]
            
            # Get features for the state-action pair
            features = self.feature_extractor(state, action)
            
            # Current Q-value estimate
            current_q = compute_q_values(features, self.w)
            
            # Monte Carlo update
            # Using (G - Q) as the error term instead of TD error
            error = G_t - current_q
            self.w += self.step_size * error * features

    def act(self, obs) -> int:
        """Returns an action based on the current policy."""
        self.prev_state = obs
        action_values = get_action_values(obs, self.feature_extractor, self.w, self.num_actions)
        action = self.explorer.select_action(action_values)
        self.prev_action = action
        return action

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Store transition in current episode and update if episode is complete."""
        # Store the transition
        if self.prev_state is not None:
            self.current_episode.append((self.prev_state, self.prev_action, reward))

        # If episode is complete
        if terminated or truncated:
            # Extract states, actions, and rewards from the episode
            states, actions, rewards = zip(*self.current_episode)
            
            # Compute returns
            returns = self.compute_returns(list(rewards))
            
            # Update weights using the complete episode
            self.update_episode(states, actions, returns)
            
            # Clear episode memory
            self.current_episode = []
            self.prev_state = None
            self.prev_action = None
        else:
            # Select next action using current policy
            self.act(obs)
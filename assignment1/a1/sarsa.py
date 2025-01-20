import numpy as np


class SARSA:
    """Class that implements SARSA."""

    def __init__(self,
                 num_states,
                 num_actions,
                 step_size,
                 explorer,
                 discount=0.99,):
        self.explorer = explorer
        self.step_size = step_size
        self.q = np.zeros((num_states, num_actions))
        self.discount = discount
        # Your code here: introduce any variables you may need
        self.prev_state = None
        self.prev_action = None
        # End your code here


    def update_q(self, obs, action, reward, next_obs, next_action, terminated):
        # Your code here
        self.q[obs, action] += self.step_size * (reward + self.discount * (1 - terminated) * self.q[next_obs, next_action] - self.q[obs, action])
        # End your code here
    

    def act(self, obs: int) -> int:
        """Returns an integer 
        """
        # Your code here
        self.prev_state = obs
        action = self.explorer.select_action(self.q[obs])
        self.prev_action = action
        # End your code here
        return action
        

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        # Your code here
        state = self.prev_state
        action = self.prev_action
        next_state = obs
        next_action = self.act(next_state)
        self.update_q(state, action, reward, next_state, next_action, terminated) # keep this line
        # End your code here

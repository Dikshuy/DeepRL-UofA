import collections
import numpy as np

class ReplayBuffer:

    def __init__(self, buffer_size, discount=0.99, n_step=1):
        self.buffer = collections.deque([], maxlen=buffer_size)
        self.discount = discount
        self.n_step = n_step

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, terminated, truncated):
        transition = {'state': state,
                      'action': action,
                      'reward': reward,
                      'next_state': next_state,
                      'discount': self.discount,
                      'terminated': terminated,
                      'truncated': truncated}
        self.buffer.append(transition)

    def create_multistep_transition(self, index):
        # your code here
        if self.n_step == 1:
            return self.buffer[index]
        
        transition = self.buffer[index]
        state = transition['state']
        action = transition['action']
        n_step_reward = transition['reward']
        next_state = transition['next_state']
        terminated = transition['terminated']
        truncated = transition['truncated']
        discount = self.discount

        for i in range(1, self.n_step):
            next_index = (index + i) % len(self.buffer)
            if next_index <= index or terminated or truncated:
                break
            next_transition = self.buffer[next_index]
            n_step_reward += discount * next_transition['reward']
            discount *= self.discount
            next_state = next_transition['next_state']
            terminated = next_transition['terminated']
            truncated = next_transition['truncated']

        return {'state': state,
                'action': action,
                'reward': n_step_reward,
                'next_state': next_state,
                'discount': discount,
                'terminated': terminated,
                'truncated': truncated}
        # end your code

    def sample(self, n_transitions):
        assert len(self.buffer) >= n_transitions
        batch_indices = np.random.choice(len(self.buffer), size=n_transitions, replace=False)
        batch = [self.create_multistep_transition(index) for index in batch_indices]
        return batch

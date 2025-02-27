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
        transition = self.buffer[index]
        state = transition['state']
        action = transition['action']
        reward = transition['reward']
        next_state = transition['next_state']
        discount = transition['discount']
        terminated = transition['terminated']
        truncated = transition['truncated']
        for i in range(1, self.n_step):
            if i >= len(self.buffer):
                break
            transition = self.buffer[index + i]
            reward += transition['reward'] * (self.discount ** i)
            next_state = transition['next_state']
            discount *= transition['discount']
            if transition['terminated']:
                terminated = True
                break
            if transition['truncated']:
                truncated = True
                break
        return {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'discount': discount, 'terminated': terminated, 'truncated': truncated}
        # end your code

    def sample(self, n_transitions):
        assert len(self.buffer) >= n_transitions
        batch_indices = np.random.choice(len(self.buffer), size=n_transitions, replace=False)
        batch = [self.create_multistep_transition(index) for index in batch_indices]
        return batch

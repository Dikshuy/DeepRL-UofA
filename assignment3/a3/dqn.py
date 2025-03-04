import numpy as np
import torch
import copy
import collections


def target_network_refresh(q_network):
    target_network = copy.deepcopy(q_network)
    return target_network


class DQN:
    """Class that implements Deep Q-networks."""

    def __init__(self,
                 q_network,
                 optimizer,
                 replay_buffer,
                 explorer,
                 discount,
                 gradient_updates_per_target_refresh,
                 gradient_update_frequency=1,
                 input_preprocessor= lambda x: x,
                 minibatch_size=32,
                 min_replay_size_before_updates=32,
                 track_statistics=False,
                 reward_phi=lambda reward: reward):
        self.q_network = q_network
        self.optimizer = optimizer
        self.target_network = target_network_refresh(self.q_network)
        self.replay_buffer = replay_buffer
        self.explorer = explorer
        self.discount = discount
        self.gradient_updates_per_target_refresh = gradient_updates_per_target_refresh
        self.gradient_update_frequency = gradient_update_frequency
        self.input_preprocessor = input_preprocessor
        self.minibatch_size = minibatch_size
        self.min_replay_size_before_updates = min_replay_size_before_updates
        self.track_statistics = track_statistics
        self.reward_phi = reward_phi
        # your code here
        self.total_steps = 0
        self.last_obs = None
        self.last_action = None
        self.update_rate = 0
        # end your code

    def act(self, obs) -> int:
        """Returns an integer 
        """
        # Your code here
        self.last_obs = obs
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        action = self.explorer.select_action(q_values.detach().numpy())
        self.last_action = action
        # End your code here
        return action

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        # your code here
        with torch.no_grad():
            next_q_values = self.target_network(batched_next_states).max(dim=1)[0]
            targets = batched_rewards + batched_discounts * next_q_values * (1 - batch_terminated)
        return targets
        # End your code here

    def gradient_update(self):
        minibatch = self.replay_buffer.sample(self.minibatch_size)
        # your code here
        batched_states = torch.tensor(np.array([transition['state'] for transition in minibatch]), dtype=torch.float32)
        batched_actions = torch.tensor(np.array([transition['action'] for transition in minibatch]), dtype=torch.int64) 
        batched_rewards = torch.tensor(np.array([transition['reward'] for transition in minibatch]), dtype=torch.float32)
        batched_next_states = torch.tensor(np.array([transition['next_state'] for transition in minibatch]), dtype=torch.float32)
        batched_discounts = torch.tensor(np.array([transition['discount'] for transition in minibatch]), dtype=torch.float32)
        batch_terminated = torch.tensor(np.array([transition['terminated'] for transition in minibatch]), dtype=torch.float32)

        q_values = self.q_network(batched_states)
        q_values = q_values.gather(1, batched_actions.unsqueeze(1)).squeeze(1)
        targets = self.compute_targets(batched_rewards, batched_next_states, batched_discounts, batch_terminated)
        
        loss = torch.nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_rate += 1
        if self.update_rate % self.gradient_updates_per_target_refresh == 0:
            self.target_network = target_network_refresh(self.q_network)
        # End your code here


    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        reward = self.reward_phi(reward)
        # append transition to buffer
        # do gradient updates if necessary
        # refresh target networks if needed, etc.
        
        # Your code here
        if self.last_obs is not None and self.last_action is not None:
            self.replay_buffer.append(self.last_obs, self.last_action, reward, obs, terminated, truncated)
            self.total_steps += 1
            if self.total_steps >= self.min_replay_size_before_updates and self.total_steps % self.gradient_update_frequency == 0:
                self.gradient_update()
        # End your code here

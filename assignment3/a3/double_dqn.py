import dqn
import torch

class DoubleDQN(dqn.DQN):

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        # begin your code
        with torch.no_grad():
            q_values = self.target_network(batched_next_states)
            q_values = q_values.max(dim=1)[0]
            q_values = q_values * batched_discounts
            q_values = q_values * (1 - batch_terminated)
            q_values = q_values + batched_rewards
        return q_values
        # end your code
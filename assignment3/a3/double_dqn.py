import dqn
import torch

class DoubleDQN(dqn.DQN):

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        # begin your code
        with torch.no_grad():
            next_q_values = self.q_network(batched_next_states)
            best_actions = next_q_values.argmax(dim=1)
            next_q_values_target = self.target_network(batched_next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets = batched_rewards + batched_discounts * next_q_values_target * (1 - batch_terminated)            
        return targets
        # end your code
import numpy as np
import torch
import copy
import collections


def soft_update(source, target, tau):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

class TD3:
    """Class that implements TD3 algorithm."""
    def __init__(
        self,
        actor,
        actor_optimizer,
        critic,
        critic_optimizer,
        replay_buffer,
        explorer,
        discount,
        gradient_updates_per_target_refresh,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_update_frequency=2,
        gradient_update_frequency=1,
        input_preprocessor=lambda x: x,
        minibatch_size=32,
        min_replay_size_before_updates=32,
        tau = 0.005,
        track_statistics=False,
        reward_phi=lambda reward: reward,
        max_action=1.0
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.actor_target = copy.deepcopy(actor)

        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.critic_target = copy.deepcopy(critic)

        self.replay_buffer = replay_buffer
        self.explorer = explorer
        self.discount = discount
        self.gradient_updates_per_target_refresh = gradient_updates_per_target_refresh
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_update_frequency = policy_update_frequency
        self.gradient_update_frequency = gradient_update_frequency
        self.input_preprocessor = input_preprocessor
        self.minibatch_size = minibatch_size
        self.min_replay_size_before_updates = min_replay_size_before_updates
        self.tau = tau
        self.track_statistics = track_statistics
        self.reward_phi = reward_phi
        self.max_action = max_action
 
        self.total_steps = 0
        self.gradient_updates = 0
        self.last_obs = None
        self.last_action = None
        self.episode_loss = 0

    def act(self, obs) -> int:
        """Returns an action and its q-value 
        """
        obs = self.input_preprocessor(obs)
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action_vals = self.actor(obs).detach().numpy()
            action = self.explorer.select_action(action_vals)
            action = np.clip(action, -self.max_action, self.max_action)
            q_value = self.critic.Q1(obs.unsqueeze(0), torch.tensor(action, dtype=torch.float32).unsqueeze(0)).item()
        self.last_obs = obs
        self.last_action = action
        return action, q_value

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        with torch.no_grad():
            noise = (torch.rand_like(batched_next_states[:, :self.actor_target(batched_next_states).size(1)])*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(batched_next_states) + noise).clamp(-self.max_action, self.max_action)
            
            target_q1, target_q2 = self.critic_target(batched_next_states, next_actions)
            target_q = torch.min(target_q1, target_q2).squeeze(-1)

            targets = batched_rewards + batched_discounts * target_q * (1 - batch_terminated)
        return targets

    def gradient_update(self):
        minibatch = self.replay_buffer.sample(self.minibatch_size)
        batched_states = torch.stack([transition['state'] for transition in minibatch])
        batched_actions = torch.tensor([transition['action'] for transition in minibatch]) 
        batched_rewards = torch.tensor([transition['reward'] for transition in minibatch])
        batched_next_states = torch.stack([transition['next_state'] for transition in minibatch])
        batched_discounts = torch.tensor([transition['discount'] for transition in minibatch])
        batch_terminated = torch.tensor([transition['terminated'] for transition in minibatch])

        targets = self.compute_targets(batched_rewards, batched_next_states, batched_discounts, batch_terminated).float()

        current_q1, current_q2 = self.critic(batched_states, batched_actions)
        current_q1 = current_q1.squeeze(-1)
        current_q2 = current_q2.squeeze(-1)

        critic_loss = torch.nn.functional.mse_loss(current_q1, targets) + torch.nn.functional.mse_loss(current_q2, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.gradient_updates % self.policy_update_frequency == 0:
            actor_loss = -self.critic.Q1(batched_states, self.actor(batched_states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.gradient_updates % self.gradient_updates_per_target_refresh == 0:
                soft_update(self.actor, self.actor_target, self.tau)
                soft_update(self.critic, self.critic_target, self.tau)

        self.gradient_updates += 1
        return critic_loss.item(), actor_loss.item()

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        reward = self.reward_phi(reward)

        if self.last_obs is not None and self.last_action is not None:
            self.replay_buffer.append(self.last_obs, self.last_action, reward, self.input_preprocessor(obs), int(terminated), int(truncated))
            self.total_steps += 1
            if self.total_steps >= self.min_replay_size_before_updates and self.total_steps % self.gradient_update_frequency == 0:
                loss = self.gradient_update()
                self.episode_loss += loss
            
            if terminated or truncated:
                episode_loss = self.episode_loss
                self.episode_loss = 0
                return episode_loss
            else:
                return 0

import torch
import numpy as np

def agent_environment_episode_loop(agent, env, num_episodes, debug=False, track_q=False):
    episode_returns = []
    mean_q_predictions = [] # the average Q-value for all state-action pairs visited in the episode
    for episode in range(num_episodes):
        if track_q:
            episode_q_values = []
        observation, info = env.reset()
        # start your code
        done = False
        episode_return = 0
        episode_loss = 0
        while not done:
            action, q_values = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            loss = agent.process_transition(next_observation, reward, terminated, truncated)
            episode_loss += loss
            observation = next_observation
            episode_return += reward
            done = terminated or truncated
            if track_q:
                episode_q_values.append(q_values)
        episode_returns.append(episode_return)
        if track_q:
            mean_q_predictions.append(np.mean(episode_q_values))
            if debug:
                print(f"Episode {episode} - Return: {episode_return} - TD-Error: {episode_loss:.8f}")
        if debug:
            print(f"Episode {episode} - Return: {episode_return}")
        # end your code
    if track_q:
        return episode_returns, mean_q_predictions
    else:
        return episode_returns, None

def agent_environment_step_loop(agent, env, num_steps, debug=False, track_q=False):
    observation, info = env.reset()
    episode_returns = []
    episodes_timesteps = []
    current_timestep = 0
    episode_return = 0
    episode_loss = 0
    mean_q_predictions = [] # the average Q-value for all state-action pairs visited in the episode
    for step in range(num_steps):
        # start your code
        action, q_values = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        loss = agent.process_transition(next_observation, reward, terminated, truncated)
        episode_loss += loss
        observation = next_observation
        episode_return += reward
        current_timestep = step + 1
        done = terminated or truncated
        if track_q:
            mean_q_predictions.append(q_values)
        if done:
            episode_returns.append(episode_return)
            episodes_timesteps.append(current_timestep)
            if debug:
                print(f"Step: {current_timestep} - Return: {episode_return}")
            episode_return = 0
            episode_loss = 0
            observation, info = env.reset()
        # end your code
    if track_q:   
        return episode_returns, episodes_timesteps, mean_q_predictions
    else:
        return episode_returns, episodes_timesteps, None

import time
def agent_environment_episode_loop(agent, env, num_episodes):
    episode_returns = []
    for episode in range(num_episodes):
        observation, info = env.reset()
        # start your code
        episode_return = 0
        done = False
        i = 0
        while not done:
            i += 1
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.process_transition(next_observation, reward, terminated, truncated)
            # print(f"---step:{i}--{action}---")
            # time.sleep(1)
            observation = next_observation
            episode_return += reward
            done = terminated or truncated
        print(f"Episode {episode} return: {episode_return}")    # remove this line
        episode_returns.append(episode_return)
        # end your code
    return episode_returns

def agent_environment_step_loop(agent, env, num_steps):
    observation, info = env.reset()
    episode_returns = []
    episode_return = 0
    for _ in range(num_steps):
        # start your code
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.process_transition(next_observation, reward, terminated, truncated)
        observation = next_observation
        episode_return += reward
        done = terminated or truncated
        if done:
            episode_returns.append(episode_return)
            episode_return = 0
            observation, info = env.reset()
        # end your code
    return episode_returns

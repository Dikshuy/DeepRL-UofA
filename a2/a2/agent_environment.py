

def agent_environment_episode_loop(agent, env, num_episodes):
    episode_returns = []
    for episode in range(num_episodes):
        observation, info = env.reset()
        # start your code
        pass
        # end your code
    return episode_returns

def agent_environment_step_loop(agent, env, num_steps):
    observation, info = env.reset()
    episode_returns = []
    episode_return = 0
    for _ in range(num_steps):
        # start your code
        pass
        # end your code
    return episode_returns

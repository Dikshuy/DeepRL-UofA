import numpy as np

def terrible_feature_extractor(obs, action):
    num_actions = 2
    num_state_features = 625
    terrible_features = np.zeros(num_state_features * num_actions)
    terrible_features[num_state_features * action: num_state_features * (action+1)] = obs[0:25, 0:25].flatten()
    return terrible_features

def good_features(obs, action):
    num_actions = 2
    num_state_features = 625
    features = np.zeros(num_state_features * num_actions)

    '''
    # visualize the observation
    RESET = "\033[0m"       # Reset to default
    GRAY = "\033[90m"       # Gray for 0.0
    GREEN = "\033[92m"      # Green for 0.5
    RED = "\033[91m"        # Red for 1.0

    for row in obs[::-1]:
        colored_row = [
            f"{GRAY}0.0{RESET}" if num == 0.0 else
            f"{GREEN}0.5{RESET}" if num == 0.5 else
            f"{RED}1.0{RESET}"
            for num in row
        ]
        print(" ".join(colored_row))
    '''

    env = np.array(obs[::-1])
    height, width = env.shape
    
    floor_height = None

    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i

    print(floor_height)

    agent_positions = np.where(env[:, :-1] == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1])
                      if y not in [0, height-1] and x not in [0, width-1] and y < floor_height]
    
    if agent_positions:
        agent_min_y = min(y for y, _ in agent_positions)
        agent_max_y = max(y for y, _ in agent_positions)
        agent_min_x = min(x for _, x in agent_positions)
        agent_max_x = max(x for _, x in agent_positions)

    agent_height = agent_max_y - agent_min_y + 1 + 1 # +1 to include the floor
    agent_width = agent_max_x - agent_min_x + 1
    
    obstacle_positions = np.where(env == 0.5)
    obstacle_height = len(set(obstacle_positions[0])) + 1 # +1 to include the floor
    obstacle_width = len(set(obstacle_positions[1]))

    print(agent_height, agent_width, obstacle_height, obstacle_width)

    exit()

    # distance between agent and obstacle
    # width of agent and obstacle
    # height of agent and obstacle
    # number of obstacle
    
    pass

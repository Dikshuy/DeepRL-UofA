import numpy as np

def terrible_feature_extractor(obs, action):
    num_actions = 2
    num_state_features = 625
    terrible_features = np.zeros(num_state_features * num_actions)
    terrible_features[num_state_features * action: num_state_features * (action+1)] = obs[0:25, 0:25].flatten()
    return terrible_features

def draw_env(obs):
    RESET = "\033[0m"       # Reset to default
    GRAY = "\033[90m"       # Gray for 0.0
    GREEN = "\033[92m"      # Green for 0.5
    RED = "\033[91m"        # Red for 1.0

    for row in obs:
        colored_row = [
            f"{GRAY}0.0{RESET}" if num == 0.0 else
            f"{GREEN}0.5{RESET}" if num == 0.5 else
            f"{RED}1.0{RESET}" for num in row
        ]
        print(" ".join(colored_row))

def good_feature_extractor(obs, action):
    num_actions = 2
    num_state_features = 9

    features = np.zeros(num_state_features * num_actions)

    env = obs[::-1]
    env = env[1:-1, 1:-1]

    height, width = env.shape

    floor_height = None
    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break

    agent_positions = np.where(env == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1]) if y < floor_height]
    
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1]) if y < floor_height]
    
    # draw_env(env)

    agent_min_y = min(y for y, _ in agent_positions)
    agent_max_y = max(y for y, _ in agent_positions) + 1
    agent_min_x = min(x for _, x in agent_positions)
    agent_max_x = max(x for _, x in agent_positions) + 1

    agent_height = agent_max_y - agent_min_y
    agent_width = agent_max_x - agent_min_x

    obstacle_min_y = min(y for y, _ in obstacle_positions)
    obstacle_max_y = max(y for y, _ in obstacle_positions) + 1
    obstacle_min_x = min(x for _, x in obstacle_positions)
    obstacle_max_x = max(x for _, x in obstacle_positions) + 1

    obstacle_height = obstacle_max_y - obstacle_min_y
    obstacle_width = obstacle_max_x - obstacle_min_x

    distance_from_goal = width - agent_max_x

    distance_from_obstacle = obstacle_min_x - agent_max_x

    jump_height = floor_height - agent_max_y

    in_air = float(jump_height > 0)

    optimal_jump_point = float(distance_from_obstacle == obstacle_height)

    for a in range(num_actions):
        idx = a * num_state_features

        features[idx] = agent_height / height
        features[idx + 1] = agent_max_y / height
        features[idx + 2] = distance_from_goal / width
        features[idx + 3] = distance_from_obstacle / width
        features[idx + 4] = jump_height / height
        features[idx + 5] = in_air
        features[idx + 6] = optimal_jump_point
        features[idx + 7] = 1.0 if a == action else 0.0

        if action == 0:
            features[idx + 8] = 1.0 if distance_from_obstacle > obstacle_height else 0.0
        else:
            features[idx + 8] = 1.0 if distance_from_obstacle == obstacle_height else 0.0
    

    return features

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
    num_state_features = 11
    features = np.zeros(num_state_features * num_actions)

    env = obs[::-1]
    env = env[1:-1, 1:-1]
    height, width = env.shape

    floor_height = None
    for i in range(1, height - 1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break

    agent_positions = np.where(env == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1]) if y < floor_height]
    agent_min_y = min(y for y, _ in agent_positions)
    agent_max_y = max(y for y, _ in agent_positions) + 1
    agent_min_x = min(x for _, x in agent_positions)
    agent_max_x = max(x for _, x in agent_positions) + 1
    agent_height = agent_max_y - agent_min_y
    agent_width = agent_max_x - agent_min_x

    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1]) if y < floor_height]

    obstacle_positions_sorted = sorted(obstacle_positions, key=lambda pos: pos[1])

    obstacle1_min_y = min(y for y, _ in obstacle_positions_sorted)
    obstacle1_max_y = max(y for y, _ in obstacle_positions_sorted) + 1
    obstacle1_min_x = min(x for _, x in obstacle_positions_sorted)
    obstacle1_max_x = max(x for _, x in obstacle_positions_sorted) + 1
    obstacle1_height = obstacle1_max_y - obstacle1_min_y
    obstacle1_width = obstacle1_max_x - obstacle1_min_x
    distance_from_obstacle1 = obstacle1_min_x - agent_max_x

    obstacle2_min_y = min(y for y, _ in obstacle_positions_sorted[1:])
    obstacle2_max_y = max(y for y, _ in obstacle_positions_sorted[1:]) + 1
    obstacle2_min_x = min(x for _, x in obstacle_positions_sorted[1:])
    obstacle2_max_x = max(x for _, x in obstacle_positions_sorted[1:]) + 1
    obstacle2_height = obstacle2_max_y - obstacle2_min_y
    obstacle2_width = obstacle2_max_x - obstacle2_min_x
    distance_from_obstacle2 = obstacle2_min_x - agent_max_x

    # Compute additional features
    distance_from_goal = width - agent_max_x
    jump_height = floor_height - agent_max_y
    in_air = float(jump_height > 0)

    # Optimal jump points for both obstacles
    optimal_jump_point1 = float(distance_from_obstacle1 == obstacle1_height)
    optimal_jump_point2 = float(distance_from_obstacle2 == obstacle2_height)

    # Populate the feature vector
    for a in range(num_actions):
        idx = a * num_state_features
        features[idx] = agent_height / height  # Normalized agent height
        features[idx + 1] = distance_from_goal / width  # Normalized distance to goal
        features[idx + 2] = distance_from_obstacle1 / width  # Normalized distance to obstacle 1
        features[idx + 3] = distance_from_obstacle2 / width  # Normalized distance to obstacle 2
        features[idx + 4] = jump_height / height  # Normalized jump height
        features[idx + 5] = in_air  # Whether the agent is in the air
        features[idx + 6] = optimal_jump_point1  # Optimal jump point for obstacle 1
        features[idx + 7] = optimal_jump_point2  # Optimal jump point for obstacle 2
        features[idx + 8] = 1.0 if a == action else 0.0  # Action indicator

        # Additional logic for action-specific features
        if action == 0:
            features[idx + 9] = 1.0 if distance_from_obstacle1 > obstacle1_height else 0.0
            features[idx + 10] = 1.0 if distance_from_obstacle2 > obstacle2_height else 0.0
        else:
            features[idx + 9] = 1.0 if distance_from_obstacle1 == obstacle1_height else 0.0
            features[idx + 10] = 1.0 if distance_from_obstacle2 == obstacle2_height else 0.0

    return features
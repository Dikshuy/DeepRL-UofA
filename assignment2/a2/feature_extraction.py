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
    num_state_features = 10
    features = np.zeros(num_state_features * num_actions)
    
    env = obs[::-1]
    env = env[1:-1, 1:-1]
    height, width = env.shape
    
    # get floor height
    floor_height = None
    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break
            
    # agent info
    agent_positions = np.where(env == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1]) if y < floor_height]
    
    agent_min_y = min(y for y, _ in agent_positions)
    agent_max_y = max(y for y, _ in agent_positions) + 1
    agent_min_x = min(x for _, x in agent_positions)
    agent_max_x = max(x for _, x in agent_positions) + 1
    agent_height = agent_max_y - agent_min_y
    agent_width = agent_max_x - agent_min_x

    # obstacle info
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1]) if y < floor_height]
    
    x_positions = sorted(set(x for _, x in obstacle_positions))
    obstacle_groups = []
    current_group = []
    
    for i in range(len(x_positions)):
        if i == 0 or x_positions[i] - x_positions[i-1] > 1:
            if current_group:
                obstacle_groups.append(current_group)
            current_group = [x_positions[i]]
        else:
            current_group.append(x_positions[i])
    if current_group:
        obstacle_groups.append(current_group)
    
    obstacles_info = []
    for group in obstacle_groups:
        group_positions = [(y, x) for y, x in obstacle_positions if x in group]
        
        obstacle_min_y = min(y for y, _ in group_positions)
        obstacle_max_y = max(y for y, _ in group_positions) + 1
        obstacle_min_x = min(x for _, x in group_positions)
        obstacle_max_x = max(x for _, x in group_positions) + 1
        obstacle_height = obstacle_max_y - obstacle_min_y
        obstacle_width = obstacle_max_x - obstacle_min_x
        distance_from_obstacle = obstacle_min_x - agent_max_x
        
        obstacles_info.append({
            'height': obstacle_height,
            'width': obstacle_width,
            'distance': distance_from_obstacle,
            'min_x': obstacle_min_x
        })
    
    obstacles_info.sort(key=lambda x: x['distance'])

    # global vars
    distance_from_goal = width - agent_max_x
    jump_height = floor_height - agent_max_y
    in_air = float(jump_height > 0)
    
    for a in range(num_actions):
        idx = a * num_state_features
        
        # get nearest obstacle info
        nearest_obstacle = obstacles_info[0]
        
        features[idx + 0] = agent_height / height
        features[idx + 1] = distance_from_goal / width
        features[idx + 2] = nearest_obstacle['distance'] / width
        features[idx + 3] = jump_height / height
        features[idx + 4] = in_air
        features[idx + 5] = 1.0 if a == action else 0.0
        
        # action specific features
        if action == 0:
            features[idx + 6] = 1.0 if nearest_obstacle['distance'] > nearest_obstacle['height'] else 0.0
        else:
            features[idx + 6] = 1.0 if nearest_obstacle['distance'] == nearest_obstacle['height'] else 0.0
        
        # features for second obstacle (if exists)
        if len(obstacles_info) > 1:
            next_obstacle = obstacles_info[1]
            optimal_jump_point_2 = float(next_obstacle['distance'] == next_obstacle['height'])

            features[idx + 7] = next_obstacle['distance'] / width
            features[idx + 8] = optimal_jump_point_2 / height
            if action == 0:
                features[idx + 9] = 1.0 if next_obstacle['distance'] > next_obstacle['height'] else 0.0
            else:
                features[idx + 9] = 1.0 if next_obstacle['distance'] == next_obstacle['height'] else 0.0
        else:
            # if no second obstacle, set all features to 0
            features[idx + 7] = 0.0
            features[idx + 8] = 0.0
            features[idx + 9] = 0.0

        # I was trying to make it more smarter so that it can work for any number of obstacles but I was not able to fully automate it :(
        # Currently, it only works for two obstacles
    
    return features
import numpy as np

def terrible_feature_extractor(obs, action):
    num_actions = 2
    num_state_features = 625
    terrible_features = np.zeros(num_state_features * num_actions)
    terrible_features[num_state_features * action: num_state_features * (action+1)] = obs[0:25, 0:25].flatten()
    return terrible_features

def normalize_height(feature, max_height):
    return feature / max_height

def normalize_width(feature, max_width):
    return feature / max_width

def good_features_extractor(obs, action):
    num_actions = 2
    num_state_features = 12
    good_features = np.zeros(num_state_features * num_actions)

    # # visualize the observation
    # RESET = "\033[0m"       # Reset to default
    # GRAY = "\033[90m"       # Gray for 0.0
    # GREEN = "\033[92m"      # Green for 0.5
    # RED = "\033[91m"        # Red for 1.0

    # for row in obs[::-1]:
    #     colored_row = [
    #         f"{GRAY}0.0{RESET}" if num == 0.0 else
    #         f"{GREEN}0.5{RESET}" if num == 0.5 else
    #         f"{RED}1.0{RESET}"
    #         for num in row
    #     ]
    #     print(" ".join(colored_row))

    env = np.array(obs[::-1])
    height, width = env.shape
    
    floor_height = None

    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i

    agent_positions = np.where(env[:, :-1] == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1])
                      if y not in [0, height-1] and x not in [0, width-1] and y < floor_height]
    
    if agent_positions:
        agent_min_x = min(y for y, _ in agent_positions)
        agent_max_x = max(y for y, _ in agent_positions)
        agent_min_y = min(x for _, x in agent_positions)
        agent_max_y = max(x for _, x in agent_positions)

    # height and width of the agent
    agent_height = agent_max_x - agent_min_x + 1 + 1    # +1 to include the floor
    agent_width = agent_max_y - agent_min_y + 1

    # normalized agent height and width
    agent_height = normalize_height(agent_height, height)
    agent_width = normalize_width(agent_width, width)
    
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1]) if y < floor_height]

    if obstacle_positions:
        obstacle_min_y = min(y for y, _ in obstacle_positions)
        obstacle_max_y = max(y for y, _ in obstacle_positions)
        obstacle_min_x = min(x for _, x in obstacle_positions)
        obstacle_max_x = max(x for _, x in obstacle_positions)

    # height and width of the obstacle
    obstacle_height = obstacle_max_y - obstacle_min_y + 1 + 1   # +1 to include the floor
    obstacle_width = obstacle_max_x - obstacle_min_x + 1

    # normalized obstacle height and width
    obstacle_height = normalize_height(obstacle_height, height)
    obstacle_width = normalize_width(obstacle_width, width)

    # current position of the agent and the obstacle
    current_agent_position = agent_min_y + agent_width // 2
    current_obstacle_position = obstacle_min_y + obstacle_width // 2

    # normalize the current position of the agent and the obstacle
    current_agent_position = normalize_width(current_agent_position, width)
    current_obstacle_position = normalize_width(current_obstacle_position, width)

    # distance between agent and obstacle
    distance = current_obstacle_position - current_agent_position

    # normalize the distance
    distance = np.exp(1 - normalize_width(distance, width))

    agent_y_position = agent_min_x + agent_height // 2
    obstacle_y_position = obstacle_min_x + obstacle_height // 2

    # normalize the y position of the agent and the obstacle
    agent_y_position = normalize_height(agent_y_position, height)
    obstacle_y_position = normalize_height(obstacle_y_position, height)

    features = [agent_width, agent_height, obstacle_width, obstacle_height, current_agent_position, current_obstacle_position, distance, agent_y_position, obstacle_y_position]

    good_features[num_state_features * action: num_state_features * (action+1)] = features
    
    return good_features

def improved_features_extractor(obs, action):
    num_actions = 2
    num_state_features = 16
    features = np.zeros(num_state_features * num_actions)
    
    env = np.array(obs[::-1])
    height, width = env.shape
    
    # Find floor height
    floor_height = None
    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break
            
    if floor_height is None:
        return features
    
    # Get agent positions
    agent_positions = np.where(env[:, :-1] == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1])
                      if y not in [0, height-1] and x not in [0, width-1] and y < floor_height]
    
    if not agent_positions:
        return features
        
    agent_min_x = min(y for y, _ in agent_positions)
    agent_max_x = max(y for y, _ in agent_positions)
    agent_min_y = min(x for _, x in agent_positions)
    agent_max_y = max(x for _, x in agent_positions)
    
    # Agent state
    agent_bottom_y = agent_min_x
    agent_right_edge = agent_max_y
    agent_height_from_ground = agent_bottom_y - floor_height
    
    # Ground contact and jump state
    is_on_ground = any(env[floor_height-1, agent_min_y:agent_max_y+1] == 1.0)
    
    # Obstacle detection
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1])
                         if y < floor_height]
    
    # Initialize obstacle features
    next_obstacle_distance = width  # Default to max distance if no obstacle
    next_obstacle_height = 0
    optimal_jump_distance = width
    is_jumping_required = False
    distance_to_landing = 0
    clear_path_to_goal = True
    in_jump_window = False
    jump_urgency = 0
    
    if obstacle_positions:
        obstacle_min_x = min(y for y, _ in obstacle_positions)
        obstacle_max_x = max(y for y, _ in obstacle_positions)
        obstacle_min_y = min(x for _, x in obstacle_positions)
        obstacle_max_y = max(x for _, x in obstacle_positions)
        
        obstacle_height = obstacle_max_x - obstacle_min_x + 1
        next_obstacle_distance = max(0, obstacle_min_y - agent_right_edge)
        next_obstacle_height = obstacle_height
        
        # Calculate optimal jump point based on obstacle height and distance
        optimal_jump_distance = max(1, obstacle_height * 1.5)  # Basic physics approximation
        
        # Determine if jump is required
        is_jumping_required = (next_obstacle_distance <= optimal_jump_distance * 1.2 and 
                             next_obstacle_distance > 0)
        
        # Calculate landing zone
        distance_to_landing = max(0, obstacle_max_y - agent_right_edge)
        
        # Check if path to goal is clear
        clear_path_to_goal = obstacle_max_y < agent_right_edge
        
        # Jump window calculation
        in_jump_window = (0 < next_obstacle_distance <= optimal_jump_distance)
        
        # Jump urgency (higher when closer to optimal jump point)
        if in_jump_window:
            jump_urgency = 1 - (next_obstacle_distance / optimal_jump_distance)
    
    # Progress features
    distance_to_goal = normalize_width(width - agent_right_edge, width)
    progress = 1 - distance_to_goal
    
    # Combine features
    features_list = [
        normalize_height(agent_height_from_ground, height),  # Height from ground
        float(is_on_ground),                                # Ground contact
        normalize_width(next_obstacle_distance, width),      # Distance to next obstacle
        normalize_height(next_obstacle_height, height),      # Height of next obstacle
        normalize_width(optimal_jump_distance, width),       # Optimal jump distance
        float(is_jumping_required),                         # Jump required flag
        normalize_width(distance_to_landing, width),         # Distance to safe landing
        float(clear_path_to_goal),                         # Clear path to goal
        float(in_jump_window),                             # In optimal jump window
        jump_urgency,                                      # Jump urgency
        progress,                                          # Progress toward goal
        normalize_height(max(0, agent_height_from_ground), height),  # Current jump height
        float(agent_height_from_ground < 0),               # Below ground level
        float(action == 1),                                # Is jump action
        float(is_on_ground and action == 1),               # Starting jump from ground
        float(is_jumping_required and action == 1)         # Jumping when needed
    ]
    
    # Set features for the current action
    features[num_state_features * action: num_state_features * (action+1)] = features_list
    
    return features
import numpy as np

def terrible_feature_extractor(obs, action):
    num_actions = 2
    num_state_features = 625
    terrible_features = np.zeros(num_state_features * num_actions)
    terrible_features[num_state_features * action: num_state_features * (action+1)] = obs[0:25, 0:25].flatten()
    return terrible_features

def get_bounding_box(mask):
    """Finds the bounding box (x, y, width, height) of the largest connected component in the mask."""
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return None  # No object found
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, (x_max - x_min + 1), (y_max - y_min + 1)


def gen_masks(obs):
    
    obstacle_mask = obs == 0.5
    
    floor_mask = obs == 1
    floor_row = np.argmax(np.sum(floor_mask, 1))
    
    agent_mask = obs == 1
    agent_mask[floor_row, :] = False
    
    return obstacle_mask, agent_mask

def terrible_feature_extractor(obs, action):
    
    # preprocess the observation
    obs = np.flip(obs, axis=0)
    obs = obs[1:-1, 1:-1 ]
        
    # generate masks
    obstacle_mask, agent_mask = gen_masks(obs)
    
    width, height = obs.shape
    
    # (x, y, width, height)
    agent_bbox = get_bounding_box(agent_mask)
    obstacle_bbox = get_bounding_box(obstacle_mask)    
    # eg (0, 39, 4, 9), (24, 39, 9, 9)
    
    on_floor = agent_bbox[1] == obstacle_bbox[1]
    dist_from_box = obstacle_bbox[0] - agent_bbox[0]
    dist_from_right = width - agent_bbox[0]
    before_jump_point = obstacle_bbox[0] - agent_bbox[0] - agent_bbox[2] > obstacle_bbox[3]
    after_jump_point = obstacle_bbox[0] - agent_bbox[0] - agent_bbox[2] < obstacle_bbox[3]
    on_jump_point = obstacle_bbox[0] - agent_bbox[0] - agent_bbox[2] == obstacle_bbox[3]
    
    # print(on_jump_point)
    # plt.imshow(obs, cmap='gray')
    # plt.show()
    
    # print(np.array([on_floor, dist_from_box, dist_from_right, before_jump_point, after_jump_point]))
    # plt.imshow(obs, cmap = 'gray')
    # plt.show()
    
    # print(obstacle_bbox[0] - agent_bbox[0] - agent_bbox[2], obstacle_bbox[3])
    # plt.imshow(obs, cmap = 'gray')
    # plt.show()
    
    feat_set = np.array([on_floor, dist_from_box, dist_from_right, before_jump_point, after_jump_point])
    feat_set0 = np.array([on_floor, dist_from_box, dist_from_right, before_jump_point, after_jump_point, on_jump_point]) * (action == 0)
    feat_set1 = np.array([on_floor, dist_from_box, dist_from_right, before_jump_point, after_jump_point, on_jump_point]) * (action == 1)
    
    final_feat_set = np.concat((feat_set0, feat_set1, np.array([0.995**dist_from_right, 1]))) / width
    
    return final_feat_set

def normalize_height(feature, max_height):
    return feature / max_height

def normalize_width(feature, max_width):
    return feature / max_width

def good_features_extractor(obs, action):
    num_actions = 2  # up and right
    num_state_features = 10  # number of features per action
    features = np.zeros(num_state_features * num_actions)

    # Screen dimensions
    scr_h, scr_w = obs.shape

    # Find the floor height
    floor_height = None
    for i in range(scr_h):
        if np.all(obs[i, :] == 1.0):  # Floor is a horizontal line of 1.0s
            floor_height = i
            break

    # Find the agent's position and dimensions
    agent_pos = np.argwhere(obs == 1.0)  # Agent is represented by 1.0
    if len(agent_pos) == 0:
        return features  # Return zero features if the agent is not found

    # Agent's bottom-left corner
    agent_y, agent_x = agent_pos[0]

    # Agent's width and height
    agent_w = np.max(agent_pos[:, 1]) - np.min(agent_pos[:, 1]) + 1
    agent_h = np.max(agent_pos[:, 0]) - np.min(agent_pos[:, 0]) + 1

    # Find the obstacle's position and dimensions
    obstacle_pos = np.argwhere(obs == 0.5)  # Obstacle is represented by 0.5
    if len(obstacle_pos) == 0:
        return features  # Return zero features if the obstacle is not found

    # Obstacle's bottom-left corner
    obstacle_y, obstacle_x = obstacle_pos[0]

    # Obstacle's width and height
    obstacle_w = np.max(obstacle_pos[:, 1]) - np.min(obstacle_pos[:, 1]) + 1
    obstacle_h = np.max(obstacle_pos[:, 0]) - np.min(obstacle_pos[:, 0]) + 1

    # Compute features
    # Feature 1: Horizontal distance between agent's right edge and obstacle's left edge (normalized)
    distance_to_obstacle = (obstacle_x - (agent_x + agent_w)) / scr_w

    # Feature 2: Whether the agent is on the ground or in the air
    agent_in_air = 1 if agent_y != floor_height else 0

    # Feature 3: Obstacle height (normalized)
    normalized_obstacle_height = obstacle_h / scr_h

    # Feature 4: Agent's speed (normalized)
    agent_speed = 1.0  # Assuming agent speed is 1 pixel per step
    normalized_agent_speed = agent_speed / scr_w

    # Feature 5: Time to collision (estimated steps until collision)
    time_to_collision = distance_to_obstacle / agent_speed

    # Feature 6: Jump trajectory height (normalized)
    jump_height = 16  # Assuming jump height equals agent height
    normalized_jump_height = jump_height / scr_h

    # Feature 7: Steps since last jump
    steps_since_last_jump = 0  # Placeholder, needs to be tracked over time

    # Feature 8: Steps until landing
    steps_until_landing = 0  # Placeholder, needs to be tracked over time

    # Feature 9: Current clearance (vertical distance between agent's top and obstacle's bottom)
    current_clearance = (obstacle_y - (agent_y + agent_h)) / scr_h

    # Feature 10: Obstacle proximity (normalized)
    obstacle_proximity = distance_to_obstacle / scr_w

    # Populate the feature vector
    features[action * num_state_features:(action + 1) * num_state_features] = [
        distance_to_obstacle,
        agent_in_air,
        normalized_obstacle_height,
        normalized_agent_speed,
        time_to_collision,
        normalized_jump_height,
        steps_since_last_jump,
        steps_until_landing,
        current_clearance,
        obstacle_proximity
    ]

    return features

def old_good_feature_extractor(obs, action):
    num_actions = 2
    num_state_features = 12
    features = np.zeros(num_state_features * num_actions)

    env = np.array(obs[::-1])
    height, width = env.shape

    # Find floor height
    floor_height = None
    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break

    # Get agent position
    agent_positions = np.where(env[:, :-1] == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1])
                      if y not in [0, height-1] and x not in [0, width-1] and y < floor_height]
    
    # Get obstacle position
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1])
                         if y < floor_height]
    
    agent_min_y = min(y for y, _ in agent_positions)
    agent_max_y = max(y for y, _ in agent_positions)
    agent_min_x = min(x for _, x in agent_positions)
    agent_max_x = max(x for _, x in agent_positions)

    agent_max_y = max(y for y, _ in agent_positions)
    is_jumping = agent_max_y + 1 < floor_height

    if is_jumping:
        agent_height = agent_max_y - agent_min_y + 1
    else:
        agent_height = agent_max_y - agent_min_y + 1 + 1 # +1 to include the floor
    
    # Obstacle features
    obstacle_min_y = min(y for y, _ in obstacle_positions)
    obstacle_max_y = max(y for y, _ in obstacle_positions)
    obstacle_min_x = min(x for _, x in obstacle_positions)
    obstacle_max_x = max(x for _, x in obstacle_positions)
    
    obstacle_height = obstacle_max_y - obstacle_min_y + 1 #+ 1 # +1 to include the floor

    distance = obstacle_min_x - agent_max_x

    close = 1.0 if abs(distance - obstacle_height) == 0 else 0.0

    # print(f"agent max x:{agent_max_x}, obstacle min x:{obstacle_min_x}, distance:{distance}, is jumping:{is_jumping}, agent height:{agent_height}, close:{close}")

    # features_list = [agent_height, agent_width, obstacle_height, agent_max_x, obstacle_min_x, agent_center_x]
    # features_list = [agent_max_x/width, obstacle_min_x/width, distance/width, float(is_jumping), close, agent_width/width]
    # features[num_state_features * action: num_state_features * (action+1)] = features_list

    for a in range(num_actions):
        feature_idx = a * num_state_features
    # 1. Distance equals obstacle height? (key jumping condition)
        features[feature_idx] = 5.0 if abs(distance - obstacle_height) < 1 else 0.0
        
        # 2. Distance to obstacle normalized
        features[feature_idx + 1] = distance / width
        
        # 3. Is distance less than obstacle height? (too close)
        features[feature_idx + 2] = 1.0 if distance < obstacle_height else 0.0
        
        # 4. Is agent currently jumping?
        features[feature_idx + 3] = 1.0 if is_jumping else 0.0
        
        # 5. Is agent too close to obstacle? (emergency feature)
        features[feature_idx + 4] = 1.0 if 0 < distance < obstacle_height/2 else 0.0
        
        # 6. Action-specific feature
        if a == 1:  # up action
            # High value when distance matches obstacle height
            features[feature_idx + 5] = 1.0 if abs(distance - obstacle_height) <= 2 else 0.0
        else:  # right action
            # High value when distance is greater than obstacle height
            features[feature_idx + 5] = 1.0 if distance > obstacle_height else 0.0

    return features

def new_features_extractor(obs, action):
    """Extract features for the jumping task.
    Key features focus on:
    1. Distance to obstacle
    2. Current height of agent
    3. Obstacle dimensions
    4. Relative positioning
    5. Jump trajectory information
    """
    num_actions = 2
    num_state_features = 8
    features = np.zeros(num_state_features * num_actions)
    
    env = np.array(obs[::-1])  # Flip to make bottom-up coordinate system
    height, width = env.shape
    
    # Find floor height
    floor_height = None
    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break
    
    # Get agent position
    agent_positions = np.where(env[:, :-1] == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1])
                      if y not in [0, height-1] and x not in [0, width-1] and y < floor_height]
    
    # Get obstacle position
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1]) 
                         if y < floor_height]
    
    if agent_positions and obstacle_positions:
        # Agent properties
        agent_x = max(x for _, x in agent_positions)  # Rightmost point of agent
        agent_y = max(y for y, _ in agent_positions)  # Highest point of agent
        agent_base_y = min(y for y, _ in agent_positions)  # Lowest point of agent
        
        # Obstacle properties
        obstacle_left_x = min(x for _, x in obstacle_positions)
        obstacle_height = max(y for y, _ in obstacle_positions) - min(y for y, _ in obstacle_positions) + 1
        
        feature_idx = action * num_state_features
        
        # Feature 1: Normalized horizontal distance to obstacle
        distance_to_obstacle = obstacle_left_x - agent_x
        features[feature_idx] = np.clip(distance_to_obstacle / width, 0, 1)
        
        # Feature 2: Is agent currently in the air?
        features[feature_idx + 1] = float(agent_base_y > floor_height)
        
        # Feature 3: Normalized current height above ground
        height_above_ground = agent_base_y - floor_height
        features[feature_idx + 2] = height_above_ground / height
        
        # Feature 4: Normalized obstacle height
        features[feature_idx + 3] = obstacle_height / height
        
        # Feature 5: Critical jump distance indicator
        # Assume optimal jump distance is between 5-15 units
        min_ = obstacle_height
        max_ = obstacle_height
        optimal_jump_range = (min_, max_)  # These values might need tuning
        features[feature_idx + 4] = 1.0 if optimal_jump_range[0] <= distance_to_obstacle <= optimal_jump_range[1] else 0.0
        
        # Feature 6: Distance to right boundary
        distance_to_end = width - agent_x
        features[feature_idx + 5] = distance_to_end / width
        
        # Feature 7: Jump trajectory feature
        # If we're at a good distance and not already jumping, this feature activates
        features[feature_idx + 6] = 1.0 if (optimal_jump_range[0] <= distance_to_obstacle <= optimal_jump_range[1] 
                                          and agent_base_y == floor_height and action == 0) else 0.0
        
        # Feature 8: Safe landing indicator
        # Distance after obstacle where it's safe to land
        landing_zone_distance = obstacle_left_x + obstacle_height - agent_x
        features[feature_idx + 7] = 1.0 if landing_zone_distance > 0 and landing_zone_distance < width/4 else 0.0
        
    return features

def advanced_features_extractor(obs, action):
    """Extract sophisticated action-specific features for the jumping task.
    Separate features for UP and RIGHT actions to better capture action-specific dynamics.
    """
    num_actions = 2  # UP (0) and RIGHT (1)
    num_state_features = 12  # Increased number of features per action
    features = np.zeros(num_state_features * num_actions)
    
    env = np.array(obs[::-1])
    height, width = env.shape
    
    # Find floor height
    floor_height = None
    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break
            
    # Get agent position
    agent_positions = np.where(env[:, :-1] == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1])
                      if y not in [0, height-1] and x not in [0, width-1] and y < floor_height]
    
    # Get obstacle position
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1]) 
                         if y < floor_height]
    
    if agent_positions and obstacle_positions:
        # Agent properties
        agent_left_x = min(x for _, x in agent_positions)
        agent_right_x = max(x for _, x in agent_positions)
        agent_top_y = max(y for y, _ in agent_positions)
        agent_bottom_y = min(y for y, _ in agent_positions)
        agent_center_x = (agent_left_x + agent_right_x) / 2
        agent_center_y = (agent_top_y + agent_bottom_y) / 2
        
        # Obstacle properties
        obstacle_left_x = min(x for _, x in obstacle_positions)
        obstacle_right_x = max(x for _, x in obstacle_positions)
        obstacle_top_y = max(y for y, _ in obstacle_positions)
        obstacle_bottom_y = min(y for y, _ in obstacle_positions)
        obstacle_center_x = (obstacle_left_x + obstacle_right_x) / 2
        obstacle_height = obstacle_top_y - obstacle_bottom_y# + 1
        
        # Common features for both actions
        for a in range(num_actions):
            feature_idx = a * num_state_features
            
            # Base position features
            features[feature_idx] = (agent_center_x - obstacle_left_x) / width  # Distance to obstacle
            features[feature_idx + 1] = (agent_bottom_y - floor_height) / height  # Height from ground
            features[feature_idx + 2] = obstacle_height / height  # Obstacle height
            
            # Progress features
            features[feature_idx + 3] = (width - agent_right_x) / width  # Distance to goal
            features[feature_idx + 4] = float(agent_bottom_y > floor_height)  # In air?
            
            # Danger zone features
            in_obstacle_x_range = obstacle_left_x <= agent_right_x <= obstacle_right_x
            features[feature_idx + 5] = float(in_obstacle_x_range)
            
        # UP action specific features (action 0)
        if action == 1:
            feature_idx = 0
            
            # Jump timing features
            optimal_jump_start = obstacle_left_x - obstacle_height  # Adjust these values based on environment
            optimal_jump_end = obstacle_right_x + obstacle_height
            # distance_to_jump_point = agent_right_x - optimal_jump_start
            distance_to_jump_point = obstacle_height
            
            features[feature_idx + 6] = float(optimal_jump_start <= agent_right_x <= optimal_jump_end)
            features[feature_idx + 7] = np.exp(-abs(distance_to_jump_point) / 10)  # Exponential decay
            
            # Jump trajectory prediction
            estimated_jump_peak = agent_bottom_y + 16  # Adjust based on jump height
            features[feature_idx + 8] = float(estimated_jump_peak > obstacle_top_y)
            
            # Safety features for jumping
            features[feature_idx + 9] = float(agent_bottom_y == floor_height)  # Can jump?
            features[feature_idx + 10] = float(agent_right_x < obstacle_left_x)  # Haven't passed obstacle?
            
            # Combination feature
            features[feature_idx + 11] = float(
                optimal_jump_start <= agent_right_x <= optimal_jump_end and 
                agent_bottom_y == floor_height and 
                agent_right_x < obstacle_left_x
            )
            
        # RIGHT action specific features (action 1)
        else:
            feature_idx = num_state_features
            
            # Movement safety features
            safe_to_move_right = agent_right_x + 1 < obstacle_left_x or agent_right_x > obstacle_right_x
            features[feature_idx + 6] = float(safe_to_move_right)
            
            # Landing trajectory features
            in_landing_zone = agent_right_x > obstacle_right_x and agent_bottom_y > floor_height
            features[feature_idx + 7] = float(in_landing_zone)
            
            # Progress encouragement
            distance_after_obstacle = max(0, agent_left_x - obstacle_right_x) / width
            features[feature_idx + 8] = distance_after_obstacle
            
            # Speed features
            features[feature_idx + 9] = float(agent_bottom_y == floor_height and agent_right_x < obstacle_left_x - 15)
            features[feature_idx + 10] = float(agent_right_x > obstacle_right_x)  # Past obstacle
            
            # Combination feature
            features[feature_idx + 11] = float(
                safe_to_move_right and 
                (agent_bottom_y == floor_height or agent_right_x > obstacle_right_x)
            )
            
    return features

def more_feature_extractor(obs, action):
    """Extract meaningful features for the jumping task.
    
    Features:
    1. Distance to obstacle (normalized)
    2. Height difference to obstacle (normalized)
    3. Current height from floor (normalized)
    4. Is jumping (binary)
    5. Distance to right edge (normalized)
    6. Is obstacle in jump trajectory (binary)
    7. Immediate collision danger (binary)
    8. Clear path to right (binary)
    9. Optimal jump distance indicator (binary)
    """
    num_actions = 2  # up and right
    num_state_features = 9
    features = np.zeros(num_state_features * num_actions)
    
    # Convert observation to numpy array and flip vertically
    env = np.array(obs[::-1])
    height, width = env.shape
    
    # Find floor height
    floor_height = None
    for i in range(height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break
    
    # Find agent position
    agent_positions = np.where(env == 1.0)
    agent_y = None
    agent_x = None
    for i in range(len(agent_positions[0])):
        y, x = agent_positions[0][i], agent_positions[1][i]
        if y > floor_height and y < height-1 and x < width-1:
            agent_y = y
            agent_x = x
            break
    
    # Find obstacle position
    obstacle_positions = np.where(env == 0.5)
    if len(obstacle_positions[1]) > 0:
        obstacle_x = np.min(obstacle_positions[1])
        obstacle_y = np.max(obstacle_positions[0])
    else:
        obstacle_x = width
        obstacle_y = floor_height
    
    # Calculate features
    base_features = np.zeros(num_state_features)
    
    # 1. Distance to obstacle (normalized)
    distance_to_obstacle = (obstacle_x - agent_x) / width
    base_features[0] = distance_to_obstacle
    
    # 2. Height difference to obstacle (normalized)
    height_diff = (obstacle_y - agent_y) / height
    base_features[1] = height_diff
    
    # 3. Current height from floor (normalized)
    height_from_floor = (agent_y - floor_height) / height
    base_features[2] = height_from_floor
    
    # 4. Is jumping (binary)
    is_jumping = height_from_floor > 0
    base_features[3] = float(is_jumping)
    
    # 5. Distance to right edge (normalized)
    distance_to_right = (width - agent_x) / width
    base_features[4] = distance_to_right
    
    # 6. Is obstacle in jump trajectory
    in_jump_trajectory = (distance_to_obstacle < 0.3) and not is_jumping
    base_features[5] = float(in_jump_trajectory)
    
    # 7. Immediate collision danger
    collision_danger = (distance_to_obstacle < 0.2) and not is_jumping
    base_features[6] = float(collision_danger)
    
    # 8. Clear path to right
    clear_path = distance_to_obstacle > 0.3 or is_jumping
    base_features[7] = float(clear_path)
    
    # 9. Optimal jump distance indicator
    optimal_jump_distance = 0.15 <= distance_to_obstacle <= 0.25
    base_features[8] = float(optimal_jump_distance)
    
    # Create action-dependent features
    if action == 0:  # right
        features[:num_state_features] = base_features
    else:  # jump
        features[num_state_features:] = base_features
        
    return features

def test_features_extractor(obs, action):
    num_actions = 2
    num_state_features = 12  # Features focused on jump timing discovery
    features = np.zeros(num_state_features * num_actions)
    
    env = np.array(obs[::-1])
    height, width = env.shape
    
    # Find floor height
    floor_height = None
    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break
    
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
    agent_height_from_ground = agent_min_x - floor_height
    is_on_ground = any(env[floor_height-1, agent_min_y:agent_max_y+1] == 1.0)
    
    # Get obstacle information
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1])
                         if y < floor_height]
    
    if obstacle_positions:
        obstacle_min_x = min(y for y, _ in obstacle_positions)
        obstacle_max_x = max(y for y, _ in obstacle_positions)
        obstacle_min_y = min(x for _, x in obstacle_positions)
        obstacle_max_y = max(x for _, x in obstacle_positions)
        
        # Critical distances and heights
        distance_to_obstacle = max(0, obstacle_min_y - agent_max_y)
        obstacle_height = obstacle_max_x - floor_height + 1
        
        # Relative position features
        relative_x_to_obstacle = obstacle_min_y - agent_max_y
        relative_y_to_obstacle = obstacle_max_x - agent_min_x
        
        # Discretized distance zones for jump timing
        close_to_obstacle = 0 < distance_to_obstacle <= 5
        very_close_to_obstacle = 0 < distance_to_obstacle <= 2
        about_to_collide = 0 < distance_to_obstacle <= 1
        
        features_list = [
            float(is_on_ground),                                # Can only jump from ground
            normalize_width(distance_to_obstacle, width),       # Distance to obstacle
            normalize_height(obstacle_height, height),          # Height of obstacle
            normalize_height(agent_height_from_ground, height), # Current height from ground
            float(close_to_obstacle),                          # In general jump zone
            float(very_close_to_obstacle),                     # Getting very close
            float(about_to_collide),                           # Critical jump zone
            float(relative_y_to_obstacle > 0),                 # Obstacle is higher than agent
            float(action == 1 and is_on_ground),              # Valid jump
            float(action == 1 and not is_on_ground),          # Invalid jump
            normalize_width(relative_x_to_obstacle, width),    # Relative x position
            normalize_height(relative_y_to_obstacle, height)   # Relative y position
        ]
    else:
        # No obstacle in view - encourage forward movement
        features_list = [float(is_on_ground), 1.0] + [0.0] * 10
        
    # Set features for the current action
    features[num_state_features * action: num_state_features * (action+1)] = features_list
    
    return features

def features_extractor(obs, action):
    num_actions = 2  # [right, up]
    num_state_features = 6
    features = np.zeros(num_state_features * num_actions)
    
    env = np.array(obs[::-1])
    height, width = env.shape
    
    # Find floor height
    floor_height = None
    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break
            
    # Get agent position
    agent_positions = np.where(env[:, :-1] == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1])
                      if y not in [0, height-1] and x not in [0, width-1] and y < floor_height]
    
    # Get obstacle position
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1])
                         if y < floor_height]
    
    if agent_positions and obstacle_positions:
        # Get rightmost bottom point of agent
        agent_bottom_points = [(y, x) for y, x in agent_positions if y == min(y for y, _ in agent_positions)]
        agent_rightmost_bottom = max(x for _, x in agent_bottom_points)
        
        # Get leftmost bottom point of obstacle
        obstacle_bottom_points = [(y, x) for y, x in obstacle_positions if y == min(y for y, _ in obstacle_positions)]
        obstacle_leftmost_bottom = min(x for _, x in obstacle_bottom_points)
        
        # Calculate obstacle height
        obstacle_min_y = min(y for y, _ in obstacle_positions)
        obstacle_max_y = max(y for y, _ in obstacle_positions)
        obstacle_height = obstacle_max_y - obstacle_min_y + 1 + 1
        
        # Calculate horizontal distance between agent and obstacle
        distance = obstacle_leftmost_bottom - agent_rightmost_bottom
        
        # Is agent currently jumping?
        agent_min_y = min(y for y, _ in agent_positions)
        is_jumping = agent_min_y > floor_height

        
        
        for a in range(num_actions):
            feature_idx = a * num_state_features
            
            # 1. Distance equals obstacle height? (key jumping condition)
            features[feature_idx] = 1.0 if abs(distance - obstacle_height) <= 1 else 0.0
            
            # 2. Distance to obstacle normalized
            features[feature_idx + 1] = distance / width
            
            # 3. Is distance less than obstacle height? (too close)
            features[feature_idx + 2] = 1.0 if distance < obstacle_height else 0.0
            
            # 4. Is agent currently jumping?
            features[feature_idx + 3] = 1.0 if is_jumping else 0.0
            
            # 5. Is agent too close to obstacle? (emergency feature)
            features[feature_idx + 4] = 1.0 if 0 < distance < obstacle_height/2 else 0.0
            
            # 6. Action-specific feature
            if a == 1:  # up action
                # High value when distance matches obstacle height
                features[feature_idx + 5] = 1.0 if abs(distance - obstacle_height) <= 2 else 0.0
            else:  # right action
                # High value when distance is greater than obstacle height
                features[feature_idx + 5] = 1.0 if distance > obstacle_height else 0.0
    
    return features
    
def god_features_extractor(obs, action):
    num_actions = 2  # [right, up]
    num_state_features = 9
    features = np.zeros(num_state_features * num_actions)
    
    env = np.array(obs[::-1])
    height, width = env.shape
    
    # Find floor height
    floor_height = None
    for i in range(1, height-1):
        if np.all(env[i] == 1.0):
            floor_height = i
            break
            
    # Get agent position
    agent_positions = np.where(env[:, :-1] == 1.0)
    agent_positions = [(y, x) for y, x in zip(agent_positions[0], agent_positions[1])
                      if y not in [0, height-1] and x not in [0, width-1] and y < floor_height]
    
    # Get obstacle position
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1])
                         if y < floor_height]
    
    if agent_positions and obstacle_positions:
        # Agent features
        agent_min_y = min(y for y, _ in agent_positions)
        agent_max_y = max(y for y, _ in agent_positions)
        agent_min_x = min(x for _, x in agent_positions)
        agent_max_x = max(x for _, x in agent_positions)
        
        agent_height = agent_max_y - agent_min_y + 1
        agent_width = agent_max_x - agent_min_x + 1
        
        # Agent center
        agent_center_x = (agent_min_x + agent_max_x) // 2
        agent_center_y = (agent_min_y + agent_max_y) // 2
        
        # Obstacle features
        obstacle_min_y = min(y for y, _ in obstacle_positions)
        obstacle_max_y = max(y for y, _ in obstacle_positions)
        obstacle_min_x = min(x for _, x in obstacle_positions)
        obstacle_max_x = max(x for _, x in obstacle_positions)
        
        obstacle_height = obstacle_max_y - obstacle_min_y + 1
        
        # Obstacle center
        obstacle_center_x = (obstacle_min_x + obstacle_max_x) // 2
        obstacle_center_y = (obstacle_min_y + obstacle_max_y) // 2
        
        # Current jump height estimation (maximum height reached so far in this jump)
        is_jumping = agent_center_y > floor_height
        current_height = agent_center_y - floor_height if is_jumping else 0
        
        # Calculate horizontal distance to obstacle
        distance_to_obstacle = obstacle_center_x - agent_center_x
        
        for a in range(num_actions):
            feature_idx = a * num_state_features
            
            # 1. Current height from ground (helps learn jump trajectory)
            features[feature_idx] = (agent_center_y - floor_height) / height
            
            # 2. Is agent currently jumping? (binary)
            features[feature_idx + 1] = 1.0 if is_jumping else 0.0
            
            # 3. Distance to obstacle normalized by current estimated jump height
            # This helps learn the relationship between distance and jump height
            features[feature_idx + 2] = distance_to_obstacle / (current_height + 1) if current_height > 0 else 0.0
            
            # 4. Is the agent at the peak of jump? (helps learn max jump height)
            # We consider it peak if agent is jumping and higher than obstacle
            features[feature_idx + 3] = 1.0 if is_jumping and agent_center_y > obstacle_max_y else 0.0
            
            # 5. Is the agent in potential jumping zone?
            # Use obstacle height as a proxy for required jump height initially
            estimated_jump_dist = obstacle_height + agent_width // 2
            features[feature_idx + 4] = 1.0 if abs(distance_to_obstacle - estimated_jump_dist) < agent_width else 0.0
            
            # 6. Is the agent too close to obstacle? (emergency jump needed)
            features[feature_idx + 5] = 1.0 if 0 < distance_to_obstacle < agent_width * 2 else 0.0
            
            # 7. Relative height difference to obstacle top (normalized)
            features[feature_idx + 6] = (agent_center_y - obstacle_max_y) / height
            
            # 8. Is agent at a good lateral position for jump?
            # This uses the formula you mentioned but with estimated jump height
            optimal_jump_distance = obstacle_height + agent_width // 2 - 2
            features[feature_idx + 7] = 1.0 if abs(distance_to_obstacle - optimal_jump_distance) < agent_width else 0.0
            
            # 9. Current velocity proxy (helps learn jump trajectory)
            # Higher value for 'up' action when jumping is necessary
            if a == 1:  # up action
                features[feature_idx + 8] = 1.0 if (is_jumping and distance_to_obstacle < optimal_jump_distance * 1.5) else 0.0
            else:  # right action
                features[feature_idx + 8] = 1.0 if not is_jumping or distance_to_obstacle > optimal_jump_distance * 1.5 else 0.0
    
    return features

def improved_features_extractor(obs, action):
    num_actions = 2
    num_state_features = 15  # Increased for jump height features
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
    agent_current_height = agent_bottom_y - floor_height
    is_on_ground = any(env[floor_height-1, agent_min_y:agent_max_y+1] == 1.0)
    is_ascending = not is_on_ground and agent_current_height > 0
    
    # Get obstacle information
    obstacle_positions = np.where(env == 0.5)
    obstacle_positions = [(y, x) for y, x in zip(obstacle_positions[0], obstacle_positions[1])
                         if y < floor_height]
    
    if obstacle_positions:
        obstacle_min_x = min(y for y, _ in obstacle_positions)
        obstacle_max_x = max(y for y, _ in obstacle_positions)
        obstacle_min_y = min(x for _, x in obstacle_positions)
        obstacle_max_y = max(x for _, x in obstacle_positions)
        
        # Critical distances and heights
        distance_to_obstacle = max(0, obstacle_min_y - agent_max_y)
        obstacle_height = obstacle_max_x - floor_height + 1
        
        # Jump height features
        height_diff_to_obstacle = obstacle_max_x - agent_bottom_y
        relative_height_ratio = agent_current_height / max(1, obstacle_height)
        
        # Jump timing windows based on distance
        in_critical_zone = 0 < distance_to_obstacle <= 3
        approaching_obstacle = 3 < distance_to_obstacle <= 6
        safe_distance = distance_to_obstacle > 6
        
        features_list = [
            float(is_on_ground),                                # Ground contact
            float(is_ascending),                               # Currently in upward jump
            normalize_height(agent_current_height, height),     # Current height from ground
            normalize_width(distance_to_obstacle, width),       # Distance to next obstacle
            normalize_height(obstacle_height, height),          # Obstacle height
            normalize_height(height_diff_to_obstacle, height),  # Height difference to obstacle
            relative_height_ratio,                             # How high we are relative to obstacle
            float(in_critical_zone),                          # In jump decision zone
            float(approaching_obstacle),                       # About to enter jump zone
            float(safe_distance),                             # Safe to continue right
            float(height_diff_to_obstacle > 0),               # Obstacle is above us
            float(obstacle_height > agent_current_height),     # Need to jump higher
            float(action == 1 and is_on_ground),              # Valid jump attempt
            float(action == 1 and not is_on_ground),          # Invalid jump attempt
            float(in_critical_zone and action == 1)           # Critical zone jump decision
        ]
    else:
        # No obstacle in view
        features_list = [float(is_on_ground), float(is_ascending), 
                        normalize_height(agent_current_height, height)] + [0.0] * 12
    
    # Set features for the current action
    features[num_state_features * action: num_state_features * (action+1)] = features_list
    
    return features

'''
dump:

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
    agent_height = agent_max_x - agent_min_x + 1 #+ 1    # +1 to include the floor
    agent_width = agent_max_y - agent_min_y + 1
    
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

    # current position of the agent and the obstacle
    current_agent_position = agent_min_y + agent_width // 2
    current_obstacle_position = obstacle_min_y + obstacle_width // 2

    # distance between agent and obstacle
    distance = current_obstacle_position - current_agent_position

    # Agent state
    agent_bottom_y = agent_min_x
    is_on_ground = any(env[floor_height-1, agent_min_y:agent_max_y+1] == 1.0)

    print(agent_width, agent_height, obstacle_width, obstacle_height, current_agent_position, current_obstacle_position, distance)
    print()

    features = [agent_width, 
                agent_height, 
                obstacle_width, 
                obstacle_height, 
                current_agent_position, 
                current_obstacle_position, 
                distance,
                ]

    good_features[num_state_features * action: num_state_features * (action+1)] = features
'''
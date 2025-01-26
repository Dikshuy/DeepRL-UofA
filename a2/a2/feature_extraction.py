import numpy as np

def terrible_feature_extractor(obs, action):
    num_actions = 2
    num_state_features = 625
    terrible_features = np.zeros(num_state_features * num_actions)
    terrible_features[num_state_features * action: num_state_features * (action+1)] = obs[0:25, 0:25].flatten()
    return terrible_features

import numpy as np
import gymnasium as gym
import agent_environment

class TestEnv(gym.Env):

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(13)
        self.action_space = gym.spaces.Discrete(2)
        self.reset_counter = 0
        self.resets = [0, 3, 6, 9]
        self.termination_obs = [2, 8]
        self.truncation_obs = [5, 12]
        self.current_obs = 0

    def reset(self, seed=None, options=None):
        if self.reset_counter > 0:
            assert self.current_obs == self.resets[self.reset_counter] - 1, f"{self.current_obs} vs {self.resets[self.reset_counter]}"
        observation = self.resets[self.reset_counter]
        self.reset_counter += 1
        self.current_obs = observation
        return observation, {}

    def step(self, action):
        self.current_obs += 1
        terminated = self.current_obs in self.termination_obs
        truncated = self.current_obs in self.truncation_obs
        return self.current_obs, self.current_obs * 0.5, terminated, truncated, {}

class TestAgent:

    def __init__(self):
        self.observation_list = []
        self.reward_list = []
        self.termination_list = []
        self.truncation_list = []
    
    def act(self, obs: int) -> int:
        return 0
        
    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        self.observation_list.append(obs)
        self.reward_list.append(reward)
        self.termination_list.append(terminated)
        self.truncation_list.append(truncated)

def test_agent_environment_step_loop():
    try:
        agent = TestAgent()
        env = TestEnv()
        num_timesteps = 8
        episode_rewards = agent_environment.agent_environment_step_loop(agent, env, num_timesteps)
        expected_results = np.array([1.5, 4.5, 7.5])
        actual_result = np.array(episode_rewards)
        test_passed = np.allclose(expected_results, actual_result)
        test_result = "PASSED" if test_passed else "FAILED"
        total_marks = 10
        marks = total_marks if test_passed else 0
    except Exception as e:
        test_result = "FAILED"
        total_marks = 10
        marks = 0
    print(f"test_agent_environment_step_loop: {test_result}. Marks: {marks}/{total_marks}")
    return marks


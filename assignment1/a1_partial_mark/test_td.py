import sarsa
import q_learning
import numpy as np
import epsilon_greedy_explorers


def test_sarsa_update_q_standard():
    try:
        # Logs the error appropriately. 
        explorer = epsilon_greedy_explorers.ConstantEpsilonGreedyExploration(0.1, 2)
        agent = sarsa.SARSA(5, 2, 0.1, explorer, 0.9)
        obs = 2
        action = 0
        reward = 1.5
        terminated = False
        next_obs = 3
        next_action = 0
        agent.q[obs, action] = 5.0 
        agent.q[next_obs, next_action] = 6.0
        agent.update_q(2, 0, reward, next_obs, next_action, terminated)
        expected_q = 5.0 + 0.1 * (1.5 + 0.9 * 6.0 - 5.0)
        test_passed = np.allclose(expected_q, agent.q[obs, action])
        test_result = "PASSED" if test_passed else "FAILED"
        total_marks = 5
        marks = total_marks if test_passed else 0
        print(f"test_sarsa_update_q_standard: {test_result}. Marks: {marks}/{total_marks}")
    except Exception as e:
        test_result = "FAILED"
        total_marks = 5
        marks = 0
        print(f"test_sarsa_update_q_standard: {test_result}. Marks: {marks}/{total_marks}")
    return marks
import epsilon_greedy_explorers
import numpy as np

def test_epsilon_greedy_standard():
	try:
		q_values = np.array([0, 5, 4, 4])
		epsilon = 0.2
		expected_results = np.array([0.05, 0.85, 0.05, 0.05])
		actual_result = epsilon_greedy_explorers.compute_epsilon_greedy_action_probs(q_values, epsilon)
		test_passed = np.allclose(expected_results, actual_result)
		test_result = "PASSED" if test_passed else "FAILED"
		total_marks = 5
		marks = total_marks if test_passed else 0
	except Exception as e:
		test_result = "FAILED"
		total_marks = 5
		marks = 0
	print(f"test_epsilon_greedy_standard: {test_result}. Marks: {marks}/{total_marks}")
	return marks

def test_epsilon_greedy_pure_greedy():
	try:
		q_values = np.array([0, 5, 5., 4, 9])
		epsilon = 0.0
		expected_results = np.array([0., 0., 0., 0., 1.0])
		actual_result = epsilon_greedy_explorers.compute_epsilon_greedy_action_probs(q_values, epsilon)
		test_passed = np.allclose(expected_results, actual_result)
		test_result = "PASSED" if test_passed else "FAILED"
		total_marks = 5
		marks = total_marks if test_passed else 0
	except Exception as e:
		test_result = "FAILED"
		total_marks = 5
		marks = 0
	print(f"test_epsilon_greedy_pure_greedy: {test_result}. Marks: {marks}/{total_marks}")
	return marks


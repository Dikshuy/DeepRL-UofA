import argparse
import numpy as np
import test_exploration
import test_td
import test_agent_environment

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ccid', required=True, type=str)
	args = parser.parse_args()
	ccid = args.ccid
	print(f"A1 Grades for CCID: {ccid}")
	epsilon_greedy_standard_marks = test_exploration.test_epsilon_greedy_standard()
	epsilon_greedy_pure_greedy = test_exploration.test_epsilon_greedy_pure_greedy()
	exploration_marks = epsilon_greedy_standard_marks + epsilon_greedy_pure_greedy
	sarsa_update_q_standard_marks = test_td.test_sarsa_update_q_standard()
	td_marks = sarsa_update_q_standard_marks
	agent_env_step_loop_marks = test_agent_environment.test_agent_environment_step_loop()
	agent_environment_marks = agent_env_step_loop_marks
	total_marks = exploration_marks + td_marks + agent_environment_marks
	print(f"{ccid} total_marks: {total_marks}")
	print("--------------------------------------------------------------------")
	print()

import argparse
import jumping_task
from jumping_task.envs import JumpTaskEnv
import pygame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Which environment", type=int, choices=[1,2,3], default=1)
    parser.add_argument("--num-episodes", help="How many episodes you want to play", default=2, type=int)
    args = parser.parse_args()

    if args.config == 1:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 25, 30],
                agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                obstacle_position=0, obstacle_size=(9,10),
                rendering=True, zoom=8, slow_motion=True, with_left_action=False,
                max_number_of_steps=300, two_obstacles=False, finish_jump=False)
    elif args.config == 2:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[30, 40],
                        agent_w=7, agent_h=7, agent_init_pos=0, agent_speed=1,
                        obstacle_position=0, obstacle_size=(11,17),
                        rendering=True, zoom=8, slow_motion=True, with_left_action=False,
                        max_number_of_steps=300, two_obstacles=False, finish_jump=False,
                        jump_height=24)
    else:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 30, 40],
                    agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                    obstacle_position=0, obstacle_size=(9,10),
                    rendering=True, zoom=8, slow_motion=True, with_left_action=False,
                    max_number_of_steps=300, two_obstacles=True, finish_jump=False)
    obs, _ = env.reset()
    for i in range(args.num_episodes):
        env.render()
        score = 0
        while not env.terminated:
            action = None
            if env.jumping[0] and env.finish_jump:
                action = 3
            else:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RIGHT:
                            action = 0
                        elif event.key == pygame.K_UP:
                            action = 1
                        elif event.key == pygame.K_e:
                            env.exit()
                        else:
                            action = 'unknown'
            if action is None:
                continue
            elif action == 'unknown':
                print('We did not recognize that action. Please use the arrows to move the agent or the \'e\' key to exit.')
                continue
            obs, r, term, trunc, _ = env.step(action)
            env.render()
            score += r
            print('Agent position: {:2f} | Reward: {:2f} | Terminal: {} | Truncate: {}'.format(
                env.agent_pos_x, r, term, trunc))
        print('---------------')
        print('Final score: {:2f}'.format(int(score)))
        print('---------------')
        obs, _ = env.reset()
import gym
import numpy as np
from ddpg_tf2 import Agent, PotentialAgent
import matplotlib.pyplot as plt
from env_dynamic_goal import CoppeliaSimEnvWrapper
from setup_flags import set_up
import sys
import os
import math
import tensorflow as tf
import json

import logging
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(devices[0], True)
    print("Success in setting memory growth")
except:
    print("Failed to set memory growth, invalid device or cannot modify virtual devices once initialized.")


FLAGS = set_up()
expname = FLAGS.exp



def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)



if __name__ == '__main__':
    env = CoppeliaSimEnvWrapper(visualize=FLAGS.show)
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    potentialagent = PotentialAgent(gamma=agent.gamma)
    
    n_games = 2000

    score_history, avg_score_history = [], []
    load_checkpoint = FLAGS.evaluate

    try:
        if load_checkpoint:
            n_steps = 0
            n_games = 3
            while n_steps <= agent.batch_size:
                observation = env.reset()
                action = env.action_space.sample()
                observation_, reward, done, info = env.step(action)
                agent.remember(observation, action, reward, observation_, done)
                n_steps += 1
            agent.learn()
            agent.load_models(FLAGS.exp)
            evaluate = True
        else:
            evaluate = False

        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0
            step_per_episode = 0
            reward_in_episode = []

            obs_floor = [math.floor(obs*100)/100.0 for obs in observation[:-3]]
            trajectory_epi = [obs_floor]
            while not done:
                action = agent.choose_action(observation, evaluate)
                observation_, reward, done, info = env.step(action)
                #breakpoint()

                score += reward
                step_per_episode += 1

                obs_floor_ = [math.floor(obs*100)/100.0 for obs in observation_[:-3]]
                trajectory_epi.append(obs_floor_)

                shaping_reward = potentialagent.reward_shaping(obs_floor, obs_floor_, evaluate)
                logging.debug(f"epi {i} step {step_per_episode}, reward: {reward} + {shaping_reward}")
                logging.debug(f"old obs: {observation}, floor {obs_floor}")
                logging.debug(f"new obs: {observation_}, floor {obs_floor_}")
                if evaluate:
                    reward_in_episode.append(f"{reward}+{shaping_reward}")
                reward += shaping_reward

                agent.remember(observation, action, reward, observation_, reward==1)
                if not load_checkpoint:
                    agent.learn()
                observation = observation_
                obs_floor = obs_floor_.copy()


            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            avg_score_history.append(avg_score)

            ## After one episode ##
            if trajectory_epi:
                potentialagent.add_trajectory(trajectory_epi, score, done)
                
            potentialagent.learn_pf()

            if potentialagent.potential_learn_cnt == 10:
                with open(f'tmp/meta.json', 'a') as f:
                    f.write(json.dumps({**{'expname': expname, \
                                            'potential_learn_cnt': potentialagent.potential_learn_cnt, \
                                            'at episode': i}}))

            print(expname, 'episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, \
                'pf_learn_cnt', potentialagent.potential_learn_cnt, 'final distance', env.distance_to_goal())
            if evaluate:
                print(reward_in_episode)

        # finish training:
        if not load_checkpoint:
            np.save(f"results/rewards_average_{expname}.npy", avg_score_history)
            agent.save_models(exp=expname)
            potentialagent.save_models(exp=expname)
            x = [i+1 for i in range(n_games)]
            plot_learning_curve(x, score_history, figure_file=f'apf_ddpg_{expname}.png')

        potentialagent.memory_traj.get_nlargest_trajectories(N_good=1)

        env.closeSim()
        print("Finish training!")

    except (Exception, KeyboardInterrupt) as error:
        print('\nTraining exited early.')
        print(error)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        breakpoint()
        env.closeSim()

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import sys
import torch as T

sys.path.insert(1, '../util/')
sys.path.insert(1, '../env/')

import DRL
from RecoveryAviary import RecoveryAviary
import numpy as np
import signal

RPM = ActionType('rpm')
KIN = ObservationType('kin')

def run(agent, env):

    num_games   = 1000 # number of total games to be run
    avg_size    = 10 # number of samples used in running average

    for i in range(num_games):
        score = 0

        # obs: Box() of shape (NUM_DRONES,12)
        obs, info = env.reset()

        term, trunc = False, False

        while not term and not trunc:

            # get action and value data for current state
            obs = np.reshape(obs, (-1, 12))[0]

            action = agent.choose_action(T.FloatTensor(obs))

            obs_, reward, term, trunc, info = env.step(np.reshape(action, (1, 4)))
            obs_ = np.reshape(obs_, (-1, 12))[0]

            # add memory
            agent.memory.append(obs, obs_, reward, term, clip=True)

            # learn
            agent.learn()

            # update observation
            obs = obs_
            score += reward

            if term or trunc:
                break

        print(f"Episode: {i + 1}\tReward: {score:.2f}\tEps: {agent.epsilon:.3f}")

        agent.scores.append(score)
        agent.avg_score = np.mean(agent.scores[-avg_size:])
        agent.avgs.append(agent.avg_score)
        agent.epsilons.append(agent.epsilon)

        # save network weights after improving
        '''
        if agent.avg_score > agent.max_score:
            agent.save_models()
            agent.max_score = agent.avg_score
            '''

    agent.save_stats()


def save_training_data(sig, frame):

    global agent

    #print(f'velo: ({agent.memory.minv}, {agent.memory.maxv}) avel: ({agent.memory.mina}, {agent.memory.maxa})')
    print('\n\nClosing training...')

    agent.save_stats()

    print('\n')
    sys.exit()

sys.path.insert(1, '../ppo/')
from ppo import PPO
if __name__ == '__main__':
    signal.signal(signal.SIGINT, save_training_data)

    env = RecoveryAviary(act=ActionType.RPM, obs=ObservationType.KIN, gui=True, ctrl_freq=480, pyb_freq=480)
    agent = PPO(env, use_network=True)
    agent.learn(100000)

    agent.save_stats()
    sys.exit()


    env = RecoveryAviary.RecoveryAviary(act=RPM, obs=KIN, gui=True)
    agent = DRL.Agent(explore=False, batch_size=16, lr_critic=1e-3, lr_actor=1e-3)

    run(agent, env)
    
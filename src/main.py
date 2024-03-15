from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import sys

sys.path.insert(1, '../util/')
sys.path.insert(1, '../env/')

import DRL
import RecoveryAviary
import numpy as np
import signal

RPM = ActionType('rpm')
KIN = ObservationType('kin')

def run(agent, env):

    num_games   = 10 # number of total games to be run

    for i in range(num_games):
        score = 0

        # obs: Box() of shape (NUM_DRONES,12)
        obs, info = env.reset()
        term, trunc = False, False

        while not term and not trunc:
            # choose action
            action = agent.choose_action()

            # get reward, new observation
            obs_, reward, term, trunc, info = env.step(action)
            score += reward

            # store memory and learn
            agent.memory.append(obs, action, reward, obs_, term or trunc)
            agent.learn()

            # update observation
            obs = obs_

        agent.scores.append(score)
        agent.avg_score = np.mean(agent.scores[-agent.avg_size:])
        agent.avgs.append(agent.avg_score)

        # save network weights after improving
        if agent.avg_score > agent.max_score:
            agent.save_weights()
            agent.max_score = agent.avg_score


def save_training_data(sig, frame):
    print('Closing training...')

    global agent
    agent.save_stats()
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, save_training_data)

    env = RecoveryAviary.RecoveryAviary(act=RPM, obs=KIN, gui=True)
    agent = DRL.Agent()
    run(agent, env)
    
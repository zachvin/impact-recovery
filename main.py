# Zach Vincent
# main.py
# Controls PPO training

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

from RecoveryAviary import RecoveryAviary
from ppo import PPO

import numpy as np
import sys
import signal
import argparse

# argument parsing
parser = argparse.ArgumentParser(
    prog='main.py',
    description='Runs PPO agent in quadcopter environment'
)

parser.add_argument('--num_epochs', '-n', help='number of epochs to run',
                    type=int)
parser.add_argument('--eval', '-e', help='whether to evaluate network',
                    action='store_true')
parser.add_argument('--checkpoints', '-c', help='whether to use trained networks',
                    action='store_true')
parser.add_argument('--plot', '-p', help='whether to save and plot output',
                    action='store_true')

args = parser.parse_args()

def end_training(sig, frame):
    """
    Signal handler saves training data on interrupt.
    """

    print('\n\nClosing training...')

    # ask to save/plot training data
    plot = True
    if args.plot != True:
        resp = input('Save training data? [y/N]: ')
        if 'y' not in resp.lower():
            print('Not saving data.')
            plot = False

    # ask to save trained networks
    networks = True
    resp = input('Save networks? [y/N]: ')
    if 'y' not in resp.lower():
        print('Not saving networks.')
        networks = False

    global agent
    agent.save_stats(plot, networks)

    print('Agent hyperparameters:')
    print(f'\tgamma {agent.gamma}')
    print(f'\tlambda {agent.lam}')
    print(f'\tentropy coeff {agent.entropy_coefficient}')


    print('\n')
    sys.exit()


if __name__ == '__main__':
    # SIMULATION CONTROL
    ctrl_freq = 240
    pyb_freq = 240
    initial_xyzs = np.expand_dims(np.random.rand(3), 0)
    initial_xyzs = np.array([[0,0,0]])
    eval = args.eval if args.eval else False
    use_checkpoint = args.checkpoints if args.checkpoints else False

    # HYPERPARAMETERS
    entropy_coefficient = 0.5 # make higher if converging on local min
    lr = 0.1

    # OTHER
    act = ActionType.RPM
    obs = ObservationType.KIN

    signal.signal(signal.SIGINT, end_training)

    # SETUP
    env = RecoveryAviary(act=act, obs=obs, gui=eval, ctrl_freq=ctrl_freq,
                         pyb_freq=pyb_freq, initial_xyzs=initial_xyzs)
    
    agent = PPO(env, eval=eval, use_checkpoint=use_checkpoint,
                entropy_coefficient=entropy_coefficient, lr=lr)
    
    num_epochs = args.num_epochs if args.num_epochs else 100
    print(f'Starting {num_epochs}')
    agent.learn(num_epochs)

    end_training(None, None)
    sys.exit()
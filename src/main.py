from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import sys
import torch as T

sys.path.insert(1, '../util/')
sys.path.insert(1, '../env/')

from RecoveryAviary import RecoveryAviary
import numpy as np
import signal
import argparse

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

args = parser.parse_args()

def sig_handler(sig, frame):
    """
    Signal handler saves training data on interrupt.
    """

    print('\n\nClosing training...')

    global agent
    agent.save_stats()

    print('\n')
    sys.exit()

sys.path.insert(1, '../ppo/')
from ppo import PPO

if __name__ == '__main__':
    # SIMULATION CONTROL
    ctrl_freq = 240
    pyb_freq = 240
    initial_xyzs = np.expand_dims(np.random.rand(3), 0)
    initial_xyzs = np.array([[0,0,0]])
    eval = args.eval if args.eval else False
    use_checkpoint = args.checkpoints if args.checkpoints else False

    # HYPERPARAMETERS
    entropy_coefficient = 0.01 # make higher if converging on local min
    lr = 0.01

    # OTHER
    act = ActionType.RPM
    obs = ObservationType.KIN

    signal.signal(signal.SIGINT, sig_handler)

    # SETUP
    env = RecoveryAviary(act=act, obs=obs, gui=eval, ctrl_freq=ctrl_freq,
                         pyb_freq=pyb_freq, initial_xyzs=initial_xyzs)
    
    agent = PPO(env, eval=eval, use_checkpoint=use_checkpoint,
                entropy_coefficient=entropy_coefficient, lr=lr)
    
    num_epochs = args.num_epochs if args.num_epochs else 100
    print(f'Starting {num_epochs}')
    agent.learn(num_epochs)

    agent.save_stats()
    sys.exit()
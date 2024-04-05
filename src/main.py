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
    ctrl_freq = 480
    pyb_freq = 480
    initial_xyzs = np.expand_dims(np.random.rand(3), 0)
    eval = True
    use_checkpoint = True

    # HYPERPARAMETERS
    entropy_coefficient = 0.005 # make higher if converging on local min

    # OTHER
    act = ActionType.RPM
    obs = ObservationType.KIN

    signal.signal(signal.SIGINT, sig_handler)

    # SETUP
    env = RecoveryAviary(act=act, obs=obs, gui=eval, ctrl_freq=ctrl_freq,
                         pyb_freq=pyb_freq, initial_xyzs=initial_xyzs)
    
    agent = PPO(env, eval=eval, use_checkpoint=use_checkpoint, entropy_coefficient=0.005)
    agent.learn(1000000)

    agent.save_stats()
    sys.exit()
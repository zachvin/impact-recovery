# Zach Vincent
# main.py
# Controls PPO training

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

from RecoveryAviary import RecoveryAviary
from ppo import PPO
from util import gen_random_position, gen_random_orientation

from tqdm import tqdm
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

    # dump hyperparameters
    tqdm.write('Agent hyperparameters:')
    tqdm.write(f'\tlearning rate (actor) {agent.max_a_lr}')
    tqdm.write(f'\tlearning rate (critic) {agent.max_c_lr}')
    tqdm.write(f'\tgamma {agent.gamma}')
    tqdm.write(f'\tlambda {agent.lam}')
    tqdm.write(f'\tentropy coeff {agent.entropy_coefficient}')

    tqdm.write('\n')
    sys.exit()

def quicksave(sig, frame):
    tqdm.write('========== Quicksaving ==========')
    global agent
    agent._save_networks()
    tqdm.write('============= Done. =============')


if __name__ == '__main__':
    # SIMULATION CONTROL
    ctrl_freq = 60
    pyb_freq = 120

    eval = args.eval if args.eval else False
    use_checkpoint = args.checkpoints if args.checkpoints else False

    # HYPERPARAMETERS
    entropy_coefficient = 0.01 # 0 -> 0.01
    a_lr = 3e-4 # 0.003 or lower
    c_lr = 3e-4
    clip = 0.2
    gamma = 0.99
    upi = 5
    epb = 5

    # OTHER
    act = ActionType.RPM
    obs = ObservationType.KIN

    signal.signal(signal.SIGINT, end_training)
    signal.signal(signal.SIGUSR1, quicksave) # kill -10

    # SETUP
    env = RecoveryAviary(act=act, obs=obs, gui=eval, ctrl_freq=ctrl_freq,
                         pyb_freq=pyb_freq)
    
    agent = PPO(env, eval=eval, use_checkpoint=use_checkpoint,
                entropy_coefficient=entropy_coefficient, a_lr=a_lr,
                c_lr=c_lr, clip=clip, gamma=gamma, upi=upi, epb=epb)
    
    num_epochs = args.num_epochs if args.num_epochs else 100
    print(f'Starting {num_epochs}')
    agent.learn(num_epochs)

    end_training(None, None)
    sys.exit()
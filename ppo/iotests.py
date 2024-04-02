from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import numpy as np
import sys
import time

sys.path.insert(1, '../env/')
from RecoveryAviary import RecoveryAviary

env = RecoveryAviary(act=ActionType.RPM, obs=ObservationType.KIN, gui=True)
num_games = 10
np.linalg.norm

# right, close, left, far
action = np.array([[0.2,0.2,0.2,0.2]])
action = np.random.rand(1,4)
for i in range(num_games):
    obs, info = env.reset()

    term, trunc = False, False
    rewards = []
    while not term and not trunc:
        obs_, reward, term, trunc, info = env.step(action)
        print(f'rotation {np.linalg.norm(env._getDroneStateVector(0)[7:10]):.2f}, reward {reward:.2f}')
        rewards.append(reward)
        time.sleep(0.1)
    print(f'\treward sum: {sum(rewards)}')
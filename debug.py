# Zach Vincent
# debug.py
# Tools for debugging and visualizing PPO

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def visualize_critic(xmin, xmax, y, z):
    #pos     = state[0:3]
    #quat    = state[3:7]
    #rpy     = state[7:10]
    #vel     = state[10:13]
    #ang_v   = state[13:16]
    #act     = state[16:20]
    #obs     = obs[0:3], obs[7:10], obs[10:13], obs[13:16]

    REAL = '#A34700'
    OBSERVED = '#FFBA85'

    critic = nn.Sequential(
        nn.Linear(12, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )
    critic.load_state_dict(torch.load(f'networks/low_critic_0.08'))

    x_vals = np.linspace(xmin, xmax, 100)
    rewards = np.zeros(100)
    real = np.zeros(100)
    
    for i,x in enumerate(x_vals):
        state = np.zeros(12)

        state[0] = x
        state[1] = y
        state[2] = z
        V = critic(torch.tensor(state, dtype=torch.float))
        rewards[i] = V.item()

        pos_reward = 2 - np.abs(np.linalg.norm(np.array([0,0,1])-state[0:3]))
        pos_reward = max(pos_reward, 0)
        real[i] = pos_reward

    plt.plot(x_vals, rewards, color=OBSERVED, label='observed rewards')
    plt.plot(x_vals, real, color=REAL, label='real rewards')
    plt.legend()
    plt.title(f'reward map for y={y}, z={z}')
    plt.savefig(f'debug/debug_{xmin}_{xmax}_{y:.1f}_{z}.png')
    plt.clf()

    print(f'saving debug/debug_{xmin}_{xmax}_{y:.1f}_{z}.png')

if __name__ == '__main__':

    for y in np.linspace(-1, 1, 10):
        visualize_critic(-1, 1, y, 0)
from util import plot_from_json

import json
import numpy as np

class Memory():
    def __init__(self):
        """
        Memory object abstracts difficulties with minibatches.
        """

        # everything we need for learning
        self.obs = []
        self.acts = []
        self.log_probs = []
        self.advantages = []
        self.rewards_to_go = []

        self.num_minibatches = 0
        self.inds = []

    def append(self, obs, act, log_prob, adv, rtg):
        """
        Append a new memory to the buffer.
        """

        self.obs.append(obs)
        self.acts.append(act)
        self.log_probs.append(log_prob)
        self.advantages.append(adv)
        self.rewards_to_go.append(rtg)

    def sample(self):
        """
        Sample random selection of memories from the buffer.
        """

        chunk = len(self.obs) // self.num_minibatches
        if max_mem < size:
            print('[ERR] Requested too many memories from buffer.')
            return None
        
        i = np.random.choice(max_mem, size=size, replace=False)
        

class Stats():
    """
    Stats object saves all training statistics.
    """

    def __init__(self):
        self.avgs = []
        self.scores = []
        self.a_lrs = []
        self.c_lrs = []
        self.ep_num = 0
        self.avg_score = -np.inf
        self.best_avg = -np.inf

        self.actor_losses = []
        self.critic_losses = []
        self.loss_epochs = []

    def append(self, score, a_lr, c_lr):
        """
        Append a new statistic set to memory.
        """

        self.scores.append(score)
        self.avgs.append(np.mean(self.scores[-10:]))
        self.a_lrs.append(a_lr)
        self.c_lrs.append(c_lr)
        self.avg_score = np.mean(self.scores[-20:])

    def append_loss(self, actor, critic):
        """
        Append new loss statistic set to memory.
        """
        self.actor_losses.append(actor.item())
        self.critic_losses.append(critic.item())
        self.loss_epochs.append(self.ep_num)
        
    def save(self):
        """
        Save statistics to JSON and plot.
        """

        # turn data into dictionary
        data = {
            'avgs': self.avgs,
            'scores': self.scores,
            'a_lrs': self.a_lrs,
            'c_lrs': self.c_lrs,

            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'loss_epochs': self.loss_epochs
        }

        fname = f'data/training_data_{int(self.avg_score)}.json'

        print(f'\nWriting to {fname}... ', end='')

        # save dictionary in json
        data_json = json.dumps(data)
        with open(fname, 'w') as f:
            f.write(data_json)

        print('done.')

        # plot json
        try:
            plot_from_json(f'data/training_data_{int(self.avg_score)}.json',
                            f'plots/{int(self.avg_score)}.png')
        except:
            print('[ERR] Plotting failed.')

from util import plot_from_json

import json
import numpy as np

class Memory():
    def __init__(self):
        """
        Memory object abstracts difficulties with minibatches.
        """

        # everything we need to learn
        self.obs = []
        self.acts = []
        self.log_probs = []
        self.advantages = []
        self.rewards_to_go = []

        self.mem_size = 5000
        self.i = 0

    def append(self, obs, act, log_prob, adv, rtg):
        """
        Append a new memory to the buffer.
        """

        i = self.i % self.mem_size

        self.obs[i] = obs
        self.acts[i] = act
        self.log_probs[i] = log_prob
        self.advantages[i] = adv
        self.rewards_to_go[i] = rtg

    def sample(self, size=1000):
        """
        Sample random selection of memories from the buffer.
        """
        max_mem = min(self.i, self.mem_size)
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

    def append(self, score, a_lr, c_lr):
        """
        Append a new statistic set to memory.
        """

        self.scores.append(score)
        self.avgs.append(np.mean(self.scores[-10:]))
        self.a_lrs.append(a_lr)
        self.c_lrs.append(c_lr)
        self.avg_score = np.mean(self.scores[-20:])
        
    def save(self):
        """
        Save statistics to JSON and plot.
        """

        # turn data into dictionary
        data = {
            'avgs': self.avgs,
            'scores': self.scores,
            'a_lrs': self.a_lrs,
            'c_lrs': self.c_lrs
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

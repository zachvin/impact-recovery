import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm

# color setup
DARK = False

TEXT = '#F3F3F4'
AVG = '#4B9DFB'
SCORE = '#AFD2FE'
LR = '#AB6D5F'
BACK = '#081821'

if not DARK:
    TEXT = '#081821'
    AVG = '#28536B'
    SCORE = '#377495'
    BACK = 'white'


def plot_from_json(src, dst):
    """
    Plot data from json file.
    
    Params:
        src : str
            json filename
        dst : str
            png filename
    """
    print(f'Creating plot of {src} at {dst}...', end=' ')
    with open(src, 'r') as s:
        data = json.load(s)

        x = range(len(data['avgs']))

        plt.figure(facecolor=BACK)

        fig, ax1 = plt.subplots()

        fig.set_facecolor(BACK)

        ax1.set_xlabel('Epoch', color=TEXT)
        ax1.set_ylabel('Score', color=SCORE)
        ax1.plot(x, data['scores'], label='Score', color=SCORE)
        ax1.plot(x, data['avgs'], label='Average score', color=AVG)
        ax1.tick_params(axis='y', labelcolor=SCORE)
        ax1.tick_params(axis='x', labelcolor=TEXT)
        ax1.set_facecolor(BACK)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning rate', color=LR)
        ax2.plot(x, data['a_lrs'], label='Learning rate (actor)', color=LR)
        ax2.plot(x, data['c_lrs'], label='Learning rate (critic)', color=LR)
        ax2.tick_params(axis='y', labelcolor=LR)

        title = src.split('/')[-1].split('.')[0]
        fig.suptitle(f'Model learning curve ({title})', color=TEXT)
        fig.tight_layout()
        fig.legend()

        plt.savefig(dst)

    print(f'done.')

def gen_random_position(scale=1):
    pos = np.random.rand(3)*scale
    pos[0] = (pos[0] - (scale/2)) * 2
    pos[1] = (pos[1] - (scale/2)) * 2

    return np.expand_dims(pos, 0)

def gen_random_orientation(scale=0.2):
    ori = np.random.rand(3)
    ori[0:2] = ori[0:2] * scale
    ori[2] = ori[2] * 2 * np.pi

    return np.expand_dims(ori, 0)


class SurfaceExplorer():
    """
    Creates a set of start locations that represent large portion of observation
    space.
    """
    def __init__(self):
        self.i = 0
        self.res = 8

        self.x = 1
        self.y = 1
        self.z = 1
        
        x_range = np.linspace(-self.x, self.x, self.res)
        y_range = np.linspace(-self.y, self.y, self.res)
        z_range = np.linspace(0, self.z, self.res//2)

        self.locs = []
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    self.locs.append(np.array([x, y, z]))

    def get_loc(self):
        try:
            return np.expand_dims(self.locs.pop(0), 0)
        except:
            tqdm.write('[WARN] Out of exploration locations')
            return np.array([[0,0,0]])




if __name__ == '__main__':
    num = -58
    plot_from_json(f'data/training_data_{num}.json', f'plots/{num}.png')
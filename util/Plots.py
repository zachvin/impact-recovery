import matplotlib.pyplot as plt
import json
import os

CHARCOAL    = '#28536B'
BLUE        = '#377495'
REDWOOD     = '#AB6D5F'

def plot_from_json(src, dst):
    print(f'Creating plot of {src} at {dst}...', end=' ')
    with open(src, 'r') as s:
        data = json.load(s)

        x = range(len(data['avgs']))

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Score', color=CHARCOAL)
        ax1.plot(x, data['scores'], ':', label='Score', color=BLUE)
        ax1.plot(x, data['avgs'], label='Average score', color=CHARCOAL)
        ax1.tick_params(axis='y', labelcolor=CHARCOAL)
        ax1.set_ylim([0, max(data['scores'])])

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon', color=REDWOOD)
        ax2.plot(x, data['epsilons'], label='Epsilon', color=REDWOOD)
        ax2.tick_params(axis='y', labelcolor=REDWOOD)
        ax2.set_ylim([0,1])

        title = src.split('/')[-1].split('.')[0]
        fig.suptitle(f'Model learning curve ({title})')
        fig.tight_layout()
        fig.legend()

        plt.savefig(dst)

    print(f'done.')

if __name__ == '__main__':
    num = 301
    plot_from_json(f'data/training_data_{num}.json', f'plots/{num}.png')
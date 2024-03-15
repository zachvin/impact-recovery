import matplotlib.pyplot as plt
import json

def plot_from_json(src, dst):
    print(f'Creating plot of {src} at {dst}...', end=' ')
    with open(src, 'r') as s:
        data = json.load(s)

        x = range(len(data['avgs']))
        plt.plot(x, data['avgs'])
        plt.plot(x, data['scores'])

        plt.savefig(dst)
    print(f'done.')
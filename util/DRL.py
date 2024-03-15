import numpy as np
import Memory
import json

class Agent():
        
    def __init__(self, epsilon=1):
        # objects
        self.memory = Memory.Memory(state_shape=12)

        # stats
        self.avgs = []
        self.scores = []

        self.max_score = -np.inf
        self.avg_score = -np.inf

        # hyperparemeters
        self.epsilon = epsilon

    def choose_action(self):
        """
        Chooses an action at random or from neural network depending on current
        epsilon value.
        Returns:
            np.array
                Set of values corresponding to motor RPMs
        """
        if np.random.rand() < self.epsilon:
            # choose random
            #return np.random.rand(1,4)
            return np.array([[.7, .7, .7, .7]])
        else:
            # choose best action
            return np.zeros((1,4))
        
    def learn(self):
        pass


    def save_stats(self):
        if self.avg_score == -np.inf:
            print('No data to be saved.')
            return

        fname = f'training_data_{int(self.avg_score)}.json'
    
        data = {
            'avgs': self.avgs,
            'scores': self.scores
        }

        data_json = json.dumps(data)
        with open(fname, 'w') as f:
            f.write(data_json)
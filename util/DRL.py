import numpy as np
import torch as T
import torch.nn as nn
import Memory
import json

class Agent():
        
    def __init__(self, epsilon=1, batch_size=16, tau=0.001, gamma=0.99,
                 min_epsilon=0.01, lr_actor=1e-3, lr_critic=1e-3,
                 explore=False, learn_threshold=50):
        """
        Initializes a deep reinforcement learning model agent.
        Params
            epsilon : float | 1, optional
                Describes beginning epsilon value for training
            batch_size : int | 32, optional
        Returns
            Agent object
        """

        # objects
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.memory = Memory.Memory(device=self.device, state_shape=12)

        self.actor = nn.Sequential(
            nn.Linear(12, 50),
            #nn.ReLU(),
            #nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 4),
            nn.Tanh()  # Ensures action outputs are between -1 and 1
        )
        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = nn.Sequential(
            nn.Linear(12, 50),
            #nn.ReLU(),
            #nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)  # Outputting a single value for the state value
        )
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # stats
        self.avgs = []
        self.scores = []

        self.max_score = -np.inf
        self.avg_score = -np.inf

        # hyperparameters
        self.epsilon = epsilon
        if not explore:
            self.epsilon = 0
        self.min_epsilon = min_epsilon
        self.epsilon_dec_rate = 0.001
        self.explore = explore

        self.batch_size = batch_size
        self.learn_threshold = learn_threshold
        self.tau = tau
        self.gamma = gamma

        # other vars
        self.rewards = []
        self.epsilons = []

    def choose_action(self, state):
        """
        Chooses an action at random or from neural network depending on current
        epsilon value.
        Returns:
            np.array
                Set of values corresponding to motor RPMs
            float
                Expected reward from given state
        """
        
        action = None
        if self.explore and np.random.rand() < self.epsilon:
            # choose random
            action = (np.random.rand(1,4) - 0.5) * 20

        else:
            # choose learned action
            action = self.actor(state).detach().numpy()

        if self.explore: self._decrement_epsilon()
        return action
        
    def back(self, obs, reward, value, value_, done):
        target = reward + self.gamma * value_ * (1-done)
        adv = target - value

        actor_loss = -T.log(self.actor(obs)) * adv.detach()
        critic_loss = adv.pow(2)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.sum().backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def _decrement_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon-self.epsilon_dec_rate)
        
    def learn(self):
        """
        Neural network back propagation based on past experiences from memory
        buffer sampling.
        """
        # determine if there are enough memories to learn
        if self.memory.i < self.batch_size:
            return
                
        # get memory
        obs, obs_, reward, done = self.memory.sample(self.batch_size)
        obs = T.FloatTensor(obs)
        obs_ = T.FloatTensor(obs_)

        for i in range(len(obs)):
            # calculate values
            value = self.critic(obs[i])
            value_ = self.critic(obs_[i])

            # self.back()
            self.back(obs[i], reward[i], value, value_, done[i])
        

    def soft_target_update(self):
        """
        Soft updates current network with target network weights according to:
        theta' = tau*theta + (1-tau)*theta'
        where theta' is the target_net weights and theta is the policy_net
        weights.
        """

        policy_net_weights = self.policy.state_dict()
        target_net_weights = self.target.state_dict()
        for key in policy_net_weights:
            target_net_weights[key] = self.tau*policy_net_weights[key] + \
                (1-self.tau)*target_net_weights[key]
            
        self.target.load_state_dict(target_net_weights)        

    def save_models(self):
        """
        Saves current state dict for policy and target nets.
        """

        self.policy_net.save()
        self.target_net.save()

    def load_models(self):
        """
        Loads state dicts for policy and target nets.
        """

        self.policy_net.load('policy_net')
        self.target_net.load('target_net')

    def save_stats(self):
        """
        Saves training statistics when training is completed or interrupted.
        """

        resp = input('Save training data? [y/N]: ')

        if 'y' not in resp.lower():
            print('Not saving data.')
            return

        if self.avg_score == -np.inf:
            print('No data to be saved.')
            return

        fname = f'../data/training_data_{int(self.avg_score)}.json'
    
        data = {
            'avgs': self.avgs,
            'scores': self.scores,
            'epsilons': self.epsilons
        }

        print(f'Writing to {fname}... ', end='')

        data_json = json.dumps(data)
        with open(fname, 'w') as f:
            f.write(data_json)

        print('done.')
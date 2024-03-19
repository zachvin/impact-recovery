import numpy as np
import torch as T
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import namedtuple
import Network
import Memory
import json

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Agent():
        
    def __init__(self, epsilon=1, batch_size=32, replace=50, tau=0.001,
                 gamma=0.99, min_epsilon=0.01, lr_actor=1e-3, lr_critic=1e-3,
                 explore=False):
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
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 4),
            nn.Tanh()  # Ensures action outputs are between -1 and 1
        )
        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = nn.Sequential(
            nn.Linear(12, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
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
        self.min_epsilon = min_epsilon
        self.epsilon_dec_rate = 0.001
        self.explore = explore

        self.batch_size = batch_size
        self.replace = replace
        self.tau = tau
        self.gamma = gamma

        # other vars
        self.learn_counter = 0
        self.saved_actions = []
        self.rewards = []

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
            action = np.random.rand(1,4)

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
        return

        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = T.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.epsilon)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, T.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = T.stack(policy_losses).sum() + T.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


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
        

    def hard_target_update(self):
        """
        Copies current network to target network.
        """
        if self.learn_counter % self.replace == 0:
            self.target.load_state_dict(self.policy.state_dict())
        

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
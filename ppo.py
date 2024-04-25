from torch.distributions import MultivariateNormal
import torch
import torch.nn as nn

import numpy as np
from util import plot_from_json
import json
import time
from tqdm import tqdm

class Stats():
    def __init__(self):
        self.current_average = -np.inf
        self.average_scores = []
        self.timesteps = []
        self.episode_nums = []

class PPO():
    def __init__(self, env, eval=False, use_checkpoint=False,
                 entropy_coefficient=0.005, lr=0.01):
        # HYPERPARAMETERS
        self.epochs_per_batch = 4
        self.gamma = 0.95
        self.lam = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.action_clip = 0.2
        self.lr = lr
        self.entropy_coefficient = entropy_coefficient
        self.use_checkpoint = use_checkpoint

        # SIMULATION CONTROL
        self.env = env
        self.obs_dim = 12
        self.act_dim = 4
        self.eval = eval

        # STATISTICS
        self.avgs = []
        self.scores = []
        self.epsilons = []
        self.avg_score = 0
        self.ep_num = 0

        # OTHER
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        print(f'Using device {self.device}')

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = torch.diag(self.cov_var)

        # ACTOR
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.act_dim),
            nn.Tanh(),
        )

        # CRITIC
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        if self.use_checkpoint:
            self.actor.load_state_dict(torch.load(f'state_dict_actor'))
            self.critic.load_state_dict(torch.load(f'state_dict_critic'))

        self.actor.to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic.to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def learn(self, target_episodes):
        """
        Runs the learning algorithm for target_episodes number of episodes.
        """

        # tqdm doesn't track to target_episodes since four episodes are run
        # in each rollout
        for batch_num in tqdm(range(target_episodes//self.epochs_per_batch)):
            # Get training data
            batch_obs, batch_acts, batch_log_probs, batch_rewards, batch_vals, \
                batch_dones = self.rollout()

            # don't learn if just evaluating networks
            if self.eval: continue

            # Find estimated values
            V, _, _ = self.evaluate(batch_obs, batch_acts)

            # Calculate and normalize generalized advantage estimate
            A_k = self.calculate_gae(batch_rewards, batch_vals, batch_dones)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()

            # Update network weights
            for _ in range(self.n_updates_per_iteration):

                # Calculate value of actions and log probabilities
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
                
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k

                entropy_loss = entropy.mean() * self.entropy_coefficient

                actor_loss = (-torch.min(surr1, surr2)).mean() - entropy_loss
                
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            # anneal learning rate
            factor = 1-(batch_num / (target_episodes//self.epochs_per_batch))
            self.lr = self.lr * factor

    def rollout(self):
        # batch statistics
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_vals = []
        batch_sums = []
        batch_dones = []

        # episode statistics
        ep_rewards = []
        ep_vals = []
        ep_dones = []

        for _ in range(self.epochs_per_batch):
            # episode statistics wiped each episode
            ep_rewards = []
            ep_vals = []
            ep_dones = []

            # randomize start location
            #self.env.INIT_XYZS = np.expand_dims(np.random.rand(3), 0)

            # initial observation
            obs, info = self.env.reset()
            obs = np.reshape(obs, (-1, 12))[0]
            term, trunc = False, False

            while not term and not trunc:
                if self.eval: time.sleep(0.001)

                # record done and obs for timestep t
                ep_dones.append(term or trunc)
                batch_obs.append(obs)

                # get action and val for timestep t
                action, log_prob = self.get_action(obs)
                action = np.reshape(action, (1,4))

                obs = torch.tensor(obs, dtype=torch.float, device=self.device)
                val = self.critic(obs)

                # get obs for timestep t+1
                obs, reward, term, trunc, _ = self.env.step(action)
                obs = np.reshape(obs, (-1, 12))[0]

                # store reward, action, log_probs for timestep t
                ep_rewards.append(reward)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

            self.scores.append(sum(ep_rewards))
            batch_sums.append(sum(ep_rewards))
            self.avg_score = np.mean(self.scores[-100:])
            self.avgs.append(self.avg_score)
            self.epsilons.append(0)

            # add episode data to batch
            batch_rewards.append(np.array(ep_rewards))
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

            tqdm.write(f'Episode {self.ep_num:04} reward: {sum(ep_rewards):.2f}\tavg reward: {np.mean(self.scores[-10:]):.2f}\tlr: {self.lr}')
            self.ep_num += 1

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float, device=self.device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float, device=self.device)

        # reward normalization
        #batch_rewards = np.array(batch_rewards)
        #batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-10)

        return batch_obs, batch_acts, batch_log_probs, batch_rewards, batch_vals, batch_dones
    

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):

            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)

        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        if isinstance(batch_obs, np.ndarray):
            batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs, dist.entropy()
    
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)

    def get_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample() # sample action from distribution
        log_prob = dist.log_prob(action) # using log_prob for network

        return action.cpu().detach().numpy() * self.action_clip, log_prob.cpu().detach()
    
    def save_stats(self, plot, networks):
        """
        Saves training statistics when training is completed or interrupted.
        """

        if plot:
            if self.avg_score == -np.inf:
                print('No data to be saved.')
            else:
                fname = f'data/training_data_{int(self.avg_score)}.json'
        
                data = {
                    'avgs': self.avgs,
                    'scores': self.scores,
                    'epsilons': self.epsilons
                }

                print(f'\nWriting to {fname}... ', end='')

                data_json = json.dumps(data)
                with open(fname, 'w') as f:
                    f.write(data_json)

                print('done.')

                plot_from_json(f'data/training_data_{int(self.avg_score)}.json',
                               f'plots/{int(self.avg_score)}.png')


        if networks:
            self._save_networks()        

    def _save_networks(self):
        """
        Save trained networks.
        """

        print('Saving networks... ', end='')
        torch.save(self.actor.state_dict(), f'networks/state_dict_actor')
        torch.save(self.critic.state_dict(), f'networks/state_dict_critic')

        print('done.')
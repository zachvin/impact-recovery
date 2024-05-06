# Zach Vincent
# ppo.py
# Implementation of Proximal Policy Optimization Algorithms (https://arxiv.org/abs/1707.06347) by Schulman, et al. 2017.
# Adapted from code by Eric Yang Yu (https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8).

from torch.distributions import MultivariateNormal
import torch
import torch.nn as nn
import argparse

import numpy as np
from util import gen_random_position, gen_random_orientation, SurfaceExplorer
import time
from tqdm import tqdm
from Record import Stats
import signal
import sys

class PPO():
    def __init__(self, env, eval=False, use_checkpoint=False,
                 entropy_coefficient=0.005, c_lr=0.1, lam=0.99,
                 gamma=0.99, clip=0.2, a_lr=0.001, obs_dim=12,
                 act_dim=4, anneal=False, upi=5, action_clip=1,
                 epb=5, save_every=50, num_minibatches=20, cp_a=None,
                 cp_c=None, pybullet=False):
        
        # HYPERPARAMETERS
        self.epochs_per_batch = epb
        self.gamma = gamma
        self.lam = lam
        self.n_updates_per_iteration = upi
        self.clip = clip
        self.action_clip = action_clip
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.max_a_lr = a_lr
        self.max_c_lr = c_lr
        self.entropy_coefficient = entropy_coefficient
        self.use_checkpoint = use_checkpoint
        self.anneal = anneal
        self.save_every = save_every
        self.num_minibatches = num_minibatches

        # SIMULATION CONTROL
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.eval = eval
        self.explorer = SurfaceExplorer()
        self.pybullet = pybullet

        # STATISTICS
        self.stats = Stats()

        # OTHER
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        print(f'Using device {self.device}')

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = torch.diag(self.cov_var)

        # ACTOR
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.act_dim),
        )

        # CRITIC
        # outputs value of a given state (expected reward)
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        actor_name = cp_a or 'good_actor'
        critic_name = cp_c or 'good_critic'
        if self.use_checkpoint:
            self.actor.load_state_dict(torch.load(f'networks/{actor_name}'))
            self.critic.load_state_dict(torch.load(f'networks/{critic_name}'))

        self.actor.to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.critic.to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

    def learn(self, target_episodes):
        """
        Runs the learning algorithm for target_episodes number of episodes.
        """

        # tqdm doesn't track to target_episodes since four episodes are run
        # per rollout
        for batch_num in tqdm(range(target_episodes//self.epochs_per_batch)):
            if batch_num > 0 and batch_num % self.save_every == 0:
                self._save_networks(suffix=f'_quicksave')

            # Get training data
            batch_obs, batch_acts, batch_log_probs, batch_rewards, batch_vals, \
                batch_dones, batch_rtgs = self.rollout()

            # don't learn if just evaluating networks
            if self.eval: continue

            # Find estimated values
            V, _, _ = self.evaluate(batch_obs, batch_acts)

            # Calculate and normalize generalized advantage estimate
            #A_k = self.calculate_gae(batch_rewards, batch_vals, batch_dones)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Update network weights
            critic_losses = []
            actor_losses = []
            for _ in range(self.n_updates_per_iteration):

                # create minibatches
                num_episodes_in_batch = sum([len(l) for l in batch_rewards])
                mb_len = num_episodes_in_batch // self.num_minibatches

                # randomize indices
                inds = np.arange(num_episodes_in_batch)
                np.random.shuffle(inds)

                for i in range(self.num_minibatches):
                    # get chunk of indices for first minibatch
                    start = mb_len*i
                    end = mb_len*(i+1)
                    mb_inds = inds[start:end]

                    # use chunk indices to make minibatches
                    mb_obs = batch_obs[mb_inds]
                    mb_acts = batch_acts[mb_inds]
                    mb_log_probs = batch_log_probs[mb_inds]
                    mb_rtgs = batch_rtgs[mb_inds]
                    mb_A_k = A_k[mb_inds]

                    # continue computing as normal
                    # Calculate value of actions and log probabilities
                    V, curr_log_probs, entropy = self.evaluate(mb_obs, mb_acts)
                    
                    ratios = torch.exp(curr_log_probs - mb_log_probs)

                    surr1 = ratios * mb_A_k
                    surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * mb_A_k

                    entropy_loss = entropy.mean() * self.entropy_coefficient

                    actor_loss = (-torch.min(surr1, surr2)).mean() - entropy_loss
                    
                    critic_loss = nn.MSELoss()(V, mb_rtgs)

                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

            # anneal learning rate
            if self.anneal:
                factor = 1-(batch_num / (target_episodes//self.epochs_per_batch))
                self.a_lr = self.max_a_lr * factor
                self.c_lr = self.max_c_lr * factor

            self.actor_optim.param_groups[0]["lr"] = self.a_lr
            self.critic_optim.param_groups[0]["lr"] = self.c_lr

            # print losses
            avg_actor_loss = sum(actor_losses)/len(actor_losses)
            avg_critic_loss = sum(critic_losses)/len(critic_losses)
            tqdm.write(f'Actor: {avg_actor_loss:.2f}\tCritic: {avg_critic_loss:.2f}')
            self.stats.append_loss(avg_actor_loss, avg_critic_loss)
            #tqdm.write(f'{self.critic[0].weight}')

    def rollout(self):
        # batch statistics
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_vals = []
        batch_dones = []
        batch_rtgs = []

        # episode statistics
        ep_rewards = []
        ep_vals = []
        ep_dones = []

        # start new episode
        for _ in range(self.epochs_per_batch):
            # episode statistics wiped each episode
            ep_rewards = []
            ep_vals = []
            ep_dones = []

            # randomize start location and position
            self.env.INIT_XYZS = gen_random_position()
            self.env.INIT_RPYS = gen_random_orientation()

            # initial observation
            obs, info = self.env.reset()
            obs = np.reshape(obs, (-1, self.obs_dim))[0]
            term, trunc = False, False

            # continue until episode ends
            while not term and not trunc:
                # slow GUI framerate for viewing
                if self.eval: time.sleep(0.001)

                batch_obs.append(obs)

                # get action and val for timestep t
                action, log_prob = self.get_action(obs)
                if self.pybullet: action = np.reshape(action, (1,self.act_dim))

                obs = torch.tensor(obs, dtype=torch.float, device=self.device)

                # get obs for timestep t+1
                obs, reward, term, trunc, _ = self.env.step(action)
                if trunc:
                    reward -= 100
                
                # store reward, action, log_probs for timestep t
                ep_dones.append(term or trunc)

                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

            self.stats.append(sum(ep_rewards), self.a_lr, self.c_lr)
            
            # add episode data to batch
            batch_rewards.append(ep_rewards)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

            # print episode statistics to screen
            tqdm.write(f'Episode {self.stats.ep_num:04} reward: {sum(ep_rewards):.2f}\tavg reward: {np.mean(self.stats.scores[-10:]):.2f}\talr: {self.a_lr:.6f}\tclr: {self.c_lr:.6f}')
            self.stats.ep_num += 1

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float, device=self.device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float, device=self.device)
        batch_rtgs = self.compute_rtgs(batch_rewards)

        return batch_obs, batch_acts, batch_log_probs, batch_rewards, batch_vals, batch_dones, batch_rtgs
    

    def compute_rtgs(self, batch_rewards):
        """
        Compute rewards-to-go for each step of each batch. RTGs are calculated
        as the sum of rewards remaining in the episode multiplied by a discount
        factor gamma.
        """
        batch_rtgs = []

        for ep_rewards in reversed(batch_rewards):

            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        return torch.tensor(np.array(batch_rtgs), dtype=torch.float, device=self.device)
    
    def evaluate(self, batch_obs, batch_acts):
        """
        Wrapper function for forward propagation in critic network.
        """
        V = self.critic(batch_obs).squeeze()

        if isinstance(batch_obs, np.ndarray):
            batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs, dist.entropy()
    
    def calculate_gae(self, rewards, values, dones):
        """
        Calculates generalized advantage estimation. Not used in Part 4 submission.
        """

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
        """
        Get mean action from probability distribution in actor network.
        """

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
            if self.stats.avg_score == -np.inf:
                tqdm.write('No data to be saved.')
            else:
                self.stats.save()


        if networks:
            self._save_networks()        

    def _save_networks(self, suffix=''):
        """
        Save trained networks.
        """

        tqdm.write(f'Saving networks at networks/state_dict_xxxxx{suffix}... ', end='')
        torch.save(self.actor.state_dict(), f'networks/state_dict_actor{suffix}')
        torch.save(self.critic.state_dict(), f'networks/state_dict_critic{suffix}')

        tqdm.write('done.')


if __name__ == '__main__':
    import gymnasium as gym

    # args
    parser = argparse.ArgumentParser(
        prog='ppo.py',
        description='Uses PPO on Gymnasium inverted pendulum.'
    )

    parser.add_argument('--num_epochs', '-n', help='number of epochs to run',
                        type=int)
    parser.add_argument('--eval', '-e', help='whether to evaluate network',
                        action='store_true')
    parser.add_argument('--checkpoints', '-c', help='whether to use trained networks',
                        action='store_true')

    args = parser.parse_args()

    eval = False if args.eval is None else args.eval
    use_checkpoint = False if args.checkpoints is None else args.checkpoints

    entropy_coefficient = 0.00 # 0 -> 0.01
    a_lr = 3e-4
    c_lr = 3e-4
    clip = 0.2
    gamma = 0.99
    upi = 10
    epb = 10
    anneal = False

    num_epochs = args.num_epochs or 2500

    env = None
    env = gym.make('Pendulum-v1')
    if eval:
        env = gym.make('Pendulum-v1', render_mode='human')

    print(env.observation_space.shape[0], env.action_space.shape[0])

    agent = PPO(env, eval=eval, use_checkpoint=use_checkpoint,
                entropy_coefficient=entropy_coefficient, a_lr=a_lr,
                c_lr=c_lr, clip=clip, obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.shape[0], gamma=gamma, upi=upi,
                epb=epb, cp_a='good_actor_pendulum', cp_c='good_actor_critic',
                anneal=anneal)
    
    def end_training(sig, frame):
        global agent
        agent._save_networks()
        sys.exit()

    signal.signal(signal.SIGINT, end_training)
    
    print(f'Starting {num_epochs}')
    agent.learn(num_epochs)
    agent._save_networks()
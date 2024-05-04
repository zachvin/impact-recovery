from torch.distributions import MultivariateNormal
import torch
import torch.nn as nn

import numpy as np
from util import gen_random_position, gen_random_orientation, SurfaceExplorer
import time
from tqdm import tqdm
from Record import Memory, Stats
import signal
import sys

class PPO():
    def __init__(self, env, eval=False, use_checkpoint=False,
                 entropy_coefficient=0.005, c_lr=0.1, lam=0.99,
                 gamma=0.99, clip=0.2, a_lr=0.001, obs_dim=12,
                 act_dim=4, anneal=False, upi=5, action_clip=1,
                 epb=5, save_every=50):
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

        # SIMULATION CONTROL
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.eval = eval
        self.explorer = SurfaceExplorer()

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
            nn.Linear(64, self.act_dim),
            #nn.Tanh(),
            #nn.Linear(64, self.act_dim),
            #nn.Tanh(),
        )

        # CRITIC
        # outputs value of a given state (expected reward)
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            #nn.Tanh(),
            #nn.Linear(64, 64),
            #nn.Tanh(),
            nn.Linear(64, 1),
        )

        if self.use_checkpoint:
            self.actor.load_state_dict(torch.load(f'networks/state_dict_actor.pth'))
            self.critic.load_state_dict(torch.load(f'networks/state_dict_critic.pth'))

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
                self._save_networks(suffix=f'{batch_num}')

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

            #V = self.critic(batch_obs).squeeze()
            #batch_rtgs = A_k + V.detach()

            #print(f'avg {torch.mean(A_k)}')

            # Update network weights
            critic_losses = []
            actor_losses = []
            for _ in range(self.n_updates_per_iteration):

                # Calculate value of actions and log probabilities
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
                
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k

                entropy_loss = entropy.mean() * self.entropy_coefficient

                actor_loss = (-torch.min(surr1, surr2)).mean() - entropy_loss
                
                critic_loss = nn.MSELoss()(V, batch_rtgs)

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
            tqdm.write(f'Actor: {sum(actor_losses)/len(actor_losses):.2f}\tCritic: {sum(critic_losses)/len(critic_losses):.2f}')
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
            #self.env.INIT_XYZS = self.explorer.get_loc()
            #self.env.INIT_RPYS = gen_random_orientation()

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
                action = np.reshape(action, (1,self.act_dim))

                obs = torch.tensor(obs, dtype=torch.float, device=self.device)
                #val = self.critic(obs)

                # get obs for timestep t+1
                obs, reward, term, trunc, _ = self.env.step(action)
                
                # store reward, action, log_probs for timestep t
                ep_dones.append(term or trunc)
                #ep_vals.append(val.flatten())

                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

            self.stats.append(sum(ep_rewards), self.a_lr, self.c_lr)
            
            # add episode data to batch
            batch_rewards.append(ep_rewards)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

            # print episode statistics to screen
            tqdm.write(f'Episode {self.stats.ep_num:04} reward: {sum(ep_rewards):.2f}\tavg reward: {np.mean(self.stats.scores[-10:]):.2f}\talr: {self.a_lr:.4f}\tclr: {self.c_lr:.4f}')
            self.stats.ep_num += 1

        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rtgs = self.compute_rtgs(batch_rewards)

        return batch_obs, batch_acts, batch_log_probs, batch_rewards, batch_vals, batch_dones, batch_rtgs
    

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

        tqdm.write(f'Saving networks at networks/state_dict_xxxxx_{suffix}...', end='')
        torch.save(self.actor.state_dict(), f'networks/state_dict_actor_{suffix}')
        torch.save(self.critic.state_dict(), f'networks/state_dict_critic_{suffix}')

        tqdm.write('done.')


import gymnasium as gym
if __name__ == '__main__':

    eval = False
    use_checkpoint = False

    entropy_coefficient = 0.00 # 0 -> 0.01
    a_lr = 3e-4
    c_lr = 3e-4
    clip = 0.2
    gamma = 0.99
    upi = 10
    epb = 4

    num_epochs = 2000

    env = None
    env = gym.make('Pendulum-v1')
    if eval:
        env = gym.make('Pendulum-v1', render_mode='human')

    print(env.observation_space.shape[0], env.action_space.shape[0])

    agent = PPO(env, eval=eval, use_checkpoint=use_checkpoint,
                entropy_coefficient=entropy_coefficient, a_lr=a_lr,
                c_lr=c_lr, clip=clip, obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.shape[0], gamma=gamma, upi=upi,
                epb=epb)
    
    def end_training(sig, frame):
        global agent
        agent._save_networks()
        sys.exit()

    signal.signal(signal.SIGINT, end_training)
    
    print(f'Starting {num_epochs}')
    agent.learn(num_epochs)
    agent._save_networks()
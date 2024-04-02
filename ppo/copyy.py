from torch.distributions import MultivariateNormal
import torch
import torch.nn as nn
import numpy as np
import json
import time

class ActorNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ActorNN, self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        return self.stack(obs)

class CriticNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CriticNN, self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        return self.stack(obs)

class PPO():
    def __init__(self, env, pendulum=False):
        self._init_hyperparameters()

        self.pendulum = pendulum

        self.env = env
        self.obs_dim = 12 if not self.pendulum else env.observation_space.shape[0]
        self.act_dim = 1 if not self.pendulum else env.action_space.shape[0]

        # statistics
        self.avgs = []
        self.scores = []
        self.epsilons = []
        self.avg_score = 0

        # chooses action
        self.actor = ActorNN(self.obs_dim, self.act_dim)
        # calculates reward
        self.critic = CriticNN(self.obs_dim, 1)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_timesteps):
        print('Starting batch learning... ', end='')
        t_so_far = 0
        losses = []

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t_so_far += np.sum(batch_lens)

            V, _ = self.evaluate(batch_obs, batch_acts)

            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                losses.append(actor_loss.detach().numpy())

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

        print('done.')

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_sums = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []

            obs, info = self.env.reset()
            if not self.pendulum: obs = np.reshape(obs, (-1, 12))[0]

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                if not self.pendulum: action = np.expand_dims(action, axis=0)

                obs, rew, term, trunc, _ = self.env.step(action)
                if not self.pendulum: obs = np.reshape(obs, (-1, 12))[0]
                time.sleep(0.05)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if term or trunc:
                    break

            #print(f'Episode reward: {sum(ep_rews)}')
            self.scores.append(sum(ep_rews))
            batch_sums.append(sum(ep_rews))
            self.avg_score = np.mean(self.scores[-50:])
            self.avgs.append(self.avg_score)
            self.epsilons.append(0)

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)

        print(f'Batch average reward: {np.mean(np.array(batch_sums)):.2f}')

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):

            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def get_action(self, obs):
        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample() # sample action from distribution
        log_prob = dist.log_prob(action) # using log_prob for network

        return action.detach().numpy(), log_prob.detach()

    def save_stats(self):
        """
        Saves training statistics when training is completed or interrupted.
        """

        resp = input('Save training data? [y/N]: ')

        if 'y' not in resp.lower():
            print('Not saving data.')
            self._save_networks()
            return

        if self.avg_score == -np.inf:
            print('No data to be saved.')
            return

        fname = f'training_data_{int(self.avg_score)}.json'

        data = {
            'avgs': self.avgs,
            'scores': self.scores,
            'epsilons': self.epsilons
        }

        print(f'Writing to {fname}... ', end='')

        data_json = json.dumps(data)
        with open(fname, 'w') as f:
            f.write(data_json)

        print('done.\n\n')

        self._save_networks()

    def _save_networks(self):
        resp = input('Save networks? [y/N]: ')

        if 'y' not in resp.lower():
            print('Not saving networks.')
            return

        print('Saving networks... ', end='')
        torch.save(self.actor.state_dict(), f'state_dict_actor')
        torch.save(self.critic.state_dict(), f'state_dict_critic')

        print('done.')

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 2
        self.clip = 0.2
        self.lr = 0.01

class RecoveryAviary(HoverAviary):
    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            return ret
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            print(ret)
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")

import gymnasium as gym
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

if __name__ == '__main__':
    env = RecoveryAviary(act=ActionType.ONE_D_RPM, obs=ObservationType.KIN, gui=True)
    model = PPO(env)
    model.learn(100000)

    model.save_stats()
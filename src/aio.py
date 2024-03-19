from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

RPM = ActionType('rpm')
KIN = ObservationType('kin')

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Ensures action outputs are between -1 and 1
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputting a single value for the state value
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

def train(env, model, optimizer_actor, optimizer_critic, max_episodes, gamma=0.99):
    for episode in range(max_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:        
            state = torch.FloatTensor(np.reshape(state, (-1, 12))[0])
            action, value = model(state)
            action = action.squeeze(0)
            value = value.squeeze(0)

            next_state, reward, term, trunc, _ = env.step(np.reshape(action.detach().numpy(), (1,4)) * 5)
            done = term or trunc
            next_state = torch.FloatTensor(np.reshape(next_state, (-1, 12))[0]).unsqueeze(0)
            next_value = model.critic(next_state).squeeze(0)

            target = reward + gamma * next_value * (1 - done)
            advantage = target - value

            # Actor loss
            actor_loss = -torch.log(model.actor(state)) * advantage.detach()

            # Critic loss
            critic_loss = advantage.pow(2)

            # Update networks
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            actor_loss.sum().backward()
            critic_loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()

            state = next_state.squeeze(0).numpy()
            episode_reward += reward

        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

# Example usage:
if __name__ == "__main__":
    # Define environment and model parameters
    input_dim = 12  # Assuming a state space of 4 dimensions
    action_dim = 4  # Four continuous actions
    hidden_dim = 64
    max_episodes = 1000
    gamma = 0.99
    learning_rate_actor = 1e-3
    learning_rate_critic = 1e-3

    # Create environment and model
    env = HoverAviary(act=RPM, obs=KIN, gui=True)  # Replace MyEnvironment() with your own environment class
    model = ActorCritic(input_dim, action_dim, hidden_dim)
    optimizer_actor = optim.Adam(model.actor.parameters(), lr=learning_rate_actor)
    optimizer_critic = optim.Adam(model.critic.parameters(), lr=learning_rate_critic)

    # Train the model
    train(env, model, optimizer_actor, optimizer_critic, max_episodes, gamma)

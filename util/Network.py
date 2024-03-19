import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, name):
        super(DNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(12, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 4)
        )

        self.name = name
        self.affine1 = nn.Linear(12, 64)

        # actor - returns 
        self.actor_head = nn.Linear(64, 4)

        # critic - returns scores
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        """
        Network forward propagation for both actor and critic.
        Returns:
            4 motor outputs and estimated value
        """

        x = F.relu(self.affine1(x))

        # actor returns probability distribution of actions for state s_t
        action_prob = F.softmax(self.actor_head(x), dim=-1)

        # critic evaluates state s_t
        state_value = self.critic_head(x)

        return action_prob, state_value
    
    def save(self):
        """
        Saves weights.
        """

        T.save(self.state_dict(), f'state_dict_{self.name}')

    def load(self):
        """
        Loads weights for evaluation.
        """

        self.load_state_dict(T.load(f'state_dict_{self.name}'))
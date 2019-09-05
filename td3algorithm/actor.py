import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        """Create the Actor neural network layers."""
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        """Forward pass in Actor neural network."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x

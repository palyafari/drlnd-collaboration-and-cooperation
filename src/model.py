import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=[128,128], use_bn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_units (array): Array of numbers of nodes in the hidden layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_units[0])
        self.layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                self.layers.append(nn.Linear(state_size, hidden_units[i]))
            else:
                self.layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
        self.layers.append(nn.Linear(hidden_units[-1], action_size))
        self.layers = nn.ModuleList(self.layers)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.layers)-1):
            self.layers[i].weight.data.uniform_(*hidden_init(self.layers[i]))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.use_bn and state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = state
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
            if self.use_bn and i== 0:
                x = self.bn(x)

        return F.tanh(self.layers[-1](x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=[128,128], use_bn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_units[0])
        self.layers = []
        for i in range(len(hidden_units)-1):
            if i == 0:
                self.layers.append(nn.Linear(state_size, hidden_units[i]))
            else:
                self.layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
        self.layers.append(nn.Linear(hidden_units[-2]+action_size, hidden_units[-1]))
        self.layers.append(nn.Linear(hidden_units[-1], 1))
        self.layers = nn.ModuleList(self.layers)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.layers)-1):
            self.layers[i].weight.data.uniform_(*hidden_init(self.layers[i]))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.use_bn and state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = state
        for i in range(len(self.layers)-2):
            x = F.relu(self.layers[i](x))
            if self.use_bn and i == 0:
                x = self.bn(x)

        x = torch.cat((x, action), dim=1)
        x = F.relu(self.layers[-2](x))
        return self.layers[-1](x)

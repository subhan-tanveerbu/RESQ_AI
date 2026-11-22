import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, observation_size, action_size=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.net(x)

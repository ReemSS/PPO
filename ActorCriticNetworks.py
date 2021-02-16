import torch
from torch import nn
import numpy as np

class SamplingNetworks(nn.Module):
    """
        Sampling Networks with 3 linear layers
    """
    def __init__(self, in_dim, out_dim):
        """
            Initialise the networks with input and output dimensions equal to the environment's dimensions
            :param in_dim:
            :param out_dim:
        """
        super(SamplingNetworks, self).__init__()

        self.layer1 = nn.Linear(in_dim, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        """
            Feeds the observation to the network
            :param obs: the observation of the enivronment
            :return: the output of the network, which is the estimated value of the given observations in case the
            critic network is, or
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        act1 = nn.ReLU(self.layer1(obs))
        act2 = nn.Relu(self.layer2(act1))
        out = nn.ReLU(self.layer3(act2))

        return out
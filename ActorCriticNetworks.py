import torch
from torch import nn
import numpy as np
from torch.distributions import MultivariateNormal
import torch.nn.functional as f

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
        # to give all the actions initially the same probability
        self.variable = torch.full(size=(in_dim,), fill_value=0.5)
        self.scalar_matrix = torch.diag(self.variable)

    def forward(self, obs, actor = False):
        """
            Feeds the observation to the network
            :param obs: the observation of the enivronment
            :return: the output of the network, which is the estimated value of the given observations in case the
            critic network is, or
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        act1 = f.relu(self.layer1(obs))
        act2 = f.relu(self.layer2(act1))
        out = f.relu(self.layer3(act2))

        # actor network we need the probability of the actions to count the ratio
        # if case of critic network we can ignore the return probabilties
        if actor:
            # see if other distribution work
            dist = MultivariateNormal(out, self.scalar_matrix)
            action = dist.sample()
            return action, dist.log_prob(action)



        return out

# actor = SamplingNetworks(3,4)
# print(actor.forward(np.array([1,4,5])))
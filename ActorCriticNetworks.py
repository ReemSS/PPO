import torch
from torch import nn
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F

class SamplingNetworks(nn.Module):
    """
        Sampling Networks with 3 linear layers using Relu as activation function also credits to
        https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea58146
    """
    def __init__(self, in_dim, out_dim):
        """
            Initialise the networks with input and output dimensions equal to the environment's dimensions
            :param in_dim:
            :param out_dim:
        """
        super(SamplingNetworks, self).__init__()
        # implemented an easy network in order to understand the work of the actor and critic networks
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        # to give all the actions initially the same probability, needed for continuous action spaces
        self.scalar = torch.full(size=(out_dim,), fill_value=0.5)
        self.scalar_matrix = torch.diag(self.scalar)

    def forward(self, obs, actor = False):
        """
            Feeds the observation to the network
            :param obs: the observation of the enivronment
            :return: the output of the network, which is the estimated value of the given observations in case the
            critic network is, or
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        act1 = F.relu(self.layer1(obs))
        act2 = F.relu(self.layer2(act1))
        # to sum the output probabilities to 1. Softmax is used here since it helps propagating multiclass information
        if actor:
            out = F.softmax(self.layer3(act2))
        else:
            out = torch.tanh(self.layer3(act2))
        return out

    def random_action(self, obs, env_continuous):
        # In Actor network we need the probability of the actions to count the ratio of

        # initially give all actions the same probability to allow the actor to explore, credits to
        # https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea58146
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        mean = self.forward(obs, actor=True)

        if env_continuous:
            # if the action space is continuous i.e. type of Box. The actions have a range of infinite values.
            # Therefore the probability distribution of the possible actions is continuous.
            # The common continuous distributions are the following:
            # Normal, Binomial, hypergeometric, Poisson, Beta, Cauchy, Exponential, Gamma, Logistic and Weibull.
            # The observation space has more than one dimension, then the distribution is a multivariate distribution,
            # In this case of uncertainty the normal distribution describes uncertain variables.
            # The mean here represents the most likely value of the uncertain actions (the first condition of the
            # underlying normal distribution.

            dist = MultivariateNormal(mean, self.scalar_matrix)
            action = dist.sample()
            entropy = dist.entropy()

        else:
            # The multinomial or the categorical distribution describes the random obs with many possible actions.
            dist = Categorical(mean)
            action = dist.sample()
            entropy = dist.entropy()

        return action.detach().numpy(), dist.log_prob(action), entropy.mean()

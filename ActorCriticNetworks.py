import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions.normal import Normal

class Actor(nn.Module):
    """
        Actor class to create a simple feedforward network, same as the critic network with a single difference is the
        number of the network output, to give tp each possible action a probability.
    """
    def __init__(self, obs_dim, act_dim, neuron_num, continuous):
        super().__init__()
        self.continuous = continuous
        if continuous:
            # Using the logarithm of the standard deviation to ensure that it won't take a negative value or a value
            # of zero. At the start all actions have a probability of 0.5.
            log_std = 0.5 * np.ones(act_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.network = nn.Sequential(nn.Linear(obs_dim, neuron_num),
                                     nn.Tanh(),
                                     nn.Linear(neuron_num, neuron_num),
                                     nn.Tanh(),
                                     nn.Linear(neuron_num, act_dim),
                                     nn.Identity())

    def _distribution(self, obs):
        """
            Internal function to create the sampling distribution.
        :param obs: the observations of the environment as input to the network
        :return: if the actions have discrete values, return a categorical distribution.
                If they have continuous values, a normal distribution is created.
        """
        mean = self.network(obs)
        if self.continuous:
            std = torch.exp(self.log_std)
            return Normal(mean, std)
        return Categorical(logits=mean)

    def _logprobability(self, pi, act):
        if self.continuous:
            return pi.log_prob(act).sum(axis=-1)
        return pi.log_prob(act)

    def forward(self, obs, action=None, sample=False):
        """
            Forward pss on the network
        :param obs: observations of the environment as input to the network.
        :param action: if an action is already sampled or not. It is needed since forward is used when interacting with
                       environment and when the policy parameters are updated.
        :param sample: if an action should be sampled or not.
        :return:
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        pi = self._distribution(obs)
        logp = None
        if action is not None:
            action = torch.as_tensor(action, dtype=torch.float32)
            logp = self._logprobability(pi, action)
        if sample:
            return pi.sample(), logp, pi.entropy()
        return pi, logp, pi.entropy()

class Critic(nn.Module):
    """
        Critic (FeedForward) network consists of three fully connected layers with two hyperbolic tangent function
        TanH between the first two layers as activation functions.
    """
    def __init__(self, obs_dim, neuron_num):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(obs_dim, neuron_num),
                                     nn.Tanh(),
                                     nn.Linear(neuron_num, neuron_num),
                                     nn.Tanh(),
                                     nn.Linear(neuron_num, 1),
                                     nn.Identity())

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.network(obs).view(-1)  # Critical to ensure v has right shape.

class ActorCritic(nn.Module):
    """
        Actor-Critic modules to build the networks of the actor and the critic.
    """

    def __init__(self, obs_dim, action_space, neurons_num, continuous):
        """
            Initliasing the networks.
        :param obs_dim: the input to the actor network and to the critic network.
        :param action_space: the output dimension of the actor network.
        :param neurons_num: the number of neurons in the layers of the network.
        :param continuous: two cases should be considered, when the action have discrete or continuous values.
        """
        super().__init__()
        self.actor = Actor(obs_dim, action_space, neurons_num, continuous)
        self.critic = Critic(obs_dim, neurons_num)

    def step(self, obs):
        """
            Enables to the actor to interact with the environment, and the critic to evaluate the actor's decisions.
        :param obs: the input to the networks
        :return: return the probability of the possible actors, the feedback to be in the current state and the
            logarithm of the probabilities.
        """
        if isinstance(obs, np.ndarray):
            # The input of the network should be a tensor.
            obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            # To improve the computation efficiency and also the computation gradient is not needed here. Hence,
            # the next lines do not require the gradient (it is need for the back and forth propagation).
            pi = self.actor._distribution(obs)
            # Choose an action with the highest probability.
            a = pi.sample()
            logp_a = self.actor._logprobability(pi, a)
            v = self.critic(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

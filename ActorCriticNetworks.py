import torch
from torch import nn
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, contintuous):
        super().__init__()
        self.continuous = contintuous
        if contintuous:
            log_std = 0.5 * np.ones(act_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.network = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)),
                                    nn.ReLU(),
                                     nn.Linear(64, 64),
                                    nn.Tanh(),
                                     nn.Linear(64, act_dim),
                                    nn.Identity())


    def _distribution(self, obs):
        mean = self.network(obs)
        if self.continuous:
            std = torch.exp(self.log_std)
            return Normal(mean, std)
        return Categorical(logits=mean)

    def _logprobability(self, pi, act):
        if self.continuous:
            return pi.log_prob(act).sum(axis=-1)
        return pi.log_prob(act)


    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        # those distributions.
        pi = self._distribution(obs)
        logp = None
        if act is not None:
            act = torch.as_tensor(act, dtype=torch.float32)
            logp = self._logprobability(pi, act)
        return pi, logp, pi.entropy()


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.network =  nn.Sequential(layer_init(nn.Linear(obs_dim, 64)),
                                            nn.ReLU(),
                                     nn.Linear(64,64),
                                            nn.Tanh(),
                                     nn.Linear(64, 1),
                                        nn.Identity())

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        return torch.squeeze(self.network(obs), -1) # Critical to ensure v has right shape.

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_space, continuous):
        super().__init__()

        self.actor = Actor(obs_dim, action_space, continuous)

        # build value function
        self.critic = Critic(obs_dim)

    def step(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            pi = self.actor._distribution(obs)
            a = pi.sample()
            logp_a =pi.log_prob(a)
            v = self.critic(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()
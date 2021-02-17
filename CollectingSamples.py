import numpy as np
import torch
import random

class Rollout():
    """
        Rollout class saves the relevant value, when an agent performs an action in the environment, changes might
        occur in the environment and the agent receive these changes as observations and rewards.

    """
    def __init__(self, size, obs_dim, act_dim):
        self.batch_size = size

        self.observations = torch.zeros(size, obs_dim) # input to the actor/agent network
        self.actions = torch.zeros(size, act_dim) # should it be the critic network
        self.logprobs = torch.zeros(size) # policy be the actor's network output?
        self.rewards = torch.zeros(size)
        self.values = torch.zeros(size) # calculated ctiis's output

        self.advs = torch.zeros(size)
        self.done = torch.zeros(size)  # shows when the episode is done, when it's the time to reset the environment
        self.t = 0

    def add(self, action, obs, reward, value, logprob):
        # can we add "i" a counter variable to the argument, if there is a loop
        """
            saves the results of the agent interaction when performing the given action
        :param action:
        :param obs:
        :param reward:
        :param value:
        :param logprob:
        :return:
        """
        if self.t > self.batch_size:
            raise Exception ("The batch has a size of " % (self.batch_size))

        self.actions[self.t] = action
        self.observations[self.t] = obs
        self.rewards[self.t] = reward
        self.values[self.t] = value
        self.logprobs[self.t] = logprob
        self.t += 1


    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        b
import numpy as np
import torch
import random

class Rollout():
    """
        Rollout class saves the relevant value, when an agent performs an action in the environment, changes might
        occur in the environment and the agent receive these changes as observations and rewards.

    """
    def __init__(self, size, obs_dim, act_dim):
        # not sure about using tensor or a numpy array or just a simple array, when returned converted into tensor for
        # the input of the actor network and crtitic network
        self.batch_size = size

        self.observations = torch.zeros(size, obs_dim) # input to the actor/agent network
        self.actions = np.zeros(shape=(size, act_dim), dtype=float) # should it be the critic network
        self.logprobs = torch.zeros(size) # policy be the actor's network output?
        self.rewards = torch.zeros(size)
        self.values = torch.zeros(size) # calculated critic's output
        self.rtgs = torch.zeros(size)

        self.advs = torch.zeros(size)
        #self.done = torch.zeros(size)  # needed when the batch
        # shows when the episode is done, when it's the time to reset the environment
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

        self.actions[self.t] = action # broadcast error, understand the shapes and solve the batch_size
        self.observations[self.t] = obs
        self.rewards[self.t] = reward
        self.values[self.t] = value
        self.logprobs[self.t] = logprob
        self.t += 1

    def compute_disc_rewards(self, gamma):
        """
            implements the 4th step in the pseudocode
        :param gamma:
        :return:
        """
        discounted_rewards = []
        batch_rewards = self.rewards.detach().numpy()

        for reward in batch_rewards[::-1]: # in a reversed order
            running_reward = reward + gamma*running_reward
            discounted_rewards.insert(0,running_reward) #together with the reversed counter, we get the correct order

        self.rtgs = torch.FloatTensor(discounted_rewards)

import torch
import numpy as np

class Rollout():
    """
        Rollout class saves the relevant data, when an agent performs an action in the environment, changes might
        occur in the environment and the agent receive these changes as observations and rewards.
        PPO is an on-policy algorithm, this class helps in collecting samples each time the agent (actor/critic
        networks) interact.
    """
    def __init__(self, size):
        self.batch_size = size

        self.observations = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rtgs = []
        self.t = 0

    def add(self, action, obs, value, logprob):
        """
            Saves the results of the agent interaction when performing the given action
        :param action:
        :param obs:
        :param reward:
        :param value:
        :param logprob:
        :return:
        """
        if self.t > self.batch_size:
            raise Exception ("The batch has a size of ",(self.batch_size))
        else:
            self.actions.append(action)
            self.observations.append(obs)
            self.values.append(value)
            self.logprobs.append(logprob)
            self.t += 1

    def compute_disc_rewards(self, gamma, batch_rewards):
        """
            Implements the fourth step in the pseudocode, compute the discounted rewards of each episode in a batch.

        :param gamma: the discounter factor
        :param batch_rewards: the epsiodes rewards collected per a batch
        :return:
        """
        batch_rtgs = []

        for episode_rewards in batch_rewards[::-1]:
            discounted_reward = 0
            for reward_t in episode_rewards[::-1]:
                discounted_reward = reward_t + gamma*discounted_reward
                # Together with the reversed counter, we get the correct order
                batch_rtgs.insert(0,discounted_reward)
        self.rtgs = batch_rtgs


    def convert_list_to_numpyarray(self):
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.logprobs = np.asarray(self.logprobs, dtype=np.float32)
        self.values = np.asarray(self.values, dtype=np.float32)
        self.rtgs = np.asarray(self.rtgs, dtype=np.float32)


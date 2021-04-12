import numpy as np
import scipy.signal as signal
import torch
class Rollout:
    """
        Rollout class saves the relevant data, when an agent performs an action in the environment, changes might
        occur in the environment and the agent receive these changes as observations and rewards.
        PPO is an on-policy algorithm, this class helps in collecting samples each time the agent (actor/critic
        networks) interact.
    """
    def __init__(self, size, env):
        self.batch_size = size
        self.observations = np.zeros((size,) + env.observation_space.shape, dtype=np.float32)
        self.actions = np.zeros((size,) + env.action_space.shape , dtype=np.float32)
        #self.actions = np.zeros(size, dtype=np.float32)
        self.logprobs = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.adv = np.zeros(size, dtype=np.float32)
        self.discounted_rew = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=int)

        self.t, self.tmp, self.trajectory_start_idx = 0, 0, 0

    def add(self, action, obs, value, r, logprob, done):
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
            self.actions[self.t] = action
            self.observations[self.t] = obs
            self.values[self.t] = value
            self.rew[self.t] = r
            self.logprobs[self.t] = logprob
            self.dones[self.t] = done
            self.t += 1

    def compute_discounted_rewards(self, gamma, last_ret, done):
        """
            Implements the fourth step in the pseudocode, compute the discounted rewards of each episode in a batch.

        :param gamma: the discounter factor
        :param batch_rewards: the epsiodes rewards collected per a batch
        :return:
        """
        for t in reversed(range(self.trajectory_start_idx, self.t)):
            last_state = (t == (self.t - 1))
            # check the last state in the episode length
            if last_state:
                next_terminal = 1 - done
                next_ret = last_ret
            else:
                next_ret = self.discounted_rew[t+1]

            self.discounted_rew[t] = self.rew[t] + gamma * next_ret * next_terminal
            if last_state:
                next_terminal = 1

        self.tmp = self.trajectory_start_idx
        self.trajectory_start_idx = self.t

    def compute_discounted_rewards2(self, gamma, r_t):
        """
            Implements the fourth step in the pseudocode, compute the discounted rewards of each episode in a batch.

        :param gamma: the discounter factor
        :param batch_rewards: the epsiodes rewards collected per a batch
        :return:
        """
        path_slice = slice(self.trajectory_start_idx, self.t)
        r = np.append(self.rew[path_slice], r_t)
        v = np.append(self.values[path_slice], r_t)

        #self.tmp = self.trajectory_start_idx
        self.trajectory_start_idx = self.t

        a = [1, float(-gamma)]
        b = [1]
        tmp = signal.lfilter(b, a, x=r[::-1], axis=0)[::-1]
        self.discounted_rew[path_slice] = tmp[:-1] # except the last element
        #self.compute_adv_mc(path_slice)
        # r[:-1] all but the last element, v[1:] all elements but the first element
        deltas = r[:-1] + gamma * v[1:] - v[:-1]
        #tmp = signal.lfilter(b, [1,float(-gamma*0.97)], x=deltas[::-1], axis=0)[::-1]
        self.adv[path_slice] = signal.lfilter(b, [1,float(-gamma*0.97)], x=deltas[::-1], axis=0)[::-1]

    def compute_gae(self, gamma, lam, last_value):
        """
            to evaluate the performance of the model
        :param last_value:
        :param last_dones:
        :return:
        """
        prev_gae_adv = 0.0
        for i in reversed(range(self.tmp, self.t)):
            # check the last state in the episode
            if i == self.t-1:
                next_val = last_value
            else:
                next_val = self.values[i+1]

            adv_t = self.rew[i] + gamma * next_val - self.values[i]
            # n-bootstrap
            self.adv[i] = prev_gae_adv = gamma * lam * prev_gae_adv + adv_t
        self.tmp = self.trajectory_start_idx

    def compute_adv_mc(self, path_slice):
        self.adv = self.discounted_rew - self.values
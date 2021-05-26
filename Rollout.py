import numpy as np

class Rollout:
    """
        Rollout class saves the relevant data, when an agent performs an action in the environment, changes
        occur in the environment and the agent receive these changes as observations and rewards.
        In this class, the advantages and the discounted rewards are calculated using one of the methods;
        The N-Steo Return or The Generalised Advantage Estimation.
    """

    def __init__(self, size, env):
        self.batch_size = size
        self.observations = np.zeros((size,) + env.observation_space.shape, dtype=np.float32)
        self.actions = np.zeros((size,) + env.action_space.shape, dtype=np.float32)
        self.logprobs = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.adv = np.zeros(size, dtype=np.float32)
        self.discounted_rew = np.zeros(size, dtype=np.float32)

        self.t, self.tmp, self.trajectory_start_idx = 0, 0, 0

    def add(self, action, obs, value, reward, logprob):
        """
            Saves the information of the agent's interaction with the environment, and increases the pointer by one.
        :param action: the taken action.
        :param obs: the observations about the environment.
        :param reward: the received reward after making an action.
        :param value: the estimated state value.
        :param logprob: the logarithm of the actions' probabilities.
        :return:
        """
        if self.t > self.batch_size:
            raise Exception("The batch has a size of ", (self.batch_size))
        else:
            self.actions[self.t] = action
            self.observations[self.t] = obs
            self.values[self.t] = value
            self.rewards[self.t] = reward
            self.logprobs[self.t] = logprob
            self.t += 1

    def compute_adv_nsteps(self, gamma, last_ret):
        """
            Computes the advantages and the discounted rewards using the N-Step Return method.

        :param gamma: the discount factor gamma.
        :param last_ret: the last received reward or an estimation of the state value.
        :return:
        """
        # Iterates over the states in reverse order.
        for t in reversed(range(self.trajectory_start_idx, self.t)):
            # Checks if it is the last state in the episode
            last_state = (t == (self.t - 1))
            if last_state:
                next_ret = last_ret
                next_terminal = 0
            else:
                next_ret = self.discounted_rew[t + 1]

            # Computes the discounted reward
            self.discounted_rew[t] = self.rewards[t] + gamma * next_ret * next_terminal

            if last_state:
                next_terminal = 1
                self.adv[t] = self.discounted_rew[t]
            else:
                self.adv[t] = self.discounted_rew[t] - self.values[t]

        # Update the index for the next episode.
        self.trajectory_start_idx = self.t

    def compute_gae(self, gamma, lam, last_value):
        """
            Computes the advantages and the discounted rewards using the Generalised Advantage Estimation method.

        :param gamma: the discount factor.
        :param lam: lambda the decay rate.
        :param last_value: the value of the terminal state.
        :return:
        """
        prev_gae_adv = 0.0

        for i in reversed(range(self.trajectory_start_idx, self.t)):
            # Checks the last state in the episode.
            if i == (self.t - 1):
                next_val = last_value
            else:
                next_val = self.values[i + 1]

            delta = self.rewards[i] + gamma * next_val - self.values[i]

            self.adv[i] = prev_gae_adv = delta + gamma * lam * prev_gae_adv

            self.discounted_rew[i] = self.adv[i] + self.values[i]

        # Update the index for the next episode.
        self.trajectory_start_idx = self.t

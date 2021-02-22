from ActorCriticNetworks import SamplingNetworks
from torch.optim import Adam
from CollectingSamples import Rollout
import torch.nn.functional as F
import gym
import torch

class PPO():
    """
        batches with fixed lengths or with different length given a max length
    """

    def __init__(self, env, batch_size):
        # extract environment information
        self.env = env # do we really need it
        self.observations_dim = env.observation_space.shape[0]  # or just shape
        self.actions_dim = env.action_space.shape[0]
        self.batch_size = batch_size # is there a better way to decide the batch size

        self.actor = SamplingNetworks(self.observations_dim, self.actions_dim)
        self.critic = SamplingNetworks(self.observations_dim, 1)

        self.trajectories = []
        self.init_hyperparameter()
        # in the original implementation of PPO by OpenAI, different lr were used for each network
        self.actor_optimiser = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=self.lr)

    # Hyperparameter



    def init_hyperparameter(self, learning_rate=0.005, gamma=0.95, clip_ratio=0.2, eps_len_T=10, entropy=0.01):
        self.lr = learning_rate
        #self.learning_steps = learning_steps # (K) timesteps
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.eps_len_T = eps_len_T # in case no batch size was added
        self.entropy_beta = entropy



    def learn(self, timesteps):
        k = 0
        # create a trajectory, or list of rollouts
        # start the first interacting
        obs = self.env.reset()

        while k < timesteps:
            rollout_t = Rollout(self.batch_size, self.observations_dim, self.actions_dim)

            for t in range(self.batch_size): # are we getting only one trajectory per batch
                # add each rollout to the trajectory list

                action, logprob = self.actor(obs, actor = True)
                value = self.critic(obs)

                next_obs, r_t, done, _ = self.env.step(action)

                obs = next_obs

                rollout_t.add(action, obs, r_t, value, logprob)
                # if a terminal state is reached, reset the environment
                if done:
                    # in the OpenAI implementation of PPO, they used timeout or a cutoff
                    obs = self.env.reset()
                    rollout_t.compute_disc_rewards(self.gamma)
                    break

            self.trajectories.append(rollout_t)
            k += 1

            self.update(rollout_t)


    def update(self, rollout):
        """
            implements the sixth step in the pseudocode
        :return:
        """
        advantages = rollout.rtgs - rollout.values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() - 1e-5) # ??

        prev_logpro = rollout.logprobs

        actions, curr_logpro, values, entropy = self.actor(rollout.observations, rollout.actions)

        ratio = torch.exp(curr_logpro - prev_logpro) # Importance Sampling in PPO & calculus trick

        surr1 = advantages * ratio
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages # why is ratio inside

        loss =  torch.min(surr1, surr2)  + F.mse(values, rollout.rtgs) + entropy*self.entropy_beta # as suggested in Paper

        actor_loss = loss.mean() # why

        self.actor_optimiser.zero_grad() # zero out the gradient before the backpropragation
        self.actor_loss.backward() # to minimise the loss
        self.actor_optimiser.step()
        #(retain_graph = True) # otherwise we get an error

        critic_value_loss = ((rollout.values - rollout.rewards)**2).mean()

        self.critic_optimiser.zero_grad()
        critic_value_loss.backward()
        self.critic_value_loss.step()

env = gym.make('Pendulum-v0')
model = PPO(env, 510)
model.learn(10000)
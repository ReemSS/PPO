from ActorCriticNetworks import SamplingNetworks
from torch.optim import Adam
from CollectingSamples import Rollout
import torch.nn.functional as F
import numpy as np
from torch import nn
import gym
import time
import torch
import wandb
#from spinup.utils.logx import EpochLogger

class PPO():
    """
        batches with fixed lengths or with different length given a max length
    """

    def __init__(self, env, batch_size):
        # extract environment information
        self.env = env # do we really need it

        # see what kind of spaces the environment has
        # print(env.action_space)
        # print(env.observation_space)
        self.observations_dim = self.adjust_according_to_space(env.observation_space)
        self.actions_dim = self.adjust_according_to_space(env.action_space)

        #print(self.observations_dim)
        self.batch_size = batch_size # is there a better way to decide the batch size

        self.actor = SamplingNetworks(self.observations_dim, self.actions_dim)
        self.critic = SamplingNetworks(self.observations_dim, 1)

        self.trajectories = []
        self.init_hyperparameter()
        # in the original implementation of PPO by OpenAI, different lr were used for each network
        self.actor_optimiser = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=self.lr)

        self.timesteps = 0
        self.env_contiuous = (type(env.action_space) == gym.spaces.Box)


        self.logger = {
            'start_time': 0,
            'current_timestep' : 0,
            'iteration_number': 0,
            'avg_loss': []
        }


    def init_hyperparameter(self, learning_rate=0.005, gamma=0.95, clip_ratio=0.2, eps_len_T=10, entropy=0.01):
        self.lr = learning_rate
        #self.learning_steps = learning_steps # (K) timesteps
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.eps_len_T = eps_len_T # in case no batch size was added
        self.entropy_beta = entropy



    def learn(self, timesteps):
        k = 0
        self.timesteps = timesteps
        # create a trajectory, or list of rollouts
        # start the first interacting
        self.logger['start_time'] = time.time()
        obs = self.env.reset()
        #obs = self.adjust_according_to_space(obs)
        #print(type(obs))

        while k < timesteps:
            rollout_t = Rollout(self.batch_size, self.observations_dim, self.actions_dim)

            self.logger['iteration_number'] += 1

            stop = False
            for t in range(self.batch_size): # are we getting only one trajectory per batch
                # add each rollout to the trajectory list

                action, logprob,_ = self.actor.random_action(obs, self.env_contiuous)
                value = self.critic(obs)
                print(action)

                next_obs, r_t, done, _ = self.env.step(action)

                obs = next_obs

                #obs = self.adjust_according_to_space(obs)

                rollout_t.add(action, obs, r_t, value, logprob)
                # if a terminal state is reached, reset the environment
                if done:
                    # in the OpenAI implementation of PPO, they used timeout or a cutoff
                    obs = self.env.reset()
                    rollout_t.compute_disc_rewards(self.gamma)
                    stop = True
                    break
            if not stop:    # if we did not reach a terminal state but the size of batch has been exhausted
                rollout_t.compute_disc_rewards(self.gamma)
                obs = self.env.reset() # should we do that to when the epoch is reached?

            rollout_t.convert_array_to_tensors()
            self.trajectories.append(rollout_t)
            k += 1

            self.update(rollout_t)
            self.logger_print()

    def update(self, rollout):
        """
            implements the sixth step in the pseudocode
        :return:
        """

        advantages = rollout.rtgs - rollout.values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() - 1e-5) # Normalise

        prev_logpro = rollout.logprobs

        actions, curr_logpro, entropy = self.actor.random_action(rollout.observations, self.env_contiuous)
        values = self.critic(rollout.observations).squeeze()

        ratio = torch.exp(curr_logpro - prev_logpro) # Importance Sampling in PPO & calculus trick

        surr1 = advantages * ratio
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages # why is ratio inside

        loss =  torch.min(surr1, surr2)  + F.mse_loss(values, rollout.rtgs) + entropy*self.entropy_beta # as suggested in Paper

        actor_loss = loss.mean()

        self.logger["avg_loss"].append(actor_loss.detach())

        self.actor_optimiser.zero_grad() # zero out the gradient before the backpropragation
        actor_loss.backward(retain_graph = True) # to minimise the loss
        self.actor_optimiser.step()
        #(retain_graph = True) # otherwise we get an error
        # ((rollout.values - rollout.rewards)**2).mean()
        critic_value_loss = nn.MSELoss()(values, rollout.rtgs)

        self.critic_optimiser.zero_grad()
        critic_value_loss.backward()
        self.critic_optimiser.step()

    # def init_wandb(self):
    #     wandb.init(project="ppo", entity="rfarah", sync_tensorboard=True)
    #
    #     wandb.config.update({
    #         "timesteps: ": self.timesteps
    #     })

    def logger_print(self):
        print(flush=True)
        iteration = self.logger["iteration_number"]
        print(f"Iteration number {iteration}", flush=True)
        avg_loss = np.mean([x.mean() for x in self.logger['avg_loss']])
        print(f"Average Loss {avg_loss}", flush=True)
        delta_t = time.time() - self.logger['start_time']
        print(f"Time {delta_t}", flush=True)
        print(flush=True)

    def adjust_according_to_space(self, env_space):
        dim = 0

        if type(env_space) != gym.spaces.Box:
            dim = env_space.n
        else:
            dim = env_space.shape[0]

        return dim


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    env1 = gym.make('MountainCar-v0') # discrete actions space
    #print(env.action_space.shape)
    #print(env1.action_space.n)
    model = PPO(env1, 100)
    model.learn(50)
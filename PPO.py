from ActorCriticNetworks import SamplingNetworks
from torch.optim import Adam
from CollectingSamples import Rollout
import torch.nn.functional as F
from torch import nn
import gym
import time
import torch
import wandb
from MyPPO.spinningup.spinup.utils import EpochLogger
from MyPPO.spinningup.spinup.utils import colorize
class PPO():
    """
        Implementation of the PPO algorithm with clipped surrogate objective.
        This implementation can be applied on environments with either discrete or continuous action spaces,
        like the original paper says.
        batches with fixed lengths or with different length given a max length
    """

    def __init__(self, env, batch_size, max_ep_len):
        # Extract environment information
        self.env = env

        # Adjust the environment and action space to turn them into a valid input for
        # the actor network and the value network
        self.observations_dim = self.adjust_according_to_space(env.observation_space)
        self.actions_dim = self.adjust_according_to_space(env.action_space)

        # in SpinUp implementation of OpenAI, they used num_procs() to divide the total number of interactions
        # of the agent within the environment. Here the batch size is given as an argument.
        # It also can be set within the hyperparameter
        self.batch_size = batch_size

        # in every batch we might have a number of trajectories, it is good to define the max length of the episode
        # in case we did not reach a terminal state, in other words we define a timeout
        self.max_epi_len = max_ep_len

        # actor and critic networks to help the agent to act and learn
        self.actor = SamplingNetworks(self.observations_dim, self.actions_dim)
        self.critic = SamplingNetworks(self.observations_dim, 1)


        self.init_hyperparameter()

        # in the original implementation of PPO by OpenAI, different lr were used for each network
        self.actor_optimiser = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=self.lr)

        self.timesteps = 0
        self.env_continuous = (type(env.action_space) == gym.spaces.Box)

        self.wandb = False
        self.logger = EpochLogger(dict())

        # self.logger = {
        #     'start_time': 0,
        #     'current_timestep' : 0,
        #     'iteration_number': 0,
        #     'avg_loss': []
        # }

    def init_hyperparameter(self, learning_rate=0.005, gamma=0.95, clip_ratio=0.2, entropy=0.01):
        self.lr = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_beta = entropy

    def learn(self, timesteps):
        """
            This method implements PPO according to the pseudocode on
            'SpinUP OpenAI <https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode>'_
            It implements it as follows:
                - First Step: Collecting Trajectories using Rollout to save the info in '~CollectingSamples.Rollout'

        :param timesteps: Number of interactions wit
        :return:
        """
        # Interaction Counter
        k = 0
        # needed for logging
        self.timesteps = timesteps

        obs = self.env.reset()
        #obs = self.adjust_according_to_space(obs)
        # Save configuration as json file
        #self.logger.save_config(locals())

        # Tracking episode length and return
        episode_return = 0
        episode_length = 0

        # To save each episode returns as an array
        episode_return_arr = []

        start_time = time.time()

        # Epoch or number of iterations. One epoch indicates that the dataset is passed forward and backward
        # through the network one time. In order to optimise the learning using an iterative process such as the
        # gradient policy, we need more than one epoch to pass through the network.
        # The required number of epochs depends on how diverse the dataset is.
        while k < timesteps:
            rollout_t = Rollout(self.batch_size)

            batch_rewards = []
            for t in range(self.batch_size):
                action, logprob,_ = self.actor.random_action(obs, self.env_continuous)
                critic = self.critic(obs)
                self.logger.store(Value = critic)

                # Apply the action on the environment
                next_obs, r_t, done, _ = self.env.step(action)

                # Move the agent after performing an act
                obs = next_obs

                # Save the data of the episode
                rollout_t.add(action, obs, critic, logprob)

                episode_length += 1


                batch_full = (t == (self.batch_size - 1))

                if not(done or batch_full):
                    episode_return += r_t
                    episode_return_arr.append(r_t)

                # If a terminal state is reached, reset the environment
                else:
                    if batch_full and not(done):
                        print(colorize("The batch is full but a terminal state has not been reached",
                                       color='yellow',bold=True))

                    # If a terminal state not reached, but max episode length is reached or the batch size
                    # is exhausted then ask for the reward of the current state, and compute the discounted rewards
                    if batch_full or (episode_length == self.max_epi_len - 1):
                        # then take a look at the target value
                        action, _, _ = self.actor.random_action(obs, self.env_continuous)
                        _, r_t, _,_ = self.env.step(action)
                        episode_return += r_t
                    else:
                        # the terminal state has been reached
                        r_t = 0
                        # Save the episode data
                        self.logger.store(EpisodeReturn = episode_return, EpisodeLength = episode_length)
                    episode_return_arr.append(r_t)

                    # See when to render the environment
                    # self.env.render()
                    # Add the complete or cuttoff episode rewards to the batch rewards
                    batch_rewards.append(episode_return_arr)

                    # Reset the environment and zero out the episode data
                    obs = self.env.reset()
                    episode_return, episode_length, episode_return, episode_return_arr = 0,0,0,[]

            rollout_t.compute_disc_rewards(self.gamma, batch_rewards)
            rollout_t.convert_array_to_tensors()
            k += 1

            # Set up model saving
            if k == (timesteps - 1):
                self.logger.save_state({'env': self.env}, None)
            #self.logger.setup_pytorch_saver(self.actor)
            self.update(rollout_t)
            self.logger_print( k, start_time)

    def update(self, rollout):
        """
            Implements the sixth step in the pseudocode
        :return:
        """

        advantages = rollout.rtgs - rollout.critic.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() - 1e-5) # Normalise

        prev_logpro = rollout.logprobs

        actions, curr_logpro, entropy = self.actor.random_action(rollout.observations, self.env_continuous)

        values = self.critic(rollout.observations).squeeze()

        ratio = torch.exp(curr_logpro - prev_logpro) # Importance Sampling in PPO & calculus trick

        surr1 = advantages * ratio
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages # why is ratio inside

        loss =  torch.min(surr1, surr2)  + F.mse_loss(values, rollout.rtgs) + entropy*self.entropy_beta # as suggested in Paper

        actor_loss = loss.mean()

        old_loss = actor_loss
        #self.logger["avg_loss"].append(actor_loss.detach())

        self.actor_optimiser.zero_grad() # zero out the gradient before the backpropragation
        # The error at the output is passed through the network
        actor_loss.backward(retain_graph = True)
        self.actor_optimiser.step()
        #(retain_graph = True) # otherwise we get an error
        # ((rollout.values - rollout.rewards)**2).mean()

        critic_value_loss = nn.MSELoss()(values, rollout.rtgs)
        old_lossV = critic_value_loss
        # Find the local minimum loss
        self.critic_optimiser.zero_grad()
        critic_value_loss.backward()
        self.critic_optimiser.step()

        self.logger.store(LossPi = old_loss, LossV = critic_value_loss, Entropy = entropy,
                          DeltaLossPi= (actor_loss.item() - old_loss),
                          DeltaLossV= (critic_value_loss.item() - old_lossV))

    def init_wandb(self):
        wandb.init(project="ppo", entity="rfarah", sync_tensorboard=True)

        wandb.config.update({
            "timesteps: ": self.timesteps,
            "Average_loss": self.logger["avg_loss"],
            "iteration": self.logger["iteration_number"],
            "delta_time": time.time() - self.logger["start_time"]
        })

        self.wandb = True

    def logger_print(self, epoch, start_time ):
        self.logger.log_tabular("Epoch",epoch)
        self.logger.log_tabular("EpisodeReturn", with_min_and_max=True)
        self.logger.log_tabular("EpisodeLength", average_only=True)
        self.logger.log_tabular("Value", with_min_and_max=True)
        # logger.log_tabular("Actor_Loss")
        # logger.log_tabular("Critic_Loss")
        self.logger.log_tabular("Time", time.time() - start_time)
        self.logger.dump_tabular()



    def adjust_according_to_space(self, env_space):
        dim = 0

        if type(env_space) != gym.spaces.Box:
            dim = env_space.n
        else:
            dim = env_space.shape[0]

        return dim


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    env1 = gym.make('CartPole-v1') # discrete actions space
    #print(env.action_space.shape)
    #print(env1.action_space.n)
    model = PPO(env.unwrapped, 256,50)
    model.learn(50)
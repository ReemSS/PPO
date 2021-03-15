from ActorCriticNetworks import SamplingNetworks
from torch.optim import Adam
from CollectingSamples import Rollout
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import numpy as np
import gym
import time
import torch
from MyPPO.spinningup.spinup.utils.logx import EpochLogger
from MyPPO.spinningup.spinup.algos.pytorch.ppo.core import MLPActorCritic
from MyPPO.spinningup.spinup.utils.logx import colorize


class PPO():
    """
        Implementation of the PPO algorithm with clipped surrogate objective
        This implementation can be applied on environments with either discrete or continuous action spaces,
        as mentioned in the original paper.
    """

    def __init__(self, env, batch_size, max_ep_len, name_of_exp):
        # Extract environment information
        self.env = env

        # Adjust the environment and action space to turn them into a valid input for
        # the actor network and the critic network
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
        # self.actor = SamplingNetworks(self.observations_dim, self.actions_dim)
        # self.critic = SamplingNetworks(self.observations_dim, 1)

        self.ac= MLPActorCritic(env.observation_space, env.action_space)

        self.init_hyperparameter()

        # in the original implementation of PPO by OpenAI, different values of lr were used for each network
        # self.actor_optimiser = Adam(self.actor.parameters(), lr=self.lr)
        # self.critic_optimiser = Adam(self.critic.parameters(), lr=self.lr)
        self.actor_optimiser = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.critic_optimiser = Adam(self.ac.v.parameters(), lr=self.lr)

        self.timesteps = 0
        self.env_continuous = (type(env.action_space) == gym.spaces.Box)

        # self.wandb = False
        self.logger = EpochLogger(output_dir="\Logger_Experiments", output_fname=env.unwrapped.spec.id)

        # Set up logging for tensorboard
        self.tb_logger = SummaryWriter(name_of_exp)


    def init_hyperparameter(self, learning_rate=0.005, gamma=0.99, clip_ratio=0.2,
                            entropy=0.01, gradient_descent_steps = 80):
        """
            Initialise the hyperparameters of the algorithm. The used default values are the same values
            suggested in the original paper.
        :param learning_rate: For the actor and critic models
        :param gamma: discounter factor
        :param clip_ratio: projection the trust region concept onto the algorithm
        :param entropy: added to the loss function to guarantee a sufficient exploration
        :param gradient_descent_steps: number of steps the gradient descent needs to take by the networks
        :return:
        """
        self.lr = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_beta = entropy
        self.gradient_descent_steps = gradient_descent_steps

    def learn(self, timesteps):
        """
            This method implements PPO according to the pseudocode on
            'SpinUP OpenAI <https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode>'_
            It implements it as follows:
                - First Step: Collecting Trajectories using Rollout to save the info in '~CollectingSamples.Rollout'

        :param timesteps: Number of interactions with the environment
        :return:
        """
        # Interaction Counter
        k = 0
        # needed for logging
        self.timesteps = timesteps

        obs = self.env.reset()
        # Save configuration as json file
        # self.logger.save_config(locals())

        # Tracking episode length and return
        episode_return = 0
        episode_length = 0

        # To save each episode returns as an array
        episode_return_arr = []

        start_time = time.time()
        info = dict()

        # Epoch or number of iterations. One epoch indicates that the dataset is passed forward and backward
        # through the network one time. In order to optimise the learning using an iterative process such as the
        # gradient policy, we need more than one epoch to pass through the network.
        # The required number of epochs depends on how diverse the dataset is.
        while k < timesteps:
            final_state_reached = False
            # save the information inside of a rollout
            rollout_t = Rollout(self.batch_size)

            batch_rewards = []
            for t in range(self.batch_size):
                # action, logprob,_ = self.actor.random_action(obs, self.env_continuous)
                # value = self.critic(obs)
                action,value, logprob = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
                #self.logger.store(Value = value.detach().numpy())
                self.logger.store(Value = value)
                # Save the data of the episode
                rollout_t.add(action, obs, value, logprob)
                # Apply the action on the environment
                next_obs, r_t, done, info = self.env.step(action)

                # Move the agent after performing an act
                obs = next_obs

                episode_length += 1

                # Timeout in the spinup implementation
                cutOff = (t == (self.batch_size - 1)) or (episode_length == (self.max_epi_len - 1))

                if not (done or cutOff):
                    episode_return += r_t
                    episode_return_arr.append(r_t)
                # If a terminal state is reached, reset the environment
                if done or cutOff:
                    # If a terminal state not reached, but max episode length is reached or the batch size
                    # is exhausted then ask for the reward of the current state, and compute the discounted rewards
                    if cutOff and not(done):
                        print(colorize("The batch is full or the max episode length has "
                                       "been reached, but a terminal state has not been reached",
                                       color='yellow',bold=True))
                        # then take a look at the target value
                        # action, _, logprob = self.actor.random_action(obs, self.env_continuous)
                        #action, _, logprob = self.ac(obs)
                        action, _, logprob = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
                        _, r_t, _,info = self.env.step(action)
                        episode_return += r_t
                    elif done:
                        # the terminal state has been reached
                        r_t = 0
                        final_state_reached = True
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
            rollout_t.convert_list_to_numpyarray()
            k += 1

            # Set up model saving
            if k == (timesteps - 1):
                self.logger.save_state({'env': self.env}, None)
            # self.logger.setup_pytorch_saver(self.actor)

            old_policy_loss, entropy = self.compute_loss(rollout_t, actor= True)[0].detach().numpy(),\
                                       self.compute_loss(rollout_t, actor= True)[1]

            old_value_loss = self.compute_loss(rollout_t, critic=True).detach().numpy()

            self.logger.store(Actor_Loss = old_policy_loss, Critic_Loss = old_value_loss)
            for i in range(self.gradient_descent_steps):
                self.update(rollout_t)

            curr_policy_loss = self.compute_loss(rollout_t, actor=True)[0].detach().numpy()
            curr_value_loss = self.compute_loss(rollout_t, critic=True).detach().numpy()

            self.logger.store(Delta_Loss_Actor= curr_policy_loss - old_policy_loss,
                              Delta_Loss_Critic= curr_value_loss - old_value_loss)
            self.logger_print(k, start_time, final_state_reached)

            # Save the model every 20 iteration, to continue with a trained model
            # if (k % 20 == 0) or (k == k-1):
            #     torch.save(self.actor.state_dict(), './PPO_author.pth')
            #     torch.save(self.critic.state_dict(), './critic_author.pth')

            for i in info:
                if 'episode' in i:
                    score = info['episode']['r']
                    self.tb_logger.add_scalar("charts/episode_reward", score, k)

            # add the logging info to the tensorboard
            self.tb_logger.add_scalar("losses/value_loss", np.round(old_value_loss, 4), k)
            self.tb_logger.add_scalar("losses/policy_loss", np.round(old_policy_loss, 4), k)
            #self.tb_logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # entropy = mpi_statistics_scalar(self.logger.get_stats("Entropy"), average_only = True)
            self.tb_logger.add_scalar("losses/approx_entropy", entropy, k)


    def update(self, rollout):
        """
            Implements the sixth and the sevenths step in the pseudocode of OpenAI.
            First we try to maximise the clip surrogate function, using Adam
            Secondly minimise the mean squared error of the value function, using

        :param rollout: the different trajectories collected in a batch
        :return:
        """
        actor_loss, entropy = self.compute_loss(rollout, actor=True)

        self.actor_optimiser.zero_grad()
        # The error at the output is passed through the network
        actor_loss.backward()
        self.actor_optimiser.step()

        critic_loss = self.compute_loss(rollout, critic= True)

        # Find the local minimum loss
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()


    def compute_loss(self, rollout, actor = False, critic = False):
        """

        :param rollout:
        :param actor:
        :param critic:
        :return:
        """
        if actor:
            # Compute the advantages after interacting with the environment
            advantages = rollout.rtgs - rollout.values  # there was a values.detach()
            # To a faster learning we need to normalise the data
            # advantages = (advantages - advantages.mean()) / (advantages.std() - 1e-5)
            advantages = (advantages - advantages.min()) / (advantages.max() - advantages.min())

            # To add Importance Sampling property
            prev_logpro = rollout.logprobs
            #actions, curr_logpro, entropy = self.actor.random_action(rollout.observations, self.env_continuous)
            actions, curr_logpro, entropy = self.ac(rollout.observations)
            # Since logarithm of the probabilities are used then subtract them instead of dividing them, get the exponent

            ratio = torch.exp(curr_logpro - torch.tensor(prev_logpro))
            advantages = torch.from_numpy(advantages)
            # Unclipped objective
            surr1 = advantages * ratio
            # Clipped objective
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # Since Actor and Critic are neural networks that share parameters (i.e. observations caused by action produced
            # by the actor network, are then passed through the critic network. Then the loss function that combines the
            # discounted reward, which is obtained by applying a policy, and the value function. Entropy is added
            # to ensure exploration as suggested in the original paper. It makes the agent more uncertain about the action
            # to choose
            loss =  -(torch.min(surr1, surr2) + entropy*self.entropy_beta)
            # Store the entropy
            self.logger.store(Entropy = entropy)

            return loss.mean(), entropy

        if critic:
            #values = self.critic(rollout.observations).squeeze()
            _ , values, _ = self.ac(rollout.observations)

            return nn.MSELoss()(values, torch.from_numpy(rollout.rtgs))


    # def init_wandb(self):
    #     wandb.init(project="ppo", entity="rfarah", sync_tensorboard=True)
    #
    #     wandb.config.update({
    #         "timesteps: ": self.timesteps,
    #         "EpisodeReturn": (self.logger.get_stats("EpisodeReturn")),
    #         "iteration": self.logger["iteration_number"],
    #         "delta_time": time.time() - self.logger["start_time"]
    #     })
    #
    #     self.wandb = True

    def logger_print(self, epoch, start_time , done):
        self.logger.log_tabular("Epoch", epoch)
        if done:
            self.logger.log_tabular("EpisodeReturn", with_min_and_max=True)
            self.logger.log_tabular("EpisodeLength", average_only=True)
        self.logger.log_tabular("Value", with_min_and_max=True)
        self.logger.log_tabular("Actor_Loss", average_only=True)
        self.logger.log_tabular("Critic_Loss", average_only=True)
        self.logger.log_tabular("Entropy", average_only=True)
        self.logger.log_tabular("Delta_Loss_Actor", average_only=True)
        self.logger.log_tabular("Delta_Loss_Critic", average_only=True)
        self.logger.log_tabular("Time", time.time() - start_time)
        self.logger.dump_tabular()

    def adjust_according_to_space(self, env_space):

        if type(env_space) != gym.spaces.Box:
            dim = env_space.n
        else:
            dim = env_space.shape[0]
        return dim


if __name__ == '__main__':
    # Specify name the experiment as an argument
    env = gym.make('MountainCar-v0')
    env1 = gym.make('CartPole-v0') # discrete actions space

    model = PPO(env.unwrapped,1000,100, name_of_exp="MountainCarMyPPO_CoreAC")
    model.learn(1000)
    model.tb_logger.flush()
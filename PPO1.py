from ActorCriticNetworks import SamplingNetworks
from torch.optim import Adam
from Rollout import Rollout
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import numpy as np
import torch.nn.functional as F
import gym
import time
import torch
from MyPPO.spinningup.spinup.utils.logx import EpochLogger
from MyPPO.spinningup.spinup.utils.logx import colorize


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class PPO1():
    """
        Implementation of the PPO algorithm with clipped surrogate objective
        This implementation can be applied on environments with either discrete or continuous action spaces,
        as mentioned in the original paper.
    """

    def __init__(self, env, batch_size, max_ep_len, name_of_exp, render):
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
        self.actor_critic = SamplingNetworks(self.observations_dim, self.actions_dim)
        #self.critic = SamplingNetworks(self.observations_dim, 1)


        self.init_hyperparameter()

        # in the original implementation of PPO by OpenAI, different values of lr were used for each network
        self.actor_optimiser = Adam(self.actor_critic.modelActor.parameters(), lr=3e-4)
        self.critic_optimiser = Adam(self.actor_critic.modelCritic.parameters(), lr=1e-3)

        self.timesteps = 0
        self.env_continuous = (type(env.action_space) == gym.spaces.Box)
        # self.wandb = False
        self.logger = EpochLogger(name_of_exp)

        # Set up logging for tensorboard
        self.tb_logger = SummaryWriter(name_of_exp)

        # render the environment
        self.render = render

        self.info = {"entropy":np.array([]), "old_policy_loss":0.0, "curr_policy_loss":0.0,
                     "old_value_loss":0.0, "curr_value_loss":0.0}


    def init_hyperparameter(self, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2,
                            entropy=0.01, gradient_descent_steps = 80, kl = 0.01):
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
        self.allowed_kl = kl

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
        self.logger.save_config(locals())

        # Tracking episode length and return
        episode_return = 0
        episode_length = 0

        var_counts = tuple(count_vars(module) for module in [self.actor_critic.modelActor, self.actor_critic.modelCritic])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # To save each episode returns as an array
        episode_return_arr = []
        start_time = time.time()
        info = dict()

        # Epoch or number of iterations. One epoch indicates that the dataset is passed forward and backward
        # through the network one time. In order to optimise the learning using an iterative process such as the
        # gradient policy, we need more than one epoch to pass through the network.
        # The required number of epochs depends on how diverse the dataset is.
        while k < timesteps:
            # Learning Rate Annealing, or should we move that to update
            frac = 1.0 - (k - 1.0) / timesteps
            lr = self.lr*frac

            #self.actor_optimiser.param_groups[0]["lr"] = lr
            #self.critic_optimiser.param_groups[0]["lr"] = lr
            # save the information inside of a rollout
            rollout_t = Rollout(self.batch_size, self.env)
            final_state_reached = False

            for t in range(self.batch_size):
                action, logprob, _ = self.actor_critic.random_action(obs, self.env_continuous)
                value = self.actor_critic(obs, critic=True)

                self.logger.store(Value = value.detach().numpy())

                if self.render:
                    self.env.render()

                # Apply the action on the environment
                next_obs, rew_t, done, _ = self.env.step(action)

                # Move the agent after performing an act
                obs = next_obs

                # Save the data of the episode
                rollout_t.add(action, obs, value, rew_t, logprob, done)

                episode_length += 1

                # Timeout in the spinup implementation
                cutOff = (t == (self.batch_size - 1)) or (episode_length == (self.max_epi_len))

                if not (done or cutOff):
                    episode_return += rew_t
                    episode_return_arr.append(rew_t)
                # If a terminal state is reached, reset the environment
                if done or cutOff:
                    # If a terminal state not reached, but max episode length is reached or the batch size
                    # is exhausted then ask for the reward of the current state, and compute the discounted rewards
                    if cutOff and not(done):
                        print(colorize("The batch is full or the max episode length has "
                                       "been reached, but a terminal state has not been reached",
                                       color='yellow',bold=True))
                        # then take a look at the target value
                        rew_t = self.actor_critic.forward(obs, critic=True).detach().numpy()
                    elif done:
                        # the terminal state has been reached
                        rew_t = 0
                        #final_state_reached = True
                    # Save the episode data
                    self.logger.store(EpisodeReturn = episode_return, EpisodeLength = episode_length)

                    # rollout_t.compute_discounted_rewards(self.gamma, rew_t, done)
                    rollout_t.compute_discounted_rewards2(self.gamma, rew_t)
                    # rollout_t.compute_gae(self.gamma, 0.97, rew_t)
                    #rollout_t.compute_adv_mc()

                    # Reset the environment and zero out the episode data
                    obs = self.env.reset()
                    episode_return, episode_length = 0,0

            k += 1
            # Set up model saving
            if k % 10 == 0:
                self.logger.save_state({'env': self.env}, None)

            self.update(rollout_t)

            self.logger.store(Actor_Loss = self.info.get("old_policy_loss"),
                              Critic_Loss = self.info.get("old_value_loss"))

            delta_policy_loss = np.subtract(self.info.get("curr_policy_loss"), self.info.get("old_policy_loss"))
            delta_value_loss = self.info.get("curr_value_loss") - self.info.get("old_value_loss")

            self.logger.store(Delta_Loss_Actor= delta_policy_loss, Delta_Loss_Critic= delta_value_loss)
            self.logger.store(Entropy = self.info.get("entropy"))


            self.logger_print(k, start_time, final_state_reached)

            # Save the model every 20 iteration, to continue with a trained model
            if (k % 20 == 0) or (k == timesteps-1):
                torch.save(self.actor_critic.state_dict(), './PPO_author.pth')
                #torch.save(self.critic.state_dict(), './critic_author.pth')

            # add the logging info to the tensorboard
            self.tb_logger.add_scalar("losses/value_loss", self.info.get("old_value_loss"), k)
            self.tb_logger.add_scalar("losses/policy_loss", self.info.get("old_policy_loss"), k)
            #self.tb_logger.add_scalar("losses/self.kl", self.kl, k)
            self.tb_logger.add_scalar("losses/approx_entropy", self.info.get("entropy"), k)

    def update(self, rollout):
        """
            Implements the sixth and the sevenths step in the pseudocode of OpenAI.
            First we try to maximise the clip surrogate function, using Adam
            Secondly minimise the mean squared error of the value function, using
        :param rollout: the different trajectories collected in a batch
        :return:
        """
        for i in range(self.gradient_descent_steps):
            self.actor_optimiser.zero_grad()
            actor_loss, entropy, kl = self.compute_loss(rollout)
            if i == 0:
                self.info["old_policy_loss"] = actor_loss.item()
            # Update actor policy only if the change is not large
            if kl > 1.5 * self.allowed_kl:
                print(colorize("Actor could not been updated due to large kl divergence", bold=True, color="blue"))
                break
            # The error at the output is passed through the network
            actor_loss.backward()
            #nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 10)
            self.actor_optimiser.step()

        self.info["entropy"] = entropy.item()
        self.info["curr_policy_loss"] = actor_loss.item()

        for k in range(self.gradient_descent_steps):
            self.critic_optimiser.zero_grad()
            critic_loss = self.compute_loss(rollout, actor= False)
            #critic_loss_clipped = rollout.values + torch.clamp()
            if k == 0:
                self.info["old_value_loss"] = critic_loss.item()
            # Find the local minimum loss
            critic_loss.backward()
            #nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 10)
            self.critic_optimiser.step()

        self.info["curr_value_loss"] = critic_loss.item()

    def compute_loss(self, rollout, actor = True):
        """
        :param rollout:
        :param actor:
        :param critic:
        :return:
        """
        if actor:
            # Compute the advantages after interacting with the environment
            #advantages = rollout.rtgs - rollout.values  # there was a values.detach()

            # To add Importance Sampling property
            prev_logpro = torch.as_tensor(rollout.logprobs, dtype=torch.float32)

            # actions, curr_logpro, entropy = self.actor_critic.random_action(rollout.observations.reshape((-1,)+self.env.observation_space.shape),
            #                                                          self.env_continuous,
            #                                                 rollout.actions.reshape((-1,)+self.env.action_space.shape))
            actions, curr_logpro, entropy = self.actor_critic.random_action(rollout.observations, self.env_continuous,
                                                                            rollout.actions)
            # Since logarithm of the probabilities are used then subtract them instead of dividing them, get the exponent
            ratio = torch.exp(curr_logpro - prev_logpro)
            # KL Divergence to extra ensure no drastic changes
            kl = (curr_logpro - prev_logpro).mean()

            advantages = torch.from_numpy(rollout.adv)
            # Normalising the advantages for a faster learning
            advantages = (advantages - advantages.mean()) / advantages.std()
            # Unclipped objective
            surr1 = advantages * ratio
            # Clipped objective
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # Since Actor and Critic are neural networks that share parameters (i.e. observations caused by action produced
            # by the actor network, are then passed through the critic network. Then the loss function that combines the
            # discounted reward, which is obtained by applying a policy, and the value function. Entropy is added
            # to ensure exploration as suggested in the original paper. It makes the agent more uncertain about the action
            # to choose
            entropy_loss = entropy.mean()
            # loss =  -(torch.min(surr1, surr2) + entropy_loss*self.entropy_beta)
            loss = -(torch.min(surr1, surr2))

            return loss.mean(), entropy_loss, kl

        else:
            #disc = (rollout.discounted_rew - rollout.discounted_rew.mean()) / (rollout.discounted_rew.std())
            values = self.actor_critic.forward(rollout.observations, critic=True)
            return ((values - torch.from_numpy(rollout.discounted_rew))**2).mean()


    def logger_print(self, epoch, start_time, done):
        self.logger.log_tabular("Epoch", epoch)
        # if done:
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
    env = gym.make('MountainCar-v0')
    env1 = gym.make('MountainCarContinuous-v0') # discrete actions space

    model = PPO1(env.unwrapped, 300, 100, name_of_exp="xxx", render=False)
    model.learn(1000)
    model.tb_logger.flush()

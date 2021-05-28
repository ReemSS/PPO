from ActorCriticNetworksA import Critic, CategoricalActor, GaussianActor
from ActorCriticNetworks import ActorCritic
from torch.optim import Adam
from Rollout import Rollout
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import time
import torch
import wandb
import random
from MyPPO.spinningup.spinup.utils.logx import EpochLogger
from MyPPO.spinningup.spinup.utils.logx import colorize

class PPO():
    """
        Implementation of the PPO algorithm with clipped surrogate objective
        This implementation can be applied on environments with either discrete or continuous action spaces,
        as mentioned in the original paper.
    """
    def __init__(self, env, max_ep_len, name_of_exp, render):
        # Extract environment information
        # Adjust the environment and action space to turn them into a valid input for
        # the actor network and the critic network
        self.observations_dim = adjust_according_to_space(env.observation_space)
        self.actions_dim = adjust_according_to_space(env.action_space)

        # In every batch we might have a number of trajectories, it is good to define the max length of the episode,
        # in case we did not reach a terminal state, in other words we define a timeout.
        self.max_epi_len = max_ep_len

        # create the networks of A2C agent to interact with the environment.
        self.env_continuous = (type(env.action_space) == gym.spaces.Box)
        if self.env_continuous:
            self.actor = GaussianActor(self.observations_dim, self.actions_dim)
        else:
            self.actor = CategoricalActor(self.observations_dim, self.actions_dim)

        self.critic = Critic(self.observations_dim)

        self.init_hyperparameter()
        self.init_wandb(project=name_of_exp)

        self.ac = ActorCritic(self.observations_dim, self.actions_dim, self.neuron_num, self.env_continuous)


        # in the original implementation of PPO by OpenAI, different values of lr were used for each network
        self.actor_optimiser = Adam(self.ac.actor.parameters(), lr=self.learning_rate)
        self.critic_optimiser = Adam(self.ac.critic.parameters(), lr=self.learning_rateV)
        self.env = gym.make(env.spec.id)
        # self.env = env
        # self.env = gym.wrappers.Monitor(self.env, name_of_exp, force=True)
        self.logger = EpochLogger(name_of_exp)

        # Set up logging for tensorboard
        self.tb_logger = SummaryWriter(name_of_exp)

        # render the environment
        self.render = render

        self.info = {"entropy":0.0, "kl":0.0, "old_policy_loss":0.0, "curr_policy_loss":0.0,
                     "old_value_loss":0.0, "curr_value_loss":0.0}

        self.init_seed(1)

    def init_seed(self, seed):
        """
            It is a control source of randomness to reproduce the same results with same data.
        :param seed:
        :return:
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_wandb(self, project):
        self.config = wandb.init(project= project, entity="rfarah", sync_tensorboard=True).config


    def init_hyperparameter(self, learning_rate=3e-2, gamma=0.99, clip_ratio=0.2, lam = 0.97, learning_rateV = 1e-2,
                            entropy=0.01, gradient_descent_steps = 80, neuron_num = 64, batch_size = 512):
        """
            Initialise the hyperparameters of the algorithm. The used default values are the same values
            suggested in the original paper.
        :param learning_rate: For the actor models
        :param learning_rate: For the critic models
        :param gamma: discounter factor
        :param lam: the decay rate
        :param clip_ratio: projection the trust region concept onto the algorithm
        :param entropy: added to the loss function to guarantee a sufficient exploration
        :param gradient_descent_steps: number of steps the gradient descent needs to take by the networks
        :param neuron_num: number of neurons in the networks
        :param batch_size: the batch size
        :return:
        """
        self.learning_rate = learning_rate
        self.learning_rateV = learning_rateV
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_beta = entropy
        self.gradient_descent_steps = gradient_descent_steps
        self.neuron_num = neuron_num
        self.batch_size = batch_size

    def learn(self, timesteps):
        """
            The method implements pPO as follows:
                - First Step: Collecting Trajectories using Rollout to save the info in '~CollectingSamples.Rollout'
                - Second Step: Compute the advantages and the discounted rewards.
                - Third Step: update the policy loss by maximising the policy loss, i.e., the objective function.
                - forth Step: update the value loss by minimising the value loss.
        :param timesteps: Number of interactions with the environment: timesteps * batch size
        """
        # Counter
        k = 0
        obs = self.env.reset()
        # Save configuration as json file
        # self.logger.save_config(locals())
        # Tracking episode length and return
        episode_length = 0

        # To save each episode returns as an array
        start_time = time.time()

        wandb.watch(self.ac)

        self.logger.setup_pytorch_saver(self.ac)
        # Epoch or number of iterations. One epoch indicates that the dataset is passed forward and backward
        # through the network one time. In order to optimise the learning using an iterative process such as the
        # gradient policy, we need more than one epoch to pass through the network.
        # The required number of epochs depends on how diverse the dataset is.
        number_of_interactions = 0
        rewards_per_episode, complete_episode = 0, 0
        episode = 0
        lr = self.learning_rate
        lrV = self.learning_rateV

        while k < timesteps - 1:
            number_of_interactions += k
            # save the information inside of a rollout
            batch_size = self.batch_size
            rollout_t = Rollout(batch_size, self.env)

            # Learning Rate Annealing
            # if k % 100 == 0 and k != 0:
            #     frac = 1.0 - (k - 1.0) / timesteps
            #     lr = lr*frac
            #     lrV = lrV*frac
            #     self.actor_optimiser.param_groups[0]["lr"] = lr
            #     self.critic_optimiser.param_groups[0]["lr"] = lrV

            for t in range(batch_size):
                action, value, logprob = self.ac.step(obs)
                self.logger.store(Value=value.item())
                # Apply the action on the environment
                next_obs, rew_t, done, _ = self.env.step(action)
                # done = (obs[0] >= 0.5) # for mountain Car only
                if self.render or done:
                    self.env.render()

                rewards_per_episode += rew_t
                complete_episode += rew_t
                episode_length += 1

                # Save the data of the episode
                rollout_t.add(action, obs, value, rew_t, logprob)
                # Move the agent after performing an act]
                obs = next_obs
                # Timeout in the spinup implementation
                cutOff =  (episode_length > (self.max_epi_len-1))

                # If a terminal state is reached, reset the  environment
                # if done or cutOff:
                if done: # in lunar lander there is a terminal state, no need to cutt off the episode
                    # If a terminal state not reached, but the batch size is exhausted then ask for the reward of the
                    # current state, and compute the discounted rewards
                    if (t == (self.batch_size - 1)) and not done:
                        print(colorize("The batch is full or the max episode length has "
                                       "been reached, but a terminal state has not been reached",
                                       color='yellow',bold=True))
                    if done: # for Pendulum, since the wrapper ends the episode, when its length is equal to 200
                        # the terminal state has been reached
                        self.logger.store(EpisodeReturn = rewards_per_episode, EpisodeLength = episode_length)
                        rew_t = 0
                        # wandb.log({"complete episode reward": rewards_per_episode, "episode length": episode_length})
                        obs = self.env.reset()
                        # wandb.log({"episode reward": rewards_per_episode})
                        episode += 1
                        complete_episode = 0
                    else:
                        # then take a look at the target value
                        _, rew_t,_ = self.ac.step(obs)

                    wandb.log({"episode reward": rewards_per_episode})

                    rewards_per_episode, episode_length = 0,0

                    # rollout_t.compute_discounted_rewards(self.gamma, rew_t)
                    # rollout_t.compute_adv_nsteps(self.gamma, rew_t)
                    rollout_t.compute_gae(self.gamma, self.lam, rew_t)
            k += 1
            # Set up model saving
            if k % 10 == 0:
                self.logger.save_state({'env': self.env}, None)

            # if k % 5 == 0:
            old_policy_loss, curr_policy_loss, old_value_loss, curr_value_loss, kl, entropy = self.update(rollout_t)
            self.logger.store(Actor_Loss = old_policy_loss,
                              Critic_Loss = old_value_loss)

            delta_policy_loss = np.subtract(curr_policy_loss, old_policy_loss)
            delta_value_loss = curr_value_loss - old_value_loss

            self.logger.store(Delta_Loss_Actor= delta_policy_loss, Delta_Loss_Critic= delta_value_loss)
            self.logger.store(Entropy = entropy)

            self.logger_print(k, start_time)

            # Save the model every 20 iteration, to continue with a trained model
            if (k % 20 == 0) or (k == timesteps-1):
                torch.save(self.ac.actor, './actor_ppo_valueClipped.pth')
                torch.save(self.ac.critic, './critic_ppo_valueClipped.pth')
                torch.save(self.ac.actor.state_dict(), './actor_ppo_valueClipped2.pth')
                torch.save(self.ac.critic.state_dict(), './critic_ppo_valueClipped2.pth')

            # add the logging info to the tensorboard
            self.tb_logger.add_scalar("losses/value_loss", old_value_loss, number_of_interactions)
            self.tb_logger.add_scalar("losses/policy_loss", old_policy_loss, number_of_interactions)
            self.tb_logger.add_scalar("losses/approx.kl", self.approx_kl, number_of_interactions)
            self.tb_logger.add_scalar("losses/approx_entropy", entropy, number_of_interactions)
            self.tb_logger.flush()

            wandb.log({"losses/value_loss": old_value_loss, "steps": number_of_interactions})
            wandb.log({"losses/policy_loss": old_policy_loss, "steps": number_of_interactions})
            wandb.log({"losses/approx.kl": self.approx_kl, "steps": number_of_interactions})
            wandb.log({"losses/approx_entropy": entropy, "steps": number_of_interactions})
        self.tb_logger.close()
        self.env.close()

    def update(self, rollout):
        """
            Implements the sixth and the sevenths step in the pseudocode of OpenAI.
            First we try to maximise the clip surrogate function, using Adam
            Secondly minimise the mean squared error of the value function, using
        :param rollout: the different trajectories collected in a batch
        :return:
        """
        old_policy_loss, entropy, _ = self.compute_loss(rollout)
        critic_loss,_ = self.compute_loss(rollout, actor=False)
        old_value_loss = critic_loss.item()
        for i in range(self.gradient_descent_steps):
            self.actor_optimiser.zero_grad()
            actor_loss, entropy, kl = self.compute_loss(rollout)
            # Update actor policy only if the change is not large
            if abs(kl) > 0.03:#0.015
                print(colorize("Actor could not been updated due to large kl divergence", bold=True, color="blue"))
                break
            self.approx_kl = kl
            # The error at the output is passed through the network
            actor_loss.backward()
            # nn.utils.clip_grad_norm(self.ac.actor.parameters(), 0.5)
            self.actor_optimiser.step()

        for i in range(self.gradient_descent_steps):
            self.critic_optimiser.zero_grad()
            critic_loss, diff = self.compute_loss(rollout, actor=False)
            # if abs(diff) > 1.5:
            #     break
            # Find the local minimum loss
            critic_loss.backward()
            # nn.utils.clip_grad_norm(self.ac.critic.parameters(), 0.5)
            self.critic_optimiser.step()

        curr_value_loss = critic_loss.item()
        curr_policy_loss = actor_loss.item()
        return old_policy_loss.item(), curr_policy_loss, old_value_loss, curr_value_loss, kl, entropy

    def compute_loss(self, rollout, actor = True):
        """
        :param rollout:
        :param actor:
        :param critic:
        :return:
        """
        if actor:
            # Compute the advantages after interacting with the environment
            # advantages = rollout.rtgs - rollout.values  # there was a values.detach()

            # To add Importance Sampling property
            prev_logpro = torch.as_tensor(rollout.logprobs, dtype=torch.float32)

            # actions, curr_logpro, entropy = self.actor_critic.random_action(rollout.observations.reshape((-1,)+self.env.observation_space.shape),
            #                                                          self.env_continuous,
            #                                                 rollout.actions.reshape((-1,)+self.env.action_space.shape))
            actions, curr_logpro, entropy = self.ac.actor(rollout.observations, rollout.actions, sample=True)
            # Since logarithm of the probabilities are used then subtract them instead of dividing them, get the exponent
            curr_logpro = torch.as_tensor(curr_logpro, dtype=torch.float32)
            ratio = torch.exp(curr_logpro - prev_logpro)

            # KL Divergence to extra ensure no drastic changes, KL Divergence and ratio (Trust region) helps in
            # not exploding the gradient
            kl = (prev_logpro - curr_logpro).mean().item()
            # discounted reward, which is obtained by applying a policy, and the value function. Entropy is added

            advantages = rollout.adv
            # Normalising the advantages for a faster learning. Important to add a small number, otherwise after a num
            # of iterations the gradient will explode and the actor loss turned to NaN
            advantages = (advantages - advantages.mean()) / (advantages.std()+ 1e-8)
            advantages = torch.as_tensor(advantages, dtype=torch.float32)
            # Unclipped objective
            surr1 = advantages * ratio
            # Clipped objective
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # Since Actor and Critic are neural networks that share parameters (i.e. observations caused by action produced
            # by the actor network, are then passed through the critic network. Then the loss function that combines the
            # to ensure exploration as suggested in the original paper. It makes the agent more uncertain about the action
            # to choose
            entropy_loss = entropy.mean()
            loss =  -torch.min(surr1, surr2).mean() - (entropy_loss*self.entropy_beta)
            # loss = -(torch.min(surr1, surr2)).mean()
            return loss , entropy_loss.item(), kl
        else:
            rtgs = torch.as_tensor(rollout.discounted_rew, dtype=torch.float32)
            obs = torch.as_tensor(rollout.observations, dtype=torch.float32)
            old_values = torch.as_tensor(rollout.values, dtype=torch.float32)

            new_values = self.ac.critic(obs)
            normalised_rtgs = (rtgs - rtgs.mean()) / rtgs.std()
            diff = (new_values - old_values)

            # Trust region for the value loss
            clipped_value = old_values + torch.clamp(diff, -self.clip_ratio, self.clip_ratio)
            loss_unclipped = (new_values - rtgs)**2
            loss_clipped = (clipped_value - rtgs)**2
            # Maximise the loss so th critic learns faster how to give a valid feedback, since value is needed
            # in computing the advantages and the discounted rewards.
            loss = 0.5*torch.max(loss_unclipped, loss_clipped).mean()
            # loss = ((new_values - rtgs)**2).mean()
            return loss, diff.mean()

    def logger_print(self, epoch, start_time):
        self.logger.log_tabular("Epoch", epoch)
        # self.logger.log_tabular("EpisodeReturn", with_min_and_max=True)
        # self.logger.log_tabular("EpisodeLength", average_only=True)
        self.logger.log_tabular("Value", with_min_and_max=True)
        self.logger.log_tabular("Actor_Loss", average_only=True)
        self.logger.log_tabular("Critic_Loss", average_only=True)
        self.logger.log_tabular("Entropy", average_only=True)
        self.logger.log_tabular("Delta_Loss_Actor", average_only=True)
        self.logger.log_tabular("Delta_Loss_Critic", average_only=True)
        self.logger.log_tabular("Time", time.time() - start_time)
        self.logger.dump_tabular()

def adjust_according_to_space(env_space):

    if type(env_space) != gym.spaces.Box:
        dim = env_space.n
    else:
        dim = env_space.shape[0]
    return dim

def adjust_rewards(name, obs, rewards):
    if name == "Mountain Car":
        if obs[0] >= 0.5:
            return 1
        else:
            return obs[0] + 0.5

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    envCon = gym.make('Pendulum-v0')
    envCon1 = gym.make('LunarLanderContinuous-v2')

    model = PPO(envCon1.unwrapped, 200, name_of_exp="LunarLanderCon", render=False)
    model.learn(10000)
    # model.tb_logger.flush()
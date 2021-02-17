from ActorCriticNetworks import SamplingNetworks
from torch.optim import Adam
from CollectingSamples import Rollout

class PPO():
    def __init__(self, env, batch_size):
        # extract environment information
        self.env = env # do we really need it
        self.observations_dim = env.observation_space.shape[0]  # or just shape
        self.actions_dim = env.action_space.shape[0]
        self.batch_size = batch_size

        self.actor = SamplingNetworks(self.observations_dim, self.actions_dim)
        self.critic = SamplingNetworks(self.observations_dim, 1)

        # in the original implementation of PPO by OpenAI, different lr were used for each network
        self.actor_optimisers = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimisers = Adam(self.critic.parameters(), lr=self.lr)

    # Hyperparameter

        self.init_hyperparameter()





    def init_hyperparameter(self, learning_rate, learning_steps, gamma, clip_ratio, eps_len_T):
        self.lr = learning_rate
        self.learning_steps = learning_steps # (K)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.eps_len_T = eps_len_T


    def learn(self, timesteps):
        k = 0
        # create a trajectory, or list of rollouts
        trajectory = []

        # start the first interacting
        obs = self.env.reset()

        while k < timesteps:
            for t in range(self.batch_size):
                # add each rollout to the trajectory list
                rollout_t = Rollout(self.batch_size, self.observations_dim, self.actions_dim)
                action, logprob = self.actor(obs)
                trajectory.append(rollout_t)

                t += 1




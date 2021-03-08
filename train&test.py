import gym
#from baselines.ppo1 import pposgd_simple as ppo_baseline
from spinningup.spinup.algos.pytorch.ppo import ppo
import spinningup.spinup.algos.pytorch.ppo.core as core
from spinningup.spinup.utils.run_utils import setup_logger_kwargs
from spinningup.spinup.utils.logx import colorize

def train(env):

    print(colorize(" Training within the environment %s"%env.unwrapped.spec.id, 'blue', bold=True))

    #agent = PPO(env)
    #ppo_baseline

    print("Now Learning with the Spinup Implementation", flush=True)

    logger_kwargs = setup_logger_kwargs(env.spec.id)

    ppo.ppo(lambda : env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[64] * 2),
            gamma=0.99, seed=0, steps_per_epoch=1500, epochs = 50)


    return 0

if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')
    env = env.unwrapped
    train(env)

    env.close()

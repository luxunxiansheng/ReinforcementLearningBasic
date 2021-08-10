import sys,os

current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

from algorithm.policy_based.actor_critic.batch_actor_critic import BatchActorCriticAgent
from algorithm.policy_based.actor_critic.online_actor_critic import OnlineCriticActorAgent
from algorithm.policy_based.actor_critic.batch_a3c import BatchA3CAgent

from env.mountain_car import MountainCarEnv

from env_setup import get_env

num_episodes = 5000


def test_batch_critic_actor_method(env):
    batch_actor_actor=BatchActorCriticAgent(env,num_episodes)
    batch_actor_actor.learn()

def test_online_critic_actor_method(env):
    online_actor_actor=OnlineCriticActorAgent(env,num_episodes)
    online_actor_actor.learn()

def test_batch_a3c_method(env):
    batch_a3c_agent=BatchA3CAgent(env,num_episodes)
    batch_a3c_agent.learn()


if __name__=='__main__':
    real_env = get_env("MountainCarEnv")
    test_batch_a3c_method(real_env)

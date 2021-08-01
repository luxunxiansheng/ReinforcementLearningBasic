import sys,os

current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

from algorithm.policy_based.actor_critic.batch_actor_critic import BatchActorCriticAgent
from algorithm.policy_based.actor_critic.online_actor_critic import OnlineCriticActorAgent

from env.mountain_car import MountainCarEnv

from test_setup import get_env

num_episodes = 500


def test_batch_critic_actor_method(env):
    batch_actor_actor=BatchActorCriticAgent(env,num_episodes)
    batch_actor_actor.learn()

def test_online_critic_actor_method(env):
    online_actor_actor=OnlineCriticActorAgent(env,num_episodes)
    online_actor_actor.learn()



if __name__=='__main__':
    real_env = get_env(MountainCarEnv.__name__)
    test_online_critic_actor_method(real_env)

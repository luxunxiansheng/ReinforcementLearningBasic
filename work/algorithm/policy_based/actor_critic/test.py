import sys,os

current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

from policy.policy import ParameterizedPolicy
from algorithm.policy_based.actor_critic.batch_actor_critic import BatchActor, BatchCritic, BatchCriticActor, PolicyEsitmator, ValueEestimator
from algorithm.policy_based.actor_critic.online_actor_critic import OnlineActor, OnlineCritic, OnlineCriticActor, PolicyEsitmator, ValueEestimator

from lib import plotting
from test_setup import get_env

num_episodes = 500

def test_batch_critic_actor_method(env):
    value_estimator = ValueEestimator(env.observation_space.shape[0])
    critic = BatchCritic(value_estimator)

    policy_estimator = PolicyEsitmator(env.observation_space.shape[0],env.action_space.n)
    policy = ParameterizedPolicy(policy_estimator)
    actor = BatchActor(policy)

    critic_actor = BatchCriticActor(critic,actor,env,num_episodes)
    critic_actor.improve()

def test_online_critic_actor_method(env):
    value_estimator = ValueEestimator(env.observation_space.shape[0])
    critic = OnlineCritic(value_estimator)

    policy_estimator = PolicyEsitmator(env.observation_space.shape[0],env.action_space.n)
    policy = ParameterizedPolicy(policy_estimator)
    actor = OnlineActor(policy,critic)

    critic_actor = OnlineCriticActor(critic,actor,env,num_episodes)
    critic_actor.improve()


real_env = get_env("mountaincar")


#test_online_critic_actor_method(real_env)
test_batch_critic_actor_method(real_env)


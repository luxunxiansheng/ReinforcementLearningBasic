import sys,os

current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

from policy.policy import ParameterizedPolicy
from algorithm.policy_based.actor_critic.batch_actor_critic import BatchActor, BatchCritic, BatchCriticActor, PolicyEsitmator, ValueEestimator
from algorithm.policy_based.actor_critic.online_actor_critic import OnlineActor, OnlineCritic, OnlineCriticActor, PolicyEsitmator, ValueEestimator
from algorithm.policy_based.actor_critic.batch_a3c import BatchA3C, GlobalPolicyEsitmator, GlobalValueEestimator
from env.mountain_car import MountainCarEnv

from lib import plotting
from test_setup import get_env

num_episodes = 500

def test_batch_a3c_method(env):
    gloal_value_estimator = GlobalValueEestimator(env.observation_space.shape[0])

    global_policy_estimator = GlobalPolicyEsitmator(env.observation_space.shape[0],env.action_space.n)
    global_policy = ParameterizedPolicy(global_policy_estimator)

    critic_actor = BatchA3C(gloal_value_estimator,global_policy,env,num_episodes)
    critic_actor.improve()

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



#test_online_critic_actor_method(real_env)
#test_batch_critic_actor_method(real_env)

if __name__=='__main__':
    real_env = get_env(MountainCarEnv.__name__)
    test_batch_a3c_method(real_env)

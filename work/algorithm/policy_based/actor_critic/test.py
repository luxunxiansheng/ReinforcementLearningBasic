import sys,os

current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

from policy.policy import ParameterizedPolicy
from algorithm.policy_based.actor_critic.actor_critic import Actor, Critic, CriticActor, PolicyEsitmator, ValueEestimator

from lib import plotting
from policy.policy import ContinuousStateValueBasedPolicy
from test_setup import get_env

num_episodes = 200

def test_critic_actor_method(env):
    value_estimator = ValueEestimator(env.observation_space.shape[0])
    critic = Critic(value_estimator)

    policy_estimator = PolicyEsitmator(env.observation_space.shape[0],env.action_space.n)
    policy = ParameterizedPolicy(policy_estimator)
    actor = Actor(policy)

    critic_actor = CriticActor(critic,actor,env,num_episodes)
    critic_actor.improve()

real_env = get_env("mountaincar")

test_critic_actor_method(real_env)


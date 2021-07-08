import sys,os
current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

import numpy as np 
from lib.plotting import EpisodeStats, plot_episode_stats
from test_setup import get_env
from algorithm.value_based.tabular_solution_method.monte_carlo_method.monte_carlo_es_control import MonteCarloESControl
from algorithm.value_based.tabular_solution_method.monte_carlo_method.monte_carlo_on_policy_control import MonteCarloOnPolicyControl

num_episodes = 500000

def test_monte_carlo_es_control_method(env):
    monte_carlo_es_control_statistics = EpisodeStats("monte_carlo_es_controal_statistics", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)
    mc_es_control = MonteCarloESControl(env,monte_carlo_es_control_statistics,num_episodes)
    mc_es_control.learn()
    return monte_carlo_es_control_statistics

def test_monte_carlo_onpolicy_control_method(env):
    monte_carlo_onpolicy_control_statistics = EpisodeStats("monte_carlo_online_controal_statistics", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)
    mc_online_control = MonteCarloOnPolicyControl(env,monte_carlo_onpolicy_control_statistics,num_episodes)
    mc_online_control.learn()
    return monte_carlo_onpolicy_control_statistics

def test_mc_control_method(env):
    plot_episode_stats([test_monte_carlo_onpolicy_control_method(env)])

real_env = get_env("BlackjackEnv")
test_mc_control_method(real_env)




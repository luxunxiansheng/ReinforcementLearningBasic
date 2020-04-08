import numpy as np

from algorithm.td_method.qlearning import QLearning
from algorithm.td_method.sarsa import SARSA
from algorithm.td_method.td0_evaluation_method import TD0_Evalutaion_Method
from lib import plotting
from policy.policy import TabularPolicy


def test_td0_evaluation_method(env):
    v_table = env.build_V_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)
    td0_method = TD0_Evalutaion_Method(v_table, b_policy, env)
    td0_method.evaluate()


def test_sarsa_method(env):
    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    num_episodes = 1000
    sarsa_statistics = plotting.EpisodeStats("sarsa", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes))
    sarsa_method = SARSA(q_table, b_policy, 0.1, env,
                         sarsa_statistics, num_episodes)
    sarsa_method.improve()
    plotting.plot_episode_stats([sarsa_statistics])


def test_qlearning_method(env):
    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    num_episodes = 10000
    q_learning_statistics = plotting.EpisodeStats("qlearning", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes))

    qlearning_method = QLearning(
        q_table, b_policy, 0.1, env, q_learning_statistics, num_episodes)
    qlearning_method.improve()

    plotting.plot_episode_stats([q_learning_statistics,sarsa_statistics])

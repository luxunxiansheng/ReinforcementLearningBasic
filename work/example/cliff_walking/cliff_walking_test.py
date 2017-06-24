import sys
import itertools
from collections import defaultdict

import matplotlib
import numpy as np

sys.path.append('/home/ornot/GymRL')
from algorithm import mc_offline_policy_control, mc_online_policy_control, q_learning, sarsa, expected_sarsa
from env import cliff_walking
from lib import plotting


matplotlib.style.use('ggplot')

env = cliff_walking.CliffWalkingEnv()

num_episodes = 200

# TD online
statistics_sara = plotting.EpisodeStats("sara", episode_lengths=np.zeros(
    num_episodes), episode_rewards=np.zeros(num_episodes))
Q_sara = sarsa.sarsa(env, num_episodes, statistics_sara)


# TD_offline
statistics_q_learning = plotting.EpisodeStats("q_learning", episode_lengths=np.zeros(
    num_episodes), episode_rewards=np.zeros(num_episodes))
Q_learning = q_learning.q_learning(env, num_episodes, statistics_q_learning)

# expected_sarsa. Note: It is hard for expected sarsa to reach the final point
'''statistics_expected_sarsa = plotting.EpisodeStats("expected_sarsa", episode_lengths=np.zeros(
    num_episodes), episode_rewards=np.zeros(num_episodes))
Q_exptected_sara= expected_sarsa.expected_sarsa(
    env, num_episodes, statistics_expected_sarsa)'''

plotting.plot_episode_stats(
    [statistics_expected_sarsa,statistics_q_learning])

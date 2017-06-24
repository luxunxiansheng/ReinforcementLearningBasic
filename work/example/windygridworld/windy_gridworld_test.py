"""
  What we learned from this windy_gridworld example is TD is far more
  better than MC for it is very difficult for MC to reach the final
  state. It takes a very very long steps which can not be unberable
"""


import itertools
import sys
from collections import defaultdict

import matplotlib
import numpy as np

sys.path.append('/home/ornot/GymRL')
from algorithm import mc_online_policy_control, q_learning, sarsa, expected_sarsa
from env import windy_gridworld
from lib import plotting


matplotlib.style.use('ggplot')

env = windy_gridworld.WindyGridworldEnv()

num_episodes = 200

# TD online
statistics_sara = plotting.EpisodeStats("sara", episode_lengths=np.zeros(
    num_episodes), episode_rewards=np.zeros(num_episodes))
Q_sara = sarsa.sarsa(env, num_episodes, statistics_sara)

# TD_offline
statistics_q_learning = plotting.EpisodeStats("q_learning", episode_lengths=np.zeros(
    num_episodes), episode_rewards=np.zeros(num_episodes))
Q_learning = q_learning.q_learning(env, num_episodes, statistics_q_learning)




plotting.plot_episode_stats(
    [statistics_sara, statistics_q_learning])

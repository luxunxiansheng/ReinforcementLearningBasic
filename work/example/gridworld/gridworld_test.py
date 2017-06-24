import itertools
import sys
from collections import defaultdict

import matplotlib
import numpy as np

sys.path.append('/home/ornot/GymRL')
from algorithm import mc_offline_policy_control, mc_online_policy_control, q_learning, sarsa
from env import grid_world
from lib import plotting


matplotlib.style.use('ggplot')

env = grid_world.GridworldEnv()

num_episodes = 200

# TD online
statistics_sara = plotting.EpisodeStats("sara", episode_lengths=np.zeros(
    num_episodes), episode_rewards=np.zeros(num_episodes))
Q_sara = sarsa.sarsa(env, num_episodes, statistics_sara)

# TD_offline
statistics_q_learning = plotting.EpisodeStats("q_learning", episode_lengths=np.zeros(
    num_episodes), episode_rewards=np.zeros(num_episodes))
Q_learning = q_learning.q_learning(env, num_episodes, statistics_q_learning)


# MC online
statistics_mc_online = plotting.EpisodeStats("mc_online", episode_lengths=np.zeros(
    num_episodes), episode_rewards=np.zeros(num_episodes))
Q_mc_online = mc_online_policy_control.mc_control_epsilon_greedy(
    env, num_episodes, statistics_mc_online)

# MC_offline
statistics_mc_offline = plotting.EpisodeStats("mc_offline", episode_lengths=np.zeros(
    num_episodes), episode_rewards=np.zeros(num_episodes))
Q_mc_offline = mc_offline_policy_control.mc_control_importance_sampling(
    env, num_episodes, statistics_mc_offline)


plotting.plot_episode_stats(
    [statistics_sara, statistics_q_learning, statistics_mc_online, statistics_mc_offline])

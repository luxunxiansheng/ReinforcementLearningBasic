import sys

import matplotlib
import numpy as np

import gym
import tile_coding_estimator


sys.path.append('/home/ornot/GymRL')
from algorithm import (expected_sarsa_tile_coding, q_learning_tile_coding,
                       sarsa_tile_coding)
from lib import plotting


def main():
    matplotlib.style.use('ggplot')

    env = gym.envs.make("MountainCar-v0")

    num_episodes = 100

    estimator_q_learning = tile_coding_estimator.Estimator(env)
    statistics_q_learning = plotting.EpisodeStats("q_learning",
                                                  episode_lengths=np.zeros(num_episodes),
                                                  episode_rewards=np.zeros(num_episodes))

    # Note: For the Mountain Car we don't actually need an epsilon > 0.0
    # because our initial estimate for all states is too "optimistic" which leads
    # to the exploration of all states.
    q_learning_tile_coding.q_learning(env, estimator_q_learning, num_episodes, statistics_q_learning, epsilon=0.0)
    plotting.plot_cost_to_go_mountain_car(env, estimator_q_learning)

    estimator_sarsa = tile_coding_estimator.Estimator(env)
    statistics_sarsa = plotting.EpisodeStats("sarsa", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes))

    # Note: For the Mountain Car we don't actually need an epsilon > 0.0
    # because our initial estimate for all states is too "optimistic" which leads
    # to the exploration of all states.
    sarsa_tile_coding.sarsa(env, estimator_sarsa, num_episodes,
                            statistics_sarsa, epsilon=0.0)
    plotting.plot_cost_to_go_mountain_car(env, estimator_sarsa)
    
    estimator_expected_sarsa = tile_coding_estimator.Estimator(env)
    statistics_expected_sarsa = plotting.EpisodeStats("expected_sarsa", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes))

    # Note: For the Mountain Car we don't actually need an epsilon > 0.0
    # because our initial estimate for all states is too "optimistic" which leads
    # to the exploration of all states.
    expected_sarsa_tile_coding.expected_sarsa(env, estimator_expected_sarsa, num_episodes,statistics_expected_sarsa, epsilon=0.0)
    plotting.plot_cost_to_go_mountain_car(env, estimator_expected_sarsa)



    plotting.plot_episode_stats(
        [statistics_q_learning, statistics_sarsa,statistics_expected_sarsa], smoothing_window=25)


if __name__ == '__main__':
    main()

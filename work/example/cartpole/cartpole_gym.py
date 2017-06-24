import sys
import itertools
import gym

import numpy as np

sys.path.append('/home/ornot/GymRL')
from algorithm import (expected_sarsa_tile_coding, q_learning_tile_coding,
                       sarsa_tile_coding)
from lib import plotting
from lib import utility
from lib import tile_coding_estimator


def run_episode(env, greedy_policy):
    observation = env.reset()
    for t in itertools.count():
        env.render()
        action = utility.make_decision(greedy_policy, observation)
        ob, reward, done, info = env.step(action)
        if done:
            break
        observation = ob


def main():
    env = gym.make('CartPole-v0')

    estimator_q_learning = tile_coding_estimator.Estimator(env)
    greedy_policy = utility.make_greedy_policy_with_fa(
        estimator_q_learning, env.action_space.n)

    num_episodes = 1000
    statistics_q_learning = plotting.EpisodeStats("q_learning", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes))

    q_learning_tile_coding.q_learning(
        env, estimator_q_learning, num_episodes, statistics_q_learning, epsilon=0.0)

    
    run_episode(env, greedy_policy)


if __name__ == '__main__':
    main()

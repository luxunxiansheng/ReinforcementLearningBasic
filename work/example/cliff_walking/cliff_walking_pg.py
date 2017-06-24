'''
  To reduce the variance, a value estimator is used to apprimate the baseline.
  In some states all actions have high values and we need a high baseline to 
  dierentiate the higher valued actions from the less highly valued ones;
  in other states all actions will have low values and a low baseline is
  appropriate
'''

import collections
import itertools
import sys

import numpy as np
import tensorflow as tf

import gym

sys.path.append('/home/ornot/GymRL')
from algorithm import actor_critic, reinforce
from env.cliff_walking import CliffWalkingEnv
from lib import plotting, policy_estimator, value_estimator


def main():

    env = CliffWalkingEnv()
    num_episodes = 500

    tf.reset_default_graph()
    tf.Variable(0, name="global_step", trainable=False)

    policyEstimatorReinforce = policy_estimator.PolicyEstimator(env,scope="Policy_Estimator_Reinforce")
    valueEstimatorReinforce = value_estimator.ValueEstimator(env,scope="Value_Estimator_Reinforce")
    statistics_reinforce = plotting.EpisodeStats("Reinforce", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes))

    policyEstimatorAC = policy_estimator.PolicyEstimator(env,scope="Policy_Estimator_AC")
    valueEstimatorAC = value_estimator.ValueEstimator(env,scope="Value_Estimator_AC")
    statistics_ac = plotting.EpisodeStats("AC", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # Note, due to randomness in the policy the number of episodes you need to learn a good
        # policy may vary. ~2000-5000 seemed to work well for me.
        reinforce.reinforce(env, statistics_reinforce, policyEstimatorReinforce,
                            valueEstimatorReinforce, num_episodes, discount_factor=1.0)
        actor_critic.actor_critic(
            env, statistics_ac, policyEstimatorAC, valueEstimatorAC, num_episodes)

    plotting.plot_episode_stats(
        [statistics_reinforce, statistics_ac], smoothing_window=25)


if __name__ == '__main__':
    main()

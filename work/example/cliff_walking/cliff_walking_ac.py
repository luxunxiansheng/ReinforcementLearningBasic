
import collections
import itertools
import sys

import numpy as np
import tensorflow as tf

import gym

sys.path.append('/home/ornot/GymRL')
from algorithm import actor_critic
from env.cliff_walking import CliffWalkingEnv
from lib import (plotting,policy_estimator,value_estimator)


env = CliffWalkingEnv()

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = policy_estimator.PolicyEstimator(env)
value_estimator = value_estimator.ValueEstimator(env)

num_episodes = 300
statistics_ac = plotting.EpisodeStats(
    "actor-critic", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    actor_critic.actor_critic(env, statistics_ac, policy_estimator,
                 value_estimator, num_episodes)


plotting.plot_episode_stats([statistics_ac], smoothing_window=10)

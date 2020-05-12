# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /

import heapq
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from lib.utility import create_distribution_epsilon_greedily


class DynaQ:
    def __init__(self, q_table, behavior_table_policy, epsilon, env, statistics, episodes, iterations= 15, step_size=0.1,  discount=1.0):
        self.q_table = q_table
        self.policy = behavior_table_policy
        self.env = env
        self.episodes = episodes
        self.step_size = step_size
        self.discount = discount
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(
            epsilon)
        self.statistics = statistics
        self.model = TrivialModel()
        self.iterations = iterations

    def improve(self):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)

    def _run_one_episode(self, episode):
         # S
        current_state_index = self.env.reset()

        while True:
            # A
            current_action_index = self.policy.get_action(current_state_index)
            observation = self.env.step(current_action_index)

            # R
            reward = observation[1]
            done = observation[2]

            self.statistics.episode_rewards[episode] += reward
            self.statistics.episode_lengths[episode] += 1

            # S'
            next_state_index = observation[0]

            q_values_next_state = self.q_table[next_state_index]
            max_value = max(q_values_next_state.values())
            delta = reward + self.discount * max_value - self.q_table[current_state_index][current_action_index]
            self.q_table[current_state_index][current_action_index] += self.step_size * delta

            self.model.feed(current_state_index, current_action_index, next_state_index, reward)
            for _ in tqdm(range(0,self.iterations)):
                sampled_current_state_index, sampled_current_action_index, sampled_next_state_index, sampled_reward = self.model.sample()
                sampled_q_values_next_state = self.q_table[sampled_next_state_index]
                max_value = max(sampled_q_values_next_state.values())
                delta = reward + self.discount * max_value - self.q_table[sampled_current_state_index][sampled_current_action_index]
                self.q_table[sampled_current_state_index][sampled_current_action_index] += self.step_size * delta

            # update policy softly
            q_values = self.q_table[current_state_index]
            distribution = self.create_distribution_epsilon_greedily(q_values)
            self.policy.policy_table[current_state_index] = distribution


            if done:
                break

            current_state_index = next_state_index


#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Trivial model for planning in Dyna-Q
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, state_index, action_index, next_state_index, reward):
        if state_index not in self.model.keys():
            self.model[state_index] = dict()
        self.model[state_index][action_index] = [next_state_index, reward]

    # randomly sample from previous experience
    def sample(self):
        state_index = list(self.model)[self.rand.choice(range(len(self.model.keys())))]
        action_index = list(self.model[state_index])[self.rand.choice(range(len(self.model[state_index].keys())))]
        next_state_index, reward = self.model[state_index][action_index]
        return state_index,action_index,next_state_index,reward


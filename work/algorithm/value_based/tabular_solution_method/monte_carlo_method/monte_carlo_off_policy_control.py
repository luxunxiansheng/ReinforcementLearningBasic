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

import copy
from collections import defaultdict

import numpy as np
from numpy.lib import utils
from tqdm import tqdm

from lib.utility import (create_distribution_epsilon_greedily,create_distribution_greedily)
from common import ActorBase

class Actor(ActorBase):
    def __init__(self, q_table, behavior_policy, target_policy,env, episodes=100, discount=1.0,epsilon=0.01):
        self.q_table = q_table
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.env = env
        self.episodes = episodes
        self.discount = discount
        self.create_distribution_greedily = create_distribution_greedily()
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon) 

    def improve(self):
        # it is necessary to keep the weight total for every state_action pair
        C = self._init_weight_total()
        for _ in tqdm(range(0, self.episodes)):
            trajectory = self._run_one_episode()
            G = 0.0
            W = 1
            for state_index, action_index, reward in trajectory[::-1]:
                # The return for current state_action pair
                G = reward + self.discount*G

                # weight total for current state_action pair
                C[state_index][action_index] += W

                # q_value calculated incrementally with off policy
                self.q_table[state_index][action_index] += W / C[state_index][action_index] * (G-self.q_table[state_index][action_index])

                distibution = self.create_distribution_greedily(self.q_table[state_index])
                self.target_policy.policy_table[state_index] =  distibution

                # If the action taken by the behavior policy is not the action
                # taken by the target policy,the probability will be 0 and we can break
                if action_index != np.argmax(self.target_policy.policy_table[state_index]):
                    break

                # probability product
                W = W * 1. / self.behavior_policy.policy_table[state_index][action_index]
            
    def _init_weight_total(self):
        weight_total = defaultdict(lambda: {})
        for state_index, action_values in self.q_table.items():
            for action_index, _ in action_values.items():
                weight_total[state_index][action_index] = 0.0
        return weight_total

    def _run_one_episode(self):
        trajectory = []
        current_state_index = self.env.reset(True)
        while True:
            self.behavior_policy.policy_table[current_state_index] = self.create_distribution_epsilon_greedily(self.q_table[current_state_index])
            action_index = self.behavior_policy.get_action(current_state_index)
            observation = self.env.step(action_index)
            reward = observation[1]
            trajectory.append((current_state_index, action_index, reward))
            done = observation[2]
            if done:
                break
            current_state_index = observation[0]
        return trajectory

    def get_optimal_policy(self):
        return self.target_policy


class MonteCarloOffPolicyControl:
    """
    As described in 5.7 section of Sutton' book 
    1) Weighted importance sampling.
    2) Incremental implementation
    """

    def __init__(self, q_table, behavior_policy, target_policy,env, episodes=10000, discount=1.0,epsilon=0.01):
        self.q_table = q_table
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.env = env
        self.episodes = episodes
        self.discount = discount
        self.epsilon  = epsilon

    def improve(self):
        actor = Actor(self.q_table,self.behavior_policy,self.target_policy,self.env,self.episodes,self.discount,self.epsilon)
        actor.improve()
        return actor.get_optimal_policy()

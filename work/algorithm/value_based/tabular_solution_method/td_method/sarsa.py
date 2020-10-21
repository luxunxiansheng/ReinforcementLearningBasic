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

from common import ActorBase
from lib.utility import (create_distribution_epsilon_greedily, create_distribution_greedily)
from policy.policy import PureTabularPolicy
from tqdm import tqdm




class Actor(ActorBase):
    """
    The reason that sarsa is on-policy is that it updates its Q-Values using the Q-value of the next state S' 
    and the current policy's action A'. It estimates the return for state-action pairs assuming the current 
    policy continues to be followed.
    """
    
    def __init__(self, q_value_function, policy, epsilon, env, statistics, episodes, step_size=0.1, discount=1.0):
        self.q_value_function = q_value_function
        self.policy = policy
        self.discount = discount
        self.step_size = step_size
        self.env = env
        self.episodes = episodes
        self.statistics = statistics
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon)
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, *args):
        for episode in tqdm(range(0, self.episodes)):
            # S
            current_state_index = self.env.reset()

            # A
            current_action_index = self.policy.get_action(current_state_index)

            while True:
                observation = self.env.step(current_action_index)

                # R
                reward = observation[1]

                # S'
                next_state_index = observation[0]
                done = observation[2]

                self.statistics.episode_rewards[episode] += reward
                self.statistics.episode_lengths[episode] += 1

                # A'
                next_action_index = self.policy.get_action(next_state_index)

                delta = reward + self.discount * self.q_value_function[next_state_index][next_action_index] - self.q_value_function[current_state_index][current_action_index]
                self.q_value_function[current_state_index][current_action_index] += self.step_size * delta

                q_values = self.q_value_function[current_state_index]
                soft_greedy_distibution = self.create_distribution_epsilon_greedily(q_values)
                self.policy.policy_table[current_state_index] = soft_greedy_distibution

                if done:
                    break

                current_state_index = next_state_index
                current_action_index = next_action_index


    def get_optimal_policy(self):
        return self.policy


class SARSA:
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    """

    def __init__(self, q_value_function, table_policy, epsilon, env, statistics, episodes, step_size=0.1, discount=1.0):

        self.actor = Actor(q_value_function, table_policy, epsilon,env, statistics, episodes, step_size, discount)

    def improve(self):
        self.actor.improve()
        return self.actor.get_optimal_policy()

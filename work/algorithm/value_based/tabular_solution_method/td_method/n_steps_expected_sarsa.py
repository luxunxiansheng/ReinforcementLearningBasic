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

from matplotlib.pyplot import disconnect
import numpy as np
from common import ActorBase
from lib.utility import (create_distribution_epsilon_greedily,create_distribution_greedily)
from policy.policy import PureTabularPolicy
from tqdm import tqdm

class Actor(ActorBase):
    def __init__(self, q_table, table_policy, epsilon, env, steps, statistics, episodes, step_size, discount):
        self.q_value_function = q_table
        self.policy = table_policy
        self.epsilon= epsilon
        self.env = env
        self.steps = steps
        self.statistics = statistics
        self.episodes = episodes
        self.step_size = step_size
        self.discount = discount
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon)
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, *args):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)

    def get_optimal_policy(self):
        policy_table = {}
        for state_index, _ in self.q_value_function.items():
            q_values = self.q_value_function[state_index]
            greedy_distibution = self.create_distribution_greedily(q_values)
            policy_table[state_index] = greedy_distibution
        table_policy = PureTabularPolicy(policy_table)
        return table_policy
    
    def _run_one_episode(self, episode):

        current_timestamp = 0
        final_timestamp = np.inf

        trajectory = []
        # S
        current_state_index = self.env.reset()
        # A
        current_action_index = self.policy.get_action(current_state_index)

        while True:
            if current_timestamp < final_timestamp:
                observation = self.env.step(current_action_index)
                # R
                reward = observation[1]
                done = observation[2]

                self.statistics.episode_rewards[episode] += reward
                self.statistics.episode_lengths[episode] += 1

                # S'
                next_state_index = observation[0]

                # A'
                next_action_index = self.policy.get_action(next_state_index)

                trajectory.append((current_state_index,current_action_index,reward))

                if done:
                    final_timestamp = current_timestamp + 1

            updated_timestamp = current_timestamp - self.steps

            if updated_timestamp >= 0:
                G = 0
                for i in range(updated_timestamp, min(updated_timestamp + self.steps, final_timestamp)):
                    G += np.power(self.discount, i - updated_timestamp) * trajectory[i][2]
                if updated_timestamp + self.steps < final_timestamp:
                    # expected Q value, actullay the v(s)
                    expected_next_q = 0
                    next_actions = self.policy.policy_table[trajectory[current_timestamp][0]]
                    for action, action_prob in next_actions.items():
                        expected_next_q += action_prob * self.q_value_function[trajectory[current_timestamp][0]][action]
                    G += np.power(self.discount, self.steps) * expected_next_q

                delta = G - self.q_value_function[trajectory[updated_timestamp][0]][trajectory[updated_timestamp][1]]
                self.q_value_function[trajectory[updated_timestamp][0]][trajectory[updated_timestamp][1]] += self.step_size*delta

                q_values = self.q_value_function[current_state_index]
                soft_greedy_distibution = self.create_distribution_epsilon_greedily(q_values)
                self.policy.policy_table[current_state_index] = soft_greedy_distibution


                if updated_timestamp == final_timestamp - 1:
                    break

            current_timestamp += 1
            current_state_index = next_state_index
            current_action_index = next_action_index

class NStepsExpectedSARSA:
    """
    N steps SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    """

    def __init__(self, q_table, table_policy, epsilon, env, steps, statistics, episodes, step_size=0.1, discount=1.0):
        self.actor = Actor(q_table, table_policy, epsilon, env, steps, statistics, episodes, step_size ,discount) 

    def improve(self):
        self.actor.improve()
        return self.actor.get_optimal_policy()
    












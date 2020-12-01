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

import numpy as np
from tqdm import tqdm
from algorithm.value_based.tabular_solution_method.td_method.td_common import TDCritic

class TDNOffPolicySARSACritic(TDCritic):
    def __init__(self,value_function,behavior_policy,target_policy,steps=1,step_size=0.1,discount=1.0):
        self.value_function=value_function
        self.steps = steps
        self.step_size = step_size
        self.discount = discount
        self.behavior_policy = behavior_policy
        self.target_policy =  target_policy

    def evaluate(self,*args):
        trajectory = args[0]
        current_timestamp = args[1]
        updated_timestamp = args[2]
        final_timestamp   = args[3]
        
        prob = 1
        for i in range(updated_timestamp, min(updated_timestamp + self.steps, final_timestamp)):
            state_index =  trajectory[i][0]
            action_index = trajectory[i][1]

            if  self.behavior_policy.policy_table[state_index][action_index] != 0:
                prob *= self.target_policy.policy_table[state_index][action_index]/self.behavior_policy.policy_table[state_index][action_index]

        G = 0
        for i in range(updated_timestamp, min(updated_timestamp + self.steps, final_timestamp)):
            reward = trajectory[i][2]
            G += np.power(self.discount, i - updated_timestamp) * reward

        if updated_timestamp + self.steps < final_timestamp:
            G += np.power(self.discount, self.steps) * self.value_function[trajectory[current_timestamp][0]][trajectory[current_timestamp][1]]

        delta = G - self.q_table[trajectory[updated_timestamp][0]][trajectory[updated_timestamp][1]]
        self.value_function[trajectory[updated_timestamp][0]][trajectory[updated_timestamp][1]] += self.step_size*delta*prob



class OffPolicyNStepsSARSA:
    """
    Section 7.3 
    """

    def __init__(self, critic, actor, env, n_steps,statistics, episodes):
        self.env = env
        self.episodes = episodes
        self.steps = n_steps
        self.statistics = statistics
        self.critic = critic
        self.actor = actor

    def improve(self):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)

    
    def _run_one_episode(self, episode):

        current_timestamp = 0
        final_timestamp = np.inf

        trajectory = []
        # S
        current_state_index = self.env.reset()
        # A
        current_action_index = self.actor.get_action(current_state_index)

        while True:
            if current_timestamp < final_timestamp:
                observation = self.env.step(current_action_index)
                # R
                reward = observation[1]
                done = observation[2]
                trajectory.append((current_state_index, current_action_index, reward))

                self.statistics.episode_rewards[episode] += reward
                self.statistics.episode_lengths[episode] += 1

                # S'
                next_state_index = observation[0]
                # A'
                next_action_index = self.behavior_policy.get_action(next_state_index)

                if done:
                    final_timestamp = current_timestamp + 1

            updated_timestamp = current_timestamp - self.steps

            if updated_timestamp >= 0:
                self.critic.evaluate(trajectory,current_timestamp,updated_timestamp,final_timestamp)
                self.actor.improve(trajectory[updated_timestamp][0])

                if updated_timestamp == final_timestamp - 1:
                    break

            current_timestamp += 1
            current_state_index = next_state_index
            current_action_index = next_action_index


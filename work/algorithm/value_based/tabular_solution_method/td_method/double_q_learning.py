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

from copy import deepcopy

import numpy as np

from tqdm import tqdm

class DoubleQLearningCritic:
    """
    Be aware that this class is not inheriting from TDCritic.   
    """

    def __init__(self,value_function,step_size=0.1,discount = 1.0):
        self.step_size = step_size
        self.discount = discount
        self.q_table_1 = deepcopy(value_function)
        self.q_table_2 = deepcopy(value_function)

        self.q_table = deepcopy(value_function)
        
    def exploit(self, *args):
        current_state_index  = args[0]
        current_action_index = args[1]
        reward = args[2]
        next_state_index = args[3]


        p = np.random.random()
        if (p < .5):
            q_values_next_state = self.q_table_1[next_state_index]
            max_action_index = max(q_values_next_state, key=q_values_next_state.get)
            max_value = self.q_table_2[next_state_index][max_action_index]
            delta = reward + self.discount * max_value - self.q_table_1[current_state_index][current_action_index]
            self.q_table_1[current_state_index][current_action_index] += self.step_size * delta
        else:
            q_values_next_state = self.q_table_2[next_state_index]
            max_action_index = max(q_values_next_state, key=q_values_next_state.get)
            max_value = self.q_table_1[next_state_index][max_action_index]
            delta = reward + self.discount * max_value - self.q_table_2[current_state_index][current_action_index]
            self.q_table_2[current_state_index][current_action_index] += self.step_size * delta
        

        # add 
        for action_index, q_value in self.q_table_1[current_state_index].items():
            self.q_table[current_state_index][action_index] = q_value + self.q_table_2[current_state_index][action_index]


    def get_value_function(self):
        return self.q_table



class DoubleQLearning:
    '''
    Refer the article 'Double Q-Learning, the Easy Way'
    https://towardsdatascience.com/double-q-learning-the-easy-way-a924c4085ec3  
    '''

    def __init__(self, critic, actor, env, statistics, episodes):
        self.env = env
        self.episodes = episodes
        self.critic = critic 
        self.actor  = actor 
        self.statistics = statistics

    def explore(self):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)
    
    def _run_one_episode(self, episode):
        # S
        current_state_index = self.env.reset()

        while True:

            # A
            current_action_index = self.actor.get_behavior_policy().get_action(current_state_index)
            observation = self.env.step(current_action_index)

            # R
            reward = observation[1]
            done = observation[2]

            self.statistics.episode_rewards[episode] += reward
            self.statistics.episode_lengths[episode] += 1

            # S'
            next_state_index = observation[0]

            self.critic.exploit(current_state_index,current_action_index,reward,next_state_index)
            self.actor.explore(current_state_index)

            if done:
                break

            current_state_index = next_state_index


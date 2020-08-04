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

from collections import defaultdict

from tqdm import tqdm

from lib.utility import create_distribution_epsilon_greedily


class MonteCarloOnPolicyControl:
    def __init__(self, q_table, table_policy, epsilon, env, episodes=1000, discount=1.0):
        self.q_table = q_table
        self.policy = table_policy
        self.env = env
        self.episodes = episodes
        self.discount = discount
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon)

    def improve(self):
        state_count = self._init_state_count()
        init_state_index = self.env.reset()
        for _ in tqdm(range(0, self.episodes)):
            trajectory = self._run_one_episode(init_state_index)
            R = 0.0
            for state_index, action_index, reward in trajectory[::-1]:
                R = reward+self.discount*R
                state_count[state_index][action_index] = (state_count[state_index][action_index][0] + 1, state_count[state_index][action_index][1] + R)
                self.q_table[state_index][action_index] = state_count[state_index][action_index][1] / state_count[state_index][action_index][0]
                
                q_values = self.q_table[state_index]
                distribution = self.create_distribution_epsilon_greedily(q_values)
                self.policy.policy_table[state_index] = distribution
                        
    def _init_state_count(self):
        state_count = defaultdict(lambda: {})
        for state_index, action_values in self.q_table.items():
            for action_index, _ in action_values.items():
                state_count[state_index][action_index] = (0, 0.0)
        return state_count

    def _run_one_episode(self,init_state_index):
        trajectory = []
        current_state_index = init_state_index
        while True:
            action_index = self.policy.get_action(current_state_index)
            observation = self.env.step(action_index)
            reward = observation[1]
            trajectory.append((current_state_index, action_index, reward))
            done = observation[2]
            if done:
                break
            current_state_index = observation[0]

        return trajectory

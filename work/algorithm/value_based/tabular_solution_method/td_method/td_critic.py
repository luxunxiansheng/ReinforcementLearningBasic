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


from common import CriticBase
from lib.distribution import create_distribution_greedily
from policy.policy import DiscreteStateValueTablePolicy


class TDCritic(CriticBase):
    def __init__(self,value_function,step_size=0.1):
        self.value_function = value_function
        self.step_size = step_size
        self.create_distribution_greedily = create_distribution_greedily()
    
    def update(self, *args):
        # V 
        if self._is_v_function(args):
            current_state_index = args[0]
            target = args[1]
            delta = target - self.value_function[current_state_index]
            self.value_function[current_state_index] += self.step_size * delta
        # Q 
        else:
            current_state_index = args[0]
            current_action_index = args[1]
            target = args[2]

            delta = target - self.value_function[current_state_index][current_action_index]
            self.value_function[current_state_index][current_action_index] += self.step_size * delta

    def _is_v_function(self, args):
        return len(args)==2

    def get_value_function(self):
        return self.value_function

    def get_optimal_policy(self):
        policy_table = {}
        for state_index, _ in self.value_function.items():
            q_values = self.value_function[state_index]
            greedy_distibution = self.create_distribution_greedily(q_values)
            policy_table[state_index] = greedy_distibution
        table_policy = DiscreteStateValueTablePolicy(policy_table)
        return table_policy
        

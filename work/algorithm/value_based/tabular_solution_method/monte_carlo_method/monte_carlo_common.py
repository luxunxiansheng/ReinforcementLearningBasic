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
from common import ExploitatorBase

class MonteCarloIncrementalCritic(ExploitatorBase):
    def __init__(self, q_value_function):
        self.q_value_function = q_value_function
    
        # it is necessary to keep the weight total for every state_action pair
        self.C = self._init_weight_total()

    def evaluate(self, *args):
        state_index  = args[0]
        action_index = args[1]
        G = args[2]
        W = args[3]
        
        # weight total for current state_action pair
        self.C[state_index][action_index] += W

        # q_value calculated incrementally with off policy
        self.q_value_function[state_index][action_index] += W / self.C[state_index][action_index] * (G-self.q_value_function[state_index][action_index])
        

    def _init_weight_total(self):
        weight_total = defaultdict(lambda: {})
        for state_index, action_values in self.q_value_function.items():
            for action_index, _ in action_values.items():
                weight_total[state_index][action_index] = 0.0
        return weight_total

    def get_value_function(self):
        return self.q_value_function
    
class MonteCarloAverageCritic(ExploitatorBase):
    def __init__(self, q_value_function):
        self.q_value_function= q_value_function
        self.state_count=self._init_state_count()
        
    def evaluate(self,*args):
        state_index= args[0]
        action_index = args[1]
        R= args[2]

        self.state_count[state_index][action_index] = (self.state_count[state_index][action_index][0] + 1, self.state_count[state_index][action_index][1] + R)
        self.q_value_function[state_index][action_index] = self.state_count[state_index][action_index][1] / self.state_count[state_index][action_index][0]
    
    def get_value_function(self):
        return self.q_value_function
    
    def _init_state_count(self):
        state_count = defaultdict(lambda: {})
        for state_index, action_values in self.q_value_function.items():
            for action_index, _ in action_values.items():
                state_count[state_index][action_index] = (0, 0.0)
        return state_count

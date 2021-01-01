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
from lib.utility import create_distribution_greedily


class ValueIterationActor(ActorBase):
    def __init__(self, value_function, policy,transition_table, delta=1e-8, discount=1.0):
        self.value_function = value_function
        self.policy = policy
        self.model = transition_table
        self.delta = delta
        self.discount = discount
    
    def improve(self, *args):
        while True:
            delta = 1e-10
            for state_index, old_value_of_state in self.value_function.items():
                value_of_optimal_action = self._get_value_of_optimal_action(state_index)
                self.value_function[state_index] = value_of_optimal_action
                delta = max(delta,abs(value_of_optimal_action-old_value_of_state))

            if delta < self.delta:
                return 

    def get_behavior_policy(self):
        return self.policy
    
    def _get_value_of_optimal_action(self, state_index):
        return max(self._get_q_values_of_state(state_index).values())

    def _get_value_of_action(self, state_index, action_index):
        transitions = self.model[state_index][action_index]
        value_of_action = 0.0
        for transition_prob, next_state_index, reward, done in transitions:  # For each next state
            # the reward is also related to the next state
            value_of_next_state = 0 if done else self.value_function[next_state_index]
            value_of_action += transition_prob * (self.discount*value_of_next_state+reward)
        return value_of_action

    def _get_q_values_of_state(self,state_index):
        q_values = {}
        actions = self.policy.policy_table[state_index]
        for action_index, _ in actions.items():
            value_of_action = self._get_value_of_action(state_index, action_index)
            q_values[action_index] = value_of_action
        return q_values    
    
    def get_value_function(self):
        return self.value_function

    def get_optimal_policy(self):
        for state_index, action_distribution in self.policy.policy_table.items():
            q_values={}
            for action_index, _ in action_distribution.items():
                q_values[action_index] = self._get_value_of_action(state_index, action_index)
            greedy_distibution = create_distribution_greedily()(q_values)
            self.policy.policy_table[state_index] = greedy_distibution
            
        return self.policy    




class ValueIteration:
    """
    One drawback of policy iteration method is each of its iterations involves policy 
    evalution, which may itself be a protracted iterative computation requiring multiple 
    sweeps through the state set. If policy evaluation is done iteratively, then convergence 
    exactly occurs in the limit.   

    In fact,the plilcy evaluation step of policy iteration can be truncated in serveral ways
    without losing the convergence guarantees of policy iteration. One important special case
    is policy evaluation is stopped after just one sweep(one update of each state). This
    alrgorithm is so called Value Iteration. 

    In value iteration, there is no explict policy. What the process does is to get closer and 
    closer to the optimal value. For a given MDP, the optimal values and optimal policy are 
    considered to be there, objectively. Therefore, the value iteration is a value based method.  
    """

    def __init__(self, actor):
        self.actor = actor
    

    def improve(self):
        self.actor.improve()
        return self.actor.get_optimal_policy()
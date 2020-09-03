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



from numpy.core.einsumfunc import _optimal_path
from common import ActorBase
from lib.utility import create_distribution_greedily
from policy.policy import TabularPolicy

class Actor(ActorBase):
    
    """
    One drawback to policy iteration method is that each of its iterations involves policy 
    evalution, which may itself be a protracted iterative computation requiring multiple 
    sweeps through the state set. If policy evaluation is done iteratively, then convergence 
    exactly occurs in the limit.   
    
    In fact,the plilcy evaluation step of policy iteration can be truncated in serveral ways
    without losing the convergence guarantees of policy iteration. One important special case
    is when policy evaluation is stopped after just one sweep(one update of each state). This
    alrgorithm is so called Value Iteration. 

    Refer to the equation 4.2 in Sutton's book 

    """
    
    
    def __init__(self, v_table, p, delta=1e-8, discount=1.0):
        self.v_table = v_table
        self.transition_table = p
        self.delta = delta
        self.discount = discount
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self):
        while True:
            delta = self._bellman_optimize()
            if delta < self.delta:
                break

    def get_optimal_policy(self):
        policy_table = {}
        for state_index, _ in self.v_table.items():
            q_values = {}  
            for action_index, transitions in self.transition_table[state_index].items():
                q_values[action_index]=self._get_value_of_action(transitions)
            greedy_distibution = self.create_distribution_greedily(q_values)
            policy_table[state_index] = greedy_distibution 
        
        table_policy = TabularPolicy(policy_table)
        return table_policy

    def _bellman_optimize(self):
        delta = 1e-10
        for state_index, old_value_of_state in self.v_table.items():
            q_values = {}
            transitions = self.transition_table[state_index]
            for action_index, transition in transitions.items():
                value_of_action = self._get_value_of_action(transition)
                q_values[action_index] = value_of_action
            optimal_value_of_action = self._get_optimal_value_of_action(q_values)
            self.v_table[state_index] = optimal_value_of_action
            delta = max(abs(old_value_of_state-optimal_value_of_action), delta)

        return delta

    def _get_value_of_action(self, transition):
        value_of_action = 0
        for transition_prob, next_state_index, reward, done in transition:  # For each next state
            # the reward is also related to the next state
            value_of_next_state = 0 if done else self.v_table[next_state_index]
            value_of_action += transition_prob * \
                (self.discount * value_of_next_state+reward)
        return value_of_action

    def _get_optimal_value_of_action(self, q_values):
        return max(q_values.values())


class VValueIteration:
    def __init__(self, v_table, p, delta=1e-8, discount=1.0):
        self.v_table = v_table
        self.transition_table = p
        self.delta = delta
        self.discount = discount
        

    def improve(self):
        actor = Actor(self.v_table,self.transition_table,self.delta,self.discount)
        actor.improve()
        optimal = actor.get_optimal_policy()
        return optimal
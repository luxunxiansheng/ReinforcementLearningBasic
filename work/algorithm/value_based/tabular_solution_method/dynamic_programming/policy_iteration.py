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


from common import ActorBase, ExploitatorBase
from lib.utility import create_distribution_greedily



class PoplicyIterationCritic(ExploitatorBase):
    """
    Given a policy, calculate the value of state with Jacobi-like itration method. The calculated value of state may not be 
    very accurate, but it doesn't mattter since our goal is to find the optimal policy after all. 
    """
    def __init__(self,policy,value_function,transition_table,delta,discount):
        self.policy = policy
        self.value_function = value_function
        self.model = transition_table
        self.delta = delta
        self.discount = discount
        
    def evaluate(self,*args):
        while True:
            delta = self._evaluate_once()
            if delta < self.delta:
                break
        
    def get_value_function(self):
        return self.value_function
    
    def _evaluate_once(self):
        delta = 1e-10
        for state_index, old_value_of_state in self.value_function.items():
            value_of_state = self._get_value_of_state(state_index)
            self.value_function[state_index] = value_of_state
            delta = max(abs(value_of_state-old_value_of_state), delta)
        return delta
    
    def _get_value_of_state(self,state_index):
        value_of_state = 0.0
        for action_index, probability in self.policy.policy_table[state_index].items():
            value_of_action = self._get_value_of_action(state_index,action_index)
            value_of_state+= probability*value_of_action
        return value_of_state

    def _get_value_of_action(self, state_index,action_index):
        transitions = self.model[state_index][action_index]
        value_of_action = 0.0
        for transition_prob, next_state_index, reward, done in transitions:  # For each next state
            # the reward is also related to the next state
            value_of_next_state = 0 if done else self.value_function[next_state_index]
            value_of_action += transition_prob * (self.discount*value_of_next_state+reward)
        return value_of_action

class PolicyIterationActor(ActorBase):
    """
    It is trival for the actor to improve the policy by sweeping the state space. 
    
    """
    def __init__(self,policy,critic,transition_table,delta,discount):
        self.policy = policy
        self.critic = critic
        self.model   = transition_table
        self.delta = delta
        self.discount = discount
        
    def improve(self,*args):
        while True:
            delta = self._improve_once()
            if delta < self.delta:
                break

    def get_behavior_policy(self):
        return self.policy

    def _improve_once(self):
        delta = 1e-10
        for state_index, action_distribution in self.policy.policy_table.items():
            old_policy = copy.deepcopy(action_distribution)
            q_values={}
            for action_index, _ in action_distribution.items():
                transition = self.model[state_index][action_index]
                q_values[action_index] = self._get_value_of_action(transition,self.critic.get_value_function())
            greedy_distibution = create_distribution_greedily()(q_values)
            self.policy.policy_table[state_index] = greedy_distibution
            new_old_policy_diff = {action_index: abs(old_policy[action_index]-greedy_distibution[action_index]) for action_index in greedy_distibution}
            delta = max(max(new_old_policy_diff.values()), delta)
        return delta    

    def _get_value_of_action(self, transitions,value_function):
        value_of_action = 0.0
        for transition_prob, next_state_index, reward, done in transitions:  # For each next state
            # the reward is also related to the next state
            value_of_next_state = 0 if done else value_function[next_state_index]
            value_of_action += transition_prob * (self.discount*value_of_next_state+reward)
        return value_of_action

    def get_optimal_policy(self):
        return self.policy


class PolicyIteration:
    """
    1. Policy iteration is a policy-based method.
    2. Because the previous values will be discarded once policy is improved, pollicy itration is
        on-policy 
    """

    def __init__(self, critic,actor):

        self.critic = critic
        self.actor  = actor 
        
    def improve(self):
        self.critic.evaluate()
        self.actor.improve()
        return  self.actor.get_optimal_policy()

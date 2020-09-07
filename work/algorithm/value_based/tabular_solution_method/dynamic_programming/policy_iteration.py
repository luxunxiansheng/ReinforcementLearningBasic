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
from argh.decorators import arg

from markdown.test_tools import Kwargs
from scipy.constants.codata import value

from common import ActorBase, CriticBase
from lib.utility import create_distribution_greedily


class Critic(CriticBase):
    """
    In the context of dynamic programming, the tabular model of the environment will be available. 
    It is possible to calculate the state or q value function for a given policy with Bellman equation 
    iteratively. 
    """
    
    def __init__(self,transition_table,delta,discount):
        self.transition_table = transition_table
        self.delta = delta
        self.discount = discount
        
    def evaluate(self,*args):
        policy = args[0]
        value_function = args[1]

        while True:
            delta = self._evaluate_once(policy,value_function)
            if delta < self.delta:
                break
        
        return value_function


    def _evaluate_once(self,policy,value_function):
        delta = 1e-10
        for state_index, old_value_of_state in value_function.items():
            value_of_state = 0.0
            action_transitions = self.transition_table[state_index]
            for action_index, transitions in action_transitions.items():
                value_of_action = self._get_value_of_action(transitions,value_function)
                value_of_state += policy.get_action_probablity(state_index,action_index) * value_of_action
            value_function[state_index] = value_of_state
            delta = max(abs(value_of_state-old_value_of_state), delta)
        return delta
    
    def _get_value_of_action(self, transitions,value_function):
        value_of_action = 0.0
        for transition_prob, next_state_index, reward, done in transitions:  # For each next state
            # the reward is also related to the next state
            value_of_next_state = 0 if done else value_function[next_state_index]
            value_of_action += transition_prob * (self.discount*value_of_next_state+reward)
        return value_of_action

class Actor(ActorBase):
    """
    It is trival for the actor to improve the policy by sweeping the state space. 
    
    """
    def __init__(self,transtion_table,delta,discount):
        self.model   = transtion_table
        self.delta = delta
        self.discount = discount
        
    
    def improve(self,*args):
        policy = args[0]
        value_function = args[1]
        while True:
            delta = self._improve_once(policy,value_function)
            if delta < self.delta:
                break
        
        return policy

    def _improve_once(self,policy,value_function):
        delta = 1e-10
        for state_index, actions in policy.policy_table.items():
            old_policy = copy.deepcopy(policy.policy_table[state_index])
            q_values = {}
            for action_index, _ in actions.items():
                transition = self.model[state_index][action_index]
                q_values[action_index] = self._get_value_of_action(transition,value_function)
            greedy_distibution = create_distribution_greedily()(q_values)
            policy.policy_table[state_index] = greedy_distibution
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



class PolicyIteration:
    """
    Once a ploicy has been improved to get a new sequence of value of states, we can
    then compute the value of the states to improve the pollicy, monotonically. 
    """

    def __init__(self, v_table, policy, transition_table, delta=1e-5, discount=1.0):
        self.v_table = v_table
        self.policy = policy
        self.transition_table= transition_table
        self.delta = delta
        self.discount = discount
        self.critic = Critic(transition_table,delta,discount)
        self.actor  = Actor(transition_table,delta,discount) 
        
    
    def improve(self):
        value_function=self.critic.evaluate(self.policy,self.v_table)
        optimal_policy=self.actor.improve(self.policy,value_function)
        return optimal_policy

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
from policy.policy import DiscreteStateValueBasedPolicy
from common import ActorBase, CriticBase, ExplorerBase
from lib.utility import create_distribution_greedily


class BellmanOptimalCritic(CriticBase):
    """
    Given a policy, calculate the value of state with Jacobi-like itration method. The calculated value of state may not be 
    very accurate, but it doesn't mattter since our goal is to find the optimal policy after all. 
    """
    def __init__(self,value_function,transition_table,discount):
        self.value_function = value_function
        self.transition_model = transition_table
        self.discount = discount
        self.create_distribution_greedily = create_distribution_greedily()
        
    def evaluate(self,*args):
        current_state_index = args[0]
        current_action_index = args[1]
        value_of_action = self._calculate_action_value(current_state_index, current_action_index)
        delta = self.value_function[current_state_index][current_action_index]-value_of_action
        self.value_function[current_state_index][current_action_index] = value_of_action
        return delta

    def _calculate_action_value(self, state_index, action_index):
        value_of_action = 0.0
        transitions = self.transition_model[state_index][action_index]
        for transition_prob, next_state_index, reward, done in transitions:  # For each next state
            # the reward is also related to the next state
            if done:
                value_of_next_state = 0
            else:
                value_of_next_state = self._get_value_of_optimal_action(next_state_index)            
            value_of_action += transition_prob * (self.discount*value_of_next_state+reward)
        return value_of_action

    def _get_value_of_optimal_action(self, state_index):
        return  max(self.value_function[state_index].values())
    
    def get_optimal_policy(self):
        policy_table = {}
        for state_index, _ in self.value_function.items():
            v_values = self.value_function[state_index]
            greedy_distibution = self.create_distribution_greedily(v_values)
            policy_table[state_index] = greedy_distibution
        table_policy = DiscreteStateValueBasedPolicy(policy_table)
        return table_policy
    
    def get_value_function(self):
        return self.value_function

class ValueIterationActor(ActorBase):
    """
    It is trival for the actor to sweep the state space. 
    """
    def __init__(self,critic,transition_table,critic_delta,discount):
        self.critic = critic
        self.model   = transition_table
        self.critic_delta = critic_delta
        self.discount = discount
    
    def act(self):
        while True:
            delta = 0
            for state_index, action_values in self.critic.get_value_function().items():
                for action_index in action_values:
                    delta = max(delta,self.critic.evaluate(state_index,action_index))
                
            if delta < self.critic_delta:
                break 

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

    def __init__(self,env,critic_delta=1e-5, discount = 1.0):
        self.env = env 
        self.policy = DiscreteStateValueBasedPolicy(self.env.build_policy_table())
        self.value_funciton = self.env.build_Q_table()
        transition_table = env.P 

        self.critic = BellmanOptimalCritic(self.value_funciton,transition_table,discount)
        self.actor = ValueIterationActor(self.critic,transition_table,critic_delta,discount)
    

    def learn(self):
        self.actor.act()
        self.env.show_policy(self.actor.critic.get_optimal_policy())
    

        
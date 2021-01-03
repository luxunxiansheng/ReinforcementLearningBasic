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


from common import CriticBase,ActorBase
from lib.utility import (create_distribution_epsilon_greedily,create_distribution_greedily,create_distribution_boltzmann)


class ApproximationExpectedSARSACritic(CriticBase):
    def __init__(self,env,estimator,policy,step_size=0.01,discount= 1.0):
        self.env = env 
        self.estimator = estimator
        self.discount = discount
        self.policy = policy 
        self.step_size = step_size

    def evaluate(self, *args):
        current_state_index = args[0]
        current_action_index = args[1]
        reward = args[2]
        next_state_index = args[3]    
        
        q_values = {}
        for action_index in range(self.env.action_space.n):
            q_values[action_index] = self.estimator.predict(next_state_index,action_index)
        distribution = self.policy.get_action_distribution(q_values)

        expected_q_value = 0
        for action_index in range(self.env.action_space.n):
            expected_q_value += distribution[action_index]*q_values[action_index]
        
        # set the target 
        target = reward + self.discount * expected_q_value

        # SGD fitting
        self.q_value_estimator.update(current_state_index, current_action_index, target)

    
    def get_value_function(self):
        return self.estimator

class ESoftActor(ActorBase):
    def __init__(self, policy,critic,epsilon=0.1):
        self.policy = policy
        self.critic = critic
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon)
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, *args):
        current_state_index = args[0]
        q_value_function = self.critic.get_value_function()
        q_values = q_value_function[current_state_index]
        soft_greedy_distibution = self.create_distribution_epsilon_greedily(q_values)
        self.policy.policy_table[current_state_index] = soft_greedy_distibution

    def get_behavior_policy(self):
        return self.policy

    def get_optimal_policy(self):
        policy_table = {}
        q_value_function = self.critic.get_value_function()
        for state_index, _ in q_value_function.items():
            q_values = q_value_function[state_index]
            greedy_distibution = self.create_distribution_greedily(q_values)
            policy_table[state_index] = greedy_distibution
        table_policy = DiscreteStateValueBasedPolicy(policy_table)
        return table_policy


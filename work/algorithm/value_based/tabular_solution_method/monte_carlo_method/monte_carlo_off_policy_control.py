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

import numpy as np
from common import ActorBase, CriticBase
from lib.utility import create_distribution_greedily
from tqdm import tqdm

class Critic(CriticBase):
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
    

class Actor(ActorBase):
    def __init__(self,behavior_policy,target_policy,critic):
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy
        self.critic = critic
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, *args):
        state_index = args[0]
        q_value_function = self.critic.get_value_function()
        greedy_distibution = self.create_distribution_greedily(q_value_function[state_index])
        self.target_policy.policy_table[state_index] = greedy_distibution

    def get_behavior_policy(self):
        return self.behavior_policy

    def get_optimal_policy(self):
        return self.target_policy

    def get_action_distribution(self,state_index):
        return self.target_policy.policy_table[state_index]

class MonteCarloOffPolicyControl:
    """
    As described in 5.7 section of Sutton' book 
    1) Weighted importance sampling.
    2) Incremental implementation
    """
    def __init__(self, q_value_function, behavior_policy, target_policy,env, episodes=500000, discount=1.0,epsilon=0.5):
        self.env = env
        self.episodes = episodes
        self.discount = discount
        self.epsilon  = epsilon
        
        self.critic= Critic(q_value_function)
        self.actor = Actor(behavior_policy,target_policy,self.critic)

    def improve(self):
        for _ in tqdm(range(0, self.episodes)):
            trajectory = self._run_one_episode()
            G = 0.0
            W = 1
            for state_index, action_index, reward in trajectory[::-1]:
                # The return for current state_action pair
                G = reward + self.discount*G
                
                # The return for current state_action pair
                self.critic.evaluate(state_index, action_index, G, W)
                self.actor.improve(state_index)
            
                # If the action taken by the behavior policy is not the action taken by the target policy,the probability will be 0 and we can break
                if action_index != np.argmax(self.actor.get_action_distribution(state_index)):
                    break

                # probability product
                W = W * 1. / self.actor.get_behavior_policy().policy_table[state_index][action_index]
        
        return self.actor.get_optimal_policy()


    def _init_weight_total(self):
        weight_total = defaultdict(lambda: {})
        for state_index, action_values in self.q_value_function.items():
            for action_index, _ in action_values.items():
                weight_total[state_index][action_index] = 0.0
        return weight_total

    def _run_one_episode(self):
        trajectory = []
        current_state_index = self.env.reset()
        while True:
            action_index = self.actor.get_behavior_policy().get_action(current_state_index)
            observation = self.env.step(action_index)
            reward = observation[1]
            trajectory.append((current_state_index, action_index, reward))
            done = observation[2]
            if done:
                break
            current_state_index = observation[0]
        return trajectory
























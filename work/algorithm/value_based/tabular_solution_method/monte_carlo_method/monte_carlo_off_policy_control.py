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
from tqdm import tqdm


from common import ActorBase, Agent
from policy.policy import DiscreteStateValueTablePolicy
from work.algorithm.value_based.tabular_solution_method.improver import ESoftImprover
from algorithm.value_based.tabular_solution_method.monte_carlo_method.monte_carlo_critic import MonteCarloIncrementalCritic


# TODO: Not follow the critic-explorer-actor patten very well. 
class MonteCarloOffPolicyAgent(Agent):
    """
    As described in 5.7 section of Sutton' book 
    1) Weighted importance sampling.
    2) Incremental implementation
    """
    class MonteCarloActor(ActorBase):
        def __init__(self,env,critic,explorer,statistics,discount):
            self.env = env 
            self.explorer =  explorer
            self.critic =  critic
            self.discount = discount
            self.statistics = statistics

        def act(self,*args):
            episode = args[0]

            trajectory = []
            current_state_index = self.env.reset()
            while True:
                action_index = self.explorer.get_behavior_policy().get_action(current_state_index)
                observation = self.env.step(action_index)
                reward = observation[1]
            
                trajectory.append((current_state_index, action_index, reward))
                done = observation[2]

                if done:
                    break
                current_state_index = observation[0]  
            
            G = 0.0
            W = 1
            for state_index, action_index, reward in trajectory[::-1]:
                # The return for current state_action pair
                G = reward + self.discount*G

                self.statistics.episode_rewards[episode] = G
                self.statistics.episode_lengths[episode] += 1
                
                # The return for current state_action pair
                self.critic.evaluate(state_index, action_index, G, W)
                self.explorer.explore(state_index)
            
                # If the action taken by the behavior policy is not the action taken by the target policy,the probability will be 0 and we can break
                if action_index != np.argmax(self.explorer.get_behavior_policy().policy_table[current_state_index]):
                    break

                # probability product
                W = W * 1. / self.explorer.get_behavior_policy().policy_table[state_index][action_index]
        
        def _init_weight_total(self):
            weight_total = defaultdict(lambda: {})
            for state_index, action_values in self.q_value_function.items():
                for action_index, _ in action_values.items():
                    weight_total[state_index][action_index] = 0.0
            return weight_total
            
    def __init__(self,env,statistics,episodes=500000,discount=1.0):
        self.env = env 
        self.episodes =episodes

        self.critic = MonteCarloIncrementalCritic(self.env.build_Q_table())
        explorer    = ESoftImprover(DiscreteStateValueTablePolicy(self.env.build_policy_table()),self.critic) 
        self.actor  = MonteCarloOffPolicyAgent.MonteCarloActor(self.env,self.critic,explorer,statistics,discount)

    def learn(self):
        for episode in tqdm(range(0, self.episodes)):
            self.actor.act(episode)

        self.env.show_policy(self.critic.get_greedy_policy())



























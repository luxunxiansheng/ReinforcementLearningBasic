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
## Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
##
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
from algorithm.value_based.tabular_solution_method.td_method.td_actor import TDActor, TDESoftExplorer
from algorithm.value_based.tabular_solution_method.td_method.td_critic import TDCritic
from tqdm import tqdm

class QLearningCritic(TDCritic):
    def __init__(self,value_function,step_size=0.1,discount = 1.0):
        super().__init__(value_function,step_size)
        self.discount = discount
    
    def evaluate(self, *args):
        current_state_index  = args[0]
        current_action_index = args[1]
        reward = args[2]
        next_state_index = args[3]
        
        # The target policy is implictly greedy 
        q_values_next_state = self.value_function[next_state_index]
        max_value = max(q_values_next_state.values())
        target = reward + self.discount*max_value
        self.update(current_state_index,current_action_index,target)



class QLearning:
    """
    The reason that Q-learning is off-policy is that it updates its Q-values using the Q-value of the 
    next state s' and the greedy action a' no matter what the current policy is .
    """
    def __init__(self,env, statistics, episodes):
        self.env = env
        self.episodes = episodes
        self.critic = QLearningCritic(self.env.build_Q_table()) 
        self.explorer  = TDESoftExplorer(DiscreteStateValueBasedPolicy(self.env.build_policy_table()),self.critic) 
        self.actor = TDActor(env,self.critic,self.explorer,statistics)


    def learn(self):
        for episode in tqdm(range(0, self.episodes)):
            self.actor.act(episode)

    
    def test(self):
        # S
        current_state_index = self.env.reset()
        optimal_policy = self.critic.get_optimal_policy()

        self.env.show_policy(optimal_policy)

        steps =  0
        returns = 0 
        
        while True:
            # A
            current_action_index = optimal_policy.get_action(current_state_index)

            print("current_state_index {} current_action_index {}".format(current_state_index,current_action_index))
            observation = self.env.step(current_action_index)
        
            # R
            reward = observation[1]
            done = observation[2]

            returns += reward
            steps += 1

            # S'
            next_state_index = observation[0]
            

            if done:
                print("Total Rewards {} with {} steps!".format(returns,steps))
                break
                
            current_state_index = next_state_index
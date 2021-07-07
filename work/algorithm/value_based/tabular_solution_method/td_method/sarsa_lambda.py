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

from tqdm import tqdm

from policy.policy import DiscreteStateValueBasedPolicy
from algorithm.value_based.tabular_solution_method.explorer import ESoftExplorer
from algorithm.value_based.tabular_solution_method.td_method.td_actor import TDActor
from algorithm.value_based.tabular_solution_method.td_method.td_lambda_critic import TDLambdaCritic


class SARSALambdaCritic(TDLambdaCritic):
    def __init__(self,value_function,policy,step_size=0.1,discount=1.0,lamb=0.0):
        super().__init__(value_function,step_size,discount=discount,lamb=lamb)
        self.discount = discount 
        self.target_policy = policy

    def evaluate(self,*args):
        current_state_index = args[0]
        current_action_index =args[1]
        reward = args[2]
        next_state_index = args[3]
        next_action_index = self.target_policy.get_action(next_state_index)
        
        # To calculate the target, it is necessary to know what is the next action for the next state according to current policy (On Policy)
        target = reward + self.discount * self.get_value_function()[next_state_index][next_action_index]
        self.update(current_state_index,current_action_index,target)

class SARSALambda:
    """
    SARSA algorithm with backward view: On-policy TD control. Finds the optimal epsilon-greedy policy
    """
    def __init__(self,env,statistics,episodes):
        self.env = env
        self.episodes = episodes

        # critic and exploler share the same policy (on-policy)
        self.policy = DiscreteStateValueBasedPolicy(self.env.build_policy_table())    
        self.critic = SARSALambdaCritic(self.env.build_Q_table(),self.policy)
        explorer  = ESoftExplorer(self.policy,self.critic) 
        self.actor = TDActor(env,self.critic,explorer,statistics) 

    def learn(self):
        for episode in tqdm(range(0, self.episodes)):
            self.actor.act(episode)
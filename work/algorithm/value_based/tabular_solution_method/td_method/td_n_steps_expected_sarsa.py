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


import numpy as np 
from tqdm import tqdm
from policy.policy import DiscreteStateValueBasedPolicy

from algorithm.value_based.tabular_solution_method.explorer import ESoftExplorer
from algorithm.value_based.tabular_solution_method.td_method.td_critic import TDCritic
from algorithm.value_based.tabular_solution_method.td_method.td_n_steps_actor import TDNStepsActor

class TDNExpectedSARSACritic(TDCritic):
    def __init__(self,value_function,policy,steps=1,step_size=0.1,discount=1.0):
        super().__init__(value_function, step_size=step_size)
        self.steps = steps
        self.discount = discount
        self.policy = policy 
    
    def evaluate(self,*args):
        '''
        Obviously, it will be very slow for expected Sarsa asymptotical to converge to the optimal values 
        '''
        trajectory = args[0]
        current_timestamp = args[1]
        updated_timestamp = args[2]
        final_timestamp   = args[3]
        
        G = 0
        for i in range(updated_timestamp, min(updated_timestamp + self.steps, final_timestamp)):
            G += np.power(self.discount, i - updated_timestamp) * trajectory[i][2]
        if updated_timestamp + self.steps < final_timestamp:
            # expected Q value, actullay the v(s)
            expected_next_q = 0
            next_actions = self.policy.policy_table[trajectory[current_timestamp][0]]
            for action, action_prob in next_actions.items():
                expected_next_q += action_prob * self.value_function[trajectory[current_timestamp][0]][action]
            G += np.power(self.discount, self.steps) * expected_next_q
            
        self.update(trajectory[updated_timestamp][0],trajectory[updated_timestamp][1],G) 


class TDNStepsExpectedSARSA:
    def __init__(self,env,steps, statistics, episodes):
        self.env = env
        self.episodes = episodes

        self.policy = DiscreteStateValueBasedPolicy(self.env.build_policy_table())
        self.critic = TDNExpectedSARSACritic(self.env.build_Q_table(),self.policy,steps)
        exloper = ESoftExplorer(self.policy,self.critic)
        self.actor  = TDNStepsActor(env,steps,self.critic,exloper,statistics)

    def learn(self):
        for episode in tqdm(range(0, self.episodes)):
            self.actor.act(episode)
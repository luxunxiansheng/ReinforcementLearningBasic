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
from numpy.core.numeric import cross
from tqdm import tqdm
from td_common import LambdaCritic



class TDLambdaEvalutaion:
    def __init__(self,  v_table, policy,env,episodes=1000, n_steps=3,discount=1.0, step_size=0.01,lamb=0):
        self.policy = policy
        self.env = env
        self.episodes = episodes
        self.discount = discount
        self.steps = n_steps
    
        self.critic = LambdaCritic(v_table,step_size,discount,lamb)

    
    def evaluate(self,*args):
        for _ in range(self.episodes):
            self._run_one_episode()
        
        return self.critic.get_value_function()
    def _run_one_episode(self):
        """
        Tabular TD(lambda) for estimating V(pi) 
        """
        current_state_index =  self.env.reset()

        while True:
            action_index = self.policy.get_action(current_state_index)
            observation = self.env.step(action_index)
            next_state_index = observation[0]
            reward = observation[1]
            done = observation[2]
            
            target = reward + self.discount*self.critic.get_value_function()[next_state_index]
            self.critic.evaluate(current_state_index,target)

            if done:
                break               

            current_state_index = next_state_index


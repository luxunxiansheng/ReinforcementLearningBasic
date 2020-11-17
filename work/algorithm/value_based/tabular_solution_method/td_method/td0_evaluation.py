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
from td_common import TDCritic
            
class TD0Critic(TDCritic):
    def __init__(self,value_function,step_size,discount):
        super().__init__(value_function,step_size)
        self.discount = discount
    
    def evaluate(self,*args):
        current_state_index = args[0]
        reward = args[1]
        next_state_index = args[2]

        target = reward + self.discount*self.get_value_function()[next_state_index]
        self.update(current_state_index,target)
        

class TD0Evalutaion:
    def __init__(self, value_function, policy, env, episodes=1000, discount=1.0, step_size=0.01):
        self.policy = policy
        self.env = env
        self.episodes = episodes
        
        self.critic= TD0Critic(value_function,step_size,discount)
        
    
    def evaluate(self,*args):
        for _ in range(0, self.episodes):
            self._run_one_episode()
        
        return self.critic.get_value_function()

    def _run_one_episode(self):
        """
        Tabular TD(0) for estimating V(pi)
        book 6.1 section
        """
        
        current_state_index = self.env.reset()

        while True:
            current_action_index = self.policy.get_action(current_state_index)
            observation = self.env.step(current_action_index)

            next_state_index = observation[0]
            reward = observation[1]
            done = observation[2]
            
            
            self.critic.evaluate(current_state_index,reward,next_state_index)
            
            if done:
                break   
            current_state_index = next_state_index
        

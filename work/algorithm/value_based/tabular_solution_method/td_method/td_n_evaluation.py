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


class TDNEvalutaion:
    def __init__(self, critic, policy, env,  n_steps=1, episodes=1000):
        self.policy = policy
        self.env = env
        self.episodes = episodes
        self.steps = n_steps
        self.critic = critic
        
    def exploit(self,*args):
        for _ in range(0, self.episodes):
            self._run_one_episode()
        
        return self.critic.get_value_function()
    
    def _run_one_episode(self):
        """
        Tabular TD(N) for estimating V(pi) book 7.1 section
        """
        current_timestamp = 0
        final_timestamp = np.inf

        trajectory = []
        current_state_index = self.env.reset()
        while True:
            if current_timestamp < final_timestamp:
                action_index = self.policy.get_action(current_state_index)
                observation = self.env.step(action_index)
                next_state_index = observation[0]
                reward = observation[1]
                done = observation[2]
                trajectory.append((current_state_index,reward))
                if done:
                    final_timestamp = current_timestamp+1
                current_state_index = next_state_index

            updated_timestamp = current_timestamp-self.steps 
            if updated_timestamp >=0:
                self.critic.exploit(trajectory,current_timestamp,updated_timestamp,final_timestamp)
                if updated_timestamp == final_timestamp - 1:
                    break

            current_timestamp += 1



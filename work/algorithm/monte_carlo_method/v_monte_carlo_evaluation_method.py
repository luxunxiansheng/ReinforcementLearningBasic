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

class V_Monte_Carlo_Evaluation_Method:
    def __init__(self, v_table, policy, env, episodes=50000, discount=1.0):
        self.v_table = v_table
        self.policy  = policy 
        self.env = env
        self.episodes = episodes
        self.discount = discount


    def evaluate(self):
        state_count = {}
        for state_index in self.v_table:
            state_count[state_index] = (0, 0.0)
                
        for _ in tqdm(range(0, self.episodes)):
            trajectory = self._run_one_episode()
            R = 0.0
            for state_index, reward in trajectory[::-1]:
                R = reward + self.discount*R
                state_count[state_index] = (state_count[state_index][0] + 1, state_count[state_index][1] + R)
                self.v_table[state_index] = state_count[state_index][1]/state_count[state_index][0]
        
            
                
    def _run_one_episode(self):
        trajectory = []
        current_state_index = self.env.reset()
        while True:
            action_index = self.policy.get_action(current_state_index)
            observation = self.env.step(action_index)
            reward = observation[1]
            trajectory.append((current_state_index, reward))
            done = observation[2]
            if done:
                break
            current_state_index = observation[0]
        
        return trajectory
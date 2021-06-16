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

# TD(lambda) algorithm

class SemiGradientTDLambdaEvaluation:
    def __init__(self, value_function, policy, env, step_size=2e-5, episodes=10000, discount=1.0, lamda=0.5):
        self.env = env
        self.policy = policy
        self.episodes = episodes
        self.discount = discount
        self.step_size = step_size
        self.estimator = value_function
        self.lamda = lamda

    def exploit(self):
        for _ in tqdm(range(0, self.episodes)):
            self._run_one_episode()

    def _run_one_episode(self):
        
        current_state = self.env.reset()
        self.estimator.eligibility = np.zeros(self.estimator.weights.size)

        while True:
            # A
            action_index = self.policy.get_action(current_state)
            observation = self.env.step(action_index)
            
            # S'
            next_state = observation[0]
            
            # R 
            reward = observation[1]
            done = observation[2]

            # set the target
            target = reward + self.discount * self.estimator.predict(next_state)
            self.estimator.update(self.step_size, current_state, target, self.discount, self.lamda)

            if done:
                break

            current_state = next_state

            
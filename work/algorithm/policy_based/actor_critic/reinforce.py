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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from common import ActorBase

class model(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.affine1 = nn.Linear(2, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class PolicyEsitmator():
    def __init__(self,model):
        self.model = model 
    
    def predict(self,state):
        self.model.forward(state)

    def update(self,state,target):
        pass 
        
class REINFORCEActor(ActorBase):
    def __init__(self,policy):
        self.policy = policy
        
    def improve(self,*args):
        state_index=args[0]
        G = args[1]
        self.policy.policy_estimator.update(state_index, G)
        
    def get_behavior_policy(self):
        return self.policy


class REINFORCE:
    def  __init__(self,actor,num_episodes,discount=1.0):
        self.actor= actor
        self.num_episodes = num_episodes
        self.discount = discount

    def improve(self):
        for _ in tqdm(range(0,self.num_episodes)):
            trajectory = self._run_one_episode()
            G = 0.0
            for state_index, _, reward in trajectory[::-1]:
                # The return for current state_action pair
                G = reward + self.discount*G
                self.actor.improve(state_index,G)
                

    def _run_one_episode(self):
        trajectory = []
        current_state_index = self.env.reset(False)
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

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

from collections import namedtuple

import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import PolicyEstimator, ValueEstimator
from policy.policy import Policy

class DeepValueEstimator(ValueEstimator):
    class Model(nn.Module):
        def __init__(self,observation_space_size):
            super().__init__()
            self.affine = nn.Linear(observation_space_size, 128)
            self.dropout = nn.Dropout(p=0.6)
            self.value_head = nn.Linear(128, 1)

        def forward(self, state):
            state = torch.from_numpy(state).float()
            state = F.relu(self.dropout(self.affine(state)))
            state_value = self.value_head(state)
            return state_value

    def __init__(self,observation_space_size):
        self.model = DeepValueEstimator.Model(observation_space_size)
        self.optimizer = optim.Adam(self.model.parameters(),lr =1e-3)
        
    def predict(self,state):
        value = self.model.forward(state)
        return value

    def update(self,*args):
        loss = args[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DeepPolicyEsitmator(PolicyEstimator):
    class Model(nn.Module):
        def __init__(self,observation_space_size,action_space_size):
            super().__init__()
            self.affine = nn.Linear(observation_space_size, 128)
            self.dropout = nn.Dropout(p=0.6)
            self.action_head = nn.Linear(128, action_space_size)
            
        def forward(self, state):
            state = torch.from_numpy(state).float()
            state = F.relu(self.dropout(self.affine(state)))
            actions_prob = F.softmax(self.action_head(state),dim=0)
            return actions_prob
    
    def __init__(self,observation_space_size,action_space_size):
        self.model = DeepPolicyEsitmator.Model(observation_space_size,action_space_size)
        self.optimizer = optim.Adam(self.model.parameters(),lr =1e-3)
    
    def predict(self,state):
        actions_prob=self.model.forward(state)
        return actions_prob

    def update(self,*args):
        loss = args[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ParameterizedPolicy(Policy):
    def __init__(self,policy_estimator):
        self.policy_estimator = policy_estimator
    
    def get_discrete_distribution(self, state):
        return self.policy_estimator.predict(state).detach().numpy()
    
    def get_discrete_distribution_tensor(self, state):
        return self.policy_estimator.predict(state)
    
    def get_action(self, state):
        distribution= self.get_discrete_distribution(state)
        action = np.random.choice(np.arange(len(distribution)), p=distribution)
        return action
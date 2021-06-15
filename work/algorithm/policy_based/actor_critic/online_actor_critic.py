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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algorithm.policy_based.actor_critic.actor_critic_common import ValueEstimator
from common import ActorBase,ExploitatorBase

'''
We take two networks in this case just for clarity but for 
sure it is ok to reduce to one for efficiency.
'''

class ValueEestimator(ValueEstimator):
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
        self.model = ValueEestimator.Model(observation_space_size)
        self.optimizer = optim.Adam(self.model.parameters(),lr =1e-3)
        
    def predict(self,state):
        value = self.model.forward(state)
        return value

    def update(self,*args):
        loss = args[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class OnlineCritic(ExploitatorBase):
    def __init__(self,value_estimator,discount=1.0):
        self.estimator = value_estimator
        self.discount  = discount
    
    def evaluate(self,*args):
        current_state_index = args[0]
        reward = args[1]
        next_state_index = args[2]    
        done = args[3]
        episode =args[4]
        writer = args[5]
        
        value_of_next_state = self.estimator.predict(next_state_index)

        # set the target 
        target = reward + self.discount * value_of_next_state
        input  = self.estimator.predict(current_state_index)
        loss  = F.smooth_l1_loss(input,target)

        if done:
            writer.add_scalar('value_loss',loss,episode)
        
        self.estimator.update(loss)
    
    def get_value_function(self):
        return self.estimator

class PolicyEsitmator:
    class Model(nn.Module):
        def __init__(self,observation_space_size,action_space_size):
            super().__init__()
            self.affine = nn.Linear(observation_space_size, 128)
            self.dropout = nn.Dropout(p=0.6)
            self.action_head = nn.Linear(128, action_space_size)
            
        def forward(self, state):
            state = torch.from_numpy(state).float()
            state = self.affine(state)
            state = self.dropout(state)
            state = F.relu(state)
            state = self.action_head(state)
            actions_prob = F.softmax(state,dim=0)
            return actions_prob
    
    def __init__(self,observation_space_size,action_space_size):
        self.model = PolicyEsitmator.Model(observation_space_size,action_space_size)
        self.optimizer = optim.Adam(self.model.parameters(),lr =1e-3)
    
    def predict(self,state):
        actions_prob=self.model.forward(state)
        return actions_prob

    def update(self,*args):
        loss = args[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class OnlineActor(ActorBase):
    def __init__(self,policy,critic,discount=1.0):
        self.policy = policy 
        self.discount = discount
        self.critic = critic
        
    def improve(self,*args):
        current_state_index = args[0]
        reward = args[1]
        action_prob = args[2]
        next_state_index = args[3]
        done = args[4]
        episode =args[5]
        writer = args[6]

        advantage = torch.tensor(reward) + self.discount* self.critic.estimator.predict(next_state_index) - self.critic.estimator.predict(current_state_index)
        policy_loss = -torch.log(torch.round(action_prob*10**3)/10**3)*advantage.detach()
    
        if done:
            writer.add_scalar('policy_loss',policy_loss.item(),episode)

        self.policy.estimator.update(policy_loss)

    def get_behavior_policy(self):
        return self.policy

class OnlineCriticActor:
    EPS = np.finfo(np.float32).eps.item()
    MAX_STEPS = 500000
    def  __init__(self,critic,actor,env,num_episodes):
        self.critic=critic 
        self.actor= actor
        self.env = env
        self.num_episodes = num_episodes
        self.writer = SummaryWriter()
            
    def improve(self):
        for episode in tqdm(range(0, self.num_episodes)):
            self._run_one_episode(episode)

    def _run_one_episode(self, episode):
        # S
        current_state_index = self.env.reset()

        for _ in tqdm(range(0, OnlineCriticActor.MAX_STEPS)):
            # A
            current_action_index = self.actor.get_behavior_policy().get_action(current_state_index)
            observation = self.env.step(current_action_index)
            self.env.render()

            # R
            reward = observation[1]
            done = observation[2]

            # S'
            next_state_index = observation[0]

            self.critic.evaluate(current_state_index,reward,next_state_index,done,episode,self.writer)
            
            action_prob=self.actor.get_behavior_policy().get_discrete_distribution_tensor(current_state_index)[current_action_index]
            self.actor.improve(current_state_index,reward,action_prob,next_state_index,done,episode,self.writer)

            if done:
                break

            current_state_index = next_state_index


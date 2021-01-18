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



import copy
from collections import namedtuple
import numpy as np 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algorithm.policy_based.actor_critic.reinforce import model
from algorithm.policy_based.actor_critic.actor_critic_common import ValueEstimator
from common import ActorBase,CriticBase

'''
It is O.K. to combine two newtworks into one for efficiency. We take two just for clarity 
'''

class PolicyEsitmator:
    class Model(nn.Module):
        def __init__(self,observation_space_size,action_space_size):
            super().__init__()
            self.affine = nn.Linear(observation_space_size, 128)
            self.dropout = nn.Dropout(p=0.6)
            self.action_head = nn.Linear(128, action_space_size)
            
        def forward(self, state):
            state = torch.from_numpy(state).float()
            state = F.relu(self.dropout(self.affine(state)))
            actions_prob = F.softmax(self.action_head(state))
            return actions_prob
    
    def __init__(self,observation_space_size,action_space_size):
        self.model = PolicyEsitmator.Model(observation_space_size,action_space_size)
        self.optimizer = optim.Adam(self.model.parameters(),lr =3e-2)
    
    def predict(self,state):
        actions_prob=self.model.forward(state)
        return actions_prob.detach().numpy()

    def update(self,*args):
        policy_losses = args[0]
        self.optimizer.zero_grad()
        policy_losses = torch.stack(policy_losses).sum()
        policy_losses.backward()
        self.optimizer.step()
        
class Actor(ActorBase):
    def __init__(self,policy,discount=1.0):
        self.policy = policy 
        self.discount = discount
        
    def improve(self,*args):
        trajectory = args[0]
    
        G = 0.0
        log_action_probs=[]
        state_values=[]
        returns = []
        policy_losses=[]

        # reduce the variance  
        for _,state_value,_,action_prob,reward in trajectory[::-1]:
            # estimate the sate value with Monte Carlo target   
            G = reward + self.discount*G
            log_action_probs.insert(0,np.log(action_prob))
            state_values.insert(0,state_value)
            returns.insert(0,G)
        returns = torch.tensor(returns)
        returns = (returns-returns.mean())/(returns.std()+CriticActor.EPS)

        for log_action_prob,value, G in zip(log_action_probs,state_values,returns):
            advantage = G - value
            policy_losses.append(-log_action_prob*advantage)

        self.policy.estimator.update(policy_losses)

    def get_behavior_policy(self):
        return self.policy

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
        self.optimizer = optim.Adam(self.model.parameters(),lr =3e-2)
        
    def predict(self,state):
        value = self.model.forward(state)
        return value

    def update(self,*args):
        value_losses = args[0]
        self.optimizer.zero_grad()
        value_loss = torch.stack(value_losses).sum()
        value_loss.backward()
        self.optimizer.step()

class Critic(CriticBase):
    def __init__(self,value_estimator,discount=1.0):
        self.estimator = value_estimator
        self.discount  = discount
    
    def evaluate(self,*args):
        state_values=[]
        returns = []
        value_losses=[]
        
        trajectory = args[0]

        G = 0.0 
        # reduce the variance  
        for _,state_value,_,_,reward in trajectory[::-1]:
            G = reward + self.discount*G
            state_values.insert(0,state_value)
            returns.insert(0,G)
        returns = torch.tensor(returns)
        returns = (returns-returns.mean())/(returns.std()+CriticActor.EPS)

        for value, G in zip(state_values,returns):
            value_losses.append(F.smooth_l1_loss(value,torch.tensor([G])))

        self.estimator.update(value_losses)
    
    def get_value_function(self):
        return self.estimator

class CriticActor:
    EPS = np.finfo(np.float32).eps.item()
    def  __init__(self,critic,actor,env,num_episodes):
        self.critic=critic 
        self.actor= actor
        self.env = env
        self.num_episodes = num_episodes
        
    def improve(self):
        for _ in tqdm(range(0,self.num_episodes)):
            trajectory = self._run_one_episode()    
            
            self.critic.evaluate(trajectory)

            trajectory = self._run_one_episode()
            self.actor.improve(trajectory)
        
    def _run_one_episode(self):
        trajectory = []
        current_state = self.env.reset()
        while True:
            action_index = self.actor.get_behavior_policy().get_action(current_state)
            action_prob=self.actor.get_behavior_policy().get_discrete_distribution(current_state)[action_index]
            state_value  = self.critic.get_value_function().predict(current_state)
            observation = self.env.step(action_index)
            reward = observation[1]
            trajectory.append((current_state,state_value,action_index,action_prob,reward))
            done = observation[2]
            if done:
                break
            current_state = observation[0]
        return trajectory

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

# http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from algorithm.policy_based.actor_critic.actor_critic_common import ValueEstimator
from common import ActorBase, CriticBase
from lib.utility import SharedAdam
from torch.utils.tensorboard import SummaryWriter


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
        self.optimizer = SharedAdam(self.model.parameters(),lr =1e-3)
        
    def predict(self,state):
        value = self.model.forward(state)
        return value

    def update(self,*args):
        loss = args[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class OnlineCritic(CriticBase):
    def __init__(self,value_estimator,discount=1.0):
        self.estimator = value_estimator
        self.discount  = discount
    
    def evaluate(self,*args):
        current_state_index = args[0]
        reward = args[1]
        next_state_index = args[2]    

        value_of_next_state = self.estimator.predict(next_state_index)

        # set the target 
        target = reward + self.discount * value_of_next_state
        input  = self.estimator.predict(current_state_index)
        loss  = F.smooth_l1_loss(input,target)
        
        self.estimator.update(loss)
    
    def get_value_function(self):
        return self.estimator


class Model(nn.Module):
    def __init__(self,observation_space_size,action_space_size):
        super().__init__()
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        
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

class GlobalPolicyEsitmator:
    def __init__(self,observation_space_size,action_space_size):
        self.model = Model(observation_space_size,action_space_size)
        self.optimizer = SharedAdam(self.model.parameters(),lr =1e-3)

    def update(self,*args):
        loss = args[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class OnlineGlobalActor(ActorBase):
    def __init__(self,policy,critic,discount=1.0):
        self.policy = policy 
        self.discount = discount
        self.critic = critic
        
    def improve(self,*args):
        current_state_index = args[0]
        reward = args[1]
        action_prob = args[2]
        next_state_index = args[3]

        advantage = torch.tensor(reward) + self.discount* self.critic.estimator.predict(next_state_index) - self.critic.estimator.predict(current_state_index)
        policy_loss = -torch.log(torch.round(action_prob*10**3)/10**3)*advantage.detach()

        self.policy.estimator.update(policy_loss)

    def get_behavior_policy(self):
        return self.policy


class LocalPolicyEstimator:
    def __init__(self,global_model):
        self.model = Model(global_model.observation_space_size,global_model.action_space_size)
        self.model.load_state_dict(global_model.state_dict())
    
    def predict(self,state):
        actions_prob=self.model.forward(state)
        return actions_prob


class OnlineLocalActor(ActorBase):
    def __init__(self,policy,critic,discount=1.0):
        self.policy = policy 
        self.discount = discount
        self.critic = critic
        
    def improve(self,*args):
        current_state_index = args[0]
        reward = args[1]
        action_prob = args[2]
        next_state_index = args[3]

        advantage = torch.tensor(reward) + self.discount* self.critic.estimator.predict(next_state_index) - self.critic.estimator.predict(current_state_index)
        policy_loss = -torch.log(torch.round(action_prob*10**3)/10**3)*advantage.detach()

        self.policy.estimator.update(policy_loss)

    def get_behavior_policy(self):
        return self.policy

class OnlineA3C:
    EPS = np.finfo(np.float32).eps.item()
    MAX_STEPS = 500000
    NUM_PROCESSES = 8
    def  __init__(self,global_critic,local_critic,global_actor,local_actor,env,num_episodes):
        self.global_critic= global_critic 
        self.local_critic = local_critic
        self.global_actor= global_actor
        self.local_actor = local_actor
        self.env = env
        self.num_episodes = num_episodes
        self.writer = SummaryWriter()

    def improve(self):
        processes=[]
        for rank in range(0, OnlineA3C.NUM_PROCESSES):
            p = mp.Process(target=OnlineA3C._improve, args=(rank, self.num_episodes,self.global_critic, self.local_critic, self.global_actor,self.local_actor))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


    @staticmethod
    def _improve(env,num_episodes,global_critic, local_critic, global_actor,local_actor):
        for episode in range(0, num_episodes):
            OnlineA3C._run_one_episode(env,global_critic,local_critic, global_actor,local_actor)
    
    @staticmethod
    def _run_one_episode(env,global_critic, local_critic, global_actor,local_actor):
        local_critic.sync(global_critic)
        local_actor.sync(global_actor)

        new_env = env()
        # S
        current_state_index = new_env.reset()

        for _ in range(0, OnlineA3C.MAX_STEPS):
            # A
            current_action_index = local_actor.get_behavior_policy().get_action(current_state_index)
            observation = new_env.step(current_action_index)
            

            # R
            reward = observation[1]
            done = observation[2]

            # S'
            next_state_index = observation[0]

            local_critic.evaluate(current_state_index,reward,next_state_index)
            
            action_prob=local_actor.get_behavior_policy().get_discrete_distribution_tensor(current_state_index)[current_action_index]
            global_actor.improve(current_state_index,reward,action_prob,next_state_index,)

            if done:
                break

            current_state_index = next_state_index

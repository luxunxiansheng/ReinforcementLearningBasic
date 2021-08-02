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

import os
from posix import environ

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from algorithm.policy_based.actor_critic.actor_critic_common import ValueEstimator
from common import ActorBase, Agent, ExplorerBase, CriticBase, PolicyEstimator
from lib.utility import SharedAdam
from policy.policy import ParameterizedPolicy


class ValueModel(nn.Module):
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

class GlobalValueEestimator(ValueEstimator):
    def __init__(self,observation_space_size):
        self.model = ValueModel(observation_space_size)
        self.model.share_memory()
        self.optimizer = SharedAdam(self.model.parameters(),lr =1e-3)
        self.optimizer.share_memory()

    def ensure_shared_grads(model, shared_model):
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def predict(self, state):
        return super().predict(state)        
    
    def update(self, *args):
        return super().update(*args)

class LocalValueEestimator(ValueEstimator):
    def __init__(self,observation_space_size):
        self.model = ValueModel(observation_space_size)
        
    def predict(self,state):
        value = self.model.forward(state)
        return value
    
    def update(self,*args):
        return super().update(*args)
        
class LocalBatchCritic(CriticBase):
    EPS = np.finfo(np.float32).eps.item()
    def __init__(self,local_value_estimator, global_value_estimator,discount=1.0):
        self.local_value_estimator =  local_value_estimator
        self.global_value_estimator = global_value_estimator
        self.discount  = discount
    
    # sync local to global
    def _sync_to_global(self,model, global_model):
        for param, shared_param in zip(model.parameters(),global_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def evaluate(self,*args): 
        trajectory = args[0]
        
        state_values=[]
        returns = []
        value_losses=[]

        G = 0.0 
        
        # reduce the variance  
        for _,state_value,_,_,_,reward in trajectory[::-1]:
            G = reward + self.discount*G
            state_values.insert(0,state_value)
            returns.insert(0,G)

        returns = torch.tensor(returns)
        returns = (returns-returns.mean())/(returns.std()+LocalBatchCritic.EPS)

        for value, G in zip(state_values,returns):
            value_losses.append(F.smooth_l1_loss(value,torch.tensor([G])))

        total_loss = torch.stack(value_losses).sum()
        self.global_value_estimator.optimizer.zero_grad()
        total_loss.backward()
        self._sync_to_global(self.local_value_estimator.model,self.global_value_estimator.model)
        self.global_value_estimator.optimizer.step()
        
    def get_value_function(self):
        return self.local_value_estimator
    
class PolicyModel(nn.Module):
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


class GlobalPolicyEsitmator(PolicyEstimator):
    def __init__(self,observation_space_size,action_space_size):
        self.model = PolicyModel(observation_space_size,action_space_size)
        self.model.share_memory()
        self.optimizer = SharedAdam(self.model.parameters(),lr =1e-3)
        self.optimizer.share_memory()
    
    def predict(self, state):
        return super().predict(state)
    
    def update(self, *args):
        return super().update(*args)
    
class LocalPolicyEsitmator(PolicyEstimator):
    def __init__(self,observation_space_size,action_space_size):
        self.model = PolicyModel(observation_space_size,action_space_size)
    
    def predict(self,state):
        actions_prob=self.model.forward(state)
        return actions_prob
    
    def update(self,*args):
        return super().update(*args)

class LocalBatchPolicyGridentExplorer(ExplorerBase):
    EPS = np.finfo(np.float32).eps.item()
    ENTROY_BETA = 0.0
    
    def __init__(self,local_policy,global_policy,discount=1.0):
        self.local_policy = local_policy 
        self.global_policy = global_policy
        self.discount = discount
    
    def _syn_to_global(self,model, shared_model):
        for param, shared_param in zip(model.parameters(),shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def explore(self,*args):
        trajectory = args[0]
        G = 0.0
        log_action_probs=[]
        entroys=[]
        state_values=[]
        returns = []
        policy_losses=[]

        # reduce the variance  
        for _,state_value,_,action_prob,entroy,reward in trajectory[::-1]:
            # estimate the sate value with Monte Carlo target   
            G = reward + self.discount*G
            log_action_probs.insert(0,torch.log(action_prob))
            entroys.insert(0,entroy)
            state_values.insert(0,state_value)
            returns.insert(0,G)
        returns = torch.tensor(returns)
        returns = (returns-returns.mean())/(returns.std()+LocalBatchPolicyGridentExplorer.EPS)

        for log_action_prob,state_value, G in zip(log_action_probs,state_values,returns):
            advantage = G - state_value.detach()
            policy_losses.append(-log_action_prob*advantage)
        
        total_policy_loss = torch.stack(policy_losses).sum()
        total_loss =  total_policy_loss - LocalBatchPolicyGridentExplorer.ENTROY_BETA*torch.stack(entroys).sum()
        
        self.global_policy.estimator.optimizer.zero_grad()
        total_loss.backward()
        self._syn_to_global(self.local_policy.estimator.model,self.global_policy.estimator.model)
        self.global_policy.estimator.optimizer.step()

    def get_behavior_policy(self):
        return self.local_policy

class LocalActor(ActorBase):
    EPS = np.finfo(np.float32).eps.item()
    MAX_STEPS = 500000
    
    def  __init__(self,env,local_critic,local_explorer):
        self.env = env
        self.local_critic = local_critic
        self.local_explorer = local_explorer
        
    def _create_env(env):
        env_class = env.__class__
        return env_class()

    def _run_one_episode(self):
        trajectory = []
        current_state = self.env.reset()
        
        for _ in range(0,LocalActor.MAX_STEPS):
            action_index = self.local_explorer.get_behavior_policy().get_action(current_state)
            distribution = self.local_explorer.get_behavior_policy().get_discrete_distribution_tensor(current_state)
            entropy = torch.distributions.Categorical(distribution).entropy()
            action_prob = distribution[action_index]
            state_value  = self.local_critic.get_value_function().predict(current_state)
            observation = self.env.step(action_index)
            reward = observation[1]
            self.env.render()
            trajectory.append((current_state,state_value,action_index,action_prob,entropy,reward))
            done = observation[2]
            if done:
                break
            current_state = observation[0]
        return trajectory

    def act(self, *args):
        trajectory = self._run_one_episode()    
        if len(trajectory)< LocalActor.MAX_STEPS:
            self.local_critic.evaluate(trajectory)
            self.local_explorer.explore(trajectory)

class BatchA3CAgent(Agent):
    EPS = np.finfo(np.float32).eps.item()
    NUM_PROCESS = 8

    def  __init__(self,env,num_episodes):
        self.env = env
        self.num_episodes = num_episodes
        self.global_value_esitmator = GlobalValueEestimator(env.observation_space_size)
        self.global_policy_esitmator = GlobalPolicyEsitmator(env.observation_space_size,env.action_space_size)
        

    def learn(self):

    
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

        processes = []

        for _ in range(0,BatchA3CAgent.NUM_PROCESS):
            p = mp.Process(target=BatchA3CAgent._explore,args=(self.global_value_esitmator,self.global_policy_esitmator,self.env,self.num_episodes))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

